import os
import dotenv

dotenv.load_dotenv()

from app_config import apply_runtime_environment, resolve_output_path

apply_runtime_environment()

import sys, json, threading, torch, time, gc, random, cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import io, shutil, subprocess
from planner_runtime import clear_accelerator_cache, get_active_plan, is_exact_fast_path, mark_success, preview_device, run_guarded, workload_from_diffusion_gui
from platform_utils import ffmpeg_executable, open_path

class DiffusionGUI:

    def __init__(self, args):

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        for arg in args:
            setattr(self, arg, args[arg])

        self.image_folder = resolve_output_path(self.image_folder)
        os.makedirs(self.image_folder, exist_ok=True)

        config_prompt = None
        config_source_image_path = None
        prompts_path = os.path.join(self.image_folder, "prompts.json")
        if os.path.isfile(prompts_path):
            with open(prompts_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            config_prompt = cfg.get("prompt")
            config_source_image_path = cfg.get("source_image_path")

        self.app = ctk.CTk()
        self.app.title(f"{self.title}")
        self.app.geometry("1200x800")

        self.prompt = (hasattr(self, 'prompt') and self.prompt) or config_prompt or (
            'A cozy modern apartment bedroom with soft morning light through sheer curtains, '
            'a fluffy cat lounging on a neatly made bed, cinematic composition, detailed, sharp focus'
        )
        self.negative_prompt = hasattr(self, 'negative_prompt') and self.negative_prompt or (
            "low quality, blurry, distorted, unnatural anatomy"
        )

        self.batch_size = hasattr(self, 'batch_size') and self.batch_size or 1
        self.width = hasattr(self, 'width') and self.width or 1280
        self.height = hasattr(self, 'height') and self.height or 768
        self.guidance_scale = hasattr(self, 'guidance_scale') and self.guidance_scale or 0.0
        self.num_inference_steps = hasattr(self, 'num_inference_steps') and self.num_inference_steps or 9
        self.num_images_per_prompt = hasattr(self, 'num_images_per_prompt') and self.num_images_per_prompt or 3
        self.callback_on_step_end = getattr(self, "callback_on_step_end", True)
        self.max_sequence_length = hasattr(self, 'max_sequence_length') and self.max_sequence_length or 512
        self.true_cfg_scale = self.guidance_scale
        self.generate_running = False
        self.stop_requested = False
        self.preview_busy = False
        self.preview_image = None
        self.preview_ctk = None
        self.left_panel = None
        self.preview_vae = None
        self.pipe = None
        self.negative_prompt_embeds = None
        self.model_lock = threading.Lock()
        self.model_loading = False
        self.cleanup_requested = False
        self.image_refs = []
        self.preview_time = time.time()
        self.active_seed = None
        self.active_plan = get_active_plan()
        if self.active_plan and not is_exact_fast_path(self.active_plan):
            self.vram_estimator = None
        if self.active_plan.get("restore_workload"):
            for key, value in self.active_plan.get("workload", {}).items():
                if hasattr(self, key) and isinstance(value, (int, float, str, bool)):
                    setattr(self, key, value)
        self.plan_status_label = None
        self.plan_phase = "Starting"
        self.plan_validated = False

        self.prompt_pipe = None
        self.prompt_tokenizer = None
        self.prompt_model_lock = threading.Lock()
        self.prompt_model_loading = False
        self.ai_prompt_btn = None
        self.use_prompt_model = getattr(self, "use_prompt_model", True)

        self.supports_source_image = getattr(self, "supports_source_image", False)
        self.source_image_path = getattr(self, "source_image_path", "") or config_source_image_path or ""
        self.source_image_pil = None
        self.source_image_ctk = None
        self.source_img_label = None
        self.source_path_label = None
        self.auto_size_from_source_on_select = getattr(self, "auto_size_from_source_on_select", True)

        self.gallery_title = getattr(self, "gallery_title", "Images")
        self.ipp_label = getattr(self, "ipp_label", "Images per prompt")
        self.ipp_attr = getattr(self, "ipp_attr", "num_images_per_prompt")
        self.show_ipp = getattr(self, "show_ipp", True)

        self.show_batch_size = getattr(self, "show_batch_size", self.gallery_title != "Videos")
        self.show_repeat_count = getattr(self, "show_repeat_count", self.gallery_title == "Videos")

        self.extra_params = getattr(self, "extra_params", [])
        self.repeat_count = getattr(self, "repeat_count", 1)

        if self.show_repeat_count:
            has_repeat = any(spec[0] == "repeat_count" for spec in self.extra_params)
            if not has_repeat:
                self.extra_params = list(self.extra_params) + [("repeat_count", "Repeat", int)]

        self.extra_entries = {}
        self.vram_estimator = getattr(self, "vram_estimator", None)
        self.vram_label = None
        self.vram_update_id = None

        self.source_size_multiple = getattr(self, "source_size_multiple", 1)
        self.source_max_dim = getattr(self, "source_max_dim", 0)
        self.lock_source_aspect_ratio = getattr(self, "lock_source_aspect_ratio", False)
        self.source_aspect_ratio = 0.0
        self.size_lock_busy = False

        self.last_locked_width = int(self.width)
        self.last_locked_height = int(self.height)

        self.prompt_model_prompt = (
            'Describe an imagined image about {idea} using only visible details. Write one tight paragraph that feels like a camera snapshot, '
            'with a concrete subject, setting, composition, lighting, materials, colours, atmosphere, and rich visual detail.'
        )

        self.progress = 0.0
        self.progress_step = 0
        self.progress_total = 0
        self.gpu_percent = 0
        self.thermal_throttling = False
        self.thermal_poll_ms = 30000
        self.thermal_poll_id = None

        self.gpu_handles = []
        self.nvml = None
        if torch.cuda.is_available():
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                    for i in range(pynvml.nvmlDeviceGetCount())
                ]
            except Exception as error:
                print(f"NVML unavailable: {error}")

        self.model_load_started = False
    
    def update_plan_status(self, phase=None):
        if phase:
            self.plan_phase = phase
        if self.plan_status_label is None:
            return
        plan_text = self.active_plan.get("status_text", "Legacy loading path") if self.active_plan else "Legacy loading path"
        self.plan_status_label.configure(text=f"Plan: {plan_text} · {self.plan_phase}")

    def load_model_guarded(self):
        self.app.after(0, self.update_plan_status, "Loading model")
        workload = workload_from_diffusion_gui(self)
        result = run_guarded(self.active_plan, "model_loading", workload, self.load_model)
        self.app.after(0, self.update_plan_status, "Ready")
        return result

    def generate_guarded(self):
        workload = workload_from_diffusion_gui(self)
        self.app.after(0, self.update_plan_status, "Generating")
        return run_guarded(self.active_plan, "inference", workload, self.generate)

    def set_prompt_text(self, text):
        self.prompt_box.delete("1.0", "end")
        self.prompt_box.insert("1.0", text)
        self.sync_params()
    
    def start_ai_prompt(self):
        if self.ai_prompt_btn:
            self.ai_prompt_btn.configure(text="...", state="disabled")
        seed_text = self.prompt_box.get("1.0", "end").strip()
        threading.Thread(target=self.generate_ai_prompt, args=(seed_text,), daemon=True).start()
    
    def finish_ai_prompt(self):
        if self.ai_prompt_btn:
            self.ai_prompt_btn.configure(text="AI prompt", state="normal")
    
    def generate_ai_prompt(self, seed_text=''):
        if not self.use_prompt_model:
            return self.app.after(0, self.finish_ai_prompt)

        self.ensure_model_loaded()

        try:
            self.load_prompt_model()

            with self.prompt_model_lock:
                prompt_pipe = self.prompt_pipe
                prompt_tokenizer = self.prompt_tokenizer

            if prompt_pipe is None or prompt_tokenizer is None:
                return

            messages = [
                {
                    "role": "system",
                    "content": "You write concise, vivid prompts for image and video generation.",
                },
                {
                    "role": "user",
                    "content": self.prompt_model_prompt.format(idea=(seed_text or "a scene")),
                },
            ]
            prompt_text = prompt_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            out = prompt_pipe(
                prompt_text,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.85,
                top_p=0.95,
                pad_token_id=prompt_tokenizer.eos_token_id,
                return_full_text=False,
            )

            text = self.pick_best_paragraph(out[0]["generated_text"])
            if not text:
                text = out[0]["generated_text"].strip()

            self.app.after(0, self.set_prompt_text, text)
        except Exception as e:
            print(f"AI prompt error: {e}")
        finally:
            self.app.after(0, self.finish_ai_prompt)

    def pick_best_paragraph(self, text):
        bad = ("instruction:", "system:", "user:", "assistant:", "###", "q:", "a:", "write a")
        best, best_score = "", -1e9
        for s in (p.strip() for p in text.replace("\r", "").split("\n") if p.strip()):
            low = s.lower()
            if len(s) < 40:
                continue
            score = len(s) + 20 * sum(c in ".,;:!?" for c in s) - 300 * any(b in low for b in bad)
            if score > best_score:
                best, best_score = s, score
        return best.strip()
    
    def on_gallery_item_double_click(self, event, path):
        open_path(path)
    
    def clear_cuda_cache(self):
        clear_accelerator_cache()
    
    def open_output_folder(self):
        os.makedirs(self.image_folder, exist_ok=True)
        open_path(self.image_folder)
    
    def open_cachelight(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.Popen([sys.executable, os.path.join(base_dir, "cachelight.py")], cwd=base_dir)

    def open_image_editor(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(base_dir, "img_editor.py")
        model_id = getattr(self, "launcher_model_id", "")

        cmd = [
            sys.executable,
            script,
            "--model-id",
            model_id,
            "--output-folder",
            os.path.abspath(self.image_folder),
        ]

        if self.source_image_path:
            cmd += ["--source-image", self.source_image_path]

        subprocess.Popen(cmd, cwd=base_dir)
    
    def unload_model(self):
        with self.model_lock:
            pipe = self.pipe
            vae = self.preview_vae

            self.pipe = None
            self.preview_vae = None
            self.model_loading = False

        with self.prompt_model_lock:
            prompt_pipe = self.prompt_pipe
            prompt_tokenizer = self.prompt_tokenizer

            self.prompt_pipe = None
            self.prompt_tokenizer = None
            self.prompt_model_loading = False

        if prompt_pipe is not None:
            del prompt_pipe
        if prompt_tokenizer is not None:
            del prompt_tokenizer
        if vae is not None:
            del vae
        if pipe is not None and hasattr(pipe, "close"):
            pipe.close()
        if pipe is not None:
            del pipe
        self.clear_cuda_cache()

    def ensure_model_loaded(self):
        if self.pipe is not None:
            return

        if self.model_loading:
            while self.model_loading and self.pipe is None:
                time.sleep(0.2)
            return

        self.load_model_guarded()

    def sync_params(self, event=None):
        self.prompt = self.prompt_box.get("1.0", "end").strip()
        self.negative_prompt = self.neg_box.get("1.0", "end").strip()

        if self.show_batch_size and hasattr(self, "batch_entry") and self.batch_entry is not None:
            self.batch_size = int(self.batch_entry.get())
        else:
            self.batch_size = 1

        self.width = int(self.width_entry.get())
        self.height = int(self.height_entry.get())

        self.guidance_scale = float(self.guidance_entry.get())
        self.true_cfg_scale = self.guidance_scale
        self.num_inference_steps = int(self.steps_entry.get())
        
        if self.show_ipp and hasattr(self, "ipp_entry") and self.ipp_entry is not None:
            setattr(self, self.ipp_attr, int(self.ipp_entry.get()))

        for name, cast, entry in self.extra_entries.values():
            text = entry.get().strip()

            if name == "seed" and text == "":
                setattr(self, name, "")
            elif name == "repeat_count" and text == "":
                setattr(self, name, 1)
            else:
                setattr(self, name, cast(text))

        self.update_vram_estimate()

    def get_entry_number(self, entry, cast):
        text = entry.get().strip()

        if cast is int:
            if not text.isdigit():
                return None
        else:
            number = text.lstrip("-")
            if not number.replace(".", "", 1).isdigit():
                return None

        return cast(text)

    def schedule_vram_update(self, event=None):
        if self.vram_estimator is None or self.vram_label is None:
            return

        if self.vram_update_id is not None:
            self.app.after_cancel(self.vram_update_id)

        self.vram_update_id = self.app.after(150, self.update_vram_estimate)

    def update_vram_estimate(self):
        self.vram_update_id = None

        if self.vram_estimator is None or self.vram_label is None:
            return

        width = self.get_entry_number(self.width_entry, int)
        height = self.get_entry_number(self.height_entry, int)
        guidance_scale = self.get_entry_number(self.guidance_entry, float)
        frames_entry = self.extra_entries.get("num_frames")

        if width is None or height is None or guidance_scale is None or frames_entry is None:
            return

        frames = self.get_entry_number(frames_entry[2], int)

        if frames is None or width <= 0 or height <= 0 or frames <= 0:
            return

        gpu0_gb, gpu1_gb = self.vram_estimator.estimate(width, height, frames, guidance_scale)
        self.vram_label.configure(text=f"GPU0: {gpu0_gb:.1f}GB, GPU1: {gpu1_gb:.1f}GB")
    
    def get_seed(self):
        seed = str(getattr(self, "seed", "")).strip()

        if seed:
            return int(seed)

        return random.randrange(1, 2**63)
    
    def has_fixed_seed(self):
        return bool(str(getattr(self, "seed", "")).strip())
    
    def get_repeat_count(self):
        if self.has_fixed_seed():
            return 1

        return max(1, int(getattr(self, "repeat_count", 1) or 1))
    
    def get_total_runs(self):
        if self.gallery_title == "Videos":
            return self.get_repeat_count()

        return max(1, int(self.batch_size))
    
    def prepare_run_seed(self):
        self.active_seed = self.get_seed()
        self.seed_everything(self.active_seed)
        return self.active_seed
    
    def seed_everything(self, seed):
        torch.manual_seed(int(seed))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(int(seed))

    def get_generator(self, device="cpu"):
        seed = self.active_seed if self.active_seed is not None else self.get_seed()
        print(f"seed {seed}")
        return torch.Generator(device=device).manual_seed(int(seed))
    
    def video_thumbnail(self, path):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        return img

    def refresh_gallery(self):
        os.makedirs(self.image_folder, exist_ok=True)
        exts = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".mkv", ".avi")

        files = []
        for f in os.listdir(self.image_folder):
            if f.lower().endswith(exts):
                files.append(os.path.join(self.image_folder, f))

        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        for w in self.gallery.winfo_children():
            w.destroy()

        self.image_refs = []

        for p in files:
            low = p.lower()
            img = None

            if low.endswith((".png", ".jpg", ".jpeg", ".webp")):
                with Image.open(p) as im:
                    img = im.copy()
            else:
                img = self.video_thumbnail(p)

            if img is None:
                item = ctk.CTkLabel(self.gallery, text=os.path.basename(p))
                item.grid(sticky="ew", padx=10, pady=10)
                continue

            img.thumbnail((520, 520))
            w, h = img.size
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))

            item = ctk.CTkLabel(self.gallery, text=os.path.basename(p), image=ctk_img, compound="top")
            item.bind("<Double-Button-1>", lambda e, path=p: self.on_gallery_item_double_click(e, path))
            item.grid(sticky="ew", padx=10, pady=10)

            self.image_refs.append(ctk_img)
    
    def set_preview_image(self, img):
        self.preview_image = img
        img.thumbnail((520, 300))
        w, h = img.size
        self.preview_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.preview_img.configure(image=self.preview_ctk, text="")
        self.preview_time_label.configure(text=f"Last update: {time.strftime('%H:%M:%S')}")
    
    def save_preview(self, latents):
        if self.preview_busy:
            return
        if self.preview_vae is None:
            return

        self.preview_busy = True
        try:
            with torch.inference_mode():
                x = latents.detach()
                if x.ndim == 5:
                    b, n, h, w, c = x.shape
                    x = x.reshape(b * n, h, w, c)
                    x = x.permute(0, 3, 1, 2).contiguous()

                elif x.ndim == 3:
                    x = self.unpack_flux_latents(x, self.height, self.width)

                x = x[:1].to(preview_device(self.active_plan), dtype=self.preview_vae.dtype)
                x = x / self.preview_vae.config.scaling_factor
                if hasattr(self.preview_vae.config, "shift_factor"):
                    x = x + self.preview_vae.config.shift_factor

                y = self.preview_vae.decode(x).sample
                y = (y / 2 + 0.5).clamp(0, 1)
                y = y[0].permute(1, 2, 0).float().cpu().numpy()
                img = Image.fromarray((y * 255).astype("uint8"))
                self.app.after(0, self.set_preview_image, img)
        finally:
            self.preview_busy = False
    
    def unpack_flux_latents(self, latents, height, width):
        vae_scale_factor = 8
        h = 2 * (int(height) // (vae_scale_factor * 2))
        w = 2 * (int(width) // (vae_scale_factor * 2))

        b, num_patches, ch = latents.shape
        x = latents.view(b, h // 2, w // 2, ch // 4, 2, 2)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(b, ch // 4, h, w)
        return x
    
    def get_gpu_percent(self):
        if self.nvml is None:
            return 0
        vals = []
        for handle in self.gpu_handles:
            vals.append(self.nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        return max(vals) if vals else 0
    
    def get_thermal_throttling(self):
        if self.nvml is None:
            return False
        thermal_reasons = (
            self.nvml.nvmlClocksThrottleReasonHwThermalSlowdown
            | self.nvml.nvmlClocksThrottleReasonSwThermalSlowdown
        )

        for handle in self.gpu_handles:
            reasons = self.nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            if reasons & thermal_reasons:
                return True

        return False
    
    def poll_util(self):
        while self.generate_running:
            self.gpu_percent = self.get_gpu_percent()
            self.app.after(0, self.update_util_widgets)
            time.sleep(1)
    
    def poll_thermal(self):
        self.thermal_poll_id = None

        if not self.generate_running:
            self.thermal_throttling = False
            self.update_util_widgets()
            return

        self.thermal_throttling = self.get_thermal_throttling()
        self.update_util_widgets()
        self.thermal_poll_id = self.app.after(self.thermal_poll_ms, self.poll_thermal)
    
    def update_util_widgets(self):
        text = f"GPU {self.gpu_percent}%"
        text_color = "#d0d0d0"

        if self.thermal_throttling:
            text += "!"
            text_color = "#ffb000"

        self.gpu_pill_top.configure(text=text, text_color=text_color)
    
    def update_progress_widgets(self):
        self.progress_bar.set(self.progress)
        self.progress_text.configure(text=f"{self.progress_step} / {self.progress_total}")
    
    def widget_inside(self, w, parent):
        if isinstance(w, str):
            try:
                w = self.app.nametowidget(w)
            except:
                return False

        while w:
            if w == parent:
                return True
            w = getattr(w, "master", None)

        return False
    
    def on_mousewheel(self, event):
        canvas = None

        if self.gallery and self.widget_inside(event.widget, self.gallery._parent_canvas):
            canvas = self.gallery._parent_canvas

        if self.left_panel and self.widget_inside(event.widget, self.left_panel._parent_canvas):
            canvas = self.left_panel._parent_canvas

        if not canvas:
            return

        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
            return
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
            return

        delta = getattr(event, "delta", 0)
        if delta:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")
    
    def load_video_thumb(self, path):
        ffmpeg = ffmpeg_executable()
        if not ffmpeg:
            return None

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", path,
            "-frames:v", "1",
            "-vf", "select=eq(n\\,0)",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]

        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not p.stdout:
            return None

        img = Image.open(io.BytesIO(p.stdout))
        return img.convert("RGB")
    
    def on_gallery_scroll(self, event):
        if event.num == 4:
            self.gallery._parent_canvas.yview_scroll(-1, "units")
            return
        if event.num == 5:
            self.gallery._parent_canvas.yview_scroll(1, "units")
            return
        self.gallery._parent_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
    
    def save_prompts_json(self):
        self.sync_params()
        data = {
            "saved_at_unix": time.time(),
            "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "diffusers",
            "image_folder": self.image_folder,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "batch_size": self.batch_size,
            "width": self.width,
            "height": self.height,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "max_sequence_length": self.max_sequence_length,
            "callback_on_step_end": self.callback_on_step_end,
            "active_seed": getattr(self, "active_seed", None),
            "cpu_initial_seed": int(torch.initial_seed()),
        }

        if self.supports_source_image:
            data["source_image_path"] = self.source_image_path

        for k in getattr(self, "pipeline_args", []):
            data["pipe_" + k] = getattr(self, k, None)

        for k in getattr(self, "extra_entries", {}).keys():
            data["extra_" + k] = getattr(self, k, None)

        if torch.cuda.is_available():
            for i, g in enumerate(torch.cuda.default_generators):
                data["cuda_initial_seed_" + str(i)] = int(g.initial_seed())

        out_path = os.path.join(self.image_folder, "prompts.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def write_file_comment(self, path):
        seed = getattr(self, "active_seed", None)

        if seed is None:
            return

        if not path or not os.path.isfile(path):
            return

        width = getattr(self, "width", "")
        height = getattr(self, "height", "")
        fps = getattr(self, "fps", "")
        num_frames = getattr(self, "num_frames", "")
        steps = getattr(self, "num_inference_steps", "")
        cfg = getattr(self, "guidance_scale", "")

        comment = str(seed)

        if width and height and fps and num_frames:
            comment += f"\n{width} x {height}p @ {fps} FPS ({num_frames})"

        if steps != "" and cfg != "":
            comment += f"\n{steps} steps, {cfg:g} CFG"

        os.setxattr(path, "user.xdg.comment", comment.encode("utf-8"))
    
    def enable_gallery_scroll(self, event=None):
        self.app.bind_all("<MouseWheel>", self.on_gallery_scroll)
        self.app.bind_all("<Button-4>", self.on_gallery_scroll)
        self.app.bind_all("<Button-5>", self.on_gallery_scroll)
    
    def disable_gallery_scroll(self, event=None):
        self.app.unbind_all("<MouseWheel>")
        self.app.unbind_all("<Button-4>")
        self.app.unbind_all("<Button-5>")
    
    def main(self):
        self.app.grid_columnconfigure(0, weight=1, minsize=520)
        self.app.grid_columnconfigure(1, weight=2, minsize=620)
        self.app.grid_rowconfigure(0, weight=1)

        if self.supports_source_image and self.source_image_path:
            self.set_source_image(self.source_image_path)

        left = ctk.CTkScrollableFrame(self.app, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left.grid_columnconfigure(0, weight=1)
        self.left_panel = left

        right = ctk.CTkFrame(self.app, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        prompt_header = ctk.CTkFrame(left, fg_color="transparent")
        prompt_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        prompt_header.grid_columnconfigure(0, weight=1)
        prompt_header.grid_columnconfigure(1, weight=0)
        prompt_header.grid_columnconfigure(2, weight=0)
        prompt_header.grid_rowconfigure(1, weight=0)

        prompt_title = ctk.CTkLabel(prompt_header, text="Prompt", font=ctk.CTkFont(size=18, weight="bold"))
        prompt_title.grid(row=0, column=0, sticky="w")

        plan_text = self.active_plan.get("status_text", "Legacy loading path") if self.active_plan else "Legacy loading path"
        self.plan_status_label = ctk.CTkLabel(
            prompt_header,
            text=f"Plan: {plan_text}",
            text_color="#a8a8a8",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.plan_status_label.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        gpu_column = 1
        if self.use_prompt_model:
            self.ai_prompt_btn = ctk.CTkButton(prompt_header, text="AI prompt", width=90, command=self.start_ai_prompt)
            self.ai_prompt_btn.grid(row=0, column=1, sticky="e", padx=(0, 8))
            gpu_column = 2

        self.gpu_pill_top = ctk.CTkLabel(prompt_header, text="GPU 0%", corner_radius=999, fg_color="#333333", text_color="#d0d0d0", padx=10, pady=4)
        self.gpu_pill_top.grid(row=0, column=gpu_column, sticky="e")

        self.prompt_box = ctk.CTkTextbox(left, height=110)
        self.prompt_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.prompt_box.insert("1.0", self.prompt)
        self.prompt_box.bind("<FocusOut>", self.sync_params)

        content_row = 2
        if self.supports_source_image:
            source_frame = ctk.CTkFrame(left, corner_radius=16)
            source_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
            source_frame.grid_columnconfigure(0, weight=1)

            source_header = ctk.CTkFrame(source_frame, fg_color="transparent")
            source_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
            source_header.grid_columnconfigure(0, weight=1)

            source_title = ctk.CTkLabel(source_header, text="Source image", font=ctk.CTkFont(size=14, weight="bold"))
            source_title.grid(row=0, column=0, sticky="w")

            source_btn = ctk.CTkButton(source_header, text="Choose file", width=110, command=self.choose_source_image)
            source_btn.grid(row=0, column=1, sticky="e")

            self.source_path_label = ctk.CTkLabel(source_frame, text="No file selected", text_color="#a0a0a0")
            self.source_path_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

            self.source_img_label = ctk.CTkLabel(source_frame, text="No source image", corner_radius=12)
            self.source_img_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
            content_row = 3

        preview_frame = ctk.CTkFrame(left, corner_radius=16)
        preview_frame.grid(row=content_row, column=0, sticky="ew", padx=14, pady=(0, 10))
        preview_frame.grid_columnconfigure(0, weight=1)

        preview_title = ctk.CTkLabel(preview_frame, text="Preview", font=ctk.CTkFont(size=14, weight="bold"))
        preview_title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.preview_img = ctk.CTkLabel(preview_frame, text="No preview yet", corner_radius=12)
        self.preview_img.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(preview_frame)
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        self.progress_bar.set(0)

        self.progress_text = ctk.CTkLabel(preview_frame, text="0 / 0")
        self.progress_text.grid(row=3, column=0, sticky="e", padx=10, pady=(0, 6))

        self.preview_time_label = ctk.CTkLabel(preview_frame, text="Last update: -", text_color="#a0a0a0")
        self.preview_time_label.grid(row=4, column=0, sticky="w", padx=10, pady=(0, 10))


        tabs = ctk.CTkTabview(left, corner_radius=16)
        tabs.grid(row=content_row + 1, column=0, sticky="ew", padx=14, pady=(0, 12))
        tabs.add("Parameters")
        tabs.add("Negative prompt")
        tabs.set("Parameters")

        params = tabs.tab("Parameters")
        params.grid_columnconfigure(0, weight=1)
        params.grid_columnconfigure(1, weight=1)

        param_row = 0

        self.batch_entry = None
        if self.show_batch_size:
            batch_row = ctk.CTkFrame(params, fg_color="transparent")
            batch_row.grid(row=param_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 6))
            batch_row.grid_columnconfigure(1, weight=1)

            batch_label = ctk.CTkLabel(batch_row, text="Total Batchs")
            batch_label.grid(row=0, column=0, sticky="w")

            self.batch_entry = ctk.CTkEntry(batch_row)
            self.batch_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
            self.batch_entry.insert(0, str(self.batch_size))
            self.batch_entry.bind("<Return>", self.sync_params)
            self.batch_entry.bind("<FocusOut>", self.sync_params)

            param_row += 1

        width_row = ctk.CTkFrame(params, fg_color="transparent")
        width_row.grid(row=param_row, column=0, sticky="ew", padx=10, pady=6)
        width_row.grid_columnconfigure(1, weight=1)

        width_label = ctk.CTkLabel(width_row, text="Width")
        width_label.grid(row=0, column=0, sticky="w")

        self.width_entry = ctk.CTkEntry(width_row)
        self.width_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.width_entry.insert(0, str(self.width))
        self.width_entry.bind("<Return>", self.sync_params)
        self.width_entry.bind("<FocusOut>", self.sync_params)

        height_row = ctk.CTkFrame(params, fg_color="transparent")
        height_row.grid(row=param_row, column=1, sticky="ew", padx=10, pady=6)
        height_row.grid_columnconfigure(1, weight=1)

        height_label = ctk.CTkLabel(height_row, text="Height")
        height_label.grid(row=0, column=0, sticky="w")

        self.height_entry = ctk.CTkEntry(height_row)
        self.height_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.height_entry.insert(0, str(self.height))
        self.height_entry.bind("<Return>", self.sync_params)
        self.height_entry.bind("<FocusOut>", self.sync_params)

        if self.supports_source_image:
            self.width_entry.configure(state="normal")
            self.height_entry.configure(state="normal")
        
        if self.supports_source_image and self.lock_source_aspect_ratio:
            self.width_entry.bind("<Return>", self.on_width_changed, add="+")
            self.height_entry.bind("<Return>", self.on_height_changed, add="+")
            self.width_entry.bind("<FocusOut>", self.on_width_changed, add="+")
            self.height_entry.bind("<FocusOut>", self.on_height_changed, add="+")

        param_row += 1

        guidance_row = ctk.CTkFrame(params, fg_color="transparent")
        guidance_row.grid(row=param_row, column=0, sticky="ew", padx=10, pady=6)
        guidance_row.grid_columnconfigure(1, weight=1)

        guidance_label = ctk.CTkLabel(guidance_row, text="Guidance scale")
        guidance_label.grid(row=0, column=0, sticky="w")

        self.guidance_entry = ctk.CTkEntry(guidance_row)
        self.guidance_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.guidance_entry.insert(0, str(self.guidance_scale))
        self.guidance_entry.bind("<Return>", self.sync_params)
        self.guidance_entry.bind("<FocusOut>", self.sync_params)

        steps_row = ctk.CTkFrame(params, fg_color="transparent")
        steps_row.grid(row=param_row, column=1, sticky="ew", padx=10, pady=6)
        steps_row.grid_columnconfigure(1, weight=1)

        steps_label = ctk.CTkLabel(steps_row, text="Inference steps")
        steps_label.grid(row=0, column=0, sticky="w")

        self.steps_entry = ctk.CTkEntry(steps_row)
        self.steps_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.steps_entry.insert(0, str(self.num_inference_steps))
        self.steps_entry.bind("<Return>", self.sync_params)
        self.steps_entry.bind("<FocusOut>", self.sync_params)

        param_row += 1

        self.ipp_entry = None
        if self.show_ipp:
            ipp_row = ctk.CTkFrame(params, fg_color="transparent")
            ipp_row.grid(row=param_row, column=0, columnspan=2, sticky="ew", padx=10, pady=(6, 10))
            ipp_row.grid_columnconfigure(1, weight=1)

            ipp_label = ctk.CTkLabel(ipp_row, text=self.ipp_label)
            ipp_label.grid(row=0, column=0, sticky="w")

            self.ipp_entry = ctk.CTkEntry(ipp_row)
            self.ipp_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))

            cur = getattr(self, self.ipp_attr, self.num_images_per_prompt)
            self.ipp_entry.insert(0, str(cur))

            self.ipp_entry.bind("<Return>", self.sync_params)
            self.ipp_entry.bind("<FocusOut>", self.sync_params)

            param_row += 1

        for idx, spec in enumerate(self.extra_params):
            name, label, cast = spec

            r = param_row + (idx // 2)
            c = idx % 2

            row = ctk.CTkFrame(params, fg_color="transparent")
            row.grid(row=r, column=c, sticky="ew", padx=10, pady=6)
            row.grid_columnconfigure(1, weight=1)

            lab = ctk.CTkLabel(row, text=label)
            lab.grid(row=0, column=0, sticky="w")

            ent = ctk.CTkEntry(row)
            ent.grid(row=0, column=1, sticky="ew", padx=(10, 0))
            ent.insert(0, str(getattr(self, name, "")))

            ent.bind("<Return>", self.sync_params)
            ent.bind("<FocusOut>", self.sync_params)

            self.extra_entries[name] = (name, cast, ent)

        if self.vram_estimator is not None:
            vram_index = len(self.extra_params)
            vram_row = param_row + (vram_index // 2)
            vram_column = vram_index % 2

            self.vram_label = ctk.CTkLabel(params, text="", anchor="w")
            self.vram_label.grid(row=vram_row, column=vram_column, sticky="w", padx=10, pady=6)

            self.width_entry.bind("<KeyRelease>", self.schedule_vram_update, add="+")
            self.height_entry.bind("<KeyRelease>", self.schedule_vram_update, add="+")
            self.guidance_entry.bind("<KeyRelease>", self.schedule_vram_update, add="+")

            if "num_frames" in self.extra_entries:
                self.extra_entries["num_frames"][2].bind("<KeyRelease>", self.schedule_vram_update, add="+")

            self.update_vram_estimate()
        
        neg = tabs.tab("Negative prompt")
        neg.grid_columnconfigure(0, weight=1)
        neg.grid_rowconfigure(1, weight=1)

        neg_title = ctk.CTkLabel(neg, text="Negative prompt (optional)", font=ctk.CTkFont(size=14, weight="bold"))
        neg_title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.neg_box = ctk.CTkTextbox(neg)
        self.neg_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.neg_box.insert("1.0", self.negative_prompt)
        self.neg_box.bind("<FocusOut>", self.sync_params)

        actions = ctk.CTkFrame(left, fg_color="transparent")
        actions.grid(row=content_row + 2, column=0, sticky="ew", padx=14, pady=(0, 14))
        actions.grid_columnconfigure(0, weight=1)

        self.generate_btn = ctk.CTkButton(actions, text="Generate", command=self.start_generate)
        self.generate_btn.grid(row=0, column=0, sticky="ew")

        self.generate_btn_fg = self.generate_btn.cget("fg_color")
        self.generate_btn_hover = self.generate_btn.cget("hover_color")
        self.generate_btn_text = self.generate_btn.cget("text_color")

        gallery_header = ctk.CTkFrame(right, fg_color="transparent")
        gallery_header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        gallery_header.grid_columnconfigure(0, weight=1)
        gallery_header.grid_columnconfigure(1, weight=0)
        gallery_header.grid_columnconfigure(2, weight=0)
        gallery_header.grid_columnconfigure(3, weight=0)
        gallery_header.grid_columnconfigure(4, weight=0)
        gallery_header.grid_columnconfigure(5, weight=0)

        gallery_title = ctk.CTkLabel(gallery_header, text=self.gallery_title, font=ctk.CTkFont(size=18, weight="bold"))
        gallery_title.grid(row=0, column=0, sticky="w")

        self.stop_btn = ctk.CTkButton(gallery_header, text="✕", width=36, fg_color="#b00020", hover_color="#8a0019", command=self.request_stop)
        self.stop_btn.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self.stop_btn.configure(state="disabled")

        image_editor_btn = ctk.CTkButton(gallery_header, text="🖼", width=36, command=self.open_image_editor)
        image_editor_btn.grid(row=0, column=2, sticky="e", padx=(0, 8))

        open_folder_btn = ctk.CTkButton(gallery_header, text="📁", width=36, command=self.open_output_folder)
        open_folder_btn.grid(row=0, column=3, sticky="e", padx=(0, 8))

        cachelight_btn = ctk.CTkButton(gallery_header, text="◫", width=36, command=self.open_cachelight)
        cachelight_btn.grid(row=0, column=4, sticky="e", padx=(0, 8))

        refresh_btn = ctk.CTkButton(gallery_header, text="↻", width=36, command=self.refresh_gallery)
        refresh_btn.grid(row=0, column=5, sticky="e")
        self.stop_btn.configure(state="disabled")

        self.gallery = ctk.CTkScrollableFrame(right, corner_radius=16)
        self.gallery.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.gallery.grid_columnconfigure(0, weight=1)

        self.refresh_gallery()
        self.sync_params()

        if self.supports_source_image and self.source_image_path:
            self.set_source_image(self.source_image_path)

        self.app.bind_all("<MouseWheel>", self.on_mousewheel)
        self.app.bind_all("<Button-4>", self.on_mousewheel)
        self.app.bind_all("<Button-5>", self.on_mousewheel)

        if not self.model_load_started:
            self.model_load_started = True
            threading.Thread(target=self.load_model_guarded, daemon=True).start()
    
    def set_generate_busy(self):
        self.generate_btn.configure(text="In progress", state="disabled", fg_color="#444444", hover_color="#444444", text_color="#d0d0d0")
    
    def set_generate_idle(self):
        self.generate_btn.configure(text="Generate", state="normal", fg_color=self.generate_btn_fg, 
            hover_color=self.generate_btn_hover, text_color=self.generate_btn_text)

    def set_generate_preparing(self):
        self.generate_btn.configure(text="Preparing model", state="disabled", fg_color="#444444", hover_color="#444444", text_color="#d0d0d0")

    def wait_for_model_ready(self):
        if self.pipe is not None and hasattr(self.pipe, "is_ready") and not self.pipe.is_ready():
            self.set_generate_preparing()
            return self.app.after(500, self.wait_for_model_ready)

        self.set_generate_idle()
    
    def finish_generate(self):
        completed = self.progress >= 0.999 and not self.stop_requested
        if completed and self.active_plan and not self.plan_validated:
            mark_success(self.active_plan, workload_from_diffusion_gui(self))
            self.plan_validated = True
            self.update_plan_status("Validated on this computer")

        self.generate_running = False
        self.stop_requested = False
        self.thermal_throttling = False
        self.progress = 0.0
        self.progress_step = 0
        self.progress_total = 0
        self.update_progress_widgets()
        self.refresh_gallery()
        self.gpu_percent = 0
        self.update_util_widgets()
        if self.pipe is not None and hasattr(self.pipe, "is_ready") and not self.pipe.is_ready():
            self.set_generate_preparing()
            self.app.after(500, self.wait_for_model_ready)
        else:
            self.set_generate_idle()
        self.stop_btn.configure(state="disabled")

        if self.cleanup_requested:
            self.cleanup_requested = False
            self.unload_model()
    
    def request_stop(self):
        if self.generate_running:
            self.stop_requested = True
            self.cleanup_requested = True
            self.stop_btn.configure(state="disabled")
            return

        self.cleanup_requested = False
        self.unload_model()
        self.gpu_percent = 0
        self.update_util_widgets()
    
    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            raise KeyboardInterrupt

        self.progress_step = step + 1
        self.progress_total = self.num_inference_steps
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)

        if self.preview_vae is not None and time.time() - self.preview_time > 30:
            self.preview_time = time.time()
            latents = callback_kwargs["latents"]
            threading.Thread(target=self.save_preview, args=(latents,), daemon=True).start()

        return callback_kwargs
    
    def start_generate(self):
        if self.generate_running:
            return print("Generation already in progress.")

        if self.pipe is not None and hasattr(self.pipe, "is_ready") and not self.pipe.is_ready():
            return print("Model is preparing for the next generation.")
        
        if self.supports_source_image and self.lock_source_aspect_ratio:
            self.on_width_changed()

        self.sync_params()
        if self.supports_source_image and self.source_image_pil is None:
            return print("Select a source image first.")

        if self.gallery_title == "Videos" and not self.has_fixed_seed():
            self.active_seed = None
        else:
            self.prepare_run_seed()
        self.save_prompts_json()

        self.stop_requested = False
        self.generate_running = True
        self.set_generate_busy()
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self.poll_util, daemon=True).start()

        if self.thermal_poll_id is not None:
            self.app.after_cancel(self.thermal_poll_id)
            self.thermal_poll_id = None

        self.poll_thermal()

        self.progress = 0.0
        self.progress_step = 0
        self.progress_total = self.callback_on_step_end and self.num_inference_steps or self.get_total_runs()
        self.update_progress_widgets()

        t1 = threading.Thread(target=self.generate_guarded, daemon=True)
        t1.start()
    
    def fit_size_constraints(self, width, height):
        w = max(1, int(width))
        h = max(1, int(height))

        max_dim = int(self.source_max_dim or 0)
        step = int(self.source_size_multiple or 1)

        if max_dim > 0 and max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))

        if step > 1:
            w = max(step, (w // step) * step)
            h = max(step, (h // step) * step)

        return w, h

    def set_size_entries(self, width, height):
        if hasattr(self, "width_entry") and self.width_entry is not None:
            self.width_entry.delete(0, "end")
            self.width_entry.insert(0, str(int(width)))

        if hasattr(self, "height_entry") and self.height_entry is not None:
            self.height_entry.delete(0, "end")
            self.height_entry.insert(0, str(int(height)))

    def apply_locked_aspect(self, changed):
        if self.size_lock_busy:
            return
        if not self.lock_source_aspect_ratio:
            return
        if self.source_image_pil is None:
            return
        if self.source_aspect_ratio <= 0:
            return

        if not hasattr(self, "last_locked_width"):
            self.last_locked_width = int(getattr(self, "width", 0) or 0)
        if not hasattr(self, "last_locked_height"):
            self.last_locked_height = int(getattr(self, "height", 0) or 0)

        if changed == "width":
            text = self.width_entry.get().strip()
            if not text.isdigit():
                return

            w_in = max(1, int(text))

            if w_in == int(self.last_locked_width):
                return

            w = w_in
            h = int(round(w * self.source_aspect_ratio))

        else:
            text = self.height_entry.get().strip()
            if not text.isdigit():
                return

            h_in = max(1, int(text))

            if h_in == int(self.last_locked_height):
                return

            h = h_in
            w = int(round(h / self.source_aspect_ratio))

        w, h = self.fit_size_constraints(w, h)

        self.size_lock_busy = True
        self.set_size_entries(w, h)
        self.size_lock_busy = False

        self.sync_params()

        self.last_locked_width = int(self.width)
        self.last_locked_height = int(self.height)

    def on_width_changed(self, event=None):
        self.apply_locked_aspect("width")

    def on_height_changed(self, event=None):
        self.apply_locked_aspect("height")
    
    def load_prompt_model(self):
        with self.prompt_model_lock:
            if self.prompt_pipe is not None:
                return
            if self.prompt_model_loading:
                return
            self.prompt_model_loading = True

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        try:
            prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)
            if prompt_tokenizer.pad_token is None:
                prompt_tokenizer.pad_token = prompt_tokenizer.eos_token

            prompt_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float32,
            )
            prompt_model.to("cpu")
            prompt_model.eval()
            prompt_pipe = pipeline(
                "text-generation",
                model=prompt_model,
                tokenizer=prompt_tokenizer,
                device=-1,
            )
        except Exception:
            with self.prompt_model_lock:
                self.prompt_model_loading = False
            raise

        with self.prompt_model_lock:
            self.prompt_pipe = prompt_pipe
            self.prompt_tokenizer = prompt_tokenizer
            self.prompt_model_loading = False

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

    def choose_source_image(self):
        path = ""

        start_dir = ""
        if self.source_image_path and os.path.isfile(self.source_image_path):
            start_dir = os.path.dirname(self.source_image_path)
        else:
            start_dir = os.path.expanduser("~/Pictures")

        kdialog = shutil.which("kdialog")
        if kdialog:
            p = subprocess.run(
                [
                    kdialog,
                    "--getopenfilename",
                    start_dir,
                    "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if p.returncode == 0:
                path = p.stdout.strip()

        if not path:
            path = filedialog.askopenfilename(
                title="Choose source image",
                initialdir=start_dir,
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                    ("All files", "*.*"),
                ],
            )

        if not path:
            return

        self.set_source_image(path)
    
    def set_source_image(self, path):
        if not path or not os.path.isfile(path):
            return

        with Image.open(path) as im:
            src = im.convert("RGB")

        self.source_image_path = path
        self.source_image_pil = src

        src_w, src_h = src.size

        if src_w > 0:
            self.source_aspect_ratio = src_h / float(src_w)

        if self.auto_size_from_source_on_select:
            self.width, self.height = self.fit_size_constraints(src_w, src_h)

            if hasattr(self, "width_entry") and self.width_entry is not None:
                self.width_entry.configure(state="normal")
                self.width_entry.delete(0, "end")
                self.width_entry.insert(0, str(self.width))

            if hasattr(self, "height_entry") and self.height_entry is not None:
                self.height_entry.configure(state="normal")
                self.height_entry.delete(0, "end")
                self.height_entry.insert(0, str(self.height))

        thumb = src.copy()
        thumb.thumbnail((520, 260))
        w, h = thumb.size

        self.source_image_ctk = ctk.CTkImage(light_image=thumb, dark_image=thumb, size=(w, h))

        if self.source_img_label is not None:
            self.source_img_label.configure(image=self.source_image_ctk, text="")

        if self.source_path_label is not None:
            self.source_path_label.configure(text=f"{os.path.basename(path)} ({src.size[0]}x{src.size[1]})")

        if hasattr(self, "prompt_box") and hasattr(self, "neg_box") and hasattr(self, "width_entry") and hasattr(self, "height_entry"):
            self.sync_params()

        self.last_locked_width = int(self.width)
        self.last_locked_height = int(self.height)
    
    def fit_source_size(self, width, height):
        block = int(getattr(self, "source_size_multiple", 1) or 1)
        max_dim = int(getattr(self, "source_max_dim", 0) or 0)

        w = int(width)
        h = int(height)

        if max_dim > 0 and max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            w = max(1, int(w * scale))
            h = max(1, int(h * scale))

        if block > 1:
            w = max(block, (w // block) * block)
            h = max(block, (h // block) * block)

        return w, h
    
    def generate_diffusers(self):
        stopped = False
        for i in range(self.batch_size):
            if self.stop_requested:
                stopped = True
                break

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            try:
                # call pipeline
                args = {i : getattr(self, i) for i in self.pipeline_args}
                if self.supports_source_image and self.source_image_pil is not None:
                    n = int(getattr(self, "num_images_per_prompt", 1))

                    if n > 1:
                        args["image"] = [self.source_image_pil.copy() for _ in range(n)]

                        if "prompt" in args and isinstance(args["prompt"], str):
                            args["prompt"] = [args["prompt"] for _ in range(n)]

                        if "negative_prompt" in args and isinstance(args["negative_prompt"], str):
                            args["negative_prompt"] = [args["negative_prompt"] for _ in range(n)]

                        if "num_images_per_prompt" in args:
                            args["num_images_per_prompt"] = 1
                    else:
                        args["image"] = self.source_image_pil.copy()

                if self.callback_on_step_end:
                    args["callback_on_step_end"] = self.on_step_end
                    args["callback_on_step_end_tensor_inputs"] = ["latents"]
                
                if "negative_prompt_embeds" in self.pipeline_args:
                    args["negative_prompt_embeds"] = None

                    if self.negative_prompt.strip():
                        args["negative_prompt_embeds"] = self.pipe.encode_prompt(
                            prompt=self.negative_prompt,
                            do_classifier_free_guidance=False,
                            num_images_per_prompt=self.num_images_per_prompt,
                            max_sequence_length=self.max_sequence_length,
                        )[0]

                images = self.pipe(**args)

            except KeyboardInterrupt:
                stopped = True
                break

            img_list = getattr(images, "images", None)
            if img_list is None:
                img_list = getattr(images, "image", None)

            for idx, img in enumerate(img_list):
                fname = f"output_{int(time.time())}{idx}.png"
                output_path = os.path.join(self.image_folder, fname)
                img.save(output_path)
                print(f"saved {fname}")

        if not stopped and hasattr(self.pipe, "prepare_next_generation"):
            self.pipe.prepare_next_generation()
        
        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)
    
    def generate(self):
        self.sync_params()
        self.ensure_model_loaded()

        return self.generate_diffusers()
