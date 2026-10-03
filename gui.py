import os, sys, subprocess, io, json, base64, threading, uuid
from copy import deepcopy
import dotenv

dotenv.load_dotenv()

import customtkinter as ctk
from PIL import Image
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tkinter import filedialog, messagebox

from app_config import apply_runtime_environment, chat_history_path, load_user_config, resolve_output_path, save_user_config
from hardware_detection import detect_hardware, hardware_summary
from hardware_planner import plan_attempts, save_plan_file
from planner_protocol import MEMORY_RETRY_EXIT_CODE
from platform_utils import ffmpeg_executable

apply_runtime_environment()

MODELS = [
    ('Z Image Turbo', 'z_image_turbo', 'image', '6B'),
    ('Kandinsky 5', 'kandinsky_5', 'image', '6B'),
    ('PixArt Sigma', 'pixart_sigma', 'image', '0.6B'),
    ('Stable Diffusion 3.5', 'stable_diffusion_3_5', 'image', '8B'),
    ('Black Forest FLUX.1', 'flux_1', 'image', '12B'),
    ('GLM Image', 'glm_image', 'image', '16B'),
    ('Qwen Image', 'qwen_image', 'image', '20B'),
    ('Black Forest FLUX.2', 'flux_2', 'image', '32B'),
    ('Kandinsky 5 I2I', 'kandinsky_5_i2i', 'image', '6B'),
    ('ChronoEdit (14B)', 'chronoedit', 'image', '14B'),
    ('Qwen Image Edit (20B 4bit)', 'qwen_image_edit', 'image', '30B'),
    ('BRIA RMBG 1.4 (Remove BG)', 'rmbg_1_4', 'image', '44M'),
    ('SD x4 Upscaler', 'sd_x4_upscaler', 'image', '1.1B'),

    ('SkyReels V2', 'skyreels_v2', 'video', '1.3B'),
    ('Kandinsky 5 T2V (Lite distilled16)', 'kandinsky_5_t2v', 'video', '6B'),
    ('Cosmos Predict2 V2W (2B)', 'nvidia_cosmos', 'video', '2B'),
    ('Allegro (T2V)', 'allegro', 'video', '2.8B'),
    ('CogVideoX 5B', 'cogvideox', 'video', '5B'),
    ('Hunyuan Video 1.5 (720p T2V)', 'hunyuan_video_1_5', 'video', '8.3B'),
    ('Hunyuan Video 1.5 (720p I2V)', 'hunyuan_video_1_5_i2v', 'video', '8.3B'),
    ('Kandinsky 5 T2V (Pro distilled 5s)', 'kandinsky_5_t2v_pro', 'video', '14B'),
    ('Kandinsky 5 T2V (Pro sft 5s)', 'kandinsky_5_t2v_pro_sft', 'video', '14B'),
    ('Kandinsky 5 I2V (Lite 5s)', 'kandinsky_5_i2v', 'video', '6B'),
    ('Kandinsky 5 I2V Pro (SFT 5s)', 'kandinsky_5_i2v_pro_sft', 'video', '14B'),

    ('GPT-2 Large (0.774B)', 'openai-community/gpt2-large', 'chat', '0.8B'),
    ('Falcon-H1 0.5B Instruct', 'tiiuae/Falcon-H1-0.5B-Instruct', 'chat', '0.5B'),
    ('Mistral 7B Instruct v0.3', 'mistralai/Mistral-7B-Instruct-v0.3', 'chat', '7B'),
    ('Command R7B (12-2024)', 'CohereLabs/c4ai-command-r7b-12-2024', 'chat', '7B'),
    ('Falcon-H1 7B Instruct', 'tiiuae/Falcon-H1-7B-Instruct', 'chat', '7B'),
    ('Meta-Llama 3.1 8B Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct', 'chat', '8B'),
    ('Gemma 2 9B IT', 'google/gemma-2-9b-it', 'chat', '9B'),
    ('GLM-4 9B Chat', 'zai-org/glm-4-9b-chat-hf', 'chat', '9B'),
    ('Falcon3 10B Instruct', 'tiiuae/Falcon3-10B-Instruct', 'chat', '10B'),
    ('OLMo-2 13B Instruct', 'allenai/OLMo-2-1124-13B-Instruct', 'chat', '13B'),
    ('Qwen2.5 14B Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'chat', '14B'),
    ('Qwen2.5 32B Instruct', 'Qwen/Qwen2.5-32B-Instruct', 'chat', '32B'),
    ('Falcon-H1 34B Instruct', 'tiiuae/Falcon-H1-34B-Instruct', 'chat', '34B'),
    ('Liquid LFM2.5 1.2B Thinking', 'LiquidAI/LFM2.5-1.2B-Thinking', 'chat', '1.2B'),
    ('Phi-4 14B (Math)', 'microsoft/Phi-4-reasoning', 'chat', '14B'),
    ('Qwen3.5 4B', 'Qwen/Qwen3.5-4B', 'chat', '4B'),
    ('Qwen3.5 9B', 'Qwen/Qwen3.5-9B', 'chat', '9B'),
    ('Qwen3 14B', 'Qwen/Qwen3-14B', 'chat', '14B'),
    ('GPT-OSS 20B', 'openai/gpt-oss-20b', 'chat', '20B-3.6A'),
    ('DeepSeek R1 Distill Qwen 32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'chat', '32B'),
    ('Qwen3.6 27B', 'Qwen/Qwen3.6-27B', 'chat', '27B'),
    ('Gemma 4 31B IT', 'google/gemma-4-31B-it', 'chat', '31B'),
    ('Qwen3.6 35B A3B', 'Qwen/Qwen3.6-35B-A3B', 'chat', '35B-3A'),
]

MODEL_KIND_BY_ID = {model_id: model_kind for _, model_id, model_kind, _ in MODELS}

REASONING_CHAT_MODEL_IDS = {
    "LiquidAI/LFM2.5-1.2B-Thinking",
    "microsoft/Phi-4-reasoning",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-14B",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen3.6-27B",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
}

MODEL_PREVIEW_DIRS = {
    "z_image_turbo": "z_image_turbo/",
    "kandinsky_5": "kandinsky/",
    "pixart_sigma": "pixart_sigma/",
    "stable_diffusion_3_5": "stable_diffusion3.5/",
    "flux_1": "black_forest/",
    "glm_image": "glm_image/",
    "qwen_image": "qwen_image/",
    "flux_2": "black_forest_flux_2/",
    "chronoedit": "chronoedit/",
    "qwen_image_edit": "qwen_image_edit/",
    "rmbg_1_4": "rmbg_1_4/",
    "sd_x4_upscaler": "sd_x4_upscaler/",

    "skyreels_v2": "skyreels_v2/",
    "kandinsky_5_t2v": "kandinsky_tv2_lite/",
    "nvidia_cosmos": "nvidia_cosmos/",
    "cosmos_predict2_v2w": "nvidia_cosmos/",
    "allegro": "allegro/",
    "cogvideox": "cogvideox/",
    "hunyuan_video_1_5": "hunyuan_video/",
    "hunyuan_video_1_5_i2v": "hunyuan_video_i2v/",
    "kandinsky_5_t2v_pro": "kandinsky_tv2_distilled/",
    "kandinsky_5_t2v_pro_sft": "kandinsky_tv2_pro/",
    "kandinsky_5_i2i": "kandinsky_i2i/",
    "kandinsky_5_i2v": "kandinsky_i2v/",
    "kandinsky_5_i2v_pro_sft": "kandinsky_i2v_pro/",
}


class SettingsDialog:
    def __init__(self, parent, on_saved=None):
        self.parent = parent
        self.on_saved = on_saved
        self.config = load_user_config()
        self.entries = {}

        self.window = ctk.CTkToplevel(parent)
        self.window.title("Application paths")
        self.window.geometry("720x360")
        self.window.minsize(680, 330)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.grid_columnconfigure(0, weight=1)

        title_text = "Application paths"
        title = ctk.CTkLabel(self.window, text=title_text, font=ctk.CTkFont(size=21, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=20, pady=(18, 4))

        description = ctk.CTkLabel(
            self.window,
            text="These paths are optional. Leave Output root blank to use the platform's default media folder.",
            text_color="#a8a8a8",
            anchor="w",
        )
        description.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 14))

        form = ctk.CTkFrame(self.window, corner_radius=14)
        form.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 14))
        form.grid_columnconfigure(1, weight=1)

        self.add_path_row(form, 0, "huggingface_cache_dir", "Hugging Face cache")
        self.add_path_row(form, 1, "output_root", "Media output root")

        buttons = ctk.CTkFrame(self.window, fg_color="transparent")
        buttons.grid(row=3, column=0, sticky="e", padx=20, pady=(0, 18))

        cancel = ctk.CTkButton(buttons, text="Cancel", width=100, fg_color="#444444", command=self.close)
        cancel.grid(row=0, column=0, padx=(0, 8))

        save = ctk.CTkButton(buttons, text="Save", width=100, command=self.save)
        save.grid(row=0, column=1)

    def add_path_row(self, parent, row, key, label_text):
        label = ctk.CTkLabel(parent, text=label_text, anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=(14, 10), pady=12)

        entry = ctk.CTkEntry(parent)
        entry.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=12)
        entry.insert(0, self.config.get("paths", {}).get(key, ""))
        self.entries[key] = entry

        button = ctk.CTkButton(parent, text="Browse", width=82, command=lambda name=key: self.browse(name))
        button.grid(row=row, column=2, padx=(0, 14), pady=12)

    def browse(self, key):
        current = self.entries[key].get().strip()
        initial = current if current and os.path.isdir(current) else os.path.expanduser("~")
        selected = filedialog.askdirectory(parent=self.window, initialdir=initial)
        if not selected:
            return
        self.entries[key].delete(0, "end")
        self.entries[key].insert(0, selected)

    def save(self):
        for key, entry in self.entries.items():
            value = entry.get().strip()
            self.config.setdefault("paths", {})[key] = value
        save_user_config(self.config)
        apply_runtime_environment(self.config)
        if self.on_saved is not None:
            self.on_saved(self.config)
        self.close()

    def close(self):
        self.window.grab_release()
        self.window.destroy()


class LauncherApp:
    def __init__(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Select Model")
        self.app.geometry("1100x800")
        self.app.minsize(1100, 800)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = load_user_config()
        self.hardware = detect_hardware()
        self.launch_sessions = {}

        self.model_items = {}
        self.selected_model_id = None
        self.primary_compute_gpu = None

        self.preview_paths = []
        self.preview_thumbs = []

        self.page_size = 10
        self.page_index = 0

        self.app.grid_columnconfigure(0, weight=1, minsize=420)
        self.app.grid_columnconfigure(1, weight=2, minsize=680)
        self.app.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.app, corner_radius=16)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 10))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        title = ctk.CTkLabel(header, text="Model Launcher", font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))

        subtitle = ctk.CTkLabel(
            header,
            text="Click a model to preview samples. The hardware planner chooses the loading strategy automatically.",
            text_color="#a8a8a8",
            font=ctk.CTkFont(size=13),
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 2))

        self.hardware_label = ctk.CTkLabel(
            header,
            text=hardware_summary(self.hardware),
            text_color="#7f9fbd",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.hardware_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 12))

        gpu_frame = ctk.CTkFrame(header, fg_color="transparent")
        gpu_frame.grid(row=0, column=1, rowspan=3, sticky="e", padx=14, pady=12)
        gpu_frame.grid_columnconfigure(0, weight=1)

        gpu_label = ctk.CTkLabel(gpu_frame, text="Preferred GPU", text_color="#a8a8a8", font=ctk.CTkFont(size=12))
        gpu_label.grid(row=0, column=0, sticky="e", pady=(0, 4))

        gpu_values = ["Automatic"]
        gpu_values.extend([f"GPU {gpu['index']}" for gpu in self.hardware.get("gpus", [])])
        self.gpu_menu = ctk.CTkOptionMenu(gpu_frame, values=gpu_values, width=150, command=self.on_gpu_changed)
        self.gpu_menu.grid(row=1, column=0, sticky="e")
        self.gpu_menu.set("Automatic")

        settings_button = ctk.CTkButton(gpu_frame, text="Settings", width=150, command=self.open_settings)
        settings_button.grid(row=2, column=0, sticky="e", pady=(8, 0))

        left = ctk.CTkFrame(self.app, corner_radius=16)
        left.grid(row=1, column=0, sticky="nsew", padx=(14, 8), pady=(0, 10))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)

        left_title = ctk.CTkLabel(left, text="Models", font=ctk.CTkFont(size=16, weight="bold"))
        left_title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        self.model_tabs = ctk.CTkTabview(left, corner_radius=14)
        self.model_tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.model_tabs.add("Images")
        self.model_tabs.add("Videos")
        self.model_tabs.add("Chat")
        self.model_tabs.set("Images")

        images_tab = self.model_tabs.tab("Images")
        videos_tab = self.model_tabs.tab("Videos")
        chat_tab = self.model_tabs.tab("Chat")

        images_tab.grid_columnconfigure(0, weight=1)
        images_tab.grid_rowconfigure(0, weight=1)
        videos_tab.grid_columnconfigure(0, weight=1)
        videos_tab.grid_rowconfigure(0, weight=1)
        chat_tab.grid_columnconfigure(0, weight=1)
        chat_tab.grid_rowconfigure(0, weight=1)

        self.image_list_frame = ctk.CTkScrollableFrame(images_tab, corner_radius=14)
        self.image_list_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.image_list_frame.grid_columnconfigure(0, weight=1)

        self.video_list_frame = ctk.CTkScrollableFrame(videos_tab, corner_radius=14)
        self.video_list_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.video_list_frame.grid_columnconfigure(0, weight=1)

        self.chat_list_frame = ctk.CTkScrollableFrame(chat_tab, corner_radius=14)
        self.chat_list_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.chat_list_frame.grid_columnconfigure(0, weight=1)

        right = ctk.CTkFrame(self.app, corner_radius=16)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 14), pady=(0, 10))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        self.preview_title = ctk.CTkLabel(right, text="Preview", font=ctk.CTkFont(size=16, weight="bold"))
        self.preview_title.grid(row=0, column=0, sticky="w", padx=14, pady=(12, 6))

        self.preview_list = ctk.CTkScrollableFrame(right, corner_radius=14)
        self.preview_list.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.preview_list.grid_columnconfigure(0, weight=1)

        nav = ctk.CTkFrame(right, corner_radius=14)
        nav.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        nav.grid_columnconfigure(1, weight=1)

        self.prev_btn = ctk.CTkButton(nav, text="◀", width=44, command=self.prev_page)
        self.prev_btn.grid(row=0, column=0, padx=(10, 6), pady=10, sticky="w")

        self.preview_counter = ctk.CTkLabel(nav, text="Page - / -", text_color="#a8a8a8")
        self.preview_counter.grid(row=0, column=1, padx=6, pady=10)

        self.next_btn = ctk.CTkButton(nav, text="▶", width=44, command=self.next_page)
        self.next_btn.grid(row=0, column=2, padx=(6, 10), pady=10, sticky="e")

        self.status = ctk.CTkLabel(self.app, text="Ready", text_color="#a8a8a8")
        self.status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 14))

        current_chat_section = None

        for name, model_id, model_kind, parameter_label in MODELS:
            if model_kind == "chat":
                chat_section = "Reasoning Models" if model_id in REASONING_CHAT_MODEL_IDS else "Dense Transformers"

                if chat_section != current_chat_section:
                    self.add_model_section_header(chat_section, self.chat_list_frame)
                    current_chat_section = chat_section

            self.add_model_item(name, model_id, model_kind, parameter_label)

        if MODELS:
            first_kind = MODELS[0][2]
            self.model_tabs.set("Images" if first_kind == "image" else ("Videos" if first_kind == "video" else "Chat"))
            self.set_selected_model(MODELS[0][1])

        self.app.bind_all("<MouseWheel>", self.on_mousewheel)
        self.app.bind_all("<Button-4>", self.on_mousewheel)
        self.app.bind_all("<Button-5>", self.on_mousewheel)

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

    def load_preview_pil(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            with Image.open(path) as img:
                return img.copy()

        if ext in (".mp4", ".webm", ".mov", ".mkv"):
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

        return None

    def get_chat_history_key(self):
        key = os.environ.get("CHAT_HISTORY_KEY_B64")
        if not key:
            return None
        return base64.b64decode(key)

    def decrypt_store_bytes(self, data):
        key = self.get_chat_history_key()
        if key is None:
            return None

        nonce = data[:12]
        encrypted = data[12:]
        return AESGCM(key).decrypt(nonce, encrypted, None)

    def load_chat_history_store(self):
        path = chat_history_path()

        if not os.path.isfile(path):
            return {}

        with open(path, "rb") as f:
            encrypted = f.read()

        data = self.decrypt_store_bytes(encrypted)
        if data is None:
            return {}

        return json.loads(data.decode("utf-8"))

    def render_chat_preview_page(self, model_id):
        self.clear_preview_list()
        self.preview_title.configure(text=f"Conversations: {model_id}")

        store = self.load_chat_history_store()
        model_store = store.get(model_id) or {}
        chats = model_store.get("chats") or {}
        chat_order = model_store.get("chat_order") or []

        if not chat_order:
            lab = ctk.CTkLabel(
                self.preview_list,
                text="No conversations found for this model.",
                text_color="#a8a8a8",
            )
            lab.grid(sticky="ew", padx=10, pady=10)
            self.preview_counter.configure(text="0 chats")
            return

        self.preview_counter.configure(text=f"{len(chat_order)} chats")

        for chat_id in chat_order:
            chat = chats.get(chat_id) or {}
            name = chat.get("name") or chat_id
            messages = chat.get("messages") or []

            item = ctk.CTkFrame(self.preview_list, corner_radius=14, fg_color="#2b2b2b")
            item.grid(sticky="ew", padx=8, pady=8)
            item.grid_columnconfigure(0, weight=1)

            title = ctk.CTkLabel(item, text=name, font=ctk.CTkFont(size=16, weight="bold"))
            title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))

            meta = ctk.CTkLabel(
                item,
                text=f"{len(messages)} messages",
                text_color="#9a9a9a",
                font=ctk.CTkFont(size=12),
            )
            meta.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 10))

    def on_mousewheel(self, event):
        canvas = None

        if self.image_list_frame and hasattr(self.image_list_frame, "_parent_canvas"):
            if self.widget_inside(event.widget, self.image_list_frame._parent_canvas):
                canvas = self.image_list_frame._parent_canvas

        if not canvas and self.video_list_frame and hasattr(self.video_list_frame, "_parent_canvas"):
            if self.widget_inside(event.widget, self.video_list_frame._parent_canvas):
                canvas = self.video_list_frame._parent_canvas

        if not canvas and self.chat_list_frame and hasattr(self.chat_list_frame, "_parent_canvas"):
            if self.widget_inside(event.widget, self.chat_list_frame._parent_canvas):
                canvas = self.chat_list_frame._parent_canvas

        if not canvas and self.preview_list and hasattr(self.preview_list, "_parent_canvas"):
            if self.widget_inside(event.widget, self.preview_list._parent_canvas):
                canvas = self.preview_list._parent_canvas

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

    def add_model_section_header(self, text, parent):
        header = ctk.CTkLabel(
            parent,
            text=text,
            text_color="#d8d8d8",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        header.grid(sticky="w", padx=14, pady=(14, 4))

    def update_model_title_wrap(self, event, label):
        label.configure(wraplength=max(140, event.width - 24))

    def add_model_item(self, name, model_id, model_kind, parameter_label):
        parent = self.image_list_frame if model_kind == "image" else (self.video_list_frame if model_kind == "video" else self.chat_list_frame)

        item = ctk.CTkFrame(parent, corner_radius=14, fg_color="#2b2b2b")
        item.grid(sticky="ew", padx=10, pady=7)
        item.grid_columnconfigure(0, weight=1)
        item.grid_columnconfigure(1, weight=0)

        title = ctk.CTkLabel(
            item,
            text=name,
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="left",
            anchor="w",
        )
        title.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=12,
            pady=(9, 3),
        )

        small = ctk.CTkLabel(
            item,
            text=f"{model_id} - {parameter_label}",
            text_color="#9a9a9a",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        small.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(12, 6),
            pady=(2, 9),
        )

        btn = ctk.CTkButton(
            item,
            text="▶",
            width=34,
            height=34,
            corner_radius=17,
            font=ctk.CTkFont(size=13),
            command=lambda: self.on_launch(model_id),
        )
        btn.grid(
            row=1,
            column=1,
            sticky="e",
            padx=(4, 10),
            pady=(2, 8),
        )

        item.bind("<Configure>", lambda e, label=title: self.update_model_title_wrap(e, label))
        item.bind("<Enter>", lambda e: self.on_item_hover(model_id, True))
        item.bind("<Leave>", lambda e: self.on_item_hover(model_id, False))
        item.bind("<Button-1>", lambda e: self.set_selected_model(model_id))

        title.bind("<Enter>", lambda e: self.on_item_hover(model_id, True))
        title.bind("<Leave>", lambda e: self.on_item_hover(model_id, False))
        title.bind("<Button-1>", lambda e: self.set_selected_model(model_id))

        small.bind("<Enter>", lambda e: self.on_item_hover(model_id, True))
        small.bind("<Leave>", lambda e: self.on_item_hover(model_id, False))
        small.bind("<Button-1>", lambda e: self.set_selected_model(model_id))

        self.model_items[model_id] = {"frame": item, "button": btn}

    def on_item_hover(self, model_id, hovered):
        if model_id == self.selected_model_id:
            return
        w = self.model_items.get(model_id)
        if not w:
            return
        w["frame"].configure(fg_color="#323232" if hovered else "#2b2b2b")

    def scan_preview_paths(self, model_id):
        rel = MODEL_PREVIEW_DIRS.get(model_id, "")
        if not rel:
            return []

        configured = resolve_output_path(rel)
        folder = configured if os.path.isabs(configured) else os.path.join(self.base_dir, configured)
        os.makedirs(folder, exist_ok=True)

        exts = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov", ".mkv")
        paths = []

        for name in os.listdir(folder):
            low = name.lower()
            if low.endswith(exts):
                paths.append(os.path.join(folder, name))

        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return paths

    def set_selected_model(self, model_id):
        if model_id == self.selected_model_id:
            return

        old = self.model_items.get(self.selected_model_id)
        if old:
            old["frame"].configure(fg_color="#2b2b2b")

        self.selected_model_id = model_id
        self.page_index = 0

        kind = MODEL_KIND_BY_ID.get(model_id)
        if kind == "image":
            self.model_tabs.set("Images")
        elif kind == "video":
            self.model_tabs.set("Videos")
        elif kind == "chat":
            self.model_tabs.set("Chat")

        cur = self.model_items.get(model_id)
        if cur:
            cur["frame"].configure(fg_color="#3a3a3a")

        self.preview_paths = self.scan_preview_paths(model_id)
        self.render_preview_page()

        if self.preview_list and hasattr(self.preview_list, "_parent_canvas"):
            self.preview_list._parent_canvas.yview_moveto(0)

    def clear_preview_list(self):
        for w in self.preview_list.winfo_children():
            w.destroy()
        self.preview_thumbs = []

    def render_preview_page(self):
        model_id = self.selected_model_id or "-"

        if MODEL_KIND_BY_ID.get(model_id) == "chat":
            self.render_chat_preview_page(model_id)
            return

        self.clear_preview_list()
        self.preview_title.configure(text=f"Preview: {model_id}")

        total = len(self.preview_paths)
        total_pages = (total + self.page_size - 1) // self.page_size if total else 0

        if total_pages == 0:
            rel = MODEL_PREVIEW_DIRS.get(self.selected_model_id, "")
            msg = f"No media found in:\n{rel}"
            lab = ctk.CTkLabel(self.preview_list, text=msg, text_color="#a8a8a8")
            lab.grid(sticky="ew", padx=10, pady=10)
            self.preview_counter.configure(text="Page - / -")
            return

        if self.page_index < 0:
            self.page_index = 0
        if self.page_index >= total_pages:
            self.page_index = total_pages - 1

        start = self.page_index * self.page_size
        end = min(start + self.page_size, total)
        page_paths = self.preview_paths[start:end]

        self.preview_counter.configure(text=f"Page {self.page_index + 1} / {total_pages}")

        for p in page_paths:
            if not os.path.exists(p):
                missing = ctk.CTkLabel(self.preview_list, text=f"Missing:\n{p}", text_color="#a8a8a8")
                missing.grid(sticky="ew", padx=10, pady=(10, 0))
                continue

            im = self.load_preview_pil(p)
            if im is None:
                lab = ctk.CTkLabel(self.preview_list, text=f"Couldn't preview:\n{os.path.basename(p)}", text_color="#a8a8a8")
                lab.grid(sticky="ew", padx=10, pady=(10, 0))
                continue

            im.thumbnail((980, 420))
            w, h = im.size

            ctk_img = ctk.CTkImage(light_image=im, dark_image=im, size=(w, h))
            self.preview_thumbs.append(ctk_img)

            tile = ctk.CTkFrame(self.preview_list, corner_radius=14, fg_color="#2b2b2b")
            tile.grid(sticky="ew", padx=8, pady=8)
            tile.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(tile, image=ctk_img, text="", corner_radius=12)
            lbl.grid(sticky="ew", padx=8, pady=8)

            fname = ctk.CTkLabel(tile, text=os.path.basename(p), text_color="#9a9a9a", font=ctk.CTkFont(size=12))
            fname.grid(sticky="w", padx=10, pady=(0, 10))

    def prev_page(self):
        if not self.preview_paths:
            return
        self.page_index -= 1
        if self.page_index < 0:
            self.page_index = 0
        self.render_preview_page()
        if self.preview_list and hasattr(self.preview_list, "_parent_canvas"):
            self.preview_list._parent_canvas.yview_moveto(0)

    def next_page(self):
        if not self.preview_paths:
            return
        total = len(self.preview_paths)
        total_pages = (total + self.page_size - 1) // self.page_size if total else 0
        self.page_index += 1
        if self.page_index >= total_pages:
            self.page_index = total_pages - 1
        self.render_preview_page()
        if self.preview_list and hasattr(self.preview_list, "_parent_canvas"):
            self.preview_list._parent_canvas.yview_moveto(0)

    def open_settings(self):
        SettingsDialog(self.app, on_saved=self.on_settings_saved)

    def on_settings_saved(self, config):
        self.config = config
        self.status.configure(text="Settings saved")
        if self.selected_model_id:
            self.preview_paths = self.scan_preview_paths(self.selected_model_id)
            self.render_preview_page()

    def on_gpu_changed(self, value):
        if value == "Automatic":
            self.primary_compute_gpu = None
            return
        self.primary_compute_gpu = int(value.replace("GPU", "").strip())

    def get_launch_env(self, plan_file, attempt):
        env = os.environ.copy()
        env["AI_WORKSTATION_PLAN_FILE"] = plan_file
        visible = attempt.get("visible_device_order", [])
        if visible:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in visible)
        return env

    def on_launch(self, model_id):
        self.set_selected_model(model_id)
        self.status.configure(text=f"Planning: {model_id}")
        self.app.update_idletasks()

        result = plan_attempts(
            model_id,
            preferred_gpu=self.primary_compute_gpu,
            hardware=self.hardware,
        )

        if result.get("status") != "ready":
            reason = result.get("reason", "No supported plan is available.")
            self.status.configure(text=f"Cannot run: {reason}")
            messagebox.showerror("Cannot run model", reason, parent=self.app)
            return

        session_id = uuid.uuid4().hex
        session = {
            "session_id": session_id,
            "model_id": model_id,
            "kind": MODEL_KIND_BY_ID.get(model_id),
            "attempts": result["attempts"],
            "attempt_index": 0,
            "process": None,
        }
        self.launch_sessions[session_id] = session
        self.start_plan_attempt(session_id)

    def apply_legacy_video_gpu_order(self, session, attempt):
        if session.get("kind") != "video":
            return attempt
        if attempt.get("template_id") != "current_exact_fast_path":
            return attempt

        physical_ids = list(attempt.get("selected_physical_gpu_ids", []))
        if len(physical_ids) != 2:
            return attempt

        preferred = self.primary_compute_gpu
        if preferred is None:
            preferred = 0
        if preferred not in physical_ids:
            return attempt

        ordered = [gpu_id for gpu_id in physical_ids if gpu_id != preferred] + [preferred]
        attempt["selected_physical_gpu_ids"] = ordered
        attempt["visible_device_order"] = ordered
        attempt["physical_to_logical_gpu"] = {gpu_id: index for index, gpu_id in enumerate(ordered)}
        return attempt

    def start_plan_attempt(self, session_id):
        session = self.launch_sessions.get(session_id)
        if session is None:
            return

        index = session["attempt_index"]
        attempts = session["attempts"]
        if index >= len(attempts):
            self.status.configure(text=f"No working plan found for {session['model_id']}")
            messagebox.showerror(
                "No working plan",
                "Every supported loading plan failed for this hardware and workload.",
                parent=self.app,
            )
            self.launch_sessions.pop(session_id, None)
            return

        attempt = deepcopy(attempts[index])
        attempt = self.apply_legacy_video_gpu_order(session, attempt)
        plan_file = save_plan_file(attempt)
        env = self.get_launch_env(plan_file, attempt)
        kind = session["kind"]

        if kind == "chat":
            script = os.path.join(self.base_dir, "chat_gui.py")
            command = [sys.executable, script, "--model-id", session["model_id"], "--plan-file", plan_file]
        else:
            script = os.path.join(self.base_dir, "model_gui.py")
            command = [sys.executable, script, "--model-id", session["model_id"], "--plan-file", plan_file]

        self.status.configure(
            text=f"Attempt {index + 1}/{len(attempts)}: {attempt.get('status_text', attempt.get('plan_id'))}"
        )

        process = subprocess.Popen(command, cwd=self.base_dir, env=env)
        session["process"] = process
        session["plan_file"] = plan_file
        session["attempt"] = attempt

        waiter = threading.Thread(target=self.wait_for_process, args=(session_id, process), daemon=True)
        waiter.start()

    def wait_for_process(self, session_id, process):
        return_code = process.wait()
        self.app.after(0, self.process_finished, session_id, return_code)

    def process_finished(self, session_id, return_code):
        session = self.launch_sessions.get(session_id)
        if session is None:
            return

        attempt = session.get("attempt", {})
        if return_code == MEMORY_RETRY_EXIT_CODE:
            result_data = self.read_attempt_result_data(attempt)
            retry_workload = result_data.get("workload") or attempt.get("workload")
            replanned = plan_attempts(
                session["model_id"],
                workload=retry_workload,
                preferred_gpu=self.primary_compute_gpu,
                hardware=self.hardware,
            )

            if replanned.get("status") != "ready":
                reason = replanned.get("reason", "No safer loading plan is available.")
                self.status.configure(text=f"Cannot run: {reason}")
                messagebox.showerror("No working plan", reason, parent=self.app)
                self.launch_sessions.pop(session_id, None)
                return

            session["attempts"] = replanned["attempts"]
            session["attempt_index"] = 0
            self.status.configure(
                text=f"Memory limit reached in {attempt.get('plan_id', 'plan')}; replanning for the failed workload..."
            )
            self.app.after(250, self.start_plan_attempt, session_id)
            return

        if return_code == 0:
            self.status.configure(text=f"Closed: {session['model_id']}")
        else:
            details = self.read_attempt_result(attempt)
            message = details or f"The model process exited with code {return_code}."
            self.status.configure(text=f"Failed: {session['model_id']}")
            messagebox.showerror("Model process failed", message, parent=self.app)

        self.launch_sessions.pop(session_id, None)

    def read_attempt_result_data(self, attempt):
        result_file = attempt.get("result_file")
        if not result_file or not os.path.isfile(result_file):
            return {}
        try:
            with open(result_file, "r", encoding="utf-8") as file:
                result = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}
        return result if isinstance(result, dict) else {}

    def read_attempt_result(self, attempt):
        result = self.read_attempt_result_data(attempt)
        error = result.get("error", "")
        phase = result.get("phase", "")
        if error and phase:
            return f"{phase}: {error}"
        return error

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    LauncherApp().run()
