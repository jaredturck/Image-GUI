import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from diffusers import Kandinsky5I2IPipeline
from PIL import Image, ImageTk


class Kandinsky5I2IGUI:
    def __init__(self):
        self.model_id = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
        self.image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kandinsky_i2i")
        self.source_size_multiple = 16
        self.source_max_dim = 1280
        self.pipe = None
        self.device = "cpu"
        self.dtype = torch.float32
        self.source_image_pil = None
        self.source_image_tk = None
        self.output_image_tk = None

        os.makedirs(self.image_folder, exist_ok=True)

        self.app = tk.Tk()
        self.app.title("Kandinsky 5 Image Editor")
        self.app.geometry("1050x720")
        self.app.minsize(900, 650)

        self.build_ui()
        threading.Thread(target=self.load_model, daemon=True).start()

    def build_ui(self):
        main = ttk.Frame(self.app, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)

        right = ttk.LabelFrame(main, text="Output", padding=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        ttk.Label(left, text="Prompt").grid(row=0, column=0, sticky="w")
        self.prompt_box = tk.Text(left, height=5, wrap="word")
        self.prompt_box.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.prompt_box.insert("1.0", "Turn the scene into a warm cinematic sunset while preserving the composition.")

        source_frame = ttk.LabelFrame(left, text="Source image", padding=10)
        source_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        source_frame.columnconfigure(0, weight=1)

        self.source_path_label = ttk.Label(source_frame, text="No file selected")
        self.source_path_label.grid(row=0, column=0, sticky="w")

        ttk.Button(source_frame, text="Choose image", command=self.choose_source_image).grid(row=0, column=1, padx=(10, 0))

        self.source_image_label = ttk.Label(source_frame, text="Choose an image to edit", anchor="center")
        self.source_image_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        settings = ttk.LabelFrame(left, text="Settings", padding=10)
        settings.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        ttk.Label(settings, text="Width").grid(row=0, column=0, sticky="w")
        self.width_entry = ttk.Entry(settings, width=10)
        self.width_entry.grid(row=0, column=1, sticky="ew", padx=(8, 14))
        self.width_entry.insert(0, "1280")

        ttk.Label(settings, text="Height").grid(row=0, column=2, sticky="w")
        self.height_entry = ttk.Entry(settings, width=10)
        self.height_entry.grid(row=0, column=3, sticky="ew", padx=(8, 0))
        self.height_entry.insert(0, "768")

        ttk.Label(settings, text="Guidance").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.guidance_entry = ttk.Entry(settings, width=10)
        self.guidance_entry.grid(row=1, column=1, sticky="ew", padx=(8, 14), pady=(8, 0))
        self.guidance_entry.insert(0, "3.5")

        ttk.Label(settings, text="Steps").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.steps_entry = ttk.Entry(settings, width=10)
        self.steps_entry.grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.steps_entry.insert(0, "50")

        ttk.Label(left, text="Negative prompt").grid(row=4, column=0, sticky="w")
        self.negative_prompt_box = tk.Text(left, height=3, wrap="word")
        self.negative_prompt_box.grid(row=5, column=0, sticky="ew", pady=(4, 10))
        self.negative_prompt_box.insert("1.0", "low quality, blurry, distorted")

        self.generate_btn = ttk.Button(left, text="Generate", command=self.start_generate, state="disabled")
        self.generate_btn.grid(row=6, column=0, sticky="ew")

        self.progress_bar = ttk.Progressbar(left, mode="determinate")
        self.progress_bar.grid(row=7, column=0, sticky="ew", pady=(10, 4))

        self.status_label = ttk.Label(left, text="Loading model... The first run also downloads the model weights.", wraplength=460)
        self.status_label.grid(row=8, column=0, sticky="w")

        self.output_image_label = ttk.Label(right, text="Generated image will appear here", anchor="center")
        self.output_image_label.grid(row=0, column=0, sticky="nsew")

    def detect_device(self):
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.float16

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return "cuda", dtype

        return "cpu", torch.float32

    def load_model(self):
        try:
            self.device, self.dtype = self.detect_device()

            pipe = Kandinsky5I2IPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )

            if self.device == "mps":
                pipe.enable_model_cpu_offload(device="mps")
            elif self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")

            self.pipe = pipe
            self.app.after(0, self.finish_model_loading)
        except Exception as error:
            self.app.after(0, self.show_error, "Model loading failed", str(error))

    def finish_model_loading(self):
        if self.device == "mps":
            device_name = "Apple GPU (MPS, float16)"
        elif self.device == "cuda":
            device_name = f"NVIDIA GPU (CUDA, {str(self.dtype).replace('torch.', '')})"
        else:
            device_name = "CPU (float32)"

        self.status_label.configure(text=f"Model ready on {device_name}.")
        self.generate_btn.configure(state="normal")

    def fit_size_constraints(self, width, height):
        width = max(1, int(width))
        height = max(1, int(height))

        if max(width, height) > self.source_max_dim:
            scale = self.source_max_dim / float(max(width, height))
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))

        width = max(self.source_size_multiple, width // self.source_size_multiple * self.source_size_multiple)
        height = max(self.source_size_multiple, height // self.source_size_multiple * self.source_size_multiple)
        return width, height

    def set_size_entries(self, width, height):
        self.width_entry.delete(0, "end")
        self.width_entry.insert(0, str(width))
        self.height_entry.delete(0, "end")
        self.height_entry.insert(0, str(height))

    def choose_source_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        with Image.open(path) as image:
            source_image = image.convert("RGB")

        self.source_image_pil = source_image
        self.source_path_label.configure(text=f"{os.path.basename(path)} ({source_image.width}x{source_image.height})")
        self.set_size_entries(*self.fit_size_constraints(source_image.width, source_image.height))

        preview = source_image.copy()
        preview.thumbnail((430, 260))
        self.source_image_tk = ImageTk.PhotoImage(preview)
        self.source_image_label.configure(image=self.source_image_tk, text="")

    def start_generate(self):
        if self.source_image_pil is None:
            return messagebox.showinfo("Choose an image", "Choose a source image first.")

        if self.pipe is None:
            return messagebox.showinfo("Model loading", "The model is still loading.")

        try:
            width, height = self.fit_size_constraints(self.width_entry.get(), self.height_entry.get())
            guidance_scale = float(self.guidance_entry.get())
            num_inference_steps = int(self.steps_entry.get())
        except ValueError:
            return messagebox.showerror("Invalid settings", "Width, height, guidance, and steps must be numbers.")

        if guidance_scale < 0 or num_inference_steps < 1:
            return messagebox.showerror("Invalid settings", "Guidance must be zero or higher and steps must be at least 1.")

        self.set_size_entries(width, height)
        prompt = self.prompt_box.get("1.0", "end").strip()
        negative_prompt = self.negative_prompt_box.get("1.0", "end").strip()

        self.generate_btn.configure(state="disabled")
        self.progress_bar.configure(maximum=num_inference_steps, value=0)
        self.status_label.configure(text=f"Generating {width}x{height} image on {self.device}...")

        threading.Thread(
            target=self.generate,
            args=(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps),
            daemon=True,
        ).start()

    @torch.inference_mode()
    def generate(self, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps):
        try:
            result = self.pipe(
                image=self.source_image_pil.copy(),
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                callback_on_step_end=self.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            )

            images = getattr(result, "images", None)
            if images is None:
                images = getattr(result, "image", None)

            image = images[0]
            output_path = os.path.join(self.image_folder, f"output_{int(time.time())}.png")
            image.save(output_path)
            self.app.after(0, self.finish_generate, image, output_path)
        except Exception as error:
            self.app.after(0, self.show_error, "Generation failed", str(error))

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        self.app.after(0, self.progress_bar.configure, {"value": step + 1})
        return callback_kwargs

    def finish_generate(self, image, output_path):
        preview = image.copy()
        preview.thumbnail((500, 620))
        self.output_image_tk = ImageTk.PhotoImage(preview)
        self.output_image_label.configure(image=self.output_image_tk, text="")
        self.status_label.configure(text=f"Saved: {output_path}")
        self.generate_btn.configure(state="normal")

    def show_error(self, title, message):
        self.status_label.configure(text=message)
        self.generate_btn.configure(state="normal" if self.pipe is not None else "disabled")
        messagebox.showerror(title, message)

    def main(self):
        self.app.mainloop()


if __name__ == "__main__":
    Kandinsky5I2IGUI().main()
