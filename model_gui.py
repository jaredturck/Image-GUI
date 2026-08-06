import os
import dotenv

dotenv.load_dotenv()

from app_config import apply_runtime_environment, get_path

apply_runtime_environment()

import torch, sys, inspect
from diffusers import ZImagePipeline, Kandinsky5T2IPipeline, AutoencoderKL, PixArtSigmaPipeline, StableDiffusion3Pipeline, FluxPipeline
from diffusers import Flux2Pipeline, GlmImagePipeline, QwenImagePipeline, ChronoEditPipeline
from diffusers import Kandinsky5I2IPipeline, Kandinsky5I2VPipeline
from transformers import AutoConfig, AutoModelForImageSegmentation, pipeline
try:
    from diffusers import Cosmos2VideoToWorldPipeline
except: 
    pass
from diffusers import DiffusionPipeline, AllegroPipeline, AutoencoderKLAllegro, CogVideoXPipeline, HunyuanVideo15Pipeline, QwenImageEditPlusPipeline
from base_gui import DiffusionGUI
from vram_estimation import Kandinsky5I2VProVramEstimator
from model_loading import Flux2Generator, GLMImageGenerator, QwenImageGenerator, QwenImageEditGenerator, ChronoEditGenerator, Kandinsky5I2VGenerator
from comfy_backend import connect_comfy_backend
import os, time
from diffusers import AutoModel, SkyReelsV2DiffusionForcingPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
import torch.nn.attention.flex_attention as flex_attention
from diffusers import Kandinsky5T2VPipeline
from PIL import Image, ImageOps
from diffusers.utils import load_image
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from diffusers import StableDiffusionUpscalePipeline
from planner_runtime import execution_device, get_active_plan, is_cuda_plan, is_exact_fast_path, load_component, load_diffusers_pipeline, plan_placement, prepare_preview_vae, torch_dtype

class ZImageTurboGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Z Image Turbo GUI",
                'image_folder': 'z_image_turbo/',
                'width': 1280, 'height': 768, 'guidance_scale': 0.0,
                'num_inference_steps': 9, 'num_images_per_prompt': 3,
                'max_sequence_length': 512, 'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps","negative_prompt","num_images_per_prompt","max_sequence_length"]
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "Tongyi-MAI/Z-Image-Turbo"
        pipe = load_diffusers_pipeline(
            ZImagePipeline,
            model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

class Kandinsky5T2ILiteSFTGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Kandinsky 5 GUI",
                'image_folder': 'kandinsky/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 50, 'num_images_per_prompt': 3,
                'max_sequence_length': 512, 'callback_on_step_end': True,
                'preview_vae': None,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "negative_prompt","num_images_per_prompt","max_sequence_length"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
        pipe = load_diffusers_pipeline(
            Kandinsky5T2IPipeline,
            model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )
        preview_vae = prepare_preview_vae(
            AutoencoderKL,
            model_id,
            {"subfolder": "vae", "torch_dtype": torch.float16},
        )
        if preview_vae is not None:
            preview_vae.enable_slicing()
            preview_vae.enable_tiling()
            preview_vae.eval()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = preview_vae
            self.model_loading = False

class PixArtSigmaGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "PixArt Sigma GUI",
                'image_folder': 'pixart_sigma/',
                'width': 1024, 'height': 1024, 'guidance_scale': 4.5,
                'num_inference_steps': 100, 'num_images_per_prompt': 1,
                'max_sequence_length': 512, 'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "negative_prompt","num_images_per_prompt","max_sequence_length"
                ]
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        self.pipe = load_diffusers_pipeline(
            PixArtSigmaPipeline,
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )

        with self.model_lock:
            self.model_loading = False

class AnimaGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Anima GUI",
                'image_folder': 'anima/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 40, 'num_images_per_prompt': 3,
                'max_sequence_length': 512,
                'backend': "comfy",
                'callback_on_step_end': False,
                'comfyui_dir': get_path("comfyui_dir"),
                'unet_name': "anima-preview.safetensors",
                'clip_name': "qwen_3_06b_base.safetensors",
                'clip_type': "qwen_image",
                'vae_name': "qwen_image_vae.safetensors",
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None and self.preview_vae is not None and self.comfy_nodes is not None:
                return
            self.model_loading = True

        nodes = connect_comfy_backend(self.comfyui_dir)
        
        model = nodes.UNETLoader(self.unet_name, "default")
        clip = nodes.CLIPLoader(self.clip_name, self.clip_type, "default")
        vae = nodes.VAELoader(self.vae_name)

        with self.model_lock:
            self.comfy_nodes = nodes
            self.pipe = (model, clip)
            self.preview_vae = vae

        with self.model_lock:
            self.model_loading = False

class StableDiffusion35GUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Stable Diffusion 3.5 GUI",
                'image_folder': 'stable_diffusion3.5/',
                'width': 1280, 'height': 768, 'guidance_scale': 4.5,
                'num_inference_steps': 40, 'num_images_per_prompt': 3,
                'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "negative_prompt","num_images_per_prompt"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "stabilityai/stable-diffusion-3.5-large"
        pipe = load_diffusers_pipeline(
            StableDiffusion3Pipeline,
            model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GB"},
            },
        )
        preview_vae = prepare_preview_vae(
            AutoencoderKL,
            model_id,
            {"subfolder": "vae", "torch_dtype": torch.float16},
        )
        if preview_vae is not None:
            preview_vae.enable_slicing()
            preview_vae.enable_tiling()
            preview_vae.eval()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = preview_vae
            self.model_loading = False

class BlackForestFluxGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Black Forest FLUX.1 GUI",
                'image_folder': 'black_forest/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 50, 'num_images_per_prompt': 3,
                'max_sequence_length': 512, 'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "negative_prompt","num_images_per_prompt","max_sequence_length"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "black-forest-labs/FLUX.1-dev"
        pipe = load_diffusers_pipeline(
            FluxPipeline,
            model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )
        preview_vae = prepare_preview_vae(
            AutoencoderKL,
            model_id,
            {"subfolder": "vae", "torch_dtype": torch.float16},
        )
        if preview_vae is not None:
            preview_vae.enable_slicing()
            preview_vae.enable_tiling()
            preview_vae.eval()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = preview_vae
            self.model_loading = False

class GLMImageGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "GLM Image GUI",
                'image_folder': 'glm_image/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 50, 'num_images_per_prompt': 3,
                'max_sequence_length': 2048, 'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "num_images_per_prompt","max_sequence_length","negative_prompt_embeds"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        placement = plan_placement(self.active_plan)
        if is_exact_fast_path(self.active_plan):
            pipe = GLMImageGenerator(plan=self.active_plan)
        else:
            pipe = load_diffusers_pipeline(
                GlmImagePipeline,
                "zai-org/GLM-Image",
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

class QwenImageGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Qwen Image GUI",
                'image_folder': 'qwen_image/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 50, 'num_images_per_prompt': 3,
                'max_sequence_length': 512, 'callback_on_step_end': True,
                'pipeline_args': [
                    "prompt","negative_prompt","true_cfg_scale","width","height",
                    "num_inference_steps","num_images_per_prompt","max_sequence_length"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_exact_fast_path(self.active_plan):
            pipe = QwenImageGenerator(plan=self.active_plan)
        else:
            pipe = load_diffusers_pipeline(
                QwenImagePipeline,
                "Qwen/Qwen-Image",
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

class BlackForestFlux2GUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Black Forest FLUX.2 GUI",
                'image_folder': 'black_forest_flux_2/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 20, 'num_images_per_prompt': 1,
                'max_sequence_length': 256, 'callback_on_step_end': True,
                'use_prompt_model': False, 'show_batch_size': False,
                'seed': '',
                'extra_params': [
                    ("seed", "Seed", int),
                ],
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "max_sequence_length","num_images_per_prompt"
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        placement = plan_placement(self.active_plan)
        if is_exact_fast_path(self.active_plan) or "model_specific_staging" in placement:
            pipe = Flux2Generator(plan=self.active_plan)
        else:
            pipe = load_diffusers_pipeline(
                Flux2Pipeline,
                "black-forest-labs/FLUX.2-dev",
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )
            pipe.vae.enable_tiling()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        stopped = False

        for i in range(self.batch_size):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()
            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            try:
                args = {i: getattr(self, i) for i in self.pipeline_args}
                args["generator"] = self.get_generator()

                if self.callback_on_step_end:
                    args["callback_on_step_end"] = self.on_step_end
                    args["callback_on_step_end_tensor_inputs"] = ["latents"]

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

class SkyReelsV2GUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "SkyReels V2 GUI",
                'gallery_title': "Videos",
                'image_folder': 'skyreels_v2/',

                'width': 960, 'height': 544,
                'guidance_scale': 0.0,
                'num_inference_steps': 30,
                'max_sequence_length': 512,
                'callback_on_step_end': False,
                'seed': '',

                'show_ipp': False,

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("ar_step", "AR step", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("flow_shift", "Flow shift", float),
                    ("seed", "Seed", int),
                ],

                # skyreels-specific
                'num_frames': 97,
                'base_num_frames': 97,
                'ar_step': 0,
                'fps': 24,
                'quality': 8,
                'flow_shift': 8.0,

                'pipeline_args': []
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
        vae = load_component(
            AutoModel,
            model_id,
            "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.float32},
            portable_base_kwargs={"subfolder": "vae"},
        )
        pipe = load_diffusers_pipeline(
            SkyReelsV2DiffusionForcingPipeline,
            model_id,
            current_kwargs={"vae": vae, "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"vae": vae},
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=self.flow_shift)
        if is_exact_fast_path(self.active_plan):
            pipe.enable_model_cpu_offload()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        if not os.path.isdir(self.image_folder):
            os.makedirs(self.image_folder, exist_ok=True)

        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            out = self.pipe(
                prompt=self.prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                base_num_frames=self.base_num_frames,
                ar_step=self.ar_step,
                generator=self.get_generator(),
            ).frames[0]

            base = f"output_{int(time.time())}_{i}"
            mp4_path = os.path.join(self.image_folder, base + ".mp4")
            png_path = os.path.join(self.image_folder, base + ".png")

            export_to_video(out, mp4_path, fps=self.fps, quality=self.quality)
            self.write_file_comment(mp4_path)
            print("saved", os.path.basename(mp4_path))

            # save a thumbnail frame so your existing image gallery + launcher previews work
            first = out[0]
            if hasattr(first, "save"):
                first.save(png_path)
            else:
                if hasattr(first, "detach"):
                    arr = first.detach().cpu().numpy()
                else:
                    arr = first
                if getattr(arr, "dtype", None) != "uint8":
                    arr = (arr * 255).clip(0, 255).astype("uint8")
                Image.fromarray(arr).save(png_path)

            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class Kandinsky5T2VLiteDistilled16GUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Kandinsky 5.0 T2V (Lite distilled16)",
                'gallery_title': "Videos",
                'image_folder': 'kandinsky_tv2_lite/',

                'width': 768, 'height': 512,
                'guidance_scale': 1.0,
                'num_inference_steps': 16,
                'batch_size': 1,
                'seed': '',

                'num_frames': 241,
                'num_videos_per_prompt': 1,
                'fps': 24,
                'quality': 9,

                'ipp_label': "Videos per prompt",
                'ipp_attr': "num_videos_per_prompt",
                'show_ipp': True,

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("seed", "Seed", int),
                ],

                'callback_on_step_end': True,
            }
        )

        self.model_id = "kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers"

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_cuda_plan(self.active_plan):
            flex_attention.flex_attention = torch.compile(
                flex_attention.flex_attention,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
        pipe = load_diffusers_pipeline(
            Kandinsky5T2VPipeline,
            self.model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GB", 1: "22GB", "cpu": "80GB"},
            },
        )
        if is_cuda_plan(self.active_plan):
            pipe.transformer.set_attention_backend("flex")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = self.num_inference_steps
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_videos_per_prompt=self.num_videos_per_prompt,
                callback_on_step_end=self.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=self.get_generator(),
            )

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            for idx, video_frames in enumerate(out.frames):
                ts = int(time.time())
                fname = f"output_{ts}{i}_{idx}.mp4"
                out_path = os.path.join(self.image_folder, fname)

                export_to_video(video_frames, out_path, fps=self.fps, quality=self.quality)
                self.write_file_comment(out_path)
                print(f"saved {out_path}")

                if video_frames and len(video_frames) > 0:
                    frame0 = video_frames[0]
                    if not isinstance(frame0, Image.Image):
                        frame0 = Image.fromarray(frame0)
                    self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class CosmosPredict2V2WGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Cosmos Predict2 V2W (2B)",
                'gallery_title': "Videos",
                'image_folder': 'nvidia_cosmos/',
                'width': 1280,
                'height': 704,
                'batch_size': 1,
                'num_inference_steps': 25,
                'show_ipp': False,
                'seed': '',

                'init_image': "",
                'num_frames': 60,
                'fps': 12,

                'extra_params': [
                    ("init_image", "Input image path", str),
                    ("num_frames", "Frames", int),
                    ("fps", "FPS", int),
                    ("seed", "Seed", int),
                ],

                'callback_on_step_end': False,
                'pipeline_args': []
            }
        )

        self.model_id = "nvidia/Cosmos-Predict2-2B-Video2World"

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        pipe = load_diffusers_pipeline(
            Cosmos2VideoToWorldPipeline,
            self.model_id,
            current_kwargs={"torch_dtype": torch.bfloat16},
        )
        pipe._exclude_from_cpu_offload = ["safety_checker"]
        if is_exact_fast_path(self.active_plan):
            pipe.enable_model_cpu_offload()

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

    def generate_diffusers(self):
        assert self.init_image and os.path.isfile(self.init_image), f"Input image missing:\n{self.init_image}\nabs: {os.path.abspath(self.init_image)}\n"

        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            image = load_image(self.init_image)
            image = ImageOps.fit(image, (self.width, self.height), method=Image.Resampling.LANCZOS)

            result = pipe(
                image=image,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                fps=self.fps,
                generator=self.get_generator(),
            )

            video = result.frames[0]

            base = f"output_{int(time.time())}_{i}"
            mp4_path = os.path.join(self.image_folder, base + ".mp4")

            export_to_video(video, mp4_path, fps=self.fps)
            self.write_file_comment(mp4_path)
            print("saved", mp4_path)

            if video and len(video) > 0:
                frame0 = video[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class AllegroGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Allegro GUI",
                'gallery_title': "Videos",
                'image_folder': 'allegro/',
                'width': 1280,
                'height': 720,
                'guidance_scale': 1.0,
                'num_inference_steps': 100,
                'batch_size': 1,
                'callback_on_step_end': False,
                'show_ipp': False,
                'seed': '',

                'max_sequence_length': 64,
                'fps': 15,
                'num_frames': 60,

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS", int),
                    ("seed", "Seed", int),
                    ("max_sequence_length", "Max sequence length", int),
                ],
            }
        )

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        model_id = "rhymes-ai/Allegro"
        vae = load_component(
            AutoencoderKLAllegro,
            model_id,
            "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.float32},
            portable_base_kwargs={"subfolder": "vae"},
        )
        pipe = load_diffusers_pipeline(
            AllegroPipeline,
            model_id,
            current_kwargs={"vae": vae, "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"vae": vae},
        )
        pipe.vae.enable_tiling()
        if is_exact_fast_path(self.active_plan):
            pipe.enable_model_cpu_offload()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            self.progress_total = self.get_total_runs()
            self.progress_step = i
            self.progress = self.progress_step / self.progress_total
            self.app.after(0, self.update_progress_widgets)

            gen = self.get_generator()

            out = pipe(
                self.prompt,
                negative_prompt=self.negative_prompt,
                guidance_scale=self.guidance_scale,
                max_sequence_length=self.max_sequence_length,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                generator=gen
            ).frames[0]

            fname = f"output_{int(time.time())}_{i}.mp4"
            out_path = os.path.join(self.image_folder, fname)
            export_to_video(out, out_path, fps=int(self.fps))
            self.write_file_comment(out_path)
            print(f"saved {out_path}")

            if out and len(out) > 0:
                frame0 = out[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

            self.progress_step = i + 1
            self.progress = self.progress_step / self.progress_total
            self.app.after(0, self.update_progress_widgets)

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class CogVideoXGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "CogVideoX 5B GUI",
                'gallery_title': "Videos",
                'image_folder': 'cogvideox/',
                'guidance_scale': 6.0,
                'num_inference_steps': 50,
                'batch_size': 1,
                'callback_on_step_end': True,
                'show_ipp': False,
                'seed': '',

                'num_frames': 49,
                'fps': 8,
                'num_videos_per_prompt': 1,

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("num_videos_per_prompt", "Videos per prompt", int),
                    ("seed", "Seed", int),
                ],
            }
        )
        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        pipe = load_diffusers_pipeline(
            CogVideoXPipeline,
            "THUDM/CogVideoX-5b",
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GB", 1: "22GB", "cpu": "80GB"},
            },
        )
        pipe.vae.enable_tiling()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = self.num_inference_steps
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            out = pipe(
                prompt=self.prompt,
                num_videos_per_prompt=self.num_videos_per_prompt,
                num_inference_steps=self.num_inference_steps,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                callback_on_step_end=self.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=self.get_generator(),
            )

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            for idx, video_frames in enumerate(out.frames):
                ts = int(time.time())
                fname = f"output_{ts}_{i}_{idx}.mp4"
                out_path = os.path.join(self.image_folder, fname)

                export_to_video(video_frames, out_path, fps=self.fps)
                self.write_file_comment(out_path)
                print(f"saved {out_path}")

                if video_frames and len(video_frames) > 0:
                    frame0 = video_frames[0]
                    if not isinstance(frame0, Image.Image):
                        frame0 = Image.fromarray(frame0)
                    self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class HunyuanVideoGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "HunyuanVideo 1.5 GUI",
                'gallery_title': "Videos",
                'image_folder': 'hunyuan_video/',
                'prompt': "alien spaceship arriving above a lush jungle planet, cinematic, 4k, high detail, realistic",
                'batch_size': 1,
                'num_inference_steps': 50,
                'num_frames': 121,
                'fps': 4,
                'seed': '',
                'callback_on_step_end': True,
                'show_ipp': False,
                'extra_params': [
                    ("num_frames", "Num frames", int),
                    ("fps", "FPS (export)", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)
        self.supports_callback = False

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        pipe = load_diffusers_pipeline(
            HunyuanVideo15Pipeline,
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        signature = inspect.signature(pipe.__call__)
        supports_callback = "callback_on_step_end" in signature.parameters and "callback_on_step_end_tensor_inputs" in signature.parameters

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.supports_callback = supports_callback
            self.callback_on_step_end = supports_callback
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = max(1, int(self.num_inference_steps))
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe
                supports_callback = self.supports_callback

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = max(1, int(self.num_inference_steps)) if supports_callback else 1
            self.app.after(0, self.update_progress_widgets)

            call_kwargs = dict(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                generator=self.get_generator(),
            )

            if supports_callback:
                call_kwargs["callback_on_step_end"] = self.on_step_end
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            out = pipe(**call_kwargs)

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            video = out.frames[0]
            ts = int(time.time())
            fname = f"output_{ts}_{i}.mp4"
            out_path = os.path.join(self.image_folder, fname)

            export_to_video(video, out_path, fps=self.fps)
            self.write_file_comment(out_path)
            print(f"saved {out_path}")

            if video and len(video) > 0:
                frame0 = video[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class Kandinsky5T2VProDistilled5sGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                "title": "Kandinsky 5.0 T2V Pro SFT 5s GUI",
                "gallery_title": "Videos",
                "image_folder": "kandinsky_tv2_distilled/",
                "prompt": "An alien spaceship invading mars",
                "negative_prompt": (
                    "low quality, blurry, distorted, deformed, bad anatomy, bad proportions, malformed limbs, extra limbs, "
                    "duplicate limbs, extra arms, extra legs, extra hands, extra feet, extra fingers, fused fingers, missing fingers, "
                    "disfigured, mutation, mutated, cloned body parts, unnatural anatomy"
                ),
                "batch_size": 1,
                "width": 1024,
                "height": 768,
                "guidance_scale": 1.0,
                "num_inference_steps": 16,
                "callback_on_step_end": True,
                "show_ipp": False,
                'seed': '',
                "num_frames": 121,
                "fps": 12,
                "quality": 9,
                "num_videos_per_prompt": 1,
                "extra_params": [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("num_videos_per_prompt", "Videos per prompt", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        self.model_id = "kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers"

    def ensure_model_loaded(self):
        if self.pipe is not None:
            return

        if self.model_loading:
            while self.model_loading and self.pipe is None:
                time.sleep(0.2)
            return

        self.load_model()

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_cuda_plan(self.active_plan):
            flex_attention.flex_attention = torch.compile(
                flex_attention.flex_attention,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
        pipe = load_diffusers_pipeline(
            Kandinsky5T2VPipeline,
            self.model_id,
            current_kwargs={"torch_dtype": torch.bfloat16},
        )
        if is_exact_fast_path(self.active_plan):
            pipe.enable_sequential_cpu_offload()
        if is_cuda_plan(self.active_plan):
            pipe.transformer.set_attention_backend("flex")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = self.num_inference_steps
        self.progress = self.progress_step / max(1, self.progress_total)
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_videos_per_prompt=self.num_videos_per_prompt,
                callback_on_step_end=self.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=self.get_generator(),
            )

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            for idx, video_frames in enumerate(out.frames):
                ts = int(time.time())
                mp4_path = os.path.join(self.image_folder, f"output_{ts}_{i}_{idx}.mp4")

                export_to_video(video_frames, mp4_path, fps=self.fps, quality=self.quality)
                self.write_file_comment(mp4_path)
                print(f"saved {mp4_path}")

                if video_frames and len(video_frames) > 0:
                    frame0 = video_frames[0]
                    if not isinstance(frame0, Image.Image):
                        frame0 = Image.fromarray(frame0)
                    self.app.after(0, self.set_preview_image, frame0.copy())

                time.sleep(1)

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class Kandinsky5T2VProSFT5sGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                "title": "Kandinsky 5.0 T2V Pro SFT 5s GUI",
                "gallery_title": "Videos",
                "image_folder": "kandinsky_tv2_pro/",
                "prompt": "An alien spaceship invading mars",
                "negative_prompt": (
                    "low quality, blurry, distorted, deformed, bad anatomy, bad proportions, malformed limbs, extra limbs, "
                    "duplicate limbs, extra arms, extra legs, extra hands, extra feet, extra fingers, fused fingers, missing fingers, "
                    "disfigured, mutation, mutated, cloned body parts, unnatural anatomy"
                ),
                "batch_size": 1,
                "width": 1024,
                "height": 768,
                "guidance_scale": 1.0,
                "num_inference_steps": 16,
                "callback_on_step_end": True,
                "show_ipp": False,
                'seed': '',
                "num_frames": 121,
                "fps": 12,
                "quality": 9,
                "num_videos_per_prompt": 1,
                "extra_params": [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("num_videos_per_prompt", "Videos per prompt", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        self.model_id = "kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers"

    def ensure_model_loaded(self):
        if self.pipe is not None:
            return

        if self.model_loading:
            while self.model_loading and self.pipe is None:
                time.sleep(0.2)
            return

        self.load_model()

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_cuda_plan(self.active_plan):
            flex_attention.flex_attention = torch.compile(
                flex_attention.flex_attention,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
        pipe = load_diffusers_pipeline(
            Kandinsky5T2VPipeline,
            self.model_id,
            current_kwargs={"torch_dtype": torch.bfloat16},
        )
        if is_exact_fast_path(self.active_plan):
            pipe.enable_sequential_cpu_offload()
        if is_cuda_plan(self.active_plan):
            pipe.transformer.set_attention_backend("flex")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = self.num_inference_steps
        self.progress = self.progress_step / max(1, self.progress_total)
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_videos_per_prompt=self.num_videos_per_prompt,
                callback_on_step_end=self.on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=self.get_generator(),
            )

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            for idx, video_frames in enumerate(out.frames):
                ts = int(time.time())
                mp4_path = os.path.join(self.image_folder, f"output_{ts}_{i}_{idx}.mp4")

                export_to_video(video_frames, mp4_path, fps=self.fps, quality=self.quality)
                self.write_file_comment(mp4_path)
                print(f"saved {mp4_path}")

                if video_frames and len(video_frames) > 0:
                    frame0 = video_frames[0]
                    if not isinstance(frame0, Image.Image):
                        frame0 = Image.fromarray(frame0)
                    self.app.after(0, self.set_preview_image, frame0.copy())

                time.sleep(1)

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class Kandinsky5I2IGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Kandinsky 5 I2I GUI",
                'image_folder': 'kandinsky_i2i/',
                'width': 1280, 'height': 768, 'guidance_scale': 3.5,
                'num_inference_steps': 50, 'num_images_per_prompt': 1,
                'callback_on_step_end': True,
                'supports_source_image': True,
                'source_image_path': '',
                'pipeline_args': [
                    "prompt","height","width","guidance_scale","num_inference_steps",
                    "negative_prompt","num_images_per_prompt"
                ],
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        pipe = load_diffusers_pipeline(
            Kandinsky5I2IPipeline,
            "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers",
            current_kwargs={"torch_dtype": torch.bfloat16},
        )
        if is_exact_fast_path(self.active_plan):
            pipe.enable_model_cpu_offload()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

class Kandinsky5I2VGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Kandinsky 5 I2V GUI",
                'gallery_title': "Videos",
                'image_folder': 'kandinsky_i2v/',
                'width': 768,
                'height': 512,
                'guidance_scale': 5.0,
                'num_inference_steps': 50,
                'batch_size': 1,
                'callback_on_step_end': True,
                'show_ipp': False,
                'seed': '',

                'supports_source_image': True,
                'source_image_path': '',
                'source_size_multiple': 16,
                'source_max_dim': 1200,
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'num_frames': 121,
                'fps': 24,
                'quality': 9,

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)
        self.model_id = "kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers"
        self.supports_callback = False

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_cuda_plan(self.active_plan):
            flex_attention.flex_attention = torch.compile(
                flex_attention.flex_attention,
                mode="max-autotune-no-cudagraphs",
                dynamic=True,
            )
        pipe = load_diffusers_pipeline(
            Kandinsky5I2VPipeline,
            self.model_id,
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )
        if is_cuda_plan(self.active_plan):
            pipe.transformer.set_attention_backend("flex")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        signature = inspect.signature(pipe.__call__)
        supports_callback = "callback_on_step_end" in signature.parameters and "callback_on_step_end_tensor_inputs" in signature.parameters

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.supports_callback = supports_callback
            self.callback_on_step_end = supports_callback
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = max(1, int(self.num_inference_steps))
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)

        if self.source_image_pil is None:
            return self.app.after(0, self.finish_generate)

        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe
                supports_callback = self.supports_callback

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = max(1, int(self.num_inference_steps)) if supports_callback else 1
            self.app.after(0, self.update_progress_widgets)

            image = self.source_image_pil.copy().resize((self.width, self.height), Image.Resampling.LANCZOS)

            call_kwargs = dict(
                image=image,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.get_generator(),
            )

            if supports_callback:
                call_kwargs["callback_on_step_end"] = self.on_step_end
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            out = pipe(**call_kwargs)

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            video_frames = out.frames[0]

            ts = int(time.time())
            mp4_path = os.path.join(self.image_folder, f"output_{ts}_{i}.mp4")

            export_to_video(video_frames, mp4_path, fps=self.fps, quality=self.quality)
            self.write_file_comment(mp4_path)
            print(f"saved {mp4_path}")

            if video_frames and len(video_frames) > 0:
                frame0 = video_frames[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class Kandinsky5I2VProSFT5sGUI(DiffusionGUI):
    def __init__(self):
        self.model_id = "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers"
        self.supports_callback = False

        super().__init__(
            args = {
                'title': "Kandinsky 5 I2V Pro SFT 5s GUI",
                'gallery_title': "Videos",
                'image_folder': 'kandinsky_i2v_pro/',
                'width': 512,
                'height': 512,
                'guidance_scale': 5.0,
                'num_inference_steps': 50,
                'batch_size': 1,
                'callback_on_step_end': True,
                'show_ipp': False,
                'use_prompt_model': False,
                'seed': '',

                'supports_source_image': True,
                'source_image_path': '',
                'source_size_multiple': 128,
                'source_max_dim': 1408,
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'num_frames': 121,
                'fps': 24,
                'quality': 9,
                'vram_estimator': Kandinsky5I2VProVramEstimator(),

                'extra_params': [
                    ("num_frames", "Frames", int),
                    ("fps", "FPS (export)", int),
                    ("quality", "MP4 quality", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        placement = plan_placement(self.active_plan)
        if is_exact_fast_path(self.active_plan) or "model_specific_staging" in placement:
            pipe = Kandinsky5I2VGenerator(plan=self.active_plan)
            supports_callback = True
        else:
            pipe = load_diffusers_pipeline(
                Kandinsky5I2VPipeline,
                self.model_id,
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            signature = inspect.signature(pipe.__call__)
            supports_callback = (
                "callback_on_step_end" in signature.parameters
                and "callback_on_step_end_tensor_inputs" in signature.parameters
            )

        with self.model_lock:
            self.pipe = pipe
            self.supports_callback = supports_callback
            self.callback_on_step_end = supports_callback
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = max(1, int(self.num_inference_steps))
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)

        if self.source_image_pil is None:
            return self.app.after(0, self.finish_generate)

        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe
                supports_callback = self.supports_callback

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = max(1, int(self.num_inference_steps)) if supports_callback else 1
            self.app.after(0, self.update_progress_widgets)

            image = self.source_image_pil.copy().resize((self.width, self.height), Image.Resampling.LANCZOS)

            call_kwargs = dict(
                image=image,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.get_generator(),
            )

            if supports_callback:
                call_kwargs["callback_on_step_end"] = self.on_step_end
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            out = pipe(**call_kwargs)

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            video_frames = out.frames[0]

            ts = int(time.time())
            mp4_path = os.path.join(self.image_folder, f"output_{ts}_{i}.mp4")

            export_to_video(video_frames, mp4_path, fps=self.fps, quality=self.quality)
            self.write_file_comment(mp4_path)
            print(f"saved {mp4_path}")

            if video_frames and len(video_frames) > 0:
                frame0 = video_frames[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

class ChronoEditGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "ChronoEdit GUI",
                'image_folder': 'chronoedit/',
                'prompt': "Make the photo look warmer and slightly brighter, natural, realistic",
                'batch_size': 1,
                'guidance_scale': 4.0,
                'num_inference_steps': 24,
                'callback_on_step_end': False,

                'supports_source_image': True,
                'source_image_path': '',
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'extra_params': [
                    ("strength", "Strength", float),
                ],
                'strength': 0.75,

                'pipeline_args': [
                    "prompt","negative_prompt","guidance_scale","num_inference_steps","strength"
                ],
            }
        )
        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        if is_exact_fast_path(self.active_plan):
            pipe = ChronoEditGenerator(plan=self.active_plan)
        else:
            pipe = load_diffusers_pipeline(
                ChronoEditPipeline,
                "nvidia/ChronoEdit-14B-Diffusers",
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        stopped = False
        for index in range(self.batch_size):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()
            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = self.num_inference_steps
            self.app.after(0, self.update_progress_widgets)

            args = {name: getattr(self, name) for name in self.pipeline_args}
            if self.source_image_pil is not None:
                args["image"] = self.source_image_pil.copy()
            args["generator"] = self.get_generator()

            output = self.pipe(**args)
            images = getattr(output, "images", None)
            if images is None:
                frames = getattr(output, "frames", [])
                images = []
                for video in frames:
                    image = video[-1]
                    if not isinstance(image, Image.Image):
                        image = np.asarray(image)
                        if image.dtype != np.uint8:
                            image = (image * 255).clip(0, 255).astype(np.uint8)
                        image = Image.fromarray(image)
                    images.append(image)

            for image_index, image in enumerate(images or []):
                filename = f"output_{int(time.time())}_{index}_{image_index}.png"
                output_path = os.path.join(self.image_folder, filename)
                image.save(output_path)
                self.app.after(0, self.set_preview_image, image.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)
        self.app.after(0, self.finish_generate)


class QwenImageEditGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "Qwen Image Edit GUI",
                'image_folder': 'qwen_image_edit/',
                'prompt': "Clean up the image and improve lighting, keep it realistic",
                'negative_prompt': " ",

                'batch_size': 1,
                'num_images_per_prompt': 1,
                'show_ipp': False,

                'auto_size_from_source_on_select': True,
                'supports_source_image': True,
                'source_image_path': '',
                'lock_source_aspect_ratio': True,

                'num_inference_steps': 40,
                'true_cfg_scale': 4.0,
                'guidance_scale': 1.0,

                'callback_on_step_end': False,

                'extra_params': [
                    ("true_cfg_scale", "True CFG", float),
                    ("guidance_scale", "Guidance", float),
                    ("num_inference_steps", "Steps", int),
                ],

                'pipeline_args': [
                    "prompt","negative_prompt","num_inference_steps","true_cfg_scale","guidance_scale"
                ],
            }
        )
        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        placement = plan_placement(self.active_plan)
        if is_exact_fast_path(self.active_plan) or "model_specific_staging" in placement:
            pipe = QwenImageEditGenerator(plan=self.active_plan)
        else:
            pipe = load_diffusers_pipeline(
                QwenImageEditPlusPipeline,
                "ovedrive/Qwen-Image-Edit-2511-4bit",
                current_kwargs={"torch_dtype": torch.bfloat16},
                plan=self.active_plan,
            )
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

def _rmbg_all_tied_weights_keys(self):
    v = getattr(self, "_tied_weights_keys", None)
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, (list, tuple, set)):
        return dict.fromkeys(list(v), True)
    return {}

class RMBG14GUI(DiffusionGUI):
    def __init__(self):
        self.rmbg_pipe = None

        super().__init__(
            args = {
                'title': "BRIA RMBG 1.4 GUI",
                'image_folder': 'rmbg_1_4/',
                'prompt': "Remove background",
                'batch_size': 1,
                'callback_on_step_end': False,
                'show_ipp': False,

                'supports_source_image': True,
                'source_image_path': '',
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': True,
                'source_size_multiple': 1,
                'source_max_dim': 0,

                'extra_params': [
                    ("invert_mask", "Invert mask (0/1)", int),
                ],
                'invert_mask': 0,

                'pipeline_args': []
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.rmbg_pipe is not None:
                return
            self.model_loading = True

        model_id = "briaai/RMBG-1.4"
        device = execution_device(self.active_plan)
        pipeline_device = 0 if device.startswith("cuda") else ("mps" if device == "mps" else -1)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        temporary_model = AutoModelForImageSegmentation.from_config(cfg, trust_remote_code=True)
        model_class = temporary_model.__class__
        del temporary_model
        if not hasattr(model_class, "all_tied_weights_keys"):
            model_class.all_tied_weights_keys = property(_rmbg_all_tied_weights_keys)
        rmbg_pipe = pipeline(
            "image-segmentation",
            model=model_id,
            trust_remote_code=True,
            device=pipeline_device,
        )

        with self.model_lock:
            self.rmbg_pipe = rmbg_pipe
            self.pipe = rmbg_pipe
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)

        input_path = (self.source_image_path and os.path.isfile(self.source_image_path)) and self.source_image_path or ""
        if not input_path:
            return self.app.after(0, self.finish_generate)

        with self.model_lock:
            rmbg_pipe = self.rmbg_pipe

        if rmbg_pipe is None:
            return self.app.after(0, self.finish_generate)

        self.progress_total = 1
        self.progress_step = 0
        self.progress = 0.0
        self.app.after(0, self.update_progress_widgets)

        mask = rmbg_pipe(input_path, return_mask=True)

        img = Image.open(input_path).convert("RGBA")
        img.putalpha(mask)

        if int(self.invert_mask or 0) == 1:
            img = img.copy()
            img.putalpha(ImageOps.invert(img.getchannel("A")))

        ts = int(time.time())
        out_path = os.path.join(self.image_folder, f"output_{ts}.png")
        img.save(out_path)

        self.app.after(0, self.set_preview_image, img.convert("RGB"))

        self.progress = 1.0
        self.progress_step = 1
        self.app.after(0, self.update_progress_widgets)
        self.app.after(0, self.finish_generate)

class RealESRGANGUI(DiffusionGUI):
    def __init__(self):
        self.upsampler = None

        super().__init__(
            args = {
                'title': "Real-ESRGAN GUI",
                'image_folder': 'real_esrgan/',
                'prompt': "Upscale",
                'batch_size': 1,
                'callback_on_step_end': False,
                'show_ipp': False,

                'supports_source_image': True,
                'source_image_path': '',
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'extra_params': [
                    ("outscale", "Outscale", float),
                ],
                'outscale': 4.0,

                'pipeline_args': []
            }
        )
        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.upsampler is not None:
                return
            self.model_loading = True

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        weights = hf_hub_download(repo_id="lllyasviel/Annotators", filename="RealESRGAN_x4plus.pth")
        use_cuda = execution_device(self.active_plan).startswith("cuda")
        upsampler = RealESRGANer(
            scale=4,
            model_path=weights,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=use_cuda,
            gpu_id=0 if use_cuda else None,
        )

        with self.model_lock:
            self.upsampler = upsampler
            self.pipe = upsampler
            self.preview_vae = None
            self.model_loading = False

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)

        if self.source_image_pil is None:
            return self.app.after(0, self.finish_generate)

        with self.model_lock:
            upsampler = self.upsampler

        if upsampler is None:
            return self.app.after(0, self.finish_generate)

        self.progress_total = 1
        self.progress_step = 0
        self.progress = 0.0
        self.app.after(0, self.update_progress_widgets)

        rgb = self.source_image_pil.copy().convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

        outscale = float(getattr(self, "outscale", 4.0) or 4.0)
        out_bgr, _ = upsampler.enhance(bgr, outscale=outscale)

        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        out_img = Image.fromarray(out_rgb)

        ts = int(time.time())
        out_path = os.path.join(self.image_folder, f"output_{ts}.png")
        out_img.save(out_path)
        self.app.after(0, self.set_preview_image, out_img.copy())

        self.progress = 1.0
        self.progress_step = 1
        self.app.after(0, self.update_progress_widgets)
        self.app.after(0, self.finish_generate)

class SDX4UpscalerGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "SD x4 Upscaler GUI",
                'image_folder': 'sd_x4_upscaler/',
                'prompt': "High quality, natural, realistic",
                'batch_size': 1,
                'guidance_scale': 7.0,
                'num_inference_steps': 30,
                'callback_on_step_end': False,
                'supports_source_image': True,
                'source_image_path': '',
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'pipeline_args': [
                    "prompt","negative_prompt","guidance_scale","num_inference_steps"
                ],
            }
        )
        os.makedirs(self.image_folder, exist_ok=True)

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        pipe = load_diffusers_pipeline(
            StableDiffusionUpscalePipeline,
            "stabilityai/stable-diffusion-x4-upscaler",
            current_kwargs={"torch_dtype": torch.float16},
        )
        if is_exact_fast_path(self.active_plan):
            pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.model_loading = False

class HunyuanVideoI2VGUI(DiffusionGUI):
    def __init__(self):
        super().__init__(
            args = {
                'title': "HunyuanVideo 1.5 I2V GUI",
                'gallery_title': "Videos",
                'image_folder': 'hunyuan_video_i2v/',
                'prompt': "a person dancing on a stage, cinematic lighting, realistic, high detail",
                'batch_size': 1,
                'width': 1280,
                'height': 720,
                'num_inference_steps': 50,
                'num_frames': 24,
                'fps': 8,
                'callback_on_step_end': True,
                'show_ipp': False,
                'seed': '',

                'supports_source_image': True,
                'source_image_path': '',
                'source_size_multiple': 16,
                'source_max_dim': 1280,
                'lock_source_aspect_ratio': True,
                'auto_size_from_source_on_select': False,

                'extra_params': [
                    ("num_frames", "Num frames", int),
                    ("fps", "FPS (export)", int),
                    ("seed", "Seed", int),
                ],
            }
        )

        os.makedirs(self.image_folder, exist_ok=True)
        self.supports_callback = False

    def load_model(self):
        with self.model_lock:
            if self.pipe is not None:
                return
            self.model_loading = True

        from diffusers import HunyuanVideo15ImageToVideoPipeline

        pipe = load_diffusers_pipeline(
            HunyuanVideo15ImageToVideoPipeline,
            "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
            current_kwargs={
                "torch_dtype": torch.bfloat16,
                "device_map": "balanced",
                "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "80GiB"},
            },
        )
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        signature = inspect.signature(pipe.__call__)
        supports_callback = "callback_on_step_end" in signature.parameters and "callback_on_step_end_tensor_inputs" in signature.parameters

        with self.model_lock:
            self.pipe = pipe
            self.preview_vae = None
            self.supports_callback = supports_callback
            self.callback_on_step_end = supports_callback
            self.model_loading = False

    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        if self.stop_requested:
            pipe._interrupt = True

        self.progress_step = step + 1
        self.progress_total = max(1, int(self.num_inference_steps))
        self.progress = self.progress_step / self.progress_total
        self.app.after(0, self.update_progress_widgets)
        return callback_kwargs

    def generate_diffusers(self):
        os.makedirs(self.image_folder, exist_ok=True)

        if self.source_image_pil is None:
            return self.app.after(0, self.finish_generate)

        stopped = False

        for i in range(self.get_total_runs()):
            if self.stop_requested:
                stopped = True
                break

            self.prepare_run_seed()

            with self.model_lock:
                pipe = self.pipe
                supports_callback = self.supports_callback

            if pipe is None:
                stopped = True
                break

            pipe._interrupt = False

            self.progress = 0.0
            self.progress_step = 0
            self.progress_total = max(1, int(self.num_inference_steps)) if supports_callback else 1
            self.app.after(0, self.update_progress_widgets)

            image = self.source_image_pil.copy().resize((self.width, self.height), Image.Resampling.LANCZOS)

            call_kwargs = dict(
                image=image,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                generator=self.get_generator(),
            )

            if supports_callback:
                call_kwargs["callback_on_step_end"] = self.on_step_end
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            out = pipe(**call_kwargs)

            if self.stop_requested or getattr(pipe, "_interrupt", False):
                stopped = True
                break

            video = out.frames[0]
            ts = int(time.time())
            fname = f"output_{ts}_{i}.mp4"
            out_path = os.path.join(self.image_folder, fname)

            export_to_video(video, out_path, fps=self.fps)
            self.write_file_comment(out_path)
            print(f"saved {out_path}")

            if video and len(video) > 0:
                frame0 = video[0]
                if not isinstance(frame0, Image.Image):
                    frame0 = Image.fromarray(frame0)
                self.app.after(0, self.set_preview_image, frame0.copy())

        if not stopped:
            self.progress = 1.0
            self.progress_step = self.progress_total
            self.app.after(0, self.update_progress_widgets)

        self.app.after(0, self.finish_generate)

MODEL_REGISTRY = {
    "z_image_turbo": ZImageTurboGUI,
    "kandinsky_5": Kandinsky5T2ILiteSFTGUI,
    "pixart_sigma": PixArtSigmaGUI,
    "anima": AnimaGUI,
    "stable_diffusion_3_5": StableDiffusion35GUI,
    "flux_1": BlackForestFluxGUI,
    "glm_image": GLMImageGUI,
    "qwen_image": QwenImageGUI,
    "flux_2": BlackForestFlux2GUI,
    "skyreels_v2": SkyReelsV2GUI,
    "kandinsky_5_t2v": Kandinsky5T2VLiteDistilled16GUI,
    "nvidia_cosmos": CosmosPredict2V2WGUI,
    "allegro": AllegroGUI,
    "cogvideox": CogVideoXGUI,
    "hunyuan_video_1_5": HunyuanVideoGUI,
    "hunyuan_video_1_5_i2v": HunyuanVideoI2VGUI,
    "kandinsky_5_t2v_pro": Kandinsky5T2VProDistilled5sGUI,
    "kandinsky_5_t2v_pro_sft": Kandinsky5T2VProSFT5sGUI,
    "kandinsky_5_i2i": Kandinsky5I2IGUI,
    "kandinsky_5_i2v": Kandinsky5I2VGUI,
    "kandinsky_5_i2v_pro_sft": Kandinsky5I2VProSFT5sGUI,
    "chronoedit": ChronoEditGUI,
    "qwen_image_edit": QwenImageEditGUI,
    "rmbg_1_4": RMBG14GUI,
    "real_esrgan": RealESRGANGUI,
    "sd_x4_upscaler": SDX4UpscalerGUI,
}

def get_flag_value(flag):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

if __name__ == "__main__":
    model_id = get_flag_value("--model-id")

    if not model_id:
        print("usage: python model_gui.py --model-id <model_id>")
        print("available:")
        for k in MODEL_REGISTRY:
            print(" -", k)
        sys.exit(0)

    gui_cls = MODEL_REGISTRY.get(model_id)
    if not gui_cls:
        print("unknown model-id:", model_id)
        print("available:")
        for k in MODEL_REGISTRY:
            print(" -", k)
        sys.exit(0)

    gui = gui_cls()
    gui.launcher_model_id = model_id
    gui.main()
    gui.app.mainloop()
