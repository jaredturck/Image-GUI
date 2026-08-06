import gc
import threading
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.attention.flex_attention as flex_attention
from PIL import Image
from diffusers import Flux2Pipeline
from diffusers import Flux2Transformer2DModel
from diffusers import AutoencoderKLFlux2
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import Kandinsky5I2VPipeline
from diffusers import Kandinsky5Transformer3DModel
from diffusers import AutoencoderKLHunyuanVideo
from diffusers import QwenImagePipeline
from diffusers import QwenImageEditPlusPipeline
from diffusers import QwenImageTransformer2DModel
from diffusers import AutoencoderKLQwenImage
from diffusers import GlmImagePipeline
from diffusers import GlmImageTransformer2DModel
from diffusers import AutoencoderKL
from diffusers import AutoencoderKLWan
from diffusers import ChronoEditPipeline
from diffusers import ChronoEditTransformer3DModel
from diffusers import UniPCMultistepScheduler
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
from transformers import AutoProcessor
from transformers import Mistral3ForConditionalGeneration
from transformers import Qwen2VLProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
from transformers import Qwen2Tokenizer
from transformers import GlmImageProcessor
from transformers import GlmImageForConditionalGeneration
from transformers import T5EncoderModel
from transformers import ByT5Tokenizer
from transformers import AutoTokenizer
from transformers import CLIPProcessor
from transformers import CLIPVisionModel
from transformers import UMT5EncoderModel
from transformers import BitsAndBytesConfig
from planner_runtime import build_block_device_map, clear_accelerator_cache, component_device_map, diffusers_quantization_config, execution_device, get_active_plan, is_cuda_plan, is_exact_fast_path, load_component, plan_max_memory, run_guarded, secondary_device, set_runtime_phase, torch_dtype, transformers_quantization_config

class Flux2Generator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "black-forest-labs/FLUX.2-dev"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB", "cpu": "48GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)

        if is_exact_fast_path(self.plan):
            self.transformer_device_map = {
                "pos_embed": 0,
                "time_guidance_embed": 0,
                "double_stream_modulation_img": 0,
                "double_stream_modulation_txt": 0,
                "single_stream_modulation": 0,
                "x_embedder": 0,
                "context_embedder": 0,
                "transformer_blocks": 0,
                "norm_out": 1,
                "proj_out": 1,
            }
            self.transformer_device_map.update({f"single_transformer_blocks.{index}": 0 for index in range(24)})
            self.transformer_device_map.update({f"single_transformer_blocks.{index}": 1 for index in range(24, 48)})
        else:
            self.transformer_device_map = build_block_device_map(
                "single_transformer_blocks",
                48,
                fixed_first=[
                    "pos_embed",
                    "time_guidance_embed",
                    "double_stream_modulation_img",
                    "double_stream_modulation_txt",
                    "single_stream_modulation",
                    "x_embedder",
                    "context_embedder",
                    "transformer_blocks",
                ],
                fixed_last=["norm_out", "proj_out"],
                plan=self.plan,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id, subfolder="tokenizer")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.vae = load_component(
            AutoencoderKLFlux2,
            self.model_id,
            "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "vae"},
            plan=self.plan,
            portable_device_map={"": "cpu"},
        )
        self.vae.eval()
        self.vae.enable_tiling()

        self.text_encoder_lock = threading.Lock()
        self.text_encoder_loading = False
        self.preload_enabled = True
        self.text_encoder = self.load_text_encoder()

    def load_text_encoder(self):
        set_runtime_phase(self.plan, "text_encoder_loading")
        current_quantization = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head"],
        )
        return load_component(
            Mistral3ForConditionalGeneration,
            self.model_id,
            "text_encoder",
            current_kwargs={
                "subfolder": "text_encoder",
                "torch_dtype": torch.bfloat16,
                "quantization_config": current_quantization,
                "device_map": "balanced",
                "max_memory": self.max_memory,
            },
            portable_base_kwargs={"subfolder": "text_encoder"},
            shardable=True,
            plan=self.plan,
        ).eval()

    def preload_text_encoder(self):
        with self.text_encoder_lock:
            if not self.preload_enabled or self.text_encoder is not None or self.text_encoder_loading:
                return
            self.text_encoder_loading = True

        threading.Thread(target=self.finish_text_encoder_preload, daemon=True).start()

    def finish_text_encoder_preload(self):
        text_encoder = run_guarded(
            self.plan,
            "text_encoder_preload",
            self.plan.get("workload", {}),
            self.load_text_encoder,
        )

        with self.text_encoder_lock:
            keep_text_encoder = self.preload_enabled
            if keep_text_encoder:
                self.text_encoder = text_encoder
            self.text_encoder_loading = False

        if not keep_text_encoder:
            del text_encoder
            gc.collect()
            clear_accelerator_cache()

    def prepare_next_generation(self):
        self.preload_text_encoder()

    def is_ready(self):
        with self.text_encoder_lock:
            return self.text_encoder is not None and not self.text_encoder_loading

    def close(self):
        with self.text_encoder_lock:
            self.preload_enabled = False
            text_encoder = self.text_encoder
            self.text_encoder = None

        if text_encoder is not None:
            del text_encoder

        gc.collect()
        clear_accelerator_cache()

    def encode_prompt(self, text_encoder, prompt, max_sequence_length):
        return Flux2Pipeline._get_mistral_3_small_prompt_embeds(
            text_encoder=text_encoder,
            tokenizer=self.processor,
            prompt=prompt,
            dtype=torch_dtype(self.plan),
            device=torch.device(self.primary_device),
            max_sequence_length=max_sequence_length,
            hidden_states_layers=(10, 20, 30),
        ).cpu()

    def load_pipeline(self):
        set_runtime_phase(self.plan, "transformer_loading")
        current_quantization = DiffusersBitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=[
                "x_embedder",
                "context_embedder",
                "time_guidance_embed",
                "double_stream_modulation_img",
                "double_stream_modulation_txt",
                "single_stream_modulation",
                "norm_out",
                "proj_out",
            ],
        )
        if is_exact_fast_path(self.plan):
            transformer = Flux2Transformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                quantization_config=current_quantization,
                device_map=self.transformer_device_map,
                max_memory=self.max_memory,
            )
        elif is_cuda_plan(self.plan):
            transformer = Flux2Transformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                torch_dtype=torch_dtype(self.plan),
                quantization_config=diffusers_quantization_config("transformer", self.plan),
                device_map=self.transformer_device_map,
                max_memory=self.max_memory,
            )
        else:
            transformer = load_component(
                Flux2Transformer2DModel,
                self.model_id,
                "transformer",
                current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16},
                portable_base_kwargs={"subfolder": "transformer"},
                shardable=True,
                plan=self.plan,
            )
        transformer.eval()
        return Flux2Pipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            transformer=transformer,
        )

    def decode_latents(self, pipe, packed_latents, height, width):
        pipe.vae.to(self.secondary_device)
        packed_latents = packed_latents.to(self.secondary_device)
        latent_stub = torch.zeros(
            packed_latents.shape[0],
            128,
            height // 16,
            width // 16,
            device=packed_latents.device,
            dtype=packed_latents.dtype,
        )
        latent_ids = Flux2Pipeline._prepare_latent_ids(latent_stub).to(packed_latents.device)
        latents = Flux2Pipeline._unpack_latents_with_ids(packed_latents, latent_ids)
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = Flux2Pipeline._unpatchify_latents(latents)
        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        height=768,
        width=768,
        guidance_scale=3.5,
        num_inference_steps=20,
        max_sequence_length=256,
        num_images_per_prompt=1,
        generator=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        with self.text_encoder_lock:
            text_encoder = self.text_encoder
            self.text_encoder = None

        set_runtime_phase(self.plan, "text_encoding")
        prompt_embeds = self.encode_prompt(text_encoder, prompt, max_sequence_length)

        del text_encoder
        gc.collect()
        clear_accelerator_cache()

        pipe = self.load_pipeline()

        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]

        set_runtime_phase(self.plan, "denoising")
        packed_latents = pipe(
            prompt_embeds=prompt_embeds.to(self.primary_device),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            output_type="latent"
        ).images

        pipe.transformer = None

        gc.collect()
        clear_accelerator_cache()

        set_runtime_phase(self.plan, "vae_decode")
        images = self.decode_latents(pipe, packed_latents, height, width)

        self.vae.to("cpu")
        clear_accelerator_cache()

        return images

    def __call__(
        self,
        prompt,
        height=768,
        width=768,
        guidance_scale=3.5,
        num_inference_steps=20,
        max_sequence_length=256,
        num_images_per_prompt=1,
        generator=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        images = self.generate(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        return SimpleNamespace(images=images)

class GLMImagePipelineGPU1(GlmImagePipeline):
    @property
    def _execution_device(self):
        return torch.device(getattr(self, "planner_execution_device", "cuda:1"))

class GLMImageGenerator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "zai-org/GLM-Image"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)
        secondary_index = 1 if self.secondary_device == "cuda:1" else 0

        processor = GlmImageProcessor.from_pretrained(self.model_id, subfolder="processor")
        tokenizer = ByT5Tokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        vision_language_encoder = load_component(
            GlmImageForConditionalGeneration,
            self.model_id,
            "vision_language_encoder",
            current_kwargs={"subfolder": "vision_language_encoder", "torch_dtype": torch.bfloat16, "device_map": {"": 0}, "max_memory": self.max_memory},
            portable_base_kwargs={"subfolder": "vision_language_encoder"},
            shardable=True,
            plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        text_encoder = load_component(
            T5EncoderModel,
            self.model_id,
            "text_encoder",
            current_kwargs={"subfolder": "text_encoder", "torch_dtype": torch.bfloat16, "device_map": {"": 1}, "max_memory": self.max_memory},
            portable_base_kwargs={"subfolder": "text_encoder"},
            plan=self.plan,
            portable_device_map={"": secondary_index} if is_cuda_plan(self.plan) else None,
        )
        transformer = load_component(
            GlmImageTransformer2DModel,
            self.model_id,
            "transformer",
            current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16, "device_map": {"": 1}, "max_memory": self.max_memory},
            portable_base_kwargs={"subfolder": "transformer"},
            shardable=True,
            plan=self.plan,
            portable_device_map={"": secondary_index} if is_cuda_plan(self.plan) else None,
        )
        vae = load_component(
            AutoencoderKL,
            self.model_id,
            "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "vae"},
            plan=self.plan,
        ).to(self.secondary_device)

        vision_language_encoder.eval()
        text_encoder.eval()
        transformer.eval()
        vae.eval()
        vae.enable_tiling()
        vae.enable_slicing()
        if is_exact_fast_path(self.plan):
            self.validate_model_devices("vision-language encoder", vision_language_encoder, {"cuda:0"})
            self.validate_model_devices("text encoder", text_encoder, {"cuda:1"})
            self.validate_model_devices("transformer", transformer, {"cuda:1"})
            self.validate_model_devices("VAE", vae, {"cuda:1"})

        self.pipe = GLMImagePipelineGPU1(
            tokenizer=tokenizer,
            processor=processor,
            text_encoder=text_encoder,
            vision_language_encoder=vision_language_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.pipe.planner_execution_device = self.secondary_device

    def validate_model_devices(self, name, model, expected_devices):
        parameter_devices = {
            str(parameter.device)
            for parameter in model.parameters()
        }

        if parameter_devices != expected_devices:
            raise RuntimeError(
                f"GLM Image {name} parameter devices are {sorted(parameter_devices)}, "
                f"expected {sorted(expected_devices)}"
            )

        return parameter_devices

    def encode_prompt(self, **kwargs):
        kwargs["device"] = torch.device(self.secondary_device)
        kwargs["dtype"] = torch_dtype(self.plan)
        return self.pipe.encode_prompt(**kwargs)

    def close(self):
        if self.pipe is not None:
            self.pipe.vision_language_encoder = None
            self.pipe.text_encoder = None
            self.pipe.transformer = None
            self.pipe.vae = None
            self.pipe = None

        clear_accelerator_cache()

    def __call__(self, **kwargs):
        if kwargs.get("prior_token_ids") is None:
            prompt = kwargs.get("prompt")
            image = kwargs.get("image")
            height = kwargs.get("height")
            width = kwargs.get("width")
            generator = kwargs.get("generator")

            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)

            normalized_image = self.pipe._validate_and_normalize_images(image, batch_size)
            ar_generator = generator[0] if isinstance(generator, list) else generator

            prior_token_ids, prior_token_image_ids, source_image_grid_thw = self.pipe.generate_prior_tokens(
                prompt=prompt,
                image=normalized_image,
                height=height,
                width=width,
                device=torch.device(self.primary_device),
                generator=ar_generator
            )

            kwargs["prior_token_ids"] = prior_token_ids.to(self.secondary_device)

            if prior_token_image_ids is not None:
                kwargs["prior_token_image_ids"] = [ids.to(self.secondary_device) for ids in prior_token_image_ids]

            if source_image_grid_thw is not None:
                kwargs["source_image_grid_thw"] = [grid.to(self.secondary_device) for grid in source_image_grid_thw]

        return self.pipe(**kwargs)

class QwenImageGenerator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "Qwen/Qwen-Image"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)
        if is_exact_fast_path(self.plan):
            self.transformer_device_map = {
                "pos_embed": 0, "time_text_embed": 0, "txt_norm": 0, "img_in": 0, "txt_in": 0,
                "norm_out": 1, "proj_out": 1,
            }
            self.transformer_device_map.update({f"transformer_blocks.{index}": 0 for index in range(16)})
            self.transformer_device_map.update({f"transformer_blocks.{index}": 1 for index in range(16, 60)})
        else:
            self.transformer_device_map = build_block_device_map(
                "transformer_blocks", 60,
                fixed_first=["pos_embed", "time_text_embed", "txt_norm", "img_in", "txt_in"],
                fixed_last=["norm_out", "proj_out"], plan=self.plan,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        text_encoder = load_component(
            Qwen2_5_VLForConditionalGeneration, self.model_id, "text_encoder",
            current_kwargs={
                "subfolder": "text_encoder", "torch_dtype": torch.bfloat16,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
                "device_map": {"": 0}, "max_memory": self.max_memory,
            },
            portable_base_kwargs={"subfolder": "text_encoder"}, plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        if is_exact_fast_path(self.plan):
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_threshold=6.0,
                    llm_int8_skip_modules=["time_text_embed", "img_in", "txt_in", "norm_out", "proj_out"],
                ),
                device_map=self.transformer_device_map, max_memory=self.max_memory,
            )
        elif is_cuda_plan(self.plan):
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=torch_dtype(self.plan),
                quantization_config=diffusers_quantization_config("transformer", self.plan),
                device_map=self.transformer_device_map, max_memory=self.max_memory,
            )
        else:
            transformer = load_component(
                QwenImageTransformer2DModel, self.model_id, "transformer",
                current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16},
                portable_base_kwargs={"subfolder": "transformer"}, shardable=True, plan=self.plan,
            )
        vae = load_component(
            AutoencoderKLQwenImage, self.model_id, "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "vae"}, plan=self.plan,
        ).to(self.primary_device)
        text_encoder.eval(); transformer.eval(); vae.eval(); vae.enable_tiling(); vae.enable_slicing()
        if is_exact_fast_path(self.plan):
            self.validate_model_devices("text encoder", text_encoder, {"cuda:0"})
            self.validate_model_devices("transformer", transformer, {"cuda:0", "cuda:1"})
            self.validate_model_devices("VAE", vae, {"cuda:0"})
        self.pipe = QwenImagePipeline(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer,
        )

    def validate_model_devices(self, name, model, expected_devices):
        parameter_devices = {
            str(parameter.device)
            for parameter in model.parameters()
        }

        if parameter_devices != expected_devices:
            raise RuntimeError(
                f"Qwen Image {name} parameter devices are {sorted(parameter_devices)}, "
                f"expected {sorted(expected_devices)}"
            )

        return parameter_devices

    def close(self):
        if self.pipe is not None:
            self.pipe.text_encoder = None
            self.pipe.transformer = None
            self.pipe.vae = None
            self.pipe = None

        clear_accelerator_cache()

    def __call__(self, **kwargs):
        return self.pipe(**kwargs)


class QwenImageEditPlusPipelineGPU0(QwenImageEditPlusPipeline):
    @property
    def _execution_device(self):
        return torch.device(getattr(self, "planner_execution_device", "cuda:0"))

class QwenImageEditGenerator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "ovedrive/Qwen-Image-Edit-2511-4bit"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)
        if is_exact_fast_path(self.plan):
            self.transformer_device_map = {
                "pos_embed": 0, "time_text_embed": 0, "txt_norm": 0, "img_in": 0, "txt_in": 0,
                "norm_out": 1, "proj_out": 1,
            }
            self.transformer_device_map.update({f"transformer_blocks.{index}": 0 for index in range(30)})
            self.transformer_device_map.update({f"transformer_blocks.{index}": 1 for index in range(30, 60)})
        else:
            self.transformer_device_map = build_block_device_map(
                "transformer_blocks", 60,
                fixed_first=["pos_embed", "time_text_embed", "txt_norm", "img_in", "txt_in"],
                fixed_last=["norm_out", "proj_out"], plan=self.plan,
            )
        processor = Qwen2VLProcessor.from_pretrained(self.model_id, subfolder="processor")
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        vae = load_component(
            AutoencoderKLQwenImage, self.model_id, "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "vae"}, plan=self.plan,
        ).to(self.primary_device)
        vae.eval(); vae.enable_tiling(); vae.enable_slicing()
        self.pipe = QwenImageEditPlusPipelineGPU0(
            scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer, processor=processor, transformer=None,
        )
        self.pipe.planner_execution_device = self.primary_device

    def clear_cuda_cache(self):
        clear_accelerator_cache()

    def validate_model_devices(self, name, model, expected_devices):
        parameter_devices = {
            str(parameter.device)
            for parameter in model.parameters()
        }

        if parameter_devices != expected_devices:
            raise RuntimeError(
                f"Qwen Image Edit {name} parameter devices are {sorted(parameter_devices)}, "
                f"expected {sorted(expected_devices)}"
            )

        return parameter_devices

    def load_text_encoder(self):
        set_runtime_phase(self.plan, "text_encoder_loading")
        text_encoder = load_component(
            Qwen2_5_VLForConditionalGeneration, self.model_id, "text_encoder",
            current_kwargs={
                "subfolder": "text_encoder", "torch_dtype": torch.bfloat16,
                "device_map": {"": 0}, "max_memory": self.max_memory,
            },
            portable_base_kwargs={"subfolder": "text_encoder"}, plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        text_encoder.eval()
        if is_exact_fast_path(self.plan):
            self.validate_model_devices("text encoder", text_encoder, {"cuda:0"})
        return text_encoder

    def load_transformer(self):
        set_runtime_phase(self.plan, "transformer_loading")
        if is_cuda_plan(self.plan):
            transformer = QwenImageTransformer2DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=torch_dtype(self.plan),
                device_map=self.transformer_device_map, max_memory=self.max_memory,
            )
        else:
            transformer = load_component(
                QwenImageTransformer2DModel, self.model_id, "transformer",
                current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16},
                portable_base_kwargs={"subfolder": "transformer"}, shardable=True, plan=self.plan,
            )
        transformer.eval()
        if is_exact_fast_path(self.plan):
            self.validate_model_devices("transformer", transformer, {"cuda:0", "cuda:1"})
        return transformer

    def prepare_condition_images(self, image):
        images = image if isinstance(image, list) else [image]
        condition_images = []

        for source_image in images:
            width, height = source_image.size
            condition_width, condition_height = calculate_dimensions(384 * 384, width / height)
            condition_images.append(
                self.pipe.image_processor.resize(source_image, condition_height, condition_width)
            )

        return condition_images

    def encode_prompts(self, image, prompt, negative_prompt, true_cfg_scale, max_sequence_length):
        text_encoder = self.load_text_encoder()
        self.pipe.text_encoder = text_encoder

        try:
            condition_images = self.prepare_condition_images(image)
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                image=condition_images,
                prompt=prompt,
                device=torch.device(self.primary_device),
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length
            )

            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

            if true_cfg_scale > 1 and negative_prompt is not None:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(
                    image=condition_images,
                    prompt=negative_prompt,
                    device=torch.device(self.primary_device),
                    num_images_per_prompt=1,
                    max_sequence_length=max_sequence_length
                )

            return (
                prompt_embeds.cpu(),
                None if prompt_embeds_mask is None else prompt_embeds_mask.cpu(),
                None if negative_prompt_embeds is None else negative_prompt_embeds.cpu(),
                None if negative_prompt_embeds_mask is None else negative_prompt_embeds_mask.cpu(),
            )
        finally:
            self.pipe.text_encoder = None
            del text_encoder
            self.clear_cuda_cache()

    def close(self):
        if self.pipe is not None:
            self.pipe.text_encoder = None
            self.pipe.transformer = None
            self.pipe.vae = None
            self.pipe = None

        self.clear_cuda_cache()

    @torch.inference_mode()
    def __call__(
        self,
        image,
        prompt,
        negative_prompt=None,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        num_inference_steps=40,
        max_sequence_length=512,
        generator=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        set_runtime_phase(self.plan, "text_encoding")
        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        ) = self.encode_prompts(
            image,
            prompt,
            negative_prompt,
            true_cfg_scale,
            max_sequence_length
        )

        transformer = self.load_transformer()
        self.pipe.transformer = transformer

        try:
            set_runtime_phase(self.plan, "denoising")
            return self.pipe(
                image=image,
                prompt=None,
                negative_prompt=None,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                prompt_embeds=prompt_embeds.to(self.primary_device),
                prompt_embeds_mask=None if prompt_embeds_mask is None else prompt_embeds_mask.to(self.primary_device),
                negative_prompt_embeds=None if negative_prompt_embeds is None else negative_prompt_embeds.to(self.primary_device),
                negative_prompt_embeds_mask=None if negative_prompt_embeds_mask is None else negative_prompt_embeds_mask.to(self.primary_device),
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs or ["latents"],
                max_sequence_length=max_sequence_length
            )
        finally:
            self.pipe.transformer = None
            del transformer
            self.clear_cuda_cache()


class ChronoEditPipelineGPU0(ChronoEditPipeline):
    @property
    def _execution_device(self):
        return torch.device(getattr(self, "planner_execution_device", "cuda:0"))

class ChronoEditGenerator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "nvidia/ChronoEdit-14B-Diffusers"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)
        if is_exact_fast_path(self.plan):
            self.transformer_device_map = {"rope": 0, "patch_embedding": 0, "condition_embedder": 0, "scale_shift_table": 0, "norm_out": 1, "proj_out": 1}
            self.transformer_device_map.update({f"blocks.{index}": 0 for index in range(10)})
            self.transformer_device_map.update({f"blocks.{index}": 1 for index in range(10, 40)})
        else:
            self.transformer_device_map = build_block_device_map(
                "blocks", 40,
                fixed_first=["rope", "patch_embedding", "condition_embedder", "scale_shift_table"],
                fixed_last=["norm_out", "proj_out"], plan=self.plan,
            )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        image_processor = CLIPProcessor.from_pretrained(self.model_id, subfolder="image_processor")
        scheduler = UniPCMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        text_encoder = load_component(
            UMT5EncoderModel, self.model_id, "text_encoder",
            current_kwargs={
                "subfolder": "text_encoder", "torch_dtype": torch.bfloat16,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0),
                "device_map": {"": 0}, "max_memory": self.max_memory,
            },
            portable_base_kwargs={"subfolder": "text_encoder"}, plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        image_encoder = load_component(
            CLIPVisionModel, self.model_id, "image_encoder",
            current_kwargs={"subfolder": "image_encoder", "torch_dtype": torch.float32},
            portable_base_kwargs={"subfolder": "image_encoder"}, plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        if is_cuda_plan(self.plan):
            transformer = ChronoEditTransformer3DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=torch_dtype(self.plan),
                quantization_config=(
                    DiffusersBitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_threshold=6.0,
                        llm_int8_skip_modules=["patch_embedding", "condition_embedder", "norm_out", "proj_out"],
                    ) if is_exact_fast_path(self.plan) else diffusers_quantization_config("transformer", self.plan)
                ),
                device_map=self.transformer_device_map, max_memory=self.max_memory,
            )
        else:
            transformer = load_component(
                ChronoEditTransformer3DModel, self.model_id, "transformer",
                current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16},
                portable_base_kwargs={"subfolder": "transformer"}, shardable=True, plan=self.plan,
            )
        vae = load_component(
            AutoencoderKLWan, self.model_id, "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.float32},
            portable_base_kwargs={"subfolder": "vae"}, plan=self.plan,
        ).to(self.primary_device)
        text_encoder.eval(); image_encoder.eval(); transformer.eval(); vae.eval(); vae.enable_tiling(); vae.enable_slicing()
        if is_exact_fast_path(self.plan):
            self.validate_model_devices("text encoder", text_encoder, {"cuda:0"})
            self.validate_model_devices("image encoder", image_encoder, {"cuda:0"})
            self.validate_model_devices("transformer", transformer, {"cuda:0", "cuda:1"})
            self.validate_model_devices("VAE", vae, {"cuda:0"})
        self.pipe = ChronoEditPipelineGPU0(
            tokenizer=tokenizer, text_encoder=text_encoder, image_encoder=image_encoder,
            image_processor=image_processor, transformer=transformer, vae=vae, scheduler=scheduler,
        )
        self.pipe.planner_execution_device = self.primary_device

    def validate_model_devices(self, name, model, expected_devices):
        parameter_devices = {
            str(parameter.device)
            for parameter in model.parameters()
        }

        if parameter_devices != expected_devices:
            raise RuntimeError(
                f"ChronoEdit {name} parameter devices are {sorted(parameter_devices)}, "
                f"expected {sorted(expected_devices)}"
            )

        return parameter_devices

    def close(self):
        if self.pipe is not None:
            self.pipe.text_encoder = None
            self.pipe.image_encoder = None
            self.pipe.transformer = None
            self.pipe.vae = None
            self.pipe = None

        clear_accelerator_cache()

    def __call__(self, **kwargs):
        kwargs.pop("strength", None)
        kwargs.setdefault("num_frames", 5)
        kwargs.setdefault("enable_temporal_reasoning", False)
        kwargs.setdefault("num_temporal_reasoning_steps", 0)

        frames = self.pipe(**kwargs).frames
        images = []

        for video in frames:
            image = video[-1]

            if not isinstance(image, Image.Image):
                image = np.asarray(image)
                if image.dtype != np.uint8:
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(image)

            images.append(image)

        return SimpleNamespace(images=images)

class Kandinsky5I2VGenerator:
    def __init__(self, plan=None):
        self.plan = plan or get_active_plan()
        self.model_id = "kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers"
        self.primary_device = execution_device(self.plan)
        self.secondary_device = secondary_device(self.plan)
        self.max_memory = {0: "22GiB", 1: "22GiB", "cpu": "48GiB"} if is_exact_fast_path(self.plan) else plan_max_memory(self.plan)
        if is_exact_fast_path(self.plan):
            self.transformer_device_map = {
                "time_embeddings": 0, "text_embeddings": 0, "pooled_text_embeddings": 0,
                "visual_embeddings": 0, "text_rope_embeddings": 0, "visual_rope_embeddings": 0,
                "text_transformer_blocks": 0, "out_layer": 1,
            }
            self.transformer_device_map.update({f"visual_transformer_blocks.{index}": 0 for index in range(29)})
            self.transformer_device_map.update({f"visual_transformer_blocks.{index}": 1 for index in range(29, 60)})
        else:
            self.transformer_device_map = build_block_device_map(
                "visual_transformer_blocks", 60,
                fixed_first=["time_embeddings", "text_embeddings", "pooled_text_embeddings", "visual_embeddings", "text_rope_embeddings", "visual_rope_embeddings", "text_transformer_blocks"],
                fixed_last=["out_layer"], plan=self.plan,
            )
        self.transformer_skip_modules = ["time_embeddings", "text_embeddings", "pooled_text_embeddings", "visual_embeddings", "out_layer"]
        self.transformer_skip_modules.extend([f"text_transformer_blocks.{index}.text_modulation" for index in range(4)])
        self.transformer_skip_modules.extend([f"visual_transformer_blocks.{index}.visual_modulation" for index in range(60)])
        self.processor = Qwen2VLProcessor.from_pretrained(self.model_id, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer_2")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.vae = load_component(
            AutoencoderKLHunyuanVideo, self.model_id, "vae",
            current_kwargs={"subfolder": "vae", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "vae"}, plan=self.plan,
            portable_device_map={"": "cpu"},
        )
        self.vae.eval(); self.vae.enable_tiling(); self.vae.enable_slicing()
        if is_cuda_plan(self.plan):
            flex_attention.flex_attention = torch.compile(
                flex_attention.flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True,
            )
        self.pipe = None
        self._interrupt = False

    def clear_cuda_cache(self):
        clear_accelerator_cache()

    def load_text_pipeline(self):
        set_runtime_phase(self.plan, "text_encoder_loading")
        text_encoder = load_component(
            Qwen2_5_VLForConditionalGeneration, self.model_id, "qwen_text_encoder",
            current_kwargs={
                "subfolder": "text_encoder", "torch_dtype": torch.bfloat16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_threshold=6.0,
                    llm_int8_skip_modules=["visual", "lm_head"],
                ),
                "device_map": "balanced", "max_memory": self.max_memory,
            },
            portable_base_kwargs={"subfolder": "text_encoder"}, shardable=True, plan=self.plan,
        )
        text_encoder_2 = load_component(
            CLIPTextModel, self.model_id, "clip_text_encoder",
            current_kwargs={"subfolder": "text_encoder_2", "torch_dtype": torch.bfloat16},
            portable_base_kwargs={"subfolder": "text_encoder_2"}, plan=self.plan,
            portable_device_map={"": 0} if is_cuda_plan(self.plan) else None,
        )
        text_encoder.eval(); text_encoder_2.eval()
        self.pipe = Kandinsky5I2VPipeline(
            scheduler=self.scheduler, vae=self.vae, text_encoder=text_encoder, tokenizer=self.processor,
            text_encoder_2=text_encoder_2, tokenizer_2=self.tokenizer_2, transformer=None,
        )

    def encode_prompts(self, prompt, negative_prompt, max_sequence_length):
        prompt_embeds_qwen, prompt_embeds_clip, prompt_cu_seqlens = self.pipe.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            device=torch.device(self.primary_device),
            dtype=torch_dtype(self.plan)
        )

        negative_prompt_embeds_qwen, negative_prompt_embeds_clip, negative_prompt_cu_seqlens = self.pipe.encode_prompt(
            prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            device=torch.device(self.primary_device),
            dtype=torch_dtype(self.plan)
        )

        return (
            prompt_embeds_qwen.cpu(),
            prompt_embeds_clip.cpu(),
            prompt_cu_seqlens.cpu(),
            negative_prompt_embeds_qwen.cpu(),
            negative_prompt_embeds_clip.cpu(),
            negative_prompt_cu_seqlens.cpu(),
        )

    def prepare_latents(self, image, height, width, num_frames, seed):
        set_runtime_phase(self.plan, "vae_source_encode")
        device = torch.device(self.secondary_device)
        generator = torch.Generator(device=device).manual_seed(seed)
        num_latent_frames = (num_frames - 1) // self.vae.config.temporal_compression_ratio + 1

        latents = torch.randn(
            1,
            num_latent_frames,
            height // self.vae.config.spatial_compression_ratio,
            width // self.vae.config.spatial_compression_ratio,
            16,
            generator=generator,
            device=device,
            dtype=torch_dtype(self.plan)
        )

        self.vae.to(device)

        image_tensor = self.pipe.video_processor.preprocess(image, height=height, width=width).to(
            device,
            dtype=torch_dtype(self.plan)
        )

        image_latents = self.vae.encode(image_tensor.unsqueeze(2)).latent_dist.sample(generator=generator)
        image_latents = image_latents * self.vae.config.scaling_factor
        image_latents = image_latents.permute(0, 2, 3, 4, 1).to(latents.device, latents.dtype)

        latents[:, 0:1] = image_latents

        visual_cond = torch.zeros_like(latents)
        visual_cond_mask = torch.zeros(
            1,
            num_latent_frames,
            height // self.vae.config.spatial_compression_ratio,
            width // self.vae.config.spatial_compression_ratio,
            1,
            device=device,
            dtype=torch_dtype(self.plan)
        )

        visual_cond_mask[:, 0:1] = 1
        visual_cond[:, 0:1] = image_latents
        latents = torch.cat([latents, visual_cond, visual_cond_mask], dim=-1).cpu()

        self.vae.to("cpu")
        self.clear_cuda_cache()

        return latents

    def release_text_encoders(self):
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        self.clear_cuda_cache()

    def load_transformer(self):
        set_runtime_phase(self.plan, "transformer_loading")
        if is_cuda_plan(self.plan):
            transformer = Kandinsky5Transformer3DModel.from_pretrained(
                self.model_id, subfolder="transformer", torch_dtype=torch_dtype(self.plan),
                quantization_config=(
                    DiffusersBitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_threshold=6.0,
                        llm_int8_skip_modules=self.transformer_skip_modules,
                    ) if is_exact_fast_path(self.plan) else diffusers_quantization_config("transformer", self.plan, self.transformer_skip_modules)
                ),
                device_map=self.transformer_device_map, max_memory=self.max_memory,
            )
        else:
            transformer = load_component(
                Kandinsky5Transformer3DModel, self.model_id, "transformer",
                current_kwargs={"subfolder": "transformer", "torch_dtype": torch.bfloat16},
                portable_base_kwargs={"subfolder": "transformer"}, shardable=True, plan=self.plan,
            )
        transformer.eval()
        if is_cuda_plan(self.plan):
            transformer.set_attention_backend("flex")
        return transformer

    def decode_latents(self, latent_video):
        set_runtime_phase(self.plan, "vae_decode")
        self.vae.to(self.secondary_device)
        video = latent_video.to(self.secondary_device, dtype=self.vae.dtype)
        video = video.permute(0, 4, 1, 2, 3)
        video = video / self.vae.config.scaling_factor
        video = self.vae.decode(video).sample
        video_frames = self.pipe.video_processor.postprocess_video(video, output_type="pil")[0]
        self.vae.to("cpu")
        self.clear_cuda_cache()
        return video_frames

    def close(self):
        if self.pipe is not None:
            self.pipe.text_encoder = None
            self.pipe.text_encoder_2 = None
            self.pipe.transformer = None
            self.pipe = None

        self.vae.to("cpu")
        self.clear_cuda_cache()

    @torch.inference_mode()
    def generate(
        self,
        image,
        prompt,
        negative_prompt="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        height=512,
        width=512,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
        max_sequence_length=512,
        generator=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        image = image.convert("RGB").resize((width, height))
        seed = generator.initial_seed() if generator is not None else 42
        self._interrupt = False

        self.load_text_pipeline()

        (
            prompt_embeds_qwen,
            prompt_embeds_clip,
            prompt_cu_seqlens,
            negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip,
            negative_prompt_cu_seqlens,
        ) = self.encode_prompts(prompt, negative_prompt, max_sequence_length)

        latents = self.prepare_latents(image, height, width, num_frames, seed)

        self.release_text_encoders()

        transformer = self.load_transformer()
        self.pipe.transformer = transformer

        set_runtime_phase(self.plan, "denoising")
        latent_video = self.pipe(
            image=image,
            prompt=None,
            negative_prompt=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents,
            prompt_embeds_qwen=prompt_embeds_qwen,
            prompt_embeds_clip=prompt_embeds_clip,
            negative_prompt_embeds_qwen=negative_prompt_embeds_qwen,
            negative_prompt_embeds_clip=negative_prompt_embeds_clip,
            prompt_cu_seqlens=prompt_cu_seqlens,
            negative_prompt_cu_seqlens=negative_prompt_cu_seqlens,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs or ["latents"],
            output_type="latent"
        ).frames

        self.pipe.transformer = None
        del transformer
        self.clear_cuda_cache()

        return self.decode_latents(latent_video)

    def __call__(
        self,
        image,
        prompt,
        negative_prompt="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        height=512,
        width=512,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
        max_sequence_length=512,
        generator=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        video_frames = self.generate(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        return SimpleNamespace(frames=[video_frames])

