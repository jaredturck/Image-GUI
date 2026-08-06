class DiffusionVramEstimator:
    gib = 1024 ** 3

    def tensor_bytes(self, dimensions, bytes_per_value=2):
        elements = 1

        for dimension in dimensions:
            elements *= dimension

        return elements * bytes_per_value

    def linear_parameters(self, in_features, out_features, bias=True):
        parameters = in_features * out_features

        if bias:
            parameters += out_features

        return parameters

    def effective_frames(self, frames, temporal_compression):
        return max(1, frames // temporal_compression * temporal_compression + 1) if frames % temporal_compression != 1 else max(1, frames)

    def latent_frames(self, frames, temporal_compression):
        frames = self.effective_frames(frames, temporal_compression)
        return (frames - 1) // temporal_compression + 1

    def quantized_static_bytes(self, int8_weights, bf16_parameters, metadata_ratio, runtime_gb):
        return int8_weights + bf16_parameters * 2 + int8_weights * metadata_ratio + runtime_gb * self.gib

    def estimated_peak_bytes(self, static_bytes, dynamic_bytes, fixed_workspace_gb, workspace_ratio, allocator_ratio):
        workspace_bytes = fixed_workspace_gb * self.gib + dynamic_bytes * workspace_ratio
        return (static_bytes + dynamic_bytes + workspace_bytes) * allocator_ratio

    def bytes_to_gb(self, value):
        return value / self.gib


class Kandinsky5I2VProVramEstimator(DiffusionVramEstimator):
    def __init__(self):
        self.spatial_compression = 8
        self.temporal_compression = 4
        self.patch_height = 2
        self.patch_width = 2
        self.model_dim = 4096
        self.ff_dim = 16384
        self.head_dim = 128
        self.num_heads = 32
        self.num_text_blocks = 4
        self.num_gpu0_visual_blocks = 29
        self.num_gpu1_visual_blocks = 31
        self.latent_channels = 16
        self.condition_channels = 33
        self.sparse_group_size = 64
        self.sparse_map_bytes = 24
        self.quantization_metadata_ratio = 0.06
        self.static_runtime_gb = 1.7
        self.fixed_workspace_gb = 0.8
        self.workspace_ratio = 0.20
        self.allocator_ratio = 1.08
        self.qwen_8bit_gb = 8.77
        self.clip_bf16_gb = 1.59
        self.text_workspace_gb = 1.0
        self.vae_static_gb = 0.92
        self.vae_workspace_gb = 1.0

    def attention_parameters(self):
        int8_weights = 4 * self.model_dim * self.model_dim
        bf16_parameters = 4 * self.model_dim + 2 * self.head_dim
        return int8_weights, bf16_parameters

    def feed_forward_parameters(self):
        return 2 * self.model_dim * self.ff_dim

    def transformer_static_bytes(self):
        attention_weights, attention_bf16 = self.attention_parameters()
        feed_forward_weights = self.feed_forward_parameters()

        text_block_weights = attention_weights + feed_forward_weights
        visual_block_weights = 2 * attention_weights + feed_forward_weights

        text_modulation = self.linear_parameters(1024, 6 * self.model_dim)
        visual_modulation = self.linear_parameters(1024, 9 * self.model_dim)

        time_embeddings = self.linear_parameters(self.model_dim, 1024) + self.linear_parameters(1024, 1024)
        text_embeddings = self.linear_parameters(3584, self.model_dim) + 2 * self.model_dim
        pooled_text_embeddings = self.linear_parameters(768, 1024) + 2 * 1024
        visual_embeddings = self.linear_parameters(4 * self.condition_channels, self.model_dim)
        output_layer = self.linear_parameters(1024, 2 * self.model_dim) + self.linear_parameters(self.model_dim, 4 * self.latent_channels)

        gpu0_int8_weights = self.num_text_blocks * text_block_weights + self.num_gpu0_visual_blocks * visual_block_weights
        gpu1_int8_weights = self.num_gpu1_visual_blocks * visual_block_weights

        gpu0_bf16_parameters = (
            time_embeddings
            + text_embeddings
            + pooled_text_embeddings
            + visual_embeddings
            + self.num_text_blocks * (text_modulation + attention_bf16)
            + self.num_gpu0_visual_blocks * (visual_modulation + 2 * attention_bf16)
        )
        gpu1_bf16_parameters = output_layer + self.num_gpu1_visual_blocks * (visual_modulation + 2 * attention_bf16)

        gpu0_bytes = self.quantized_static_bytes(
            gpu0_int8_weights,
            gpu0_bf16_parameters,
            self.quantization_metadata_ratio,
            self.static_runtime_gb,
        )
        gpu1_bytes = self.quantized_static_bytes(
            gpu1_int8_weights,
            gpu1_bf16_parameters,
            self.quantization_metadata_ratio,
            self.static_runtime_gb,
        )

        return gpu0_bytes, gpu1_bytes

    def transformer_dynamic_bytes(self, width, height, frames, guidance_scale):
        latent_frames = self.latent_frames(frames, self.temporal_compression)
        latent_height = height // self.spatial_compression
        latent_width = width // self.spatial_compression
        token_height = latent_height // self.patch_height
        token_width = latent_width // self.patch_width
        visual_tokens = latent_frames * token_height * token_width

        hidden_bytes = self.tensor_bytes((visual_tokens, self.model_dim))
        feed_forward_bytes = self.tensor_bytes((visual_tokens, self.ff_dim))
        feed_forward_peak = 2 * feed_forward_bytes + 3 * hidden_bytes

        sparse_groups = (visual_tokens + self.sparse_group_size - 1) // self.sparse_group_size
        sparse_map_bytes = self.num_heads * sparse_groups * sparse_groups * self.sparse_map_bytes
        rope_bytes = visual_tokens * 1024
        attention_peak = 8 * hidden_bytes + sparse_map_bytes + rope_bytes

        conditioning_latents = self.tensor_bytes(
            (latent_frames, latent_height, latent_width, self.condition_channels)
        )
        prediction = self.tensor_bytes(
            (latent_frames, latent_height, latent_width, self.latent_channels)
        )

        core_peak = max(feed_forward_peak, attention_peak) + 2 * conditioning_latents
        gpu0_dynamic = core_peak
        gpu1_dynamic = core_peak + prediction

        if guidance_scale > 1.0:
            gpu1_dynamic += 2 * prediction

        return gpu0_dynamic, gpu1_dynamic

    def text_stage_bytes(self):
        qwen_gpu_bytes = self.qwen_8bit_gb * self.gib / 2
        gpu0_bytes = qwen_gpu_bytes + (self.clip_bf16_gb + self.text_workspace_gb) * self.gib
        gpu1_bytes = qwen_gpu_bytes + self.text_workspace_gb * self.gib
        return gpu0_bytes, gpu1_bytes

    def image_encode_stage_bytes(self, width, height):
        text_gpu0, text_gpu1 = self.text_stage_bytes()
        image_bytes = self.tensor_bytes((1, 3, height, width))
        tile_activation_bytes = self.tensor_bytes((1, 512, 1, min(height, 256), min(width, 256)), 4)
        gpu0_bytes = text_gpu0
        gpu1_bytes = text_gpu1 + (self.vae_static_gb + self.vae_workspace_gb) * self.gib + image_bytes + tile_activation_bytes
        return gpu0_bytes, gpu1_bytes

    def video_decode_stage_bytes(self, width, height, frames):
        effective_frames = self.effective_frames(frames, self.temporal_compression)
        output_bytes = self.tensor_bytes((1, 3, effective_frames, height, width))
        tile_activation_bytes = self.tensor_bytes((1, 512, min(effective_frames, 16), min(height, 256), min(width, 256)), 2)
        gpu0_bytes = 0.5 * self.gib
        gpu1_bytes = (self.vae_static_gb + self.vae_workspace_gb) * self.gib + output_bytes + tile_activation_bytes
        return gpu0_bytes, gpu1_bytes

    def estimate(self, width, height, frames, guidance_scale):
        static_gpu0, static_gpu1 = self.transformer_static_bytes()
        dynamic_gpu0, dynamic_gpu1 = self.transformer_dynamic_bytes(width, height, frames, guidance_scale)

        transformer_gpu0 = self.estimated_peak_bytes(
            static_gpu0,
            dynamic_gpu0,
            self.fixed_workspace_gb,
            self.workspace_ratio,
            self.allocator_ratio,
        )
        transformer_gpu1 = self.estimated_peak_bytes(
            static_gpu1,
            dynamic_gpu1,
            self.fixed_workspace_gb,
            self.workspace_ratio,
            self.allocator_ratio,
        )

        text_gpu0, text_gpu1 = self.text_stage_bytes()
        encode_gpu0, encode_gpu1 = self.image_encode_stage_bytes(width, height)
        decode_gpu0, decode_gpu1 = self.video_decode_stage_bytes(width, height, frames)

        gpu0_bytes = max(transformer_gpu0, text_gpu0, encode_gpu0, decode_gpu0)
        gpu1_bytes = max(transformer_gpu1, text_gpu1, encode_gpu1, decode_gpu1)

        return self.bytes_to_gb(gpu0_bytes), self.bytes_to_gb(gpu1_bytes)
