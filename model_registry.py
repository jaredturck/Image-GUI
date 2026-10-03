# Comprehensive model metadata for the hardware-aware loading planner.
#
# This file combines source-derived architecture facts, checkpoint inventory,
# code-path analysis, quantization research, and explicit engineering estimates.
# Exact runtime peaks remain allocator, kernel, workload, and backend dependent;
# these values choose initial candidates and order retries rather than promising
# that a future allocation will succeed.

from copy import deepcopy
from math import ceil

REGISTRY_VERSION = '2026-08-06'
SCHEMA_VERSION = 1

GENERAL_POLICY = {
    'capacity_source': 'physical_total_memory_minus_fixed_reserve',
    'use_instantaneous_free_vram_for_initial_plan': False,
    'default_cuda_gpu_reserve_gib': 2.0,
    'default_small_gpu_reserve_gib': 1.0,
    'small_gpu_threshold_gib': 12.0,
    'default_system_ram_reserve_gib': 6.0,
    'persist_runtime_results': True,
    'validate_with_real_inference': True,
    'retry_in_fresh_process_after_cuda_oom': True,
    'minimum_automatic_weight_bits': 4,
    'allow_cannot_run_result': True,
    'quality_priority': 'high',
    'speed_priority': 'high',
    'tie_breaker': 'prefer_higher_precision_when_performance_classes_are_adjacent',
    'plan_success_definition': 'model_load_and_one_complete_default_workload_inference',
    'static_memory_role': 'coarse_rejection_and_candidate_ordering_not_a_runtime_guarantee'
}

QUANTIZATION_METHODS = {
    'bnb_int8': {
        'backend': 'bitsandbytes',
        'weight_bits': 8,
        'activation_bits': 16,
        'compute_dtype': ['bfloat16', 'float16'],
        'description': 'LLM.int8-style weight quantization with mixed-precision outlier handling.',
        'default_quality_risk': 'low',
        'planner_role': 'normal_automatic_fallback'
    },
    'bnb_nf4': {
        'backend': 'bitsandbytes',
        'weight_bits': 4,
        'activation_bits': 16,
        'compute_dtype': ['bfloat16', 'float16'],
        'double_quantization': True,
        'description': 'NF4 weight-only quantization with optional double quantization.',
        'default_quality_risk': 'medium',
        'planner_role': 'model_and_component_specific_late_fallback'
    },
    'native_fp8': {
        'backend': 'checkpoint_native',
        'weight_bits': 8,
        'activation_bits': 'backend_dependent',
        'description': 'Publisher or conversion-provided FP8 checkpoint.',
        'default_quality_risk': 'low',
        'planner_role': 'preferred_when_checkpoint_is_the_application_target'
    },
    'native_mxfp4': {
        'backend': 'checkpoint_native',
        'weight_bits': 4,
        'activation_bits': 'mixed',
        'description': 'Model-native microscaling FP4 expert representation with sensitive modules retained at higher precision.',
        'default_quality_risk': 'low',
        'planner_role': 'native_representation_not_a_fallback'
    },
    'gguf_q4_k_m': {
        'backend': 'gguf',
        'weight_bits': 4,
        'activation_bits': 'backend_dependent',
        'description': 'GGUF Q4_K_M mixed-block quantized checkpoint.',
        'default_quality_risk': 'medium',
        'planner_role': 'native_artifact'
    },
    'comfy_checkpoint_native': {
        'backend': 'comfyui',
        'weight_bits': 'checkpoint_defined',
        'activation_bits': 'runtime_defined',
        'description': 'ComfyUI single-file checkpoint loaded according to its stored tensor types.',
        'default_quality_risk': 'checkpoint_defined',
        'planner_role': 'native_artifact'
    }
}

COMPONENT_POLICIES = {
    'llm_decoder': {
        'int8_default': 'recommended_under_memory_pressure',
        'int4_default': 'size_and_task_dependent',
        'keep_high_precision': ['embeddings', 'normalization', 'lm_head_unless_verified']
    },
    'large_text_encoder': {
        'int8_default': 'preferred_under_memory_pressure',
        'int4_default': 'acceptable_when_needed',
        'quality_failure_mode': 'prompt_alignment_and_instruction_interpretation'
    },
    'small_text_encoder': {
        'int8_default': 'usually_not_worthwhile',
        'int4_default': 'avoid',
        'quality_failure_mode': 'prompt_alignment'
    },
    'image_dit': {
        'int8_default': 'reasonable_fallback',
        'int4_default': 'high_caution_without_model_specific_evidence',
        'keep_high_precision': ['input_projections', 'condition_embedders', 'modulation', 'norm_out', 'proj_out']
    },
    'video_dit': {
        'int8_default': 'conditional_fallback',
        'int4_default': 'last_resort_or_native_checkpoint_only',
        'keep_high_precision': [
            'input_projections',
            'condition_embedders',
            'temporal_modulation',
            'norm_out',
            'proj_out'
        ]
    },
    'unet': {
        'int8_default': 'limited_benefit_for_convolution_heavy_models',
        'int4_default': 'avoid_generic_bitsandbytes'
    },
    'vae': {
        'int8_default': 'disabled',
        'int4_default': 'disabled',
        'preferred_memory_tools': ['tiling', 'slicing', 'staging', 'alternate_gpu', 'cpu_decode']
    },
    'vision_encoder': {'int8_default': 'model_specific', 'int4_default': 'avoid_unless_large_and_verified'}
}

PLAN_TEMPLATES = {
    'current_exact_fast_path': {
        'placement': 'existing_code_path',
        'speed_class': 'excellent',
        'quality_class': 'model_intended',
        'fresh_process_retry': True
    },
    'resident_native_single': {
        'placement': 'single_gpu_resident',
        'quantization_mode': 'native',
        'speed_class': 'excellent',
        'quality_class': 'native',
        'minimum_gpu_count': 1
    },
    'resident_native_multi': {
        'placement': 'multi_gpu_device_map',
        'quantization_mode': 'native',
        'speed_class': 'very_good',
        'quality_class': 'native',
        'minimum_gpu_count': 2
    },
    'balanced_native_cpu_overflow': {
        'placement': 'device_map_with_cpu_overflow',
        'quantization_mode': 'native',
        'speed_class': 'usable',
        'quality_class': 'native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'resident_int8_single': {
        'placement': 'single_gpu_resident',
        'quantization_mode': 'int8',
        'speed_class': 'very_good',
        'quality_class': 'near_native',
        'minimum_gpu_count': 1
    },
    'resident_int8_multi': {
        'placement': 'multi_gpu_device_map',
        'quantization_mode': 'int8',
        'speed_class': 'good',
        'quality_class': 'near_native',
        'minimum_gpu_count': 2
    },
    'int8_cpu_overflow': {
        'placement': 'device_map_with_cpu_overflow',
        'quantization_mode': 'int8',
        'speed_class': 'usable',
        'quality_class': 'near_native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'resident_int4_single': {
        'placement': 'single_gpu_resident',
        'quantization_mode': 'int4',
        'speed_class': 'good',
        'quality_class': 'reduced',
        'minimum_gpu_count': 1
    },
    'resident_int4_multi': {
        'placement': 'multi_gpu_device_map',
        'quantization_mode': 'int4',
        'speed_class': 'good',
        'quality_class': 'reduced',
        'minimum_gpu_count': 2
    },
    'int4_cpu_overflow': {
        'placement': 'device_map_with_cpu_overflow',
        'quantization_mode': 'int4',
        'speed_class': 'slow',
        'quality_class': 'reduced',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'staged_native_single': {
        'placement': 'single_gpu_model_specific_staging',
        'quantization_mode': 'native',
        'speed_class': 'good',
        'quality_class': 'native',
        'minimum_gpu_count': 1
    },
    'staged_native_multi': {
        'placement': 'multi_gpu_model_specific_staging',
        'quantization_mode': 'native',
        'speed_class': 'very_good',
        'quality_class': 'native',
        'minimum_gpu_count': 2
    },
    'staged_int8_single': {
        'placement': 'single_gpu_model_specific_staging',
        'quantization_mode': 'int8',
        'speed_class': 'good',
        'quality_class': 'near_native',
        'minimum_gpu_count': 1
    },
    'staged_int8_multi': {
        'placement': 'multi_gpu_model_specific_staging',
        'quantization_mode': 'int8',
        'speed_class': 'very_good',
        'quality_class': 'near_native',
        'minimum_gpu_count': 2
    },
    'staged_int4_single': {
        'placement': 'single_gpu_model_specific_staging',
        'quantization_mode': 'int4',
        'speed_class': 'usable',
        'quality_class': 'reduced',
        'minimum_gpu_count': 1
    },
    'staged_int4_multi': {
        'placement': 'multi_gpu_model_specific_staging',
        'quantization_mode': 'int4',
        'speed_class': 'good',
        'quality_class': 'reduced',
        'minimum_gpu_count': 2
    },
    'model_cpu_offload_native': {
        'placement': 'diffusers_model_cpu_offload',
        'quantization_mode': 'native',
        'speed_class': 'usable',
        'quality_class': 'native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'model_cpu_offload_int8': {
        'placement': 'diffusers_model_cpu_offload',
        'quantization_mode': 'int8',
        'speed_class': 'usable',
        'quality_class': 'near_native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'model_cpu_offload_int4': {
        'placement': 'diffusers_model_cpu_offload',
        'quantization_mode': 'int4',
        'speed_class': 'slow',
        'quality_class': 'reduced',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'sequential_cpu_offload_native': {
        'placement': 'diffusers_sequential_cpu_offload',
        'quantization_mode': 'native',
        'speed_class': 'very_slow',
        'quality_class': 'native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'sequential_cpu_offload_int8': {
        'placement': 'diffusers_sequential_cpu_offload',
        'quantization_mode': 'int8',
        'speed_class': 'very_slow',
        'quality_class': 'near_native',
        'minimum_gpu_count': 1,
        'allows_cpu_overflow': True
    },
    'cpu_native': {
        'placement': 'cpu_only',
        'quantization_mode': 'native',
        'speed_class': 'emergency',
        'quality_class': 'native',
        'minimum_gpu_count': 0
    },
    'cpu_int8': {
        'placement': 'cpu_only',
        'quantization_mode': 'int8',
        'speed_class': 'emergency',
        'quality_class': 'near_native',
        'minimum_gpu_count': 0
    },
    'mps_native': {
        'placement': 'mps_resident_or_unified_memory',
        'quantization_mode': 'native',
        'speed_class': 'hardware_dependent',
        'quality_class': 'native',
        'minimum_gpu_count': 0
    },
    'comfy_native': {
        'placement': 'comfyui_managed',
        'quantization_mode': 'checkpoint_native',
        'speed_class': 'hardware_dependent',
        'quality_class': 'checkpoint_native',
        'minimum_gpu_count': 0
    },
    'vllm_native': {
        'placement': 'vllm_tensor_parallel',
        'quantization_mode': 'checkpoint_native',
        'speed_class': 'excellent',
        'quality_class': 'checkpoint_native',
        'minimum_gpu_count': 1
    }
}

SOURCE_CATALOG = {
    'bitsandbytes_docs': 'https://huggingface.co/docs/transformers/quantization/bitsandbytes',
    'diffusers_bnb_docs': 'https://huggingface.co/docs/diffusers/quantization/bitsandbytes',
    'accelerate_big_model_inference': 'https://huggingface.co/docs/accelerate/usage_guides/big_modeling',
    'diffusers_memory': 'https://huggingface.co/docs/diffusers/optimization/memory',
    'llm_int8_paper': 'https://arxiv.org/abs/2208.07339',
    'qlora_paper': 'https://arxiv.org/abs/2305.14314',
    'qdiffusion_paper': 'https://arxiv.org/abs/2302.04304',
    'viditq_paper': 'https://arxiv.org/abs/2406.02540'
}

LAUNCHER_MODEL_IDS = [
    'z_image_turbo',
    'kandinsky_5',
    'pixart_sigma',
    'anima',
    'stable_diffusion_3_5',
    'flux_1',
    'glm_image',
    'qwen_image',
    'flux_2',
    'kandinsky_5_i2i',
    'chronoedit',
    'qwen_image_edit',
    'rmbg_1_4',
    'sd_x4_upscaler',
    'skyreels_v2',
    'kandinsky_5_t2v',
    'nvidia_cosmos',
    'allegro',
    'cogvideox',
    'hunyuan_video_1_5',
    'hunyuan_video_1_5_i2v',
    'kandinsky_5_t2v_pro',
    'kandinsky_5_t2v_pro_sft',
    'kandinsky_5_i2v',
    'kandinsky_5_i2v_pro_sft',
    'openai-community/gpt2-large',
    'tiiuae/Falcon-H1-0.5B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'CohereLabs/c4ai-command-r7b-12-2024',
    'tiiuae/Falcon-H1-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-2-9b-it',
    'zai-org/glm-4-9b-chat-hf',
    'tiiuae/Falcon3-10B-Instruct',
    'allenai/OLMo-2-1124-13B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-32B-Instruct',
    'tiiuae/Falcon-H1-34B-Instruct',
    'LiquidAI/LFM2.5-1.2B-Thinking',
    'microsoft/Phi-4-reasoning',
    'Qwen/Qwen3.5-4B',
    'Qwen/Qwen3.5-9B',
    'Qwen/Qwen3-14B',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'Qwen/Qwen3.6-27B',
    'google/gemma-4-31B-it',
    'Qwen/Qwen3.6-35B-A3B',
    'Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M'
]

RAW_MODEL_PROFILES = [
    {
        'key': 'gpt2_large',
        'model_id': 'openai-community/gpt2-large',
        'runtime_model_id': 'openai-community/gpt2-large',
        'launcher_id': 'openai-community/gpt2-large',
        'display_name': 'GPT-2 Large',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'float32', 'already_quantized': False},
        'architecture': {
            'family': 'gpt2_dense_decoder',
            'parameter_billions': 0.774,
            'active_parameter_billions': 0.774,
            'hidden_size': 1280,
            'intermediate_size': 5120,
            'num_hidden_layers': 36,
            'num_attention_heads': 20,
            'num_key_value_heads': 20,
            'head_dim': 64,
            'max_context_tokens': 1024,
            'vocab_size': 50257,
            'uses_kv_cache': True,
            'cache_dtype': 'float32'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'gpt2_dense_decoder',
                'parameter_billions': 0.774,
                'active_parameter_billions': 0.774,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 768, 'max_new_tokens': 256},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 1.5
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'practical',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'high',
            'int4_priority': 'disabled_for_automatic_planning',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'gpt2_large_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/openai-community/gpt2-large',
            'https://huggingface.co/openai-community/gpt2-large/blob/main/config.json',
            'https://huggingface.co/openai-community/gpt2-large/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Small enough that native FP16/BF16 or CPU execution should normally outrank 4-bit quantization.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'falcon_h1_0_5b',
        'model_id': 'tiiuae/Falcon-H1-0.5B-Instruct',
        'runtime_model_id': 'tiiuae/Falcon-H1-0.5B-Instruct',
        'launcher_id': 'tiiuae/Falcon-H1-0.5B-Instruct',
        'display_name': 'Falcon-H1 0.5B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'falcon_h1_hybrid_attention_mamba',
            'parameter_billions': 0.5,
            'active_parameter_billions': 0.5,
            'hidden_size': 1024,
            'intermediate_size': 4096,
            'num_hidden_layers': 24,
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'head_dim': 128,
            'max_context_tokens': 262144,
            'vocab_size': 65024,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'falcon_h1_hybrid_attention_mamba',
                'parameter_billions': 0.5,
                'active_parameter_billions': 0.5,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 512},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 1.5
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'practical',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'unacceptable_for_automatic_use',
            'int4_priority': 'disabled_for_automatic_planning',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'falcon_h1_0_5b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct',
            'https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct/blob/main/config.json',
            'https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [
            'Hybrid attention/state-space architecture; use high precision because the model is already very small.'
        ],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'mistral_7b_instruct_v0_3',
        'model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'runtime_model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'launcher_id': 'mistralai/Mistral-7B-Instruct-v0.3',
        'display_name': 'Mistral 7B Instruct v0.3',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'mistral_dense_gqa',
            'parameter_billions': 7.25,
            'active_parameter_billions': 7.25,
            'hidden_size': 4096,
            'intermediate_size': 14336,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 32768,
            'vocab_size': 32768,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'mistral_dense_gqa',
                'parameter_billions': 7.25,
                'active_parameter_billions': 7.25,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'mistral_7b_instruct_v0_3_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3',
            'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json',
            'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'command_r7b',
        'model_id': 'CohereLabs/c4ai-command-r7b-12-2024',
        'runtime_model_id': 'CohereLabs/c4ai-command-r7b-12-2024',
        'launcher_id': 'CohereLabs/c4ai-command-r7b-12-2024',
        'display_name': 'Command R7B 12-2024',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'cohere_command_r_gqa_sliding_attention',
            'parameter_billions': 8.0,
            'active_parameter_billions': 8.0,
            'hidden_size': 4096,
            'intermediate_size': 14336,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 131072,
            'vocab_size': 256000,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'cohere_command_r_gqa_sliding_attention',
                'parameter_billions': 8.0,
                'active_parameter_billions': 8.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'command_r7b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024',
            'https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024/blob/main/config.json',
            'https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['The public checkpoint is commonly classified near 8B despite the R7B product name.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'falcon_h1_7b',
        'model_id': 'tiiuae/Falcon-H1-7B-Instruct',
        'runtime_model_id': 'tiiuae/Falcon-H1-7B-Instruct',
        'launcher_id': 'tiiuae/Falcon-H1-7B-Instruct',
        'display_name': 'Falcon-H1 7B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'falcon_h1_hybrid_attention_mamba',
            'parameter_billions': 7.0,
            'active_parameter_billions': 7.0,
            'hidden_size': 3072,
            'intermediate_size': 12288,
            'num_hidden_layers': 44,
            'num_attention_heads': 12,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 65024,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'falcon_h1_hybrid_attention_mamba',
                'parameter_billions': 7.0,
                'active_parameter_billions': 7.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'falcon_h1_7b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct',
            'https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/blob/main/config.json',
            'https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'llama_3_1_8b',
        'model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'runtime_model_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'launcher_id': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'display_name': 'Meta-Llama 3.1 8B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'llama3_dense_gqa',
            'parameter_billions': 8.03,
            'active_parameter_billions': 8.03,
            'hidden_size': 4096,
            'intermediate_size': 14336,
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 131072,
            'vocab_size': 128256,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'llama3_dense_gqa',
                'parameter_billions': 8.03,
                'active_parameter_billions': 8.03,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'llama_3_1_8b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct',
            'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/blob/main/config.json',
            'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'gemma_2_9b',
        'model_id': 'google/gemma-2-9b-it',
        'runtime_model_id': 'google/gemma-2-9b-it',
        'launcher_id': 'google/gemma-2-9b-it',
        'display_name': 'Gemma 2 9B IT',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'gemma2_dense_sliding_global_attention',
            'parameter_billions': 9.24,
            'active_parameter_billions': 9.24,
            'hidden_size': 3584,
            'intermediate_size': 14336,
            'num_hidden_layers': 42,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'head_dim': 224,
            'max_context_tokens': 8192,
            'vocab_size': 256000,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'gemma2_dense_sliding_global_attention',
                'parameter_billions': 9.24,
                'active_parameter_billions': 9.24,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'gemma_2_9b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/google/gemma-2-9b-it',
            'https://huggingface.co/google/gemma-2-9b-it/blob/main/config.json',
            'https://huggingface.co/google/gemma-2-9b-it/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'glm_4_9b',
        'model_id': 'zai-org/glm-4-9b-chat-hf',
        'runtime_model_id': 'zai-org/glm-4-9b-chat-hf',
        'launcher_id': 'zai-org/glm-4-9b-chat-hf',
        'display_name': 'GLM-4 9B Chat',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'glm4_dense_mqa',
            'parameter_billions': 9.4,
            'active_parameter_billions': 9.4,
            'hidden_size': 4096,
            'intermediate_size': 13696,
            'num_hidden_layers': 40,
            'num_attention_heads': 32,
            'num_key_value_heads': 2,
            'head_dim': 128,
            'max_context_tokens': 131072,
            'vocab_size': 151552,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'glm4_dense_mqa',
                'parameter_billions': 9.4,
                'active_parameter_billions': 9.4,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'glm_4_9b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/zai-org/glm-4-9b-chat-hf',
            'https://huggingface.co/zai-org/glm-4-9b-chat-hf/blob/main/config.json',
            'https://huggingface.co/zai-org/glm-4-9b-chat-hf/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'falcon3_10b',
        'model_id': 'tiiuae/Falcon3-10B-Instruct',
        'runtime_model_id': 'tiiuae/Falcon3-10B-Instruct',
        'launcher_id': 'tiiuae/Falcon3-10B-Instruct',
        'display_name': 'Falcon3 10B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'falcon3_dense_gqa',
            'parameter_billions': 10.0,
            'active_parameter_billions': 10.0,
            'hidden_size': 3072,
            'intermediate_size': 23040,
            'num_hidden_layers': 40,
            'num_attention_heads': 12,
            'num_key_value_heads': 4,
            'head_dim': 256,
            'max_context_tokens': 32768,
            'vocab_size': 131072,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'falcon3_dense_gqa',
                'parameter_billions': 10.0,
                'active_parameter_billions': 10.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'falcon3_10b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/tiiuae/Falcon3-10B-Instruct',
            'https://huggingface.co/tiiuae/Falcon3-10B-Instruct/blob/main/config.json',
            'https://huggingface.co/tiiuae/Falcon3-10B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'olmo_2_13b',
        'model_id': 'allenai/OLMo-2-1124-13B-Instruct',
        'runtime_model_id': 'allenai/OLMo-2-1124-13B-Instruct',
        'launcher_id': 'allenai/OLMo-2-1124-13B-Instruct',
        'display_name': 'OLMo 2 13B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'olmo2_dense_mha',
            'parameter_billions': 13.0,
            'active_parameter_billions': 13.0,
            'hidden_size': 5120,
            'intermediate_size': 13824,
            'num_hidden_layers': 40,
            'num_attention_heads': 40,
            'num_key_value_heads': 40,
            'head_dim': 128,
            'max_context_tokens': 4096,
            'vocab_size': 100352,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'olmo2_dense_mha',
                'parameter_billions': 13.0,
                'active_parameter_billions': 13.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'last_resort_after_int8_and_native_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'olmo_2_13b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct',
            'https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct/blob/main/config.json',
            'https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen2_5_14b',
        'model_id': 'Qwen/Qwen2.5-14B-Instruct',
        'runtime_model_id': 'Qwen/Qwen2.5-14B-Instruct',
        'launcher_id': 'Qwen/Qwen2.5-14B-Instruct',
        'display_name': 'Qwen2.5 14B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen2_5_dense_gqa',
            'parameter_billions': 14.7,
            'active_parameter_billions': 14.7,
            'hidden_size': 5120,
            'intermediate_size': 13824,
            'num_hidden_layers': 48,
            'num_attention_heads': 40,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 32768,
            'vocab_size': 152064,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen2_5_dense_gqa',
                'parameter_billions': 14.7,
                'active_parameter_billions': 14.7,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium_high',
            'int4_priority': 'after_int8_plans_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'qwen2_5_14b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct',
            'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen2_5_32b',
        'model_id': 'Qwen/Qwen2.5-32B-Instruct',
        'runtime_model_id': 'Qwen/Qwen2.5-32B-Instruct',
        'launcher_id': 'Qwen/Qwen2.5-32B-Instruct',
        'display_name': 'Qwen2.5 32B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen2_5_dense_gqa',
            'parameter_billions': 32.5,
            'active_parameter_billions': 32.5,
            'hidden_size': 5120,
            'intermediate_size': 27648,
            'num_hidden_layers': 64,
            'num_attention_heads': 40,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 32768,
            'vocab_size': 152064,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen2_5_dense_gqa',
                'parameter_billions': 32.5,
                'active_parameter_billions': 32.5,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'after_int8_resident_and_before_sequential_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '8bit',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'qwen2_5_32b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': '8bit'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct',
            'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'falcon_h1_34b',
        'model_id': 'tiiuae/Falcon-H1-34B-Instruct',
        'runtime_model_id': 'tiiuae/Falcon-H1-34B-Instruct',
        'launcher_id': 'tiiuae/Falcon-H1-34B-Instruct',
        'display_name': 'Falcon-H1 34B Instruct',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'falcon_h1_hybrid_attention_mamba',
            'parameter_billions': 34.0,
            'active_parameter_billions': 34.0,
            'hidden_size': 5120,
            'intermediate_size': 21504,
            'num_hidden_layers': 72,
            'num_attention_heads': 20,
            'num_key_value_heads': 4,
            'head_dim': 128,
            'max_context_tokens': 262144,
            'vocab_size': 65024,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'falcon_h1_hybrid_attention_mamba',
                'parameter_billions': 34.0,
                'active_parameter_billions': 34.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 1024},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'after_int8_resident_and_before_sequential_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '8bit',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'falcon_h1_34b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': '8bit'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct',
            'https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct/blob/main/config.json',
            'https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'lfm2_5_1_2b_thinking',
        'model_id': 'LiquidAI/LFM2.5-1.2B-Thinking',
        'runtime_model_id': 'LiquidAI/LFM2.5-1.2B-Thinking',
        'launcher_id': 'LiquidAI/LFM2.5-1.2B-Thinking',
        'display_name': 'Liquid LFM2.5 1.2B Thinking',
        'category': 'llm',
        'task': 'chat',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'lfm2_5_hybrid_convolution_attention',
            'parameter_billions': 1.2,
            'active_parameter_billions': 1.2,
            'hidden_size': 2048,
            'intermediate_size': 12288,
            'num_hidden_layers': 16,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 64,
            'max_context_tokens': 131072,
            'vocab_size': 65536,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'lfm2_5_hybrid_convolution_attention',
                'parameter_billions': 1.2,
                'active_parameter_billions': 1.2,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 2048},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 1.5
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'practical',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'unacceptable_for_reasoning_model',
            'int4_priority': 'disabled_for_automatic_planning',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {'loader': 'transformers_pipeline', 'quantization': 'native', 'plan': 'single_gpu_cuda_0'},
        'existing_fast_path': {
            'plan_id': 'lfm2_5_1_2b_thinking_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1, 'minimum_total_vram_gib_each': 0},
            'loader': 'transformers_pipeline',
            'placement': 'single_gpu_cuda_0',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking',
            'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/blob/main/config.json',
            'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'phi_4_reasoning',
        'model_id': 'microsoft/Phi-4-reasoning',
        'runtime_model_id': 'microsoft/Phi-4-reasoning',
        'launcher_id': 'microsoft/Phi-4-reasoning',
        'display_name': 'Phi-4 14B Reasoning',
        'category': 'llm',
        'task': 'reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'phi4_dense_gqa',
            'parameter_billions': 14.7,
            'active_parameter_billions': 14.7,
            'hidden_size': 5120,
            'intermediate_size': 17920,
            'num_hidden_layers': 40,
            'num_attention_heads': 40,
            'num_key_value_heads': 10,
            'head_dim': 128,
            'max_context_tokens': 32768,
            'vocab_size': 100352,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'phi4_dense_gqa',
                'parameter_billions': 14.7,
                'active_parameter_billions': 14.7,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'only_after_int8_and_native_offload_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'phi_4_reasoning_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/microsoft/Phi-4-reasoning',
            'https://huggingface.co/microsoft/Phi-4-reasoning/blob/main/config.json',
            'https://huggingface.co/microsoft/Phi-4-reasoning/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_5_4b',
        'model_id': 'Qwen/Qwen3.5-4B',
        'runtime_model_id': 'Qwen/Qwen3.5-4B',
        'launcher_id': 'Qwen/Qwen3.5-4B',
        'display_name': 'Qwen3.5 4B',
        'category': 'multimodal_llm',
        'task': 'multimodal_reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen3_5_hybrid_linear_full_attention_multimodal',
            'parameter_billions': 4.0,
            'active_parameter_billions': 4.0,
            'hidden_size': 2560,
            'intermediate_size': 9216,
            'num_hidden_layers': 32,
            'num_attention_heads': 16,
            'num_key_value_heads': 4,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 248320,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_5_hybrid_linear_full_attention_multimodal',
                'parameter_billions': 4.0,
                'active_parameter_billions': 4.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'official_qwen_model_overview_and_configuration',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'qwen3_5_vision_transformer',
                'parameter_billions': 0.66,
                'active_parameter_billions': 0.66,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.75,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vision_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'estimated_from_official_checkpoint_and_configuration',
                'evidence': 'official_checkpoint_total_size_minus_published_language_model_parameter_count',
                'notes': ['24 layers, hidden size 1024, 16 attention heads, patch size 16.']
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model', 'vision_encoder'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'hybrid_attention_state',
                'kv_cache',
                'prefill_mlp_activations',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * full_attention_layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'low_to_medium',
            'default_runtime_headroom_gib': 1.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'only_after_int8_and_native_offload_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '8bit',
            'plan': 'planner_selected_cuda_or_mps'
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3.5-4B',
            'https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/model.safetensors.index.json'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Chat GUI uses the Qwen3.5 processor path with thinking enabled by default.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_5_9b',
        'model_id': 'Qwen/Qwen3.5-9B',
        'runtime_model_id': 'Qwen/Qwen3.5-9B',
        'launcher_id': 'Qwen/Qwen3.5-9B',
        'display_name': 'Qwen3.5 9B',
        'category': 'multimodal_llm',
        'task': 'multimodal_reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen3_5_hybrid_linear_full_attention_multimodal',
            'parameter_billions': 9.0,
            'active_parameter_billions': 9.0,
            'hidden_size': 4096,
            'intermediate_size': 12288,
            'num_hidden_layers': 32,
            'num_attention_heads': 16,
            'num_key_value_heads': 4,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 248320,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_5_hybrid_linear_full_attention_multimodal',
                'parameter_billions': 9.0,
                'active_parameter_billions': 9.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'official_qwen_model_overview_and_configuration',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'qwen3_5_vision_transformer',
                'parameter_billions': 0.65,
                'active_parameter_billions': 0.65,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.75,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vision_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'estimated_from_official_checkpoint_and_configuration',
                'evidence': 'official_checkpoint_total_size_minus_published_language_model_parameter_count',
                'notes': ['24 layers, hidden size 1024, 16 attention heads, patch size 16.']
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model', 'vision_encoder'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'hybrid_attention_state',
                'kv_cache',
                'prefill_mlp_activations',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * full_attention_layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'low_to_medium',
            'default_runtime_headroom_gib': 1.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'only_after_int8_and_native_offload_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '4bit',
            'plan': 'planner_selected_cuda_or_mps'
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3.5-9B',
            'https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/model.safetensors.index.json'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Chat GUI uses the Qwen3.5 processor path with thinking enabled by default.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_14b',
        'model_id': 'Qwen/Qwen3-14B',
        'runtime_model_id': 'Qwen/Qwen3-14B',
        'launcher_id': 'Qwen/Qwen3-14B',
        'display_name': 'Qwen3 14B',
        'category': 'llm',
        'task': 'reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen3_dense_gqa',
            'parameter_billions': 14.8,
            'active_parameter_billions': 14.8,
            'hidden_size': 5120,
            'intermediate_size': 17408,
            'num_hidden_layers': 40,
            'num_attention_heads': 40,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 40960,
            'vocab_size': 151936,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_dense_gqa',
                'parameter_billions': 14.8,
                'active_parameter_billions': 14.8,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'high',
            'int4_priority': 'only_after_int8_and_native_offload_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'qwen3_14b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3-14B',
            'https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3-14B/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'gpt_oss_20b',
        'model_id': 'openai/gpt-oss-20b',
        'runtime_model_id': 'openai/gpt-oss-20b',
        'launcher_id': 'openai/gpt-oss-20b',
        'display_name': 'GPT-OSS 20B',
        'category': 'llm',
        'task': 'reasoning',
        'checkpoint': {'format': 'native_mxfp4', 'native_dtype': 'bfloat16', 'already_quantized': True},
        'architecture': {
            'family': 'gpt_oss_moe',
            'parameter_billions': 20.9,
            'active_parameter_billions': 3.6,
            'hidden_size': 2880,
            'intermediate_size': 2880,
            'num_hidden_layers': 24,
            'num_attention_heads': 64,
            'num_key_value_heads': 8,
            'head_dim': 64,
            'max_context_tokens': 131072,
            'vocab_size': 201088,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'gpt_oss_moe',
                'parameter_billions': 20.9,
                'active_parameter_billions': 3.6,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'artifact_defined', 'int4': 'artifact_defined'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': False,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': False,
            'decoder_block_sharding': False,
            'cpu_overflow_device_map': False,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': False,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'low_for_native_checkpoint',
            'int4_priority': 'native_representation',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['attention', 'routers', 'embeddings', 'lm_head']
        },
        'current_code': {'loader': 'transformers_pipeline', 'quantization': 'native_mxfp4', 'plan': 'single_gpu_cuda_0'},
        'existing_fast_path': {
            'plan_id': 'gpt_oss_20b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1, 'minimum_total_vram_gib_each': 0},
            'loader': 'transformers_pipeline',
            'placement': 'single_gpu_cuda_0',
            'quantization': {'language_model': 'native_mxfp4'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/openai/gpt-oss-20b',
            'https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json',
            'https://huggingface.co/openai/gpt-oss-20b/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['The application checkpoint is natively MXFP4; do not reinterpret it as generic BitsAndBytes NF4.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'deepseek_r1_distill_qwen_32b',
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'runtime_model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'launcher_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'display_name': 'DeepSeek R1 Distill Qwen 32B',
        'category': 'llm',
        'task': 'reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen2_5_dense_gqa_distilled_reasoning',
            'parameter_billions': 32.8,
            'active_parameter_billions': 32.8,
            'hidden_size': 5120,
            'intermediate_size': 27648,
            'num_hidden_layers': 64,
            'num_attention_heads': 40,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'max_context_tokens': 131072,
            'vocab_size': 152064,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen2_5_dense_gqa_distilled_reasoning',
                'parameter_billions': 32.8,
                'active_parameter_billions': 32.8,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium_high',
            'int4_priority': 'after_int8_resident_and_before_sequential_offload',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '8bit',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'deepseek_r1_distill_qwen_32b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': '8bit'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
            'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/config.json',
            'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_6_27b',
        'model_id': 'Qwen/Qwen3.6-27B',
        'runtime_model_id': 'Qwen/Qwen3.6-27B',
        'launcher_id': 'Qwen/Qwen3.6-27B',
        'display_name': 'Qwen3.6 27B',
        'category': 'multimodal_llm',
        'task': 'multimodal_reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen3_5_hybrid_linear_full_attention_multimodal',
            'parameter_billions': 27.0,
            'active_parameter_billions': 27.0,
            'hidden_size': 5120,
            'intermediate_size': 17408,
            'num_hidden_layers': 64,
            'num_attention_heads': 24,
            'num_key_value_heads': 4,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 248320,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_5_hybrid_linear_full_attention_multimodal',
                'parameter_billions': 27.0,
                'active_parameter_billions': 27.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'qwen3_5_vision_transformer',
                'parameter_billions': 0.45,
                'active_parameter_billions': 0.45,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.75,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vision_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'model_specific', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['27 layers, hidden size 1152, 16 attention heads, patch size 16.']
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model', 'vision_encoder'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 3.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'application_current_default_due_large_multimodal_checkpoint',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '4bit',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'qwen3_6_27b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': '4bit'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3.6-27B',
            'https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.6-27B/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'gemma_4_31b_it',
        'model_id': 'google/gemma-4-31B-it',
        'runtime_model_id': 'google/gemma-4-31B-it',
        'launcher_id': 'google/gemma-4-31B-it',
        'display_name': 'Gemma 4 31B IT',
        'category': 'multimodal_llm',
        'task': 'multimodal_reasoning',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'gemma4_multimodal_dense_sliding_global_attention',
            'parameter_billions': 31.0,
            'active_parameter_billions': 31.0,
            'hidden_size': 5376,
            'intermediate_size': 21504,
            'num_hidden_layers': 60,
            'num_attention_heads': 32,
            'num_key_value_heads': 16,
            'head_dim': 256,
            'max_context_tokens': 131072,
            'vocab_size': 262144,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'gemma4_multimodal_dense_sliding_global_attention',
                'parameter_billions': 31.0,
                'active_parameter_billions': 31.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'gemma4_vision_transformer',
                'parameter_billions': 0.45,
                'active_parameter_billions': 0.45,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.75,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vision_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'model_specific', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['27 layers, hidden size 1152, intermediate size 4304, 16 heads, patch size 16.']
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model', 'vision_encoder'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium_high',
            'int4_priority': 'after_int8_plans_fail',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': '8bit',
            'plan': 'balanced_two_gpu_22gib_each'
        },
        'existing_fast_path': {
            'plan_id': 'gemma_4_31b_it_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'transformers_pipeline',
            'placement': 'balanced_two_gpu_22gib_each',
            'quantization': {'language_model': '8bit'},
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/google/gemma-4-31B-it',
            'https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json',
            'https://huggingface.co/google/gemma-4-31B-it/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [
            'Text dimensions are architecture-derived from the published configuration; multimodal preprocessing adds image-token memory.'
        ],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_6_35b_a3b',
        'model_id': 'Qwen/Qwen3.6-35B-A3B',
        'runtime_model_id': 'Qwen/Qwen3.6-35B-A3B-FP8',
        'launcher_id': 'Qwen/Qwen3.6-35B-A3B',
        'display_name': 'Qwen3.6 35B A3B',
        'category': 'multimodal_llm',
        'task': 'multimodal_reasoning',
        'checkpoint': {'format': 'native_fp8', 'native_dtype': 'bfloat16', 'already_quantized': True},
        'architecture': {
            'family': 'qwen3_5_moe_multimodal',
            'parameter_billions': 35.0,
            'active_parameter_billions': 3.0,
            'hidden_size': 2048,
            'intermediate_size': 5120,
            'num_hidden_layers': 40,
            'num_attention_heads': 16,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 248320,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_5_moe_multimodal',
                'parameter_billions': 35.0,
                'active_parameter_billions': 3.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'artifact_defined', 'int4': 'artifact_defined'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'qwen3_5_vision_transformer',
                'parameter_billions': 0.45,
                'active_parameter_billions': 0.45,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.75,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vision_encoding'],
                'sharding': 'whole_component',
                'offload': 'backend_managed',
                'quantization_support': {'int8': 'native_checkpoint', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['27 layers, hidden size 1152, 16 attention heads.']
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model', 'vision_encoder'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 8192, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': False,
            'vllm': True
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': False,
            'decoder_block_sharding': False,
            'cpu_overflow_device_map': False,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low_for_native_fp8',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'high',
            'int4_priority': 'not_used_for_current_vllm_checkpoint',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {'loader': 'vllm_async_engine', 'quantization': 'native_fp8', 'plan': 'tensor_parallel_2_gpu'},
        'existing_fast_path': {
            'plan_id': 'qwen3_6_35b_a3b_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'vllm_async_engine',
            'placement': 'tensor_parallel_2_gpu',
            'quantization': {'language_model': 'native_fp8'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/main',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['The launcher logical ID is remapped by chat_gui.py to the FP8 runtime checkpoint.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_coder_next_gguf',
        'model_id': 'Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M',
        'runtime_model_id': 'Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M',
        'launcher_id': 'Qwen/Qwen3-Coder-Next-GGUF:Q4_K_M',
        'display_name': 'Qwen3 Coder Next 80B Q4_K_M',
        'category': 'llm',
        'task': 'code_reasoning',
        'checkpoint': {'format': 'gguf_q4_k_m', 'native_dtype': 'bfloat16', 'already_quantized': True},
        'architecture': {
            'family': 'qwen3_coder_next_moe',
            'parameter_billions': 80.0,
            'active_parameter_billions': 3.0,
            'hidden_size': 2048,
            'intermediate_size': 5120,
            'num_hidden_layers': 48,
            'num_attention_heads': 16,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 151936,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_coder_next_moe',
                'parameter_billions': 80.0,
                'active_parameter_billions': 3.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'artifact_defined', 'int4': 'artifact_defined'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': False,
            'vllm': True
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': False,
            'decoder_block_sharding': False,
            'cpu_overflow_device_map': False,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': False,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium_for_native_gguf',
            'int4_priority': 'native_artifact',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': True,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'vllm_async_engine',
            'quantization': 'gguf_q4_k_m',
            'plan': 'tensor_parallel_2_gpu_gguf'
        },
        'existing_fast_path': {
            'plan_id': 'qwen3_coder_next_gguf_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'vllm_async_engine',
            'placement': 'tensor_parallel_2_gpu_gguf',
            'quantization': {'language_model': 'gguf_q4_k_m'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3-Coder-Next',
            'https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main',
            'https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': [
            'The application selects the Q4_K_M GGUF artifact and uses the unquantized repository for tokenizer and configuration metadata.'
        ],
        'tokenizer_model_id': 'Qwen/Qwen3-Coder-Next',
        'config_model_id': 'Qwen/Qwen3-Coder-Next',
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen2_5_0_5b_auxiliary',
        'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
        'runtime_model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
        'launcher_id': None,
        'display_name': 'Qwen2.5 0.5B Auxiliary Prompt/Reasoning Model',
        'category': 'llm',
        'task': 'auxiliary_text_generation',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen2_5_dense_gqa',
            'parameter_billions': 0.494,
            'active_parameter_billions': 0.494,
            'hidden_size': 896,
            'intermediate_size': 4864,
            'num_hidden_layers': 24,
            'num_attention_heads': 14,
            'num_key_value_heads': 2,
            'head_dim': 64,
            'max_context_tokens': 32768,
            'vocab_size': 151936,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen2_5_dense_gqa',
                'parameter_billions': 0.494,
                'active_parameter_billions': 0.494,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 2048, 'max_new_tokens': 512},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'medium',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 1.5
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'practical',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'unacceptable_for_automatic_use',
            'int4_priority': 'disabled_for_automatic_planning',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'transformers_pipeline',
            'quantization': 'native',
            'plan': 'cpu_or_small_cuda_component'
        },
        'existing_fast_path': {
            'plan_id': 'qwen2_5_0_5b_auxiliary_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1, 'minimum_total_vram_gib_each': 0},
            'loader': 'transformers_pipeline',
            'placement': 'cpu_or_small_cuda_component',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct',
            'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Used by both prompt generation in base_gui.py and reasoning summaries in chat_gui.py.'],
        'application_source_files': ['base_gui.py', 'chat_gui.py']
    },
    {
        'key': 'z_image_turbo',
        'model_id': 'Tongyi-MAI/Z-Image-Turbo',
        'runtime_model_id': 'Tongyi-MAI/Z-Image-Turbo',
        'launcher_id': 'z_image_turbo',
        'display_name': 'Z Image Turbo',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 's3_dit_with_qwen3_conditioner', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen3_4b_text_encoder',
                'parameter_billions': 4.0,
                'active_parameter_billions': 4.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 7.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_sharding_and_device_map',
                'offload': 'component_offload',
                'quantization_support': {
                    'int8': 'recommended',
                    'int4': 'supported_with_low_to_medium_prompt_alignment_risk'
                },
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['36 layers, hidden size 2560, intermediate size 9728, 32 attention heads and 8 KV heads.']
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'z_image_s3_dit',
                'parameter_billions': 6.15,
                'active_parameter_billions': 6.15,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 22.9,
                'checkpoint_storage_basis': 'repository_transformer_files_are_float32_sized',
                'phases': ['denoising'],
                'sharding': '32_transformer_blocks_can_be_distributed',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'recommended_when_needed', 'int4': 'high_caution'},
                'skip_modules': ['context_embedder', 'x_embedder', 'time_text_embed', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['Model dimension 3840, 30 attention heads, 30 main layers plus 2 refinement layers.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'autoencoder_kl_image',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.16,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 9,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'ZImagePipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'z_image_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo',
            'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/config.json',
            'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main',
            'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json',
            'https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_t2i_lite_sft',
        'model_id': 'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers',
        'launcher_id': 'kandinsky_5',
        'display_name': 'Kandinsky 5 T2I Lite SFT',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_lite_multimodal_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_text_and_vision_encoder',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_sharding_and_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'supported_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'kandinsky5_lite_dit',
                'parameter_billions': 6.0,
                'active_parameter_billions': 6.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 11.2,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': 'transformer_block_device_map',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'recommended_when_needed', 'int4': 'high_caution'},
                'skip_modules': ['time_embeddings', 'text_embeddings', 'visual_embeddings', 'modulation', 'out_layer'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'flux_autoencoder_kl',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.16,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'Kandinsky5T2IPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow_and_preview_vae_on_gpu1',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_t2i_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preview_vae_device': 'cuda:1',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'pixart_sigma',
        'model_id': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        'runtime_model_id': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        'launcher_id': 'pixart_sigma',
        'display_name': 'PixArt Sigma XL 1024',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'pixart_sigma_dit_with_t5_xxl', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 17.7,
                'checkpoint_storage_basis': 'repository_contains_float32_sized_t5_weights',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'pixart_sigma_transformer_2d',
                'parameter_billions': 0.61,
                'active_parameter_billions': 0.61,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 2.27,
                'checkpoint_storage_basis': 'repository_float32_transformer_payload',
                'phases': ['denoising'],
                'sharding': '28_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'usually_unnecessary', 'int4': 'avoid_for_small_denoiser'},
                'skip_modules': ['pos_embed', 'adaln_single', 'caption_projection', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['28 layers, 16 attention heads, head dimension 72, patch size 2, caption channels 4096.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'sdxl_autoencoder_kl',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.31,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1024,
            'height': 1024,
            'batch_size': 1,
            'inference_steps': 100,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'PixArtSigmaPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'pixart_sigma_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
            'https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/blob/main/config.json',
            'https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'anima',
        'model_id': 'local/Anima-ComfyUI',
        'runtime_model_id': 'local/Anima-ComfyUI',
        'launcher_id': 'anima',
        'display_name': 'Anima',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'comfy_checkpoint_native',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': True
        },
        'architecture': {'family': 'comfyui_anima_preview_workflow', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen3_0_6b_comfy_text_encoder',
                'parameter_billions': 0.6,
                'active_parameter_billions': 0.6,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 1.11,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'comfyui_managed',
                'offload': 'comfyui_managed',
                'quantization_support': {
                    'int8': 'checkpoint_or_comfy_runtime_defined',
                    'int4': 'avoid_for_small_encoder'
                },
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'filename_and_public_qwen3_0_6b_artifact_size',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'anima_preview_dit',
                'parameter_billions': 2.0,
                'active_parameter_billions': 2.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 3.8,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': 'comfyui_managed',
                'offload': 'comfyui_managed',
                'quantization_support': {'int8': 'comfyui_runtime_defined', 'int4': 'checkpoint_defined'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'launcher_parameter_label_and_local_checkpoint_name',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'qwen_image_vae',
                'parameter_billions': 0.127,
                'active_parameter_billions': 0.127,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'comfyui_managed',
                'offload': 'comfyui_managed',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 40,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'comfyui_supported',
            'mps': 'comfyui_installation_dependent',
            'cpu': 'comfyui_installation_dependent',
            'bitsandbytes_cuda': False,
            'comfyui': True
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': False,
            'multi_gpu_block_sharding': False,
            'device_map': False,
            'model_cpu_offload': False,
            'sequential_cpu_offload': False,
            'custom_staging': False,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': False,
            'int4_auto_allowed': False,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True
        },
        'current_code': {
            'loader': 'ComfyUI_UNETLoader_CLIPLoader_VAELoader',
            'plan': 'comfyui_managed_native_checkpoints',
            'quantization': 'checkpoint_native'
        },
        'existing_fast_path': {
            'plan_id': 'anima_current_comfyui_local_workflow',
            'hardware_match': {'backend': 'comfyui', 'gpu_count': 0},
            'loader': 'ComfyUI_UNETLoader_CLIPLoader_VAELoader',
            'placement': 'comfyui_managed',
            'artifacts': ['anima-preview.safetensors', 'qwen_3_06b_base.safetensors', 'qwen_image_vae.safetensors'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://github.com/comfyanonymous/ComfyUI',
            'https://huggingface.co/Qwen/Qwen3-0.6B',
            'https://huggingface.co/Qwen/Qwen-Image/tree/main/vae'
        ],
        'confidence': {
            'architecture': 'medium',
            'component_weight_memory': 'medium',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [
            'The exact Anima preview checkpoint is local to the configured ComfyUI installation. The registry uses the launcher 2B label and public supporting checkpoint sizes rather than leaving the profile empty.'
        ],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'stable_diffusion_3_5_large',
        'model_id': 'stabilityai/stable-diffusion-3.5-large',
        'runtime_model_id': 'stabilityai/stable-diffusion-3.5-large',
        'launcher_id': 'stable_diffusion_3_5',
        'display_name': 'Stable Diffusion 3.5 Large',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'stable_diffusion_3_5_mmdit_three_text_encoders', 'component_count': 5},
        'components': {
            'clip_l': {
                'role': 'small_text_encoder',
                'architecture': 'clip_l_text_encoder',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.23,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_g': {
                'role': 'small_text_encoder',
                'architecture': 'openclip_big_g_text_encoder',
                'parameter_billions': 0.695,
                'active_parameter_billions': 0.695,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 1.3,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            't5_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 9.12,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'stable_diffusion_3_5_mmdit',
                'parameter_billions': 8.1,
                'active_parameter_billions': 8.1,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.2,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': '38_joint_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {
                    'int8': 'recommended_when_needed',
                    'int4': 'official_nf4_examples_exist_but_quality_risk_medium_high'
                },
                'skip_modules': ['context_embedder', 'x_embedder', 'time_text_embed', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['38 layers, 38 heads, head dimension 64, joint attention dimension 4096.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'sd3_autoencoder_kl',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.16,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['clip_l', 'clip_g', 't5_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 40,
            'prompt_tokens': 256
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['t5_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'StableDiffusion3Pipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow_and_preview_vae_on_gpu1',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'sd35_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preview_vae_device': 'cuda:1',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/stabilityai/stable-diffusion-3.5-large',
            'https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/config.json',
            'https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'flux_1_dev',
        'model_id': 'black-forest-labs/FLUX.1-dev',
        'runtime_model_id': 'black-forest-labs/FLUX.1-dev',
        'launcher_id': 'flux_1',
        'display_name': 'FLUX.1 dev',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'flux1_transformer_with_t5_and_clip', 'component_count': 4},
        'components': {
            'clip_l': {
                'role': 'small_text_encoder',
                'architecture': 'clip_l_text_encoder',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.23,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            't5_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 8.87,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'flux1_rectified_flow_transformer',
                'parameter_billions': 11.9,
                'active_parameter_billions': 11.9,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 22.2,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': 'double_and_single_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {
                    'int8': 'recommended_when_needed',
                    'int4': 'supported_with_high_caution_or_official_checkpoint'
                },
                'skip_modules': ['x_embedder', 'context_embedder', 'time_text_embed', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'flux_autoencoder_kl',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.31,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['clip_l', 't5_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['t5_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'FluxPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow_and_preview_vae_on_gpu1',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'flux1_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preview_vae_device': 'cuda:1',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/black-forest-labs/FLUX.1-dev',
            'https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/config.json',
            'https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'glm_image',
        'model_id': 'zai-org/GLM-Image',
        'runtime_model_id': 'zai-org/GLM-Image',
        'launcher_id': 'glm_image',
        'display_name': 'GLM Image',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'glm_hybrid_autoregressive_prior_and_dit', 'component_count': 4},
        'components': {
            'vision_language_encoder': {
                'role': 'vision_language_encoder',
                'architecture': 'glm4v_autoregressive_multimodal_encoder',
                'parameter_billions': 9.0,
                'active_parameter_billions': 9.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 18.8,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['autoregressive_conditioning'],
                'sharding': 'decoder_block_device_map_but_current_fast_path_whole_gpu0',
                'offload': 'staged_component_offload',
                'quantization_support': {
                    'int8': 'reasonable_future_fallback',
                    'int4': 'high_caution_for_autoregressive_conditioner'
                },
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Text hidden size 4096, 40 layers, 32 attention heads, 2 KV heads; vision encoder depth 40 and hidden size 1536.'
                ]
            },
            't5_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_base_custom_1472',
                'parameter_billions': 0.435,
                'active_parameter_billions': 0.435,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.81,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid_for_small_encoder'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['d_model 1472, d_ff 3584, 12 encoder layers.']
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'glm_image_transformer_2d',
                'parameter_billions': 7.0,
                'active_parameter_billions': 7.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 12.95,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': '30_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'recommended_when_needed', 'int4': 'high_caution'},
                'skip_modules': ['context_embedder', 'x_embedder', 'time_embedder', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['30 layers, 32 attention heads, head dimension 128, text dimension 1472.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'autoencoder_kl_image',
                'parameter_billions': 0.2,
                'active_parameter_billions': 0.2,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.76,
                'checkpoint_storage_basis': 'repository_float32_vae_payload',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'autoregressive_conditioning',
                'required_components': ['vision_language_encoder'],
                'output_can_move_between_gpus': True,
                'dynamic_memory_scales_with': ['prompt_tokens', 'conditioning_image_tokens']
            },
            {
                'name': 'text_encoding',
                'required_components': ['t5_text_encoder'],
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 50,
            'prompt_tokens': 2048
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['vision_language_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'GLMImageGenerator',
            'plan': 'custom_component_placement_ar_gpu0_t5_dit_vae_gpu1',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'glm_image_current_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'component_devices': {
                'vision_language_encoder': 'cuda:0',
                't5_text_encoder': 'cuda:1',
                'transformer': 'cuda:1',
                'vae': 'cuda:1'
            },
            'max_memory_gib': {0: 22, 1: 22},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/zai-org/GLM-Image',
            'https://huggingface.co/zai-org/GLM-Image/blob/main/config.json',
            'https://huggingface.co/zai-org/GLM-Image/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'qwen_image',
        'model_id': 'Qwen/Qwen-Image',
        'runtime_model_id': 'Qwen/Qwen-Image',
        'launcher_id': 'qwen_image',
        'display_name': 'Qwen Image',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'qwen_image_dit_with_qwen2_5_vl', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Text hidden size 3584, intermediate size 18944, 28 layers, 28 attention heads and 4 KV heads; includes a vision tower.'
                ]
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'qwen_image_transformer_2d',
                'parameter_billions': 20.4,
                'active_parameter_billions': 20.4,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 38.1,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': '60_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'high_caution_unless_native_checkpoint'},
                'skip_modules': ['time_text_embed', 'img_in', 'txt_in', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    '60 transformer blocks, 24 attention heads, head dimension 128, joint attention dimension 3584.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'qwen_image_autoencoder',
                'parameter_billions': 0.127,
                'active_parameter_billions': 0.127,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 3,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'QwenImageGenerator',
            'plan': 'resident_int8_text_encoder_gpu0_int8_transformer_split_16_44_vae_gpu0',
            'quantization': 'component_int8'
        },
        'existing_fast_path': {
            'plan_id': 'qwen_image_current_dual_3090_int8',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'component_devices': {'text_encoder': 'cuda:0', 'vae': 'cuda:0'},
            'transformer_block_map': {
                'cuda:0': [0, 15],
                'cuda:1': [16, 59]
            },
            'max_memory_gib': {0: 22, 1: 22},
            'quantization': {'text_encoder': 'bnb_int8', 'transformer': 'bnb_int8', 'vae': 'bfloat16'},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen-Image',
            'https://huggingface.co/Qwen/Qwen-Image/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen-Image/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'flux_2_dev',
        'model_id': 'black-forest-labs/FLUX.2-dev',
        'runtime_model_id': 'black-forest-labs/FLUX.2-dev',
        'launcher_id': 'flux_2',
        'display_name': 'FLUX.2 dev',
        'category': 'image_diffusion',
        'task': 'image_text_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'flux2_large_dit_with_mistral3_conditioner', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'mistral3_multimodal_conditioner',
                'parameter_billions': 24.0,
                'active_parameter_billions': 24.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 44.7,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'staged_destroy_and_reload',
                'quantization_support': {
                    'int8': 'current_fast_path',
                    'int4': 'acceptable_for_conditioning_under_pressure'
                },
                'skip_modules': ['lm_head'],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Large Mistral3-family conditioner; exact parameter inventory is inferred from repository payload and architecture class.'
                ]
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'flux2_transformer_2d',
                'parameter_billions': 32.2,
                'active_parameter_billions': 32.2,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 60.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': '48_single_transformer_blocks_plus_double_stream_blocks',
                'offload': 'staged_destroy_after_denoising',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'official_bnb_4bit_checkpoint_available'},
                'skip_modules': [
                    'x_embedder',
                    'context_embedder',
                    'time_guidance_embed',
                    'double_stream_modulation_img',
                    'double_stream_modulation_txt',
                    'single_stream_modulation',
                    'norm_out',
                    'proj_out'
                ],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'flux2_autoencoder_kl',
                'parameter_billions': 0.168,
                'active_parameter_billions': 0.168,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.31,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_then_decode_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': ['text_encoder'],
                'output_can_move_to_cpu': True,
                'dynamic_memory_scales_with': ['prompt_tokens', 'batch_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': ['transformer'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'batch_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 1,
            'inference_steps': 20,
            'prompt_tokens': 256
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder', 'transformer'],
            'native_int4_checkpoint': 'diffusers/FLUX.2-dev-bnb-4bit'
        },
        'current_code': {
            'loader': 'Flux2Generator',
            'plan': 'custom_staged_int8_text_then_int8_transformer_24_24_split_then_vae_gpu1',
            'quantization': 'component_int8'
        },
        'existing_fast_path': {
            'plan_id': 'flux2_current_dual_3090_staged_int8',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 48},
            'phase_order': [
                'text_encoding',
                'release_text_encoder',
                'load_transformer',
                'denoise',
                'release_transformer',
                'vae_decode_gpu1',
                'preload_text_encoder'
            ],
            'transformer_block_map': {
                'cuda:0': [0, 23],
                'cuda:1': [24, 47]
            },
            'quantization': {'text_encoder': 'bnb_int8', 'transformer': 'bnb_int8', 'vae': 'bfloat16'},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/black-forest-labs/FLUX.2-dev',
            'https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/config.json',
            'https://huggingface.co/black-forest-labs/FLUX.2-dev/tree/main',
            'https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit',
            'https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit/blob/main/config.json',
            'https://huggingface.co/diffusers/FLUX.2-dev-bnb-4bit/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'kandinsky_5_i2i_lite_sft',
        'model_id': 'kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers',
        'launcher_id': 'kandinsky_5_i2i',
        'display_name': 'Kandinsky 5 I2I Lite SFT',
        'category': 'image_diffusion',
        'task': 'image_image_to_image',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_lite_i2i_multimodal_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_text_and_vision_encoder',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding', 'source_conditioning'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'supported_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'kandinsky5_lite_i2i_dit',
                'parameter_billions': 6.0,
                'active_parameter_billions': 6.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 11.2,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['denoising'],
                'sharding': 'transformer_block_device_map',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'recommended_when_needed', 'int4': 'high_caution'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'flux_autoencoder_kl',
                'parameter_billions': 0.084,
                'active_parameter_billions': 0.084,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.16,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'vae_decode'],
                'sharding': 'whole_component',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_and_source_conditioning',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder', 'vae'],
                'dynamic_memory_scales_with': ['prompt_tokens', 'source_image_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'dynamic_memory_scales_with': ['width', 'height', 'strength']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder', 'transformer']
        },
        'current_code': {
            'loader': 'Kandinsky5I2IPipeline.from_pretrained',
            'plan': 'diffusers_model_cpu_offload',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_i2i_current_model_cpu_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_model_cpu_offload',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'chronoedit_14b',
        'model_id': 'nvidia/ChronoEdit-14B-Diffusers',
        'runtime_model_id': 'nvidia/ChronoEdit-14B-Diffusers',
        'launcher_id': 'chronoedit',
        'display_name': 'ChronoEdit 14B',
        'category': 'image_diffusion',
        'task': 'image_edit',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'chronoedit_video_dit_image_edit', 'component_count': 4},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'umt5_xxl_encoder',
                'parameter_billions': 5.7,
                'active_parameter_billions': 5.7,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 10.6,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['conditioning'],
                'sharding': 'encoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'acceptable'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'image_encoder': {
                'role': 'vision_encoder',
                'architecture': 'clip_vit_huge',
                'parameter_billions': 0.63,
                'active_parameter_billions': 0.63,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 2.35,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['conditioning'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'chronoedit_3d_transformer',
                'parameter_billions': 14.0,
                'active_parameter_billions': 14.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 26.1,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['temporal_edit_denoising'],
                'sharding': '40_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'last_resort_only'},
                'skip_modules': ['patch_embedding', 'condition_embedder', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'wan_autoencoder_kl',
                'parameter_billions': 0.127,
                'active_parameter_billions': 0.127,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.47,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'conditioning',
                'required_components': ['text_encoder', 'image_encoder', 'vae'],
                'dynamic_memory_scales_with': ['prompt_tokens', 'source_image_size']
            },
            {
                'name': 'temporal_edit_denoising',
                'required_components': ['transformer'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 1,
            'num_frames': 5,
            'inference_steps': 40,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder', 'transformer'],
            'video_transformer_int4_priority': 'last_resort'
        },
        'current_code': {
            'loader': 'ChronoEditGenerator',
            'plan': 'resident_text_image_vae_gpu0_int8_transformer_split_10_30',
            'quantization': 'component_int8'
        },
        'existing_fast_path': {
            'plan_id': 'chronoedit_current_dual_3090_int8',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'component_devices': {'text_encoder': 'cuda:0', 'image_encoder': 'cuda:0', 'vae': 'cuda:0'},
            'transformer_block_map': {
                'cuda:0': [0, 9],
                'cuda:1': [10, 39]
            },
            'quantization': {
                'text_encoder': 'bnb_int8',
                'transformer': 'bnb_int8',
                'image_encoder': 'float32',
                'vae': 'float32'
            },
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers',
            'https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers/blob/main/config.json',
            'https://huggingface.co/nvidia/ChronoEdit-14B-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'qwen_image_edit_2511_4bit',
        'model_id': 'ovedrive/Qwen-Image-Edit-2511-4bit',
        'runtime_model_id': 'ovedrive/Qwen-Image-Edit-2511-4bit',
        'launcher_id': 'qwen_image_edit',
        'display_name': 'Qwen Image Edit 2511 4-bit',
        'category': 'image_diffusion',
        'task': 'image_edit',
        'checkpoint': {
            'format': 'native_bnb_nf4',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': True
        },
        'architecture': {'family': 'qwen_image_edit_native_nf4_dit', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_and_image_conditioning'],
                'sharding': 'decoder_block_device_map',
                'offload': 'staged_destroy_after_conditioning',
                'quantization_support': {'int8': 'supported', 'int4': 'native_checkpoint_or_supported'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'image_dit',
                'architecture': 'qwen_image_edit_transformer_2d_native_nf4',
                'parameter_billions': 20.4,
                'active_parameter_billions': 20.4,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 11.3,
                'checkpoint_storage_basis': 'native_bitsandbytes_nf4_repository_payload',
                'phases': ['denoising'],
                'sharding': '60_transformer_blocks',
                'offload': 'staged_destroy_after_denoising',
                'quantization_support': {'int8': 'source_checkpoint_alternative', 'int4': 'native_checkpoint'},
                'skip_modules': [
                    'pos_embed',
                    'time_text_embed',
                    'txt_norm',
                    'img_in',
                    'txt_in',
                    'norm_out',
                    'proj_out'
                ],
                'memory_overrides_gib': {'int4': 11.3},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'qwen_image_autoencoder',
                'parameter_billions': 0.127,
                'active_parameter_billions': 0.127,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_and_image_conditioning',
                'required_components': ['text_encoder', 'vae'],
                'releasable_after': ['text_encoder'],
                'output_can_move_to_cpu': True,
                'dynamic_memory_scales_with': ['prompt_tokens', 'source_image_size']
            },
            {
                'name': 'denoising',
                'required_components': ['transformer'],
                'releasable_after': ['transformer'],
                'dynamic_memory_scales_with': ['source_image_size', 'inference_steps']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['output_image_size']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 768,
            'batch_size': 1,
            'inference_steps': 40,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 4.0,
            'workload_scaling': ['width', 'height', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['transformer'],
            'native_4bit_checkpoint_is_intended_application_model': True,
            'quality_class': 'checkpoint_intended_not_generic_fallback'
        },
        'current_code': {
            'loader': 'QwenImageEditGenerator',
            'plan': 'custom_staged_native_4bit_checkpoint_transformer_split_30_30',
            'quantization': 'native_bnb_nf4'
        },
        'existing_fast_path': {
            'plan_id': 'qwen_image_edit_current_dual_3090_native_4bit',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'phase_order': [
                'load_text_encoder_gpu0',
                'encode_to_cpu',
                'release_text_encoder',
                'load_transformer',
                'denoise',
                'release_transformer'
            ],
            'transformer_block_map': {
                'cuda:0': [0, 29],
                'cuda:1': [30, 59]
            },
            'quantization': {'transformer': 'checkpoint_native_nf4', 'vae': 'bfloat16'},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit',
            'https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit/blob/main/config.json',
            'https://huggingface.co/ovedrive/Qwen-Image-Edit-2511-4bit/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'bria_rmbg_1_4',
        'model_id': 'briaai/RMBG-1.4',
        'runtime_model_id': 'briaai/RMBG-1.4',
        'launcher_id': 'rmbg_1_4',
        'display_name': 'BRIA RMBG 1.4',
        'category': 'utility_image_segmentation',
        'task': 'image_segmentation',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'bria_rmbg_isnet', 'component_count': 1},
        'components': {
            'segmentation_model': {
                'role': 'segmentation_network',
                'architecture': 'isnet_enhanced_bria_rmbg',
                'parameter_billions': 0.044,
                'active_parameter_billions': 0.044,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.25,
                'checkpoint_storage_gib': 0.17,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['segmentation'],
                'sharding': 'whole_component',
                'offload': 'whole_model_cpu_or_gpu',
                'quantization_support': {
                    'int8': 'not_recommended_due_small_size_and_remote_custom_code',
                    'int4': 'disabled'
                },
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'segmentation',
                'required_components': ['segmentation_model'],
                'dynamic_memory_scales_with': ['input_width', 'input_height']
            }
        ],
        'default_workload': {'input_max_dimension': 2048, 'batch_size': 1},
        'runtime_memory': {
            'dominant_terms': ['feature_maps', 'input_resolution'],
            'default_runtime_headroom_gib': 1.0,
            'workload_scaling': ['input_width', 'input_height'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'likely_supported_but_remote_code_dependent',
            'cpu': 'practical',
            'bitsandbytes_cuda': False,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': False,
            'multi_gpu_block_sharding': False,
            'device_map': False,
            'model_cpu_offload': False,
            'sequential_cpu_offload': False,
            'custom_staging': False,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': False,
            'int4_auto_allowed': False,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['segmentation_network'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True
        },
        'current_code': {
            'loader': 'transformers_image_segmentation_pipeline',
            'plan': 'single_cuda_gpu_or_cpu',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'bria_rmbg_current_cuda_or_cpu_pipeline',
            'hardware_match': {'backend': 'cuda_or_cpu', 'gpu_count': 0},
            'loader': 'transformers_image_segmentation_pipeline',
            'placement': 'cuda0_when_available_else_cpu',
            'quantization': {'segmentation_model': 'float32'},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/briaai/RMBG-1.4',
            'https://huggingface.co/briaai/RMBG-1.4/blob/main/config.json',
            'https://huggingface.co/briaai/RMBG-1.4/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'stable_diffusion_x4_upscaler',
        'model_id': 'stabilityai/stable-diffusion-x4-upscaler',
        'runtime_model_id': 'stabilityai/stable-diffusion-x4-upscaler',
        'launcher_id': 'sd_x4_upscaler',
        'display_name': 'Stable Diffusion x4 Upscaler',
        'category': 'image_diffusion_upscale',
        'task': 'image_upscale_diffusion',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'stable_diffusion_x4_convolutional_unet', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_encoder_1024',
                'parameter_billions': 0.34,
                'active_parameter_billions': 0.34,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.63,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'unet': {
                'role': 'unet',
                'architecture': 'stable_diffusion_x4_convolutional_unet',
                'parameter_billions': 0.74,
                'active_parameter_billions': 0.74,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 2.8,
                'checkpoint_storage_basis': 'repository_contains_multiple_precision_variants',
                'phases': ['denoising'],
                'sharding': 'whole_component_or_attention_block_device_map',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'limited_bitsandbytes_benefit', 'int4': 'avoid_generic_bitsandbytes'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Convolution-heavy U-Net with channels 256, 512, 512 and 1024; BitsAndBytes only affects eligible Linear layers.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'stable_diffusion_autoencoder_kl',
                'parameter_billions': 0.083,
                'active_parameter_billions': 0.083,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.32,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'vae_decode'],
                'sharding': 'whole_component',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'source_encode_and_denoise',
                'required_components': ['vae', 'unet'],
                'dynamic_memory_scales_with': ['input_width', 'input_height', 'guidance']
            },
            {
                'name': 'vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['output_width', 'output_height']
            }
        ],
        'default_workload': {
            'input_width': 256,
            'input_height': 256,
            'scale': 4,
            'batch_size': 1,
            'inference_steps': 30
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 3.0,
            'workload_scaling': ['input_width', 'input_height', 'scale'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'recommended_fallback': 'tiling_and_model_offload'
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': False,
            'int4_auto_allowed': False,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'reason': 'small_convolution_heavy_pipeline_with_limited_linear_weight_savings'
        },
        'current_code': {
            'loader': 'StableDiffusionUpscalePipeline.from_pretrained',
            'plan': 'diffusers_model_cpu_offload_fp16',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'sd_x4_current_model_cpu_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_model_cpu_offload',
            'vae_optimizations': ['tiling', 'slicing'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler',
            'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/config.json',
            'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/main'
        ],
        'confidence': {
            'architecture': 'medium',
            'component_weight_memory': 'medium',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'skyreels_v2_df_1_3b',
        'model_id': 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers',
        'runtime_model_id': 'Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers',
        'launcher_id': 'skyreels_v2',
        'display_name': 'SkyReels V2 DF 1.3B 540P',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'skyreels_diffusion_forcing_video_dit', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'umt5_xxl_encoder',
                'parameter_billions': 5.7,
                'active_parameter_billions': 5.7,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 21.1,
                'checkpoint_storage_basis': 'repository_float32_sized_encoder_payload',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'skyreels_diffusion_forcing_transformer',
                'parameter_billions': 1.43,
                'active_parameter_billions': 1.43,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 5.35,
                'checkpoint_storage_basis': 'repository_float32_sized_transformer_payload',
                'phases': ['video_denoising'],
                'sharding': '30_transformer_blocks',
                'offload': 'model_or_sequential_cpu_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid_for_small_video_denoiser'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['Model dimension 1536, 30 layers, 12 attention heads, FFN size 8960, 16 latent channels.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'wan_video_autoencoder',
                'parameter_billions': 0.127,
                'active_parameter_billions': 0.127,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.47,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 960,
            'height': 544,
            'num_frames': 97,
            'batch_size': 1,
            'inference_steps': 30,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'SkyReelsV2DiffusionForcingPipeline.from_pretrained',
            'plan': 'diffusers_model_cpu_offload',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'skyreels_current_model_cpu_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_model_cpu_offload',
            'vae_dtype': 'float32',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers',
            'https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers/blob/main/config.json',
            'https://huggingface.co/Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_t2v_lite_distilled16',
        'model_id': 'kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers',
        'launcher_id': 'kandinsky_5_t2v',
        'display_name': 'Kandinsky 5 T2V Lite distilled16',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_lite_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'kandinsky5_lite_video_transformer',
                'parameter_billions': 2.0,
                'active_parameter_billions': 2.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 3.73,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_denoising'],
                'sharding': 'visual_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': ['time_embeddings', 'text_embeddings', 'visual_embeddings', 'modulation', 'out_layer'],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'The official Lite family is approximately 2B; the launcher display label of 6B includes or predates broader pipeline accounting.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_autoencoder',
                'parameter_billions': 0.247,
                'active_parameter_billions': 0.247,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.46,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 768,
            'height': 512,
            'num_frames': 241,
            'batch_size': 1,
            'inference_steps': 16,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'Kandinsky5T2VPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow_flex_attention',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_t2v_lite_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'attention_backend': 'flex',
            'vae_optimizations': ['tiling', 'slicing'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Lite-distilled16steps-10s-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'medium',
            'component_weight_memory': 'medium',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'cosmos_predict2_2b_video2world',
        'model_id': 'nvidia/Cosmos-Predict2-2B-Video2World',
        'runtime_model_id': 'nvidia/Cosmos-Predict2-2B-Video2World',
        'launcher_id': 'nvidia_cosmos',
        'display_name': 'Cosmos Predict2 2B Video2World',
        'category': 'video_diffusion',
        'task': 'video_image_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'cosmos_predict2_video2world_dit', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl_or_cosmos_text_encoder',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 8.9,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'cosmos_predict2_video2world_transformer',
                'parameter_billions': 2.0,
                'active_parameter_billions': 2.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 3.64,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_denoising'],
                'sharding': 'transformer_block_device_map',
                'offload': 'model_or_sequential_cpu_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'cosmos_video_tokenizer_decoder',
                'parameter_billions': 0.25,
                'active_parameter_billions': 0.25,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'source_image_encoding',
                'required_components': ['vision_encoder'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['input_image_size']
            },
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 704,
            'num_frames': 60,
            'batch_size': 1,
            'inference_steps': 25,
            'fps': 12
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'Cosmos2VideoToWorldPipeline.from_pretrained',
            'plan': 'diffusers_model_cpu_offload',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'cosmos_current_model_cpu_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_model_cpu_offload',
            'offload_exclusions': ['safety_checker'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World',
            'https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/blob/main/config.json',
            'https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/tree/main'
        ],
        'confidence': {
            'architecture': 'medium',
            'component_weight_memory': 'medium',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'allegro_t2v',
        'model_id': 'rhymes-ai/Allegro',
        'runtime_model_id': 'rhymes-ai/Allegro',
        'launcher_id': 'allegro',
        'display_name': 'Allegro T2V',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'allegro_video_dit', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 8.9,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'allegro_video_transformer',
                'parameter_billions': 2.8,
                'active_parameter_billions': 2.8,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 5.16,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_denoising'],
                'sharding': 'transformer_block_device_map',
                'offload': 'model_or_sequential_cpu_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'allegro_video_autoencoder',
                'parameter_billions': 0.175,
                'active_parameter_billions': 0.175,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.65,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'model_cpu_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 720,
            'num_frames': 60,
            'batch_size': 1,
            'inference_steps': 100,
            'prompt_tokens': 64
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': False,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'AllegroPipeline.from_pretrained',
            'plan': 'diffusers_model_cpu_offload',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'allegro_current_model_cpu_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_model_cpu_offload',
            'vae_dtype': 'float32',
            'vae_optimizations': ['tiling'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/rhymes-ai/Allegro',
            'https://huggingface.co/rhymes-ai/Allegro/blob/main/config.json',
            'https://huggingface.co/rhymes-ai/Allegro/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'cogvideox_5b',
        'model_id': 'THUDM/CogVideoX-5b',
        'runtime_model_id': 'THUDM/CogVideoX-5b',
        'launcher_id': 'cogvideox',
        'display_name': 'CogVideoX 5B',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'cogvideox_5b_video_dit', 'component_count': 3},
        'components': {
            'text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_xxl',
                'parameter_billions': 4.76,
                'active_parameter_billions': 4.76,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 8.9,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'encoder_block_device_map',
                'offload': 'component_or_pipeline_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'cogvideox_3d_transformer',
                'parameter_billions': 5.0,
                'active_parameter_billions': 5.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 9.31,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_denoising'],
                'sharding': '42_transformer_blocks',
                'offload': 'component_or_sequential_cpu_offload',
                'quantization_support': {
                    'int8': 'supported_and_reported_near_11_4gib_pipeline_memory',
                    'int4': 'not_officially_recommended'
                },
                'skip_modules': ['patch_embed', 'time_embedding', 'norm_final', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    '42 layers, 48 attention heads, head dimension 64, 16 latent channels, temporal compression ratio 4.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'cogvideox_autoencoder_kl',
                'parameter_billions': 0.2,
                'active_parameter_billions': 0.2,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.38,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'component_or_pipeline_offload',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 720,
            'height': 480,
            'num_frames': 49,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 226
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'CogVideoXPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'cogvideox_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'vae_optimizations': ['tiling'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/THUDM/CogVideoX-5b',
            'https://huggingface.co/THUDM/CogVideoX-5b/blob/main/config.json',
            'https://huggingface.co/THUDM/CogVideoX-5b/tree/main',
            'https://huggingface.co/zai-org/CogVideoX-5b',
            'https://huggingface.co/zai-org/CogVideoX-5b/blob/main/config.json',
            'https://huggingface.co/zai-org/CogVideoX-5b/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [
            'The original THUDM identifier is retained because it is used by the code; the repository is also published under the current zai-org namespace.'
        ],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'hunyuan_video_1_5_t2v',
        'model_id': 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v',
        'runtime_model_id': 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v',
        'launcher_id': 'hunyuan_video_1_5',
        'display_name': 'Hunyuan Video 1.5 720p T2V',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'hunyuan_video_1_5_large_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 7.05,
                'active_parameter_billions': 7.05,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 13.1,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            't5_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_custom_1472',
                'parameter_billions': 0.435,
                'active_parameter_billions': 0.435,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 1.62,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid_for_small_encoder'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'hunyuan_video_1_5_transformer',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 31.0,
                'checkpoint_storage_basis': 'repository_float32_sized_transformer_payload',
                'phases': ['video_denoising'],
                'sharding': '54_double_stream_transformer_blocks',
                'offload': 'component_or_sequential_cpu_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': ['x_embedder', 'condition_embedder', 'time_embedder', 'norm_out', 'proj_out'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Hidden size 2048, 16 attention heads, 54 double-stream blocks, latent input/output channels 32.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_1_5_autoencoder',
                'parameter_billions': 1.26,
                'active_parameter_billions': 1.26,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 4.7,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['Spatial compression ratio 16, temporal compression ratio 4, 32 latent channels.']
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 't5_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 720,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 10.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'guidance'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'vae_peak_risk': 'very_high'
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'HunyuanVideo15Pipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'hunyuan15_t2v_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v',
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v/blob/main/config.json',
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'hunyuan_video_1_5_i2v',
        'model_id': 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v',
        'runtime_model_id': 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v',
        'launcher_id': 'hunyuan_video_1_5_i2v',
        'display_name': 'Hunyuan Video 1.5 720p I2V',
        'category': 'video_diffusion',
        'task': 'video_image_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'hunyuan_video_1_5_i2v_large_video_dit', 'component_count': 5},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 7.05,
                'active_parameter_billions': 7.05,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 13.1,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            't5_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 't5_encoder_custom_1472',
                'parameter_billions': 0.435,
                'active_parameter_billions': 0.435,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 1.62,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vision_encoder': {
                'role': 'vision_encoder',
                'architecture': 'siglip_vision_transformer',
                'parameter_billions': 0.4,
                'active_parameter_billions': 0.4,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.75,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_image_conditioning'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'optional', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'hunyuan_video_1_5_i2v_transformer',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 31.0,
                'checkpoint_storage_basis': 'repository_float32_sized_transformer_payload',
                'phases': ['video_denoising'],
                'sharding': '54_double_stream_transformer_blocks',
                'offload': 'component_or_sequential_cpu_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_1_5_autoencoder',
                'parameter_billions': 1.26,
                'active_parameter_billions': 1.26,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 4.7,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'source_image_encoding',
                'required_components': ['vision_encoder'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['input_image_size']
            },
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 't5_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1280,
            'height': 720,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 11.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'guidance'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'vae_peak_risk': 'very_high'
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'HunyuanVideo15ImageToVideoPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'hunyuan15_i2v_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v',
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v/blob/main/config.json',
            'https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_t2v_pro_distilled_5s',
        'model_id': 'kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers',
        'launcher_id': 'kandinsky_5_t2v_pro',
        'display_name': 'Kandinsky 5 T2V Pro distilled 5s',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_pro_19b_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'sequential_or_component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'kandinsky5_pro_video_transformer',
                'parameter_billions': 19.3,
                'active_parameter_billions': 19.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 71.9,
                'checkpoint_storage_basis': 'repository_float32_sized_payload_and_official_19b_family_description',
                'phases': ['video_denoising'],
                'sharding': '4_text_blocks_and_60_visual_blocks',
                'offload': 'sequential_cpu_offload',
                'quantization_support': {'int8': 'strong_fallback_candidate', 'int4': 'last_resort_only'},
                'skip_modules': [
                    'time_embeddings',
                    'text_embeddings',
                    'pooled_text_embeddings',
                    'visual_embeddings',
                    'modulation',
                    'out_layer'
                ],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['Model dimension 4096, FFN dimension 16384, 4 text blocks and 60 visual blocks.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_autoencoder',
                'parameter_billions': 0.247,
                'active_parameter_billions': 0.247,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.46,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'sequential_or_staged_cpu_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1024,
            'height': 768,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 16,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 12.0,
            'workload_scaling': ['width', 'height', 'num_frames'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'sequential_offload_expected': True
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder', 'transformer'],
            'video_transformer_int4_priority': 'last_resort'
        },
        'current_code': {
            'loader': 'Kandinsky5T2VPipeline.from_pretrained',
            'plan': 'diffusers_sequential_cpu_offload_flex_attention',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_t2v_pro_distilled_current_sequential_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_sequential_cpu_offload',
            'attention_backend': 'flex',
            'vae_optimizations': ['tiling', 'slicing'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-distilled-5s-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_t2v_pro_sft_5s',
        'model_id': 'kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers',
        'launcher_id': 'kandinsky_5_t2v_pro_sft',
        'display_name': 'Kandinsky 5 T2V Pro SFT 5s',
        'category': 'video_diffusion',
        'task': 'video_text_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_pro_19b_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'sequential_or_component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'kandinsky5_pro_video_transformer',
                'parameter_billions': 19.3,
                'active_parameter_billions': 19.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 71.9,
                'checkpoint_storage_basis': 'repository_float32_sized_payload_and_official_19b_family_description',
                'phases': ['video_denoising'],
                'sharding': '4_text_blocks_and_60_visual_blocks',
                'offload': 'sequential_cpu_offload',
                'quantization_support': {'int8': 'strong_fallback_candidate', 'int4': 'last_resort_only'},
                'skip_modules': [
                    'time_embeddings',
                    'text_embeddings',
                    'pooled_text_embeddings',
                    'visual_embeddings',
                    'modulation',
                    'out_layer'
                ],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_autoencoder',
                'parameter_billions': 0.247,
                'active_parameter_billions': 0.247,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.46,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'sequential_or_staged_cpu_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 1024,
            'height': 768,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 16,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 12.0,
            'workload_scaling': ['width', 'height', 'num_frames'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'sequential_offload_expected': True
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder', 'transformer'],
            'video_transformer_int4_priority': 'last_resort'
        },
        'current_code': {
            'loader': 'Kandinsky5T2VPipeline.from_pretrained',
            'plan': 'diffusers_sequential_cpu_offload_flex_attention',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_t2v_pro_sft_current_sequential_offload',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1},
            'offload': 'enable_sequential_cpu_offload',
            'attention_backend': 'flex',
            'vae_optimizations': ['tiling', 'slicing'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-T2V-Pro-sft-5s-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_i2v_lite_5s',
        'model_id': 'kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers',
        'launcher_id': 'kandinsky_5_i2v',
        'display_name': 'Kandinsky 5 I2V Lite 5s',
        'category': 'video_diffusion',
        'task': 'video_image_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_lite_i2v_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'recommended', 'int4': 'acceptable'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'component_offload',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'kandinsky5_lite_i2v_transformer',
                'parameter_billions': 2.0,
                'active_parameter_billions': 2.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 4.36,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['video_denoising'],
                'sharding': 'visual_transformer_blocks',
                'offload': 'component_or_sequential_offload',
                'quantization_support': {'int8': 'conditional_fallback', 'int4': 'last_resort_only'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'Official Lite family is approximately 2B; source image conditioning increases runtime activation pressure.'
                ]
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_autoencoder',
                'parameter_billions': 0.247,
                'active_parameter_billions': 0.247,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.46,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_or_gpu',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'source_image_encoding',
                'required_components': ['vision_encoder'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['input_image_size']
            },
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': [],
                'output_can_move_to_cpu': False,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': [],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 768,
            'height': 512,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 7.0,
            'workload_scaling': ['width', 'height', 'num_frames', 'batch_size'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ]
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder'],
            'int4_denoiser_allowed': False
        },
        'current_code': {
            'loader': 'Kandinsky5I2VPipeline.from_pretrained',
            'plan': 'balanced_device_map_two_22gib_gpus_with_cpu_overflow_flex_attention',
            'quantization': 'native'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_i2v_lite_current_balanced_dual_3090',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 80},
            'attention_backend': 'flex',
            'vae_optimizations': ['tiling', 'slicing'],
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Lite-5s-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'medium',
            'component_weight_memory': 'medium',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py']
    },
    {
        'key': 'kandinsky_5_i2v_pro_sft_5s',
        'model_id': 'kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers',
        'runtime_model_id': 'kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers',
        'launcher_id': 'kandinsky_5_i2v_pro_sft',
        'display_name': 'Kandinsky 5 I2V Pro SFT 5s',
        'category': 'video_diffusion',
        'task': 'video_image_to_video',
        'checkpoint': {
            'format': 'diffusers_repository',
            'native_dtype': 'bfloat16_or_component_specific',
            'already_quantized': False
        },
        'architecture': {'family': 'kandinsky5_pro_19b_i2v_video_dit', 'component_count': 4},
        'components': {
            'qwen_text_encoder': {
                'role': 'large_text_encoder',
                'architecture': 'qwen2_5_vl_7b_conditioner',
                'parameter_billions': 8.3,
                'active_parameter_billions': 8.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 15.5,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'decoder_block_device_map',
                'offload': 'staged_destroy_after_prompt_encoding',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'acceptable_with_prompt_alignment_risk'},
                'skip_modules': ['visual', 'lm_head'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'clip_text_encoder': {
                'role': 'small_text_encoder',
                'architecture': 'clip_text_transformer',
                'parameter_billions': 0.123,
                'active_parameter_billions': 0.123,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 0.24,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['text_encoding'],
                'sharding': 'whole_component',
                'offload': 'staged_destroy_after_prompt_encoding',
                'quantization_support': {'int8': 'usually_not_worthwhile', 'int4': 'avoid'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            },
            'transformer': {
                'role': 'video_dit',
                'architecture': 'kandinsky5_pro_i2v_transformer',
                'parameter_billions': 19.3,
                'active_parameter_billions': 19.3,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.9,
                'checkpoint_storage_gib': 71.9,
                'checkpoint_storage_basis': 'repository_float32_sized_payload_and_official_19b_family_description',
                'phases': ['video_denoising'],
                'sharding': '4_text_blocks_and_60_visual_blocks',
                'offload': 'staged_destroy_after_denoising',
                'quantization_support': {'int8': 'current_fast_path', 'int4': 'last_resort_only'},
                'skip_modules': [
                    'time_embeddings',
                    'text_embeddings',
                    'pooled_text_embeddings',
                    'visual_embeddings',
                    'out_layer',
                    'text_modulation',
                    'visual_modulation'
                ],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': ['Custom fast path maps visual blocks 0-28 to GPU0 and 29-59 to GPU1.']
            },
            'vae': {
                'role': 'vae',
                'architecture': 'hunyuan_video_autoencoder',
                'parameter_billions': 0.247,
                'active_parameter_billions': 0.247,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.02,
                'checkpoint_storage_gib': 0.46,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['source_encode', 'video_vae_decode'],
                'sharding': 'whole_component',
                'offload': 'staged_cpu_then_gpu1',
                'quantization_support': {'int8': 'disabled', 'int4': 'disabled'},
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'medium',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'source_image_encoding',
                'required_components': ['vision_encoder'],
                'releasable_after': ['vision_encoder'],
                'dynamic_memory_scales_with': ['input_image_size']
            },
            {
                'name': 'text_encoding',
                'required_components': ['qwen_text_encoder', 'clip_text_encoder'],
                'releasable_after': ['qwen_text_encoder', 'clip_text_encoder'],
                'output_can_move_to_cpu': True,
                'dynamic_memory_scales_with': ['prompt_tokens']
            },
            {
                'name': 'latent_or_source_encode',
                'required_components': ['vae'],
                'releasable_after': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            },
            {
                'name': 'video_denoising',
                'required_components': ['transformer'],
                'releasable_after': ['transformer'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames', 'guidance']
            },
            {
                'name': 'video_vae_decode',
                'required_components': ['vae'],
                'dynamic_memory_scales_with': ['width', 'height', 'num_frames']
            }
        ],
        'default_workload': {
            'width': 512,
            'height': 512,
            'num_frames': 121,
            'batch_size': 1,
            'inference_steps': 50,
            'prompt_tokens': 512
        },
        'runtime_memory': {
            'dominant_terms': [
                'denoiser_weights',
                'denoiser_activations',
                'attention_workspace',
                'latents',
                'vae_encode_decode_activations'
            ],
            'default_runtime_headroom_gib': 12.0,
            'workload_scaling': ['width', 'height', 'num_frames'],
            'weight_quantization_does_not_reduce': [
                'latents',
                'attention_workspace',
                'vae_activations',
                'most_non_linear_activations'
            ],
            'vae_peak_risk': 'very_high'
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'pipeline_dependent_and_unverified',
            'cpu': 'supported_but_task_may_be_impractical',
            'bitsandbytes_cuda': True,
            'comfyui': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_component_placement': True,
            'multi_gpu_block_sharding': True,
            'device_map': True,
            'model_cpu_offload': True,
            'sequential_cpu_offload': True,
            'custom_staging': True,
            'cpu_only': True
        },
        'quantization_policy': {
            'component_specific': True,
            'int8_auto_allowed': True,
            'int4_auto_allowed': True,
            'int4_default_position': 'after_int8_and_native_model_offload',
            'never_quantize_roles': ['vae', 'scheduler', 'tokenizer', 'processor'],
            'prefer_text_encoder_int4_over_denoiser_int4': True,
            'prefer_resident_int8_over_native_sequential_offload': True,
            'int4_target_order': ['qwen_text_encoder', 'transformer'],
            'video_transformer_int4_priority': 'last_resort'
        },
        'current_code': {
            'loader': 'Kandinsky5I2VGenerator',
            'plan': 'custom_staged_int8_text_vae_source_encode_int8_transformer_29_31_split_vae_gpu1_decode',
            'quantization': 'component_int8'
        },
        'existing_fast_path': {
            'plan_id': 'kandinsky5_i2v_pro_current_dual_3090_staged_int8',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'max_memory_gib': {0: 22, 1: 22, 'cpu': 48},
            'phase_order': [
                'load_text_encoders',
                'encode_prompts_to_cpu',
                'vae_source_encode_gpu1',
                'release_text_encoders',
                'load_int8_transformer',
                'denoise',
                'release_transformer',
                'vae_decode_gpu1'
            ],
            'transformer_block_map': {
                'cuda:0': [0, 28],
                'cuda:1': [29, 59]
            },
            'quantization': {
                'qwen_text_encoder': 'bnb_int8',
                'transformer': 'bnb_int8',
                'clip_text_encoder': 'bfloat16',
                'vae': 'bfloat16'
            },
            'attention_backend': 'flex',
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers/blob/main/config.json',
            'https://huggingface.co/kandinskylab/Kandinsky-5.0-I2V-Pro-sft-5s-Diffusers/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_policy_and_model_specific_evidence'
        },
        'notes': [],
        'application_source_files': ['model_gui.py', 'model_loading.py']
    },
    {
        'key': 'whisper_medium_voice',
        'model_id': 'openai/whisper-medium',
        'runtime_model_id': 'openai/whisper-medium',
        'launcher_id': None,
        'display_name': 'Whisper Medium Voice Transcription',
        'category': 'speech_to_text',
        'task': 'automatic_speech_recognition',
        'checkpoint': {'format': 'safetensors_or_pytorch', 'native_dtype': 'float32', 'already_quantized': False},
        'architecture': {
            'family': 'whisper_encoder_decoder_transformer',
            'parameter_billions': 0.769,
            'hidden_size': 1024,
            'intermediate_size': 4096,
            'encoder_layers': 24,
            'decoder_layers': 24,
            'attention_heads': 16,
            'max_source_positions': 1500,
            'max_target_positions': 448,
            'vocab_size': 51865
        },
        'components': {
            'speech_model': {
                'role': 'encoder_decoder_transformer',
                'architecture': 'whisper_medium',
                'parameter_billions': 0.769,
                'active_parameter_billions': 0.769,
                'native_dtype': 'float32',
                'quantizable_fraction': 0.88,
                'checkpoint_storage_gib': 2.86,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['audio_encoding', 'text_decoding'],
                'sharding': 'encoder_decoder_layer_device_map',
                'offload': 'whole_model_cpu_or_gpu',
                'quantization_support': {
                    'int8': 'supported_but_usually_unnecessary',
                    'int4': 'avoid_for_medium_speech_model'
                },
                'skip_modules': [],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'architecture_and_checkpoint_metadata',
                'notes': [
                    'The application loads this lazily for voice transcription and prefers GPU1 only when two CUDA devices are visible.'
                ]
            }
        },
        'execution_phases': [
            {
                'name': 'audio_encoding',
                'required_components': ['speech_model'],
                'dynamic_memory_scales_with': ['audio_duration']
            },
            {
                'name': 'text_decoding',
                'required_components': ['speech_model'],
                'dynamic_memory_scales_with': ['generated_tokens']
            }
        ],
        'default_workload': {'sample_rate': 16000, 'audio_seconds': 60, 'batch_size': 1},
        'runtime_memory': {
            'dominant_terms': ['weights', 'audio_encoder_activations', 'decoder_kv_cache'],
            'default_runtime_headroom_gib': 1.5
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers',
            'cpu': 'practical',
            'bitsandbytes_cuda': True
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'cpu_only': True,
            'model_cpu_offload': False,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low_to_medium_for_transcription',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'high'
        },
        'current_code': {
            'loader': 'transformers_pipeline_automatic_speech_recognition',
            'plan': 'gpu1_if_two_cuda_devices_else_cpu',
            'quantization': 'native'
        },
        'sources': [
            'https://huggingface.co/openai/whisper-medium',
            'https://huggingface.co/openai/whisper-medium/blob/main/config.json',
            'https://huggingface.co/openai/whisper-medium/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'component_weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'research_inference'
        },
        'notes': [],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_6_35b_a3b_fp8_runtime',
        'model_id': 'Qwen/Qwen3.6-35B-A3B-FP8',
        'runtime_model_id': 'Qwen/Qwen3.6-35B-A3B-FP8',
        'launcher_id': None,
        'display_name': 'Qwen3.6 35B A3B FP8 Runtime Artifact',
        'category': 'llm',
        'task': 'runtime_artifact',
        'checkpoint': {'format': 'native_fp8', 'native_dtype': 'bfloat16', 'already_quantized': True},
        'architecture': {
            'family': 'qwen3_5_moe_multimodal',
            'parameter_billions': 35.0,
            'active_parameter_billions': 3.0,
            'hidden_size': 2048,
            'intermediate_size': 5120,
            'num_hidden_layers': 40,
            'num_attention_heads': 16,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 248320,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_5_moe_multimodal',
                'parameter_billions': 35.0,
                'active_parameter_billions': 3.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'artifact_defined', 'int4': 'artifact_defined'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 8192, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': False,
            'vllm': True
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': False,
            'decoder_block_sharding': False,
            'cpu_overflow_device_map': False,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': False,
            'int4_quality_risk': 'medium',
            'int4_priority': 'not_applicable_to_fp8_runtime_artifact',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {'loader': 'vllm_async_engine', 'quantization': 'native_fp8', 'plan': 'tensor_parallel_2_gpu'},
        'existing_fast_path': {
            'plan_id': 'qwen3_6_35b_a3b_fp8_runtime_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 2, 'minimum_total_vram_gib_each': 24},
            'loader': 'vllm_async_engine',
            'placement': 'tensor_parallel_2_gpu',
            'quantization': {'language_model': 'native_fp8'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Underlying runtime artifact selected by chat_gui.py for the logical Qwen3.6 35B A3B launcher entry.'],
        'application_source_files': ['chat_gui.py']
    },
    {
        'key': 'qwen3_coder_next_source',
        'model_id': 'Qwen/Qwen3-Coder-Next',
        'runtime_model_id': 'Qwen/Qwen3-Coder-Next',
        'launcher_id': None,
        'display_name': 'Qwen3 Coder Next Source Configuration',
        'category': 'llm',
        'task': 'tokenizer_and_config_source',
        'checkpoint': {'format': 'safetensors', 'native_dtype': 'bfloat16', 'already_quantized': False},
        'architecture': {
            'family': 'qwen3_coder_next_moe',
            'parameter_billions': 80.0,
            'active_parameter_billions': 3.0,
            'hidden_size': 2048,
            'intermediate_size': 5120,
            'num_hidden_layers': 48,
            'num_attention_heads': 16,
            'num_key_value_heads': 2,
            'head_dim': 256,
            'max_context_tokens': 262144,
            'vocab_size': 151936,
            'uses_kv_cache': True,
            'cache_dtype': 'bfloat16'
        },
        'components': {
            'language_model': {
                'role': 'llm_decoder',
                'architecture': 'qwen3_coder_next_moe',
                'parameter_billions': 80.0,
                'active_parameter_billions': 3.0,
                'native_dtype': 'bfloat16',
                'quantizable_fraction': 0.93,
                'checkpoint_storage_gib': 0.0,
                'checkpoint_storage_basis': 'repository_file_inventory',
                'phases': ['prefill', 'decode'],
                'sharding': 'decoder_block_sharding_and_automatic_device_map',
                'offload': 'device_map_cpu_overflow_or_cpu_execution',
                'quantization_support': {'int8': 'supported_with_bitsandbytes', 'int4': 'supported_with_bitsandbytes'},
                'skip_modules': ['model.embed_tokens', 'lm_head', 'normalization_layers'],
                'memory_overrides_gib': {},
                'confidence': 'high',
                'evidence': 'published_configuration_and_parameter_class',
                'notes': []
            }
        },
        'execution_phases': [
            {
                'name': 'prefill',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'padded_prompt_tokens', 'attention_backend']
            },
            {
                'name': 'decode',
                'required_components': ['language_model'],
                'dynamic_memory_scales_with': ['batch_size', 'cached_tokens', 'generated_tokens']
            }
        ],
        'default_workload': {'batch_size': 1, 'prompt_tokens': 4096, 'max_new_tokens': 4096},
        'runtime_memory': {
            'dominant_terms': [
                'weights',
                'kv_cache',
                'prefill_mlp_activations',
                'attention_workspace',
                'quantization_temporaries'
            ],
            'kv_cache_formula_bytes': '2 * batch * cached_tokens * layers * kv_heads * head_dim * cache_dtype_bytes',
            'prefill_risk': 'high',
            'decode_risk': 'medium',
            'default_runtime_headroom_gib': 4.0
        },
        'backend_support': {
            'cuda': 'supported',
            'mps': 'supported_with_transformers_operator_and_memory_limits',
            'cpu': 'supported',
            'cpu_practicality': 'usable',
            'bitsandbytes_cuda': True,
            'vllm': False
        },
        'placement_support': {
            'single_gpu': True,
            'multi_gpu_device_map': True,
            'decoder_block_sharding': True,
            'cpu_overflow_device_map': True,
            'cpu_only': True,
            'sequential_cpu_offload': False
        },
        'quantization_policy': {
            'int8_auto_allowed': True,
            'int8_quality_risk': 'low',
            'int4_auto_allowed': True,
            'int4_quality_risk': 'medium',
            'int4_priority': 'prefer_native_gguf_application_artifact',
            'minimum_automatic_bits': 4,
            'reasoning_or_precision_sensitive': False,
            'keep_high_precision': ['embeddings', 'normalization', 'lm_head']
        },
        'current_code': {
            'loader': 'metadata_only_for_gguf_runtime',
            'quantization': 'native',
            'plan': 'not_loaded_by_application'
        },
        'existing_fast_path': {
            'plan_id': 'qwen3_coder_next_source_current_application_path',
            'hardware_match': {'backend': 'cuda', 'gpu_count': 1, 'minimum_total_vram_gib_each': 0},
            'loader': 'metadata_only_for_gguf_runtime',
            'placement': 'not_loaded_by_application',
            'quantization': {'language_model': 'native'},
            'max_memory_gib': {},
            'preserve_exact_loader': True
        },
        'sources': [
            'https://huggingface.co/Qwen/Qwen3-Coder-Next',
            'https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json',
            'https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main'
        ],
        'confidence': {
            'architecture': 'high',
            'weight_memory': 'high',
            'runtime_peak': 'estimated',
            'quantization_quality': 'policy_and_research_inference'
        },
        'notes': ['Used as tokenizer and Hugging Face configuration source for the Q4_K_M GGUF application model.'],
        'application_source_files': ['chat_gui.py']
    }
]


DTYPE_BYTES = {
    'float32': 4.0,
    'float16': 2.0,
    'bfloat16': 2.0,
    'bfloat16_or_component_specific': 2.0
}


def round_gib(value):
    return round(float(value), 3)


def native_bytes_per_parameter(component):
    return DTYPE_BYTES.get(component.get('native_dtype'), 2.0)


def component_memory_for_precision(profile, component, precision):
    parameters = float(component.get('parameter_billions', 0.0)) * 1_000_000_000
    quantizable_fraction = min(1.0, max(0.0, float(component.get('quantizable_fraction', 0.0))))
    native_bytes = native_bytes_per_parameter(component)
    overrides = component.get('memory_overrides_gib', {})

    if precision in overrides:
        return round_gib(overrides[precision])

    checkpoint_format = profile.get('checkpoint', {}).get('format')

    if precision in ['native', 'checkpoint_native']:
        if checkpoint_format == 'native_fp8':
            bytes_per_parameter = quantizable_fraction * 1.05 + (1.0 - quantizable_fraction) * native_bytes
        elif checkpoint_format == 'native_mxfp4':
            bytes_per_parameter = quantizable_fraction * 0.58 + (1.0 - quantizable_fraction) * native_bytes
        elif checkpoint_format == 'gguf_q4_k_m':
            bytes_per_parameter = quantizable_fraction * 0.62 + (1.0 - quantizable_fraction) * native_bytes
        elif checkpoint_format == 'native_bnb_nf4':
            bytes_per_parameter = quantizable_fraction * 0.56 + (1.0 - quantizable_fraction) * native_bytes
        else:
            bytes_per_parameter = native_bytes
    elif precision == 'int8':
        bytes_per_parameter = quantizable_fraction * 1.10 + (1.0 - quantizable_fraction) * native_bytes
    elif precision == 'int4':
        bytes_per_parameter = quantizable_fraction * 0.56 + (1.0 - quantizable_fraction) * native_bytes
    else:
        bytes_per_parameter = native_bytes

    return round_gib(parameters * bytes_per_parameter / (1024 ** 3))


def enrich_component_memory(profile, component):
    enriched = deepcopy(component)
    enriched['weight_memory_gib'] = {
        'native': component_memory_for_precision(profile, component, 'native'),
        'int8': component_memory_for_precision(profile, component, 'int8'),
        'int4': component_memory_for_precision(profile, component, 'int4'),
    }
    checkpoint_storage = float(component.get('checkpoint_storage_gib', 0.0))
    if checkpoint_storage:
        enriched['weight_memory_gib']['checkpoint_storage'] = round_gib(checkpoint_storage)
    return enriched


def support_allows(value, precision):
    text = str(value).lower()
    blocked = ['disabled', 'avoid', 'not_recommended', 'unnecessary', 'not_officially_recommended']
    if any(word in text for word in blocked):
        return False
    if precision == 'int4' and any(word in text for word in ['last_resort', 'high_caution', 'official', 'available']):
        return True
    return any(word in text for word in ['supported', 'recommended', 'current', 'native', 'acceptable', 'optional', 'fallback', 'artifact'])


def component_precision_map(profile, mode):
    precision_map = {}
    checkpoint_format = profile.get('checkpoint', {}).get('format')
    int4_targets = profile.get('quantization_policy', {}).get('int4_target_order', [])

    for component_name, component in profile['components'].items():
        role = component.get('role')
        support = component.get('quantization_support', {})

        if checkpoint_format in ['native_fp8', 'native_mxfp4', 'gguf_q4_k_m', 'native_bnb_nf4']:
            if mode in ['native', 'checkpoint_native']:
                precision_map[component_name] = 'checkpoint_native'
                continue

        if mode == 'native':
            precision_map[component_name] = 'native'
            continue

        if mode == 'int8':
            if role in ['vae', 'small_text_encoder', 'scheduler', 'tokenizer', 'processor']:
                precision_map[component_name] = 'native'
            elif support_allows(support.get('int8'), 'int8'):
                precision_map[component_name] = 'int8'
            else:
                precision_map[component_name] = 'native'
            continue

        if mode == 'int4':
            if role in ['vae', 'small_text_encoder', 'scheduler', 'tokenizer', 'processor']:
                precision_map[component_name] = 'native'
            elif int4_targets and component_name not in int4_targets and role not in ['llm_decoder']:
                if support_allows(support.get('int8'), 'int8'):
                    precision_map[component_name] = 'int8'
                else:
                    precision_map[component_name] = 'native'
            elif support_allows(support.get('int4'), 'int4'):
                precision_map[component_name] = 'int4'
            elif support_allows(support.get('int8'), 'int8'):
                precision_map[component_name] = 'int8'
            else:
                precision_map[component_name] = 'native'
            continue

        precision_map[component_name] = 'native'

    return precision_map


def weight_precision_key(precision):
    if precision in ['int8', '8bit', 'bnb_int8']:
        return 'int8'
    if precision in ['int4', '4bit', 'bnb_nf4', 'gguf_q4_k_m', 'native_mxfp4', 'checkpoint_native_nf4']:
        return 'int4'
    return 'native'


def component_map_weight_gib(profile, precision_map, component_names=None):
    names = component_names or list(profile['components'])
    total = 0.0
    for component_name in names:
        component = profile['components'].get(component_name)
        if not component:
            continue
        precision = weight_precision_key(precision_map.get(component_name, 'native'))
        total += component['weight_memory_gib'][precision]
    return round_gib(total)


def largest_component_weight_gib(profile, precision_map):
    values = []
    for component_name, component in profile['components'].items():
        precision = weight_precision_key(precision_map.get(component_name, 'native'))
        values.append(component['weight_memory_gib'][precision])
    return round_gib(max(values or [0.0]))


def workload_ratio(value, default):
    if not isinstance(value, (int, float)) or not isinstance(default, (int, float)) or default <= 0:
        return 1.0
    return max(0.1, float(value) / float(default))


def phase_workload_scale(profile, phase, workload=None):
    workload = workload or profile.get('default_workload', {})
    defaults = profile.get('default_workload', {})
    scales_with = set(phase.get('dynamic_memory_scales_with', []))
    scale = 1.0

    if 'width' in scales_with or 'height' in scales_with:
        width_ratio = workload_ratio(workload.get('width'), defaults.get('width'))
        height_ratio = workload_ratio(workload.get('height'), defaults.get('height'))
        scale *= width_ratio * height_ratio

    if 'num_frames' in scales_with:
        scale *= workload_ratio(workload.get('num_frames'), defaults.get('num_frames'))

    if 'prompt_tokens' in scales_with:
        scale *= workload_ratio(workload.get('prompt_tokens'), defaults.get('prompt_tokens'))

    for key in ['num_images_per_prompt', 'num_videos_per_prompt']:
        if key in workload or key in defaults:
            scale *= workload_ratio(workload.get(key, defaults.get(key, 1)), defaults.get(key, 1))

    return max(0.25, scale)


def calculate_phase_memory(profile, precision_map, workload=None):
    phase_memory = {}
    headroom = float(profile.get('runtime_memory', {}).get('default_runtime_headroom_gib', 2.0))

    for phase in profile.get('execution_phases', []):
        phase_name = phase['name']
        required = phase.get('required_components', [])
        static_weight = component_map_weight_gib(profile, precision_map, required)
        workload_scale = phase_workload_scale(profile, phase, workload)
        scaled_headroom = max(0.75, headroom * workload_scale)
        phase_memory[phase_name] = {
            'static_weight_gib': static_weight,
            'estimated_runtime_headroom_gib': round_gib(scaled_headroom),
            'estimated_peak_gib': round_gib(static_weight + scaled_headroom),
            'workload_scale': round(workload_scale, 3),
            'scales_with': phase.get('dynamic_memory_scales_with', []),
        }

    return phase_memory


def memory_summary(profile):
    result = {}
    for mode in ['native', 'int8', 'int4']:
        precision_map = component_precision_map(profile, mode)
        phase_memory = calculate_phase_memory(profile, precision_map)
        result[mode] = {
            'component_precision': precision_map,
            'fully_resident_weight_gib': component_map_weight_gib(profile, precision_map),
            'largest_component_weight_gib': largest_component_weight_gib(profile, precision_map),
            'phases': phase_memory,
            'largest_estimated_phase_peak_gib': round_gib(max([value['estimated_peak_gib'] for value in phase_memory.values()] or [0.0])),
        }
    return result


def plan_system_ram(profile, precision_map, placement):
    weight_gib = component_map_weight_gib(profile, precision_map)
    phase_memory = calculate_phase_memory(profile, precision_map)
    largest_phase_weight = max([value['static_weight_gib'] for value in phase_memory.values()] or [weight_gib])
    base_reserve = float(GENERAL_POLICY['default_system_ram_reserve_gib'])
    if placement in ['single_gpu_resident', 'multi_gpu_device_map', 'vllm_tensor_parallel']:
        minimum = max(8.0, min(24.0, weight_gib * 0.25 + base_reserve))
        recommended = max(16.0, min(48.0, weight_gib * 0.5 + base_reserve))
    elif placement in ['single_gpu_model_specific_staging', 'multi_gpu_model_specific_staging', 'mps_model_specific_staging']:
        minimum = max(4.0, largest_phase_weight + 2.0)
        recommended = max(8.0, largest_phase_weight + 6.0)
    elif placement in ['device_map_with_cpu_overflow', 'diffusers_model_cpu_offload', 'diffusers_sequential_cpu_offload']:
        minimum = max(16.0, weight_gib + base_reserve)
        recommended = max(24.0, weight_gib * 1.35 + base_reserve)
    elif placement == 'cpu_only':
        minimum = max(16.0, weight_gib * 1.2 + base_reserve)
        recommended = max(24.0, weight_gib * 1.5 + base_reserve)
    elif placement == 'mps_resident':
        minimum = weight_gib
        recommended = weight_gib
    else:
        minimum = max(8.0, weight_gib * 0.5 + base_reserve)
        recommended = max(16.0, weight_gib + base_reserve)
    return round_gib(minimum), round_gib(recommended)


def candidate_memory_estimate(profile, precision_map, placement, workload=None):
    phase_memory = calculate_phase_memory(profile, precision_map, workload)
    resident_weight = component_map_weight_gib(profile, precision_map)
    largest_component = largest_component_weight_gib(profile, precision_map)
    phase_peak = round_gib(max([value['estimated_peak_gib'] for value in phase_memory.values()] or [0.0]))
    runtime_headroom = max([value['estimated_runtime_headroom_gib'] for value in phase_memory.values()] or [float(profile.get('runtime_memory', {}).get('default_runtime_headroom_gib', 2.0))])

    if placement in ['single_gpu_resident', 'mps_resident']:
        required_total_vram = resident_weight + runtime_headroom
        largest_gpu = required_total_vram
    elif placement == 'multi_gpu_device_map':
        required_total_vram = resident_weight + runtime_headroom
        largest_gpu = max(runtime_headroom + 2.0, largest_component * 0.35)
    elif placement == 'device_map_with_cpu_overflow':
        required_total_vram = max(runtime_headroom + 3.0, largest_component * 0.35)
        largest_gpu = required_total_vram
    elif placement in ['single_gpu_model_specific_staging', 'mps_model_specific_staging']:
        required_total_vram = phase_peak
        largest_gpu = phase_peak
    elif placement == 'multi_gpu_model_specific_staging':
        required_total_vram = phase_peak
        largest_gpu = max(runtime_headroom + 3.0, largest_component * 0.35)
    elif placement == 'diffusers_model_cpu_offload':
        required_total_vram = largest_component + runtime_headroom
        largest_gpu = required_total_vram
    elif placement == 'diffusers_sequential_cpu_offload':
        required_total_vram = runtime_headroom + 3.0
        largest_gpu = required_total_vram
    elif placement == 'cpu_only':
        required_total_vram = 0.0
        largest_gpu = 0.0
    elif placement == 'vllm_tensor_parallel':
        required_total_vram = resident_weight + runtime_headroom
        largest_gpu = max(runtime_headroom + 2.0, required_total_vram / 2.0)
    elif placement in ['mps_resident_or_unified_memory', 'comfyui_managed']:
        required_total_vram = resident_weight + runtime_headroom
        largest_gpu = required_total_vram
    else:
        required_total_vram = phase_peak
        largest_gpu = phase_peak

    minimum_ram, recommended_ram = plan_system_ram(profile, precision_map, placement)
    return {
        'fully_resident_weight_gib': resident_weight,
        'largest_component_weight_gib': largest_component,
        'required_total_usable_vram_gib': round_gib(required_total_vram),
        'minimum_largest_gpu_usable_vram_gib': round_gib(largest_gpu),
        'minimum_system_ram_gib': minimum_ram,
        'recommended_system_ram_gib': recommended_ram,
        'phases': phase_memory,
        'basis': 'component_weight_inventory_plus_phase_and_workload_runtime_estimate',
        'runtime_guarantee': False,
    }


def make_candidate_plan(profile, template_id, precision_mode, plan_id=None, notes=None, precision_map=None):
    template = deepcopy(PLAN_TEMPLATES[template_id])
    precision_map = deepcopy(precision_map) if precision_map is not None else component_precision_map(profile, precision_mode)
    placement = template['placement']
    plan = template
    plan['plan_id'] = plan_id or f"{profile['key']}__{template_id}"
    plan['template_id'] = template_id
    plan['component_precision'] = precision_map
    plan['estimated_memory'] = candidate_memory_estimate(profile, precision_map, placement)
    plan['requirements'] = {
        'backend': 'cuda' if template.get('minimum_gpu_count', 0) else 'cpu_or_accelerator',
        'minimum_gpu_count': template.get('minimum_gpu_count', 0),
        'supports_unequal_gpu_capacities': placement in ['multi_gpu_device_map', 'multi_gpu_model_specific_staging'],
        'requires_system_ram_for_weights': placement in ['device_map_with_cpu_overflow', 'single_gpu_model_specific_staging', 'multi_gpu_model_specific_staging', 'diffusers_model_cpu_offload', 'diffusers_sequential_cpu_offload', 'cpu_only'],
    }
    plan['notes'] = notes or []
    return plan

def existing_fast_path_plan(profile):
    fast_path = profile.get('existing_fast_path', {})
    if not fast_path:
        return None
    plan = deepcopy(PLAN_TEMPLATES['current_exact_fast_path'])
    plan['plan_id'] = fast_path['plan_id']
    plan['template_id'] = 'current_exact_fast_path'
    plan['component_precision'] = fast_path.get('quantization', component_precision_map(profile, 'native'))
    plan['hardware_match'] = deepcopy(fast_path.get('hardware_match', {}))
    plan['loader_details'] = deepcopy(fast_path)
    plan['estimated_memory'] = {
        'basis': 'existing_application_path_preserved_exactly',
        'runtime_guarantee': False,
    }
    plan['requirements'] = {
        'backend': fast_path.get('hardware_match', {}).get('backend', 'cuda'),
        'minimum_gpu_count': fast_path.get('hardware_match', {}).get('gpu_count', 1),
    }
    plan['notes'] = ['Select before portable alternatives when the exact hardware match succeeds.']
    return plan


def append_llm_resident_plans(plans, profile, precision):
    plans.append(make_candidate_plan(profile, f'resident_{precision}_single', precision))
    if profile.get('placement_support', {}).get('multi_gpu_device_map'):
        plans.append(make_candidate_plan(profile, f'resident_{precision}_multi', precision))


def append_llm_cpu_overflow_plan(plans, profile, precision):
    if profile.get('placement_support', {}).get('cpu_overflow_device_map'):
        plans.append(make_candidate_plan(profile, f'{precision}_cpu_overflow', precision))


def build_llm_plans(profile):
    plans = []
    fast_path = existing_fast_path_plan(profile)
    if fast_path:
        plans.append(fast_path)

    checkpoint_format = profile.get('checkpoint', {}).get('format')
    current_loader = profile.get('current_code', {}).get('loader')
    quantization_policy = profile.get('quantization_policy', {})
    int4_allowed = quantization_policy.get('int4_auto_allowed', False)
    int4_priority = quantization_policy.get('int4_priority', '')
    int4_before_cpu_overflow = int4_priority in [
        'after_int8_resident_and_before_sequential_offload',
        'application_current_default_due_large_multimodal_checkpoint',
    ]

    if current_loader == 'vllm_async_engine':
        plans.append(make_candidate_plan(profile, 'vllm_native', 'native', f"{profile['key']}__vllm_native"))

    append_llm_resident_plans(plans, profile, 'native')

    if checkpoint_format == 'safetensors':
        append_llm_resident_plans(plans, profile, 'int8')

        if int4_allowed and int4_before_cpu_overflow:
            append_llm_resident_plans(plans, profile, 'int4')

        append_llm_cpu_overflow_plan(plans, profile, 'int8')

        if int4_allowed and not int4_before_cpu_overflow:
            append_llm_resident_plans(plans, profile, 'int4')

        if int4_allowed:
            append_llm_cpu_overflow_plan(plans, profile, 'int4')

    if profile.get('backend_support', {}).get('mps', '').startswith('supported'):
        plans.append(make_candidate_plan(profile, 'mps_native', 'native'))

    cpu_practicality = profile.get('backend_support', {}).get('cpu_practicality')
    if cpu_practicality in ['practical', 'usable', 'emergency']:
        plans.append(make_candidate_plan(profile, 'cpu_native', 'native'))
        if checkpoint_format == 'safetensors' and float(profile['architecture'].get('parameter_billions', 0.0)) >= 7.0:
            plans.append(make_candidate_plan(profile, 'cpu_int8', 'int8'))

    return plans


def progressive_int4_precision_maps(profile):
    target_order = profile.get('quantization_policy', {}).get('int4_target_order', [])
    if not target_order:
        return [component_precision_map(profile, 'int4')]

    base_map = component_precision_map(profile, 'int8')
    result = []
    for target_count in range(1, len(target_order) + 1):
        precision_map = deepcopy(base_map)
        for component_name in target_order[:target_count]:
            component = profile.get('components', {}).get(component_name)
            if component and support_allows(component.get('quantization_support', {}).get('int4'), 'int4'):
                precision_map[component_name] = 'int4'
        if precision_map not in result:
            result.append(precision_map)

    return result or [component_precision_map(profile, 'int4')]


def append_progressive_int4_plans(plans, profile, template_id):
    precision_maps = progressive_int4_precision_maps(profile)
    for index, precision_map in enumerate(precision_maps, start=1):
        plan_id = f"{profile['key']}__{template_id}__step_{index}"
        targets = [
            name
            for name, precision in precision_map.items()
            if precision == 'int4'
        ]
        plans.append(make_candidate_plan(
            profile,
            template_id,
            'int4',
            plan_id=plan_id,
            notes=[f"Progressive INT4 targets: {', '.join(targets)}"],
            precision_map=precision_map,
        ))


def build_diffusion_plans(profile):
    plans = []
    fast_path = existing_fast_path_plan(profile)
    if fast_path:
        plans.append(fast_path)

    placement = profile.get('placement_support', {})
    checkpoint = profile.get('checkpoint', {})
    checkpoint_format = checkpoint.get('format')
    already_quantized = checkpoint.get('already_quantized', False)
    int8_allowed = profile.get('quantization_policy', {}).get('int8_auto_allowed', True) and not already_quantized
    int4_allowed = profile.get('quantization_policy', {}).get('int4_auto_allowed', False) and not already_quantized
    multi_gpu = placement.get('multi_gpu_component_placement') or placement.get('multi_gpu_block_sharding')

    if checkpoint_format == 'comfy_checkpoint_native':
        plans.append(make_candidate_plan(profile, 'comfy_native', 'native'))
        return plans

    plans.append(make_candidate_plan(profile, 'resident_native_single', 'native'))
    if multi_gpu:
        plans.append(make_candidate_plan(profile, 'resident_native_multi', 'native'))

    if placement.get('custom_staging'):
        plans.append(make_candidate_plan(profile, 'staged_native_single', 'native'))
        if multi_gpu:
            plans.append(make_candidate_plan(profile, 'staged_native_multi', 'native'))

    if int8_allowed:
        plans.append(make_candidate_plan(profile, 'resident_int8_single', 'int8'))
        if multi_gpu:
            plans.append(make_candidate_plan(profile, 'resident_int8_multi', 'int8'))
        if placement.get('custom_staging'):
            plans.append(make_candidate_plan(profile, 'staged_int8_single', 'int8'))
            if multi_gpu:
                plans.append(make_candidate_plan(profile, 'staged_int8_multi', 'int8'))

    if placement.get('model_cpu_offload'):
        plans.append(make_candidate_plan(profile, 'model_cpu_offload_native', 'native'))
        if int8_allowed:
            plans.append(make_candidate_plan(profile, 'model_cpu_offload_int8', 'int8'))

    if placement.get('device_map'):
        plans.append(make_candidate_plan(profile, 'balanced_native_cpu_overflow', 'native'))
        if int8_allowed:
            plans.append(make_candidate_plan(profile, 'int8_cpu_overflow', 'int8'))

    if int4_allowed:
        append_progressive_int4_plans(plans, profile, 'resident_int4_single')
        if multi_gpu:
            append_progressive_int4_plans(plans, profile, 'resident_int4_multi')
        if placement.get('custom_staging'):
            append_progressive_int4_plans(plans, profile, 'staged_int4_single')
            if multi_gpu:
                append_progressive_int4_plans(plans, profile, 'staged_int4_multi')
        if placement.get('model_cpu_offload'):
            append_progressive_int4_plans(plans, profile, 'model_cpu_offload_int4')
        if placement.get('device_map'):
            append_progressive_int4_plans(plans, profile, 'int4_cpu_overflow')

    if placement.get('sequential_cpu_offload'):
        plans.append(make_candidate_plan(profile, 'sequential_cpu_offload_native', 'native'))
        if int8_allowed:
            plans.append(make_candidate_plan(profile, 'sequential_cpu_offload_int8', 'int8'))

    cpu_support = profile.get('backend_support', {}).get('cpu', '')
    if 'supported' in cpu_support or 'practical' in cpu_support:
        plans.append(make_candidate_plan(profile, 'cpu_native', 'native'))

    return plans


def build_candidate_plans(profile):
    if profile['category'] in ['llm', 'multimodal_llm']:
        return build_llm_plans(profile)
    return build_diffusion_plans(profile)


def apply_workload_policy(profile):
    workload = deepcopy(profile.get('default_workload', {}))
    category = profile.get('category')

    if category == 'image_diffusion':
        if 'num_images_per_prompt' not in workload:
            workload['num_images_per_prompt'] = max(1, int(workload.get('batch_size', 1)))
        workload['batch_size'] = 1
        profile['workload_policy'] = {
            'fixed': ['width', 'height'],
            'adjustable': {
                'num_images_per_prompt': {'minimum': 1},
            },
        }
    elif category == 'image_diffusion_upscale':
        profile['workload_policy'] = {
            'fixed': ['input_width', 'input_height', 'scale'],
            'adjustable': {},
        }
    elif category == 'video_diffusion':
        profile['workload_policy'] = {
            'fixed': ['width', 'height'],
            'adjustable': {
                'num_videos_per_prompt': {'minimum': 1},
            },
        }

    profile['default_workload'] = workload


def enrich_profile(raw_profile):
    profile = deepcopy(raw_profile)
    apply_workload_policy(profile)
    profile['components'] = {
        name: enrich_component_memory(profile, component)
        for name, component in profile['components'].items()
    }
    profile['memory_summary'] = memory_summary(profile)
    profile['candidate_plans'] = build_candidate_plans(profile)
    profile['cannot_run_policy'] = {
        'allowed': True,
        'conditions': [
            'no_candidate_supports_the_detected_backend',
            'system_ram_is_below_every_candidate_minimum',
            'all_supported_candidates_have_failed_for_this_hardware_and_workload',
            'only_remaining_cpu_plan_is_marked_impractical_for_the_task',
            'smallest_supported_precision_is_four_bits_and_it_still_cannot_fit',
        ],
        'message_must_include': ['model_name', 'hardware_summary', 'last_failed_phase', 'smallest_supported_plan'],
    }
    return profile


def build_model_profiles():
    return {profile['key']: enrich_profile(profile) for profile in RAW_MODEL_PROFILES}


def build_model_aliases(profiles):
    aliases = {}
    for key, profile in profiles.items():
        aliases[key] = key
        model_id = profile.get('model_id')
        runtime_model_id = profile.get('runtime_model_id')
        launcher_id = profile.get('launcher_id')
        for alias in [model_id, runtime_model_id, launcher_id]:
            if alias:
                aliases[alias] = key
    return aliases


def resolve_model_key(model_reference):
    return MODEL_ALIASES.get(model_reference)


def get_model_profile(model_reference):
    key = resolve_model_key(model_reference)
    if key is None:
        return None
    return deepcopy(MODEL_PROFILES[key])


def get_candidate_plans(model_reference):
    profile = get_model_profile(model_reference)
    if profile is None:
        return []
    return profile['candidate_plans']


def get_launcher_profiles():
    profiles = {}
    for launcher_id in LAUNCHER_MODEL_IDS:
        key = resolve_model_key(launcher_id)
        if key:
            profiles[launcher_id] = MODEL_PROFILES[key]
    return profiles


def validate_registry():
    errors = []
    if len(MODEL_PROFILES) != len(RAW_MODEL_PROFILES):
        errors.append('Duplicate model profile keys exist.')

    for key, profile in MODEL_PROFILES.items():
        required = ['key', 'model_id', 'display_name', 'category', 'components', 'execution_phases', 'backend_support', 'placement_support', 'quantization_policy', 'memory_summary', 'candidate_plans', 'sources']
        for field in required:
            if field not in profile:
                errors.append(f'{key}: missing {field}')
        if not profile.get('components'):
            errors.append(f'{key}: no components')
        if not profile.get('candidate_plans'):
            errors.append(f'{key}: no candidate plans')
        for component_name, component in profile.get('components', {}).items():
            if float(component.get('parameter_billions', 0.0)) <= 0:
                errors.append(f'{key}.{component_name}: invalid parameter estimate')
            for precision in ['native', 'int8', 'int4']:
                if component.get('weight_memory_gib', {}).get(precision, 0.0) <= 0:
                    errors.append(f'{key}.{component_name}: invalid {precision} memory estimate')
        plan_ids = set()
        for plan in profile.get('candidate_plans', []):
            plan_id = plan.get('plan_id')
            if not plan_id:
                errors.append(f'{key}: plan missing plan_id')
            elif plan_id in plan_ids:
                errors.append(f'{key}: duplicate plan_id {plan_id}')
            else:
                plan_ids.add(plan_id)

            for component_name, precision in plan.get('component_precision', {}).items():
                if component_name not in profile.get('components', {}):
                    errors.append(f'{key}.{plan_id}: unknown component {component_name}')
                    continue
                role = profile['components'][component_name].get('role')
                normalized = precision
                if precision in ['int8', '8bit', 'bnb_int8', 'native_fp8']:
                    normalized = 'int8'
                elif precision in ['int4', '4bit', 'bnb_nf4', 'gguf_q4_k_m', 'native_mxfp4', 'checkpoint_native_nf4']:
                    normalized = 'int4'
                elif precision in ['checkpoint_native', 'bfloat16', 'float16', 'float32']:
                    normalized = 'native'
                if normalized not in ['native', 'int8', 'int4']:
                    errors.append(f'{key}.{plan_id}.{component_name}: unknown precision {precision}')
                if role == 'vae' and normalized in ['int8', 'int4']:
                    errors.append(f'{key}.{plan_id}.{component_name}: VAE quantization is not permitted')

    for launcher_id in LAUNCHER_MODEL_IDS:
        if launcher_id not in MODEL_ALIASES:
            errors.append(f'Launcher model has no profile: {launcher_id}')

    return errors


MODEL_PROFILES = build_model_profiles()
MODEL_ALIASES = build_model_aliases(MODEL_PROFILES)
VALIDATION_ERRORS = validate_registry()
