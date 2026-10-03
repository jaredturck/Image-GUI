import gc
import importlib
import json
import os
import sys

import torch
from copy import deepcopy

from model_registry import get_model_profile
from planner_protocol import exit_for_fatal_failure, exit_for_memory_retry, is_memory_error, report_success

_ACTIVE_PLAN = None


class ConfigProxy:
    def __init__(self, values=None):
        self.values = values or {}

    def __getattr__(self, name):
        if name not in self.values:
            raise AttributeError(name)
        value = self.values[name]
        if isinstance(value, dict):
            return ConfigProxy(value)
        return value

    def __getitem__(self, name):
        value = self.values[name]
        if isinstance(value, dict):
            return ConfigProxy(value)
        return value

    def get(self, name, default=None):
        value = self.values.get(name, default)
        if isinstance(value, dict):
            return ConfigProxy(value)
        return value

    def to_dict(self):
        return deepcopy(self.values)


class StagedComponentManager:
    def __init__(self, model_id, plan, profile):
        self.model_id = model_id
        self.plan = plan
        self.profile = profile
        self.proxies = {}
        self.current_phase_index = -1

    def add_proxy(self, registry_name, proxy):
        self.proxies[registry_name] = proxy

    def component_phase_index(self, registry_name):
        phases = self.profile.get("execution_phases", [])
        for index in range(max(0, self.current_phase_index), len(phases)):
            if registry_name in phases[index].get("required_components", []):
                return index
        for index, phase in enumerate(phases):
            if registry_name in phase.get("required_components", []):
                return index
        return self.current_phase_index

    def activate(self, registry_name):
        phase_index = self.component_phase_index(registry_name)
        if phase_index != self.current_phase_index:
            self.release_for_phase(phase_index)
            self.current_phase_index = phase_index
            phases = self.profile.get("execution_phases", [])
            if 0 <= phase_index < len(phases):
                set_runtime_phase(self.plan, phases[phase_index].get("name", "inference"))
        return self.proxies[registry_name].load_real_component()

    def release_for_phase(self, phase_index):
        phases = self.profile.get("execution_phases", [])
        required = set()
        if 0 <= phase_index < len(phases):
            required = set(phases[phase_index].get("required_components", []))
        released = False
        for registry_name, proxy in self.proxies.items():
            if registry_name not in required and proxy.real_component is not None:
                proxy.release()
                released = True
        if released:
            clear_accelerator_cache()

    def release_all(self):
        released = False
        for proxy in self.proxies.values():
            if proxy.real_component is not None:
                proxy.release()
                released = True
        self.current_phase_index = -1
        if released:
            clear_accelerator_cache()


class LazyStagedComponent(torch.nn.Module):
    def __init__(self, manager, registry_name, pipeline_name, component_class, model_id, config_values, shardable=False):
        super().__init__()
        object.__setattr__(self, "manager", manager)
        object.__setattr__(self, "registry_name", registry_name)
        object.__setattr__(self, "pipeline_name", pipeline_name)
        object.__setattr__(self, "component_class", component_class)
        object.__setattr__(self, "__module__", component_class.__module__)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "config_proxy", ConfigProxy(config_values))
        object.__setattr__(self, "shardable", shardable)
        object.__setattr__(self, "real_component", None)
        object.__setattr__(self, "deferred_calls", [])

    @property
    def __class__(self):
        return object.__getattribute__(self, "component_class")

    @property
    def config(self):
        return object.__getattribute__(self, "config_proxy")

    @property
    def dtype(self):
        real_component = object.__getattribute__(self, "real_component")
        if real_component is not None:
            return real_component.dtype
        return component_torch_dtype(object.__getattribute__(self, "registry_name"), object.__getattribute__(self, "manager").plan)

    @property
    def device(self):
        real_component = object.__getattribute__(self, "real_component")
        if real_component is not None and hasattr(real_component, "device"):
            return real_component.device
        manager = object.__getattribute__(self, "manager")
        return torch.device(component_target_device(object.__getattribute__(self, "registry_name"), manager.plan))

    def load_real_component(self):
        real_component = object.__getattribute__(self, "real_component")
        if real_component is not None:
            return real_component

        manager = object.__getattribute__(self, "manager")
        registry_name = object.__getattribute__(self, "registry_name")
        component_class = object.__getattribute__(self, "component_class")
        pipeline_name = object.__getattribute__(self, "pipeline_name")
        real_component = load_component(
            component_class,
            object.__getattribute__(self, "model_id"),
            registry_name,
            portable_base_kwargs={"subfolder": pipeline_name},
            shardable=object.__getattribute__(self, "shardable"),
            plan=manager.plan,
        )
        object.__setattr__(self, "real_component", real_component)
        for name, args, kwargs in object.__getattribute__(self, "deferred_calls"):
            method = getattr(real_component, name, None)
            if method is not None:
                method(*args, **kwargs)
        return real_component

    def release(self):
        object.__setattr__(self, "real_component", None)

    def defer_call(self, name, *args, **kwargs):
        object.__getattribute__(self, "deferred_calls").append((name, args, kwargs))
        return self

    def enable_tiling(self, *args, **kwargs):
        return self.defer_call("enable_tiling", *args, **kwargs)

    def enable_slicing(self, *args, **kwargs):
        return self.defer_call("enable_slicing", *args, **kwargs)

    def set_attention_backend(self, *args, **kwargs):
        return self.defer_call("set_attention_backend", *args, **kwargs)

    def encode(self, *args, **kwargs):
        component = object.__getattribute__(self, "manager").activate(object.__getattribute__(self, "registry_name"))
        return component.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        component = object.__getattribute__(self, "manager").activate(object.__getattribute__(self, "registry_name"))
        return component.decode(*args, **kwargs)

    def forward(self, *args, **kwargs):
        component = object.__getattribute__(self, "manager").activate(object.__getattribute__(self, "registry_name"))
        return component(*args, **kwargs)

    def __getattr__(self, name):
        modules = object.__getattribute__(self, "_modules")
        parameters = object.__getattribute__(self, "_parameters")
        buffers = object.__getattribute__(self, "_buffers")
        if name in modules:
            return modules[name]
        if name in parameters:
            return parameters[name]
        if name in buffers:
            return buffers[name]
        if name.startswith("_"):
            raise AttributeError(name)
        config_proxy = object.__getattribute__(self, "config_proxy")
        if name in config_proxy.values:
            return getattr(config_proxy, name)
        component = object.__getattribute__(self, "manager").activate(object.__getattribute__(self, "registry_name"))
        return getattr(component, name)


def flag_value(flag):
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


def load_active_plan():
    global _ACTIVE_PLAN
    if _ACTIVE_PLAN is not None:
        return _ACTIVE_PLAN

    path = flag_value("--plan-file") or os.environ.get("AI_WORKSTATION_PLAN_FILE", "")
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as file:
            _ACTIVE_PLAN = json.load(file)
    else:
        _ACTIVE_PLAN = {}
    return _ACTIVE_PLAN


def get_active_plan():
    return load_active_plan()


def is_exact_fast_path(plan=None):
    plan = plan or get_active_plan()
    if not plan:
        return True
    return plan.get("template_id") == "current_exact_fast_path" or bool(plan.get("preserve_exact_loader"))


def plan_placement(plan=None):
    return (plan or get_active_plan()).get("placement", "existing_code_path")


def torch_dtype(plan=None):
    import torch

    name = (plan or get_active_plan()).get("torch_dtype", "bfloat16")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def execution_device(plan=None):
    return (plan or get_active_plan()).get("execution_device", "cuda:0")


def secondary_device(plan=None):
    return (plan or get_active_plan()).get("secondary_device", execution_device(plan))


def preview_device(plan=None):
    return (plan or get_active_plan()).get("preview_device", secondary_device(plan))


def logical_gpu_count(plan=None):
    return int((plan or get_active_plan()).get("logical_gpu_count", 0))


def plan_max_memory(plan=None):
    plan = plan or get_active_plan()
    result = {}
    for key, value in plan.get("max_memory", {}).items():
        if str(key).isdigit():
            result[int(key)] = value
        else:
            result[key] = value
    return result


def normalize_precision(value):
    if value in ["int8", "8bit", "bnb_int8", "native_fp8"]:
        return "int8"
    if value in ["int4", "4bit", "bnb_nf4", "native_mxfp4", "checkpoint_native_nf4"]:
        return "int4"
    if value in ["checkpoint_native", "bfloat16", "float16", "float32"]:
        return "native"
    return value or "native"


def component_precision(component_name, plan=None):
    plan = plan or get_active_plan()
    values = plan.get("component_precision", {})
    if component_name in values:
        return normalize_precision(values[component_name])

    aliases = component_aliases(component_name)
    for alias in aliases:
        if alias in values:
            return normalize_precision(values[alias])
    return "native"


def component_aliases(component_name):
    aliases = {component_name}
    low = component_name.lower()

    if "transformer" in low or low in ["denoiser", "language_model"]:
        aliases.update(["transformer", "language_model", "denoiser"])
    if "unet" in low:
        aliases.add("unet")
    if "vae" in low or "autoencoder" in low:
        aliases.add("vae")
    if "text_encoder_2" in low or "clip_text" in low:
        aliases.update(["text_encoder_2", "clip_text_encoder"])
    elif "text" in low or "conditioner" in low:
        aliases.update(["text_encoder", "qwen_text_encoder", "large_text_encoder"])
    if "vision" in low or "image_encoder" in low:
        aliases.update(["vision_encoder", "image_encoder", "vision_language_encoder"])
    return aliases


def profile_for_plan(plan=None):
    plan = plan or get_active_plan()
    reference = plan.get("model_key") or plan.get("model_reference")
    if not reference:
        return None
    return get_model_profile(reference)


def component_metadata(component_name, plan=None):
    plan = plan or get_active_plan()
    profile = profile_for_plan(plan)
    if not profile:
        return {}

    for alias in component_aliases(component_name):
        component = profile.get("components", {}).get(alias)
        if component:
            return component
    return {}


def component_role(component_name, plan=None):
    return component_metadata(component_name, plan).get("role", "")


def component_torch_dtype(component_name, plan=None):
    import torch

    plan = plan or get_active_plan()
    if component_precision(component_name, plan) != "native":
        return torch_dtype(plan)

    native_dtype = component_metadata(component_name, plan).get("native_dtype")
    if native_dtype == "float32":
        return torch.float32
    if native_dtype == "float16":
        return torch.float16
    return torch_dtype(plan)


def component_skip_modules(component_name, plan=None, extra=None):
    plan = plan or get_active_plan()
    profile = profile_for_plan(plan)
    result = list(extra or [])
    if not profile:
        return result

    for alias in component_aliases(component_name):
        component = profile.get("components", {}).get(alias)
        if component:
            result.extend(component.get("skip_modules", []))
    return list(dict.fromkeys(result))


def transformers_quantization_config(component_name, plan=None, skip_modules=None):
    precision = component_precision(component_name, plan)
    if precision == "native":
        return None

    import torch
    from transformers import BitsAndBytesConfig

    skip = component_skip_modules(component_name, plan, skip_modules)
    if precision == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=skip or None,
            llm_int8_enable_fp32_cpu_offload=plan_placement(plan) == "device_map_with_cpu_overflow",
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype(plan),
        llm_int8_skip_modules=skip or None,
    )


def diffusers_quantization_config(component_name, plan=None, skip_modules=None):
    precision = component_precision(component_name, plan)
    if precision == "native":
        return None

    from diffusers import BitsAndBytesConfig

    skip = component_skip_modules(component_name, plan, skip_modules)
    if precision == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=skip or None,
            llm_int8_enable_fp32_cpu_offload=plan_placement(plan) == "device_map_with_cpu_overflow",
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype(plan),
        llm_int8_skip_modules=skip or None,
    )


def pipeline_component_name(registry_component):
    low = registry_component.lower()
    if registry_component in ["transformer", "unet", "vae", "text_encoder", "text_encoder_2", "image_encoder", "vision_language_encoder"]:
        return registry_component
    if "transformer" in low or low in ["denoiser", "language_model"]:
        return "transformer"
    if "unet" in low:
        return "unet"
    if "vae" in low or "autoencoder" in low:
        return "vae"
    if "clip_text" in low or "text_encoder_2" in low:
        return "text_encoder_2"
    if "text" in low or "conditioner" in low:
        return "text_encoder"
    if "vision_language" in low:
        return "vision_language_encoder"
    if "vision" in low or "image_encoder" in low:
        return "image_encoder"
    return registry_component


def pipeline_component_specs(pipeline_class, model_id):
    config = pipeline_class.load_config(model_id)
    result = {}
    for name, value in config.items():
        if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] and value[1]:
            result[name] = value
    return result


def resolve_component_class(spec):
    library_name, class_name = spec
    module = importlib.import_module(library_name)
    return getattr(module, class_name)


def component_config_values(model_id, pipeline_name):
    if os.path.isdir(model_id):
        path = os.path.join(model_id, pipeline_name, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, os.path.join(pipeline_name, "config.json"))
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def text_pipeline_names(specs):
    names = [name for name in specs if name == "text_encoder" or name.startswith("text_encoder_")]
    return sorted(names, key=lambda name: 1 if name == "text_encoder" else int(name.rsplit("_", 1)[-1]))


def registry_pipeline_component_map(profile, specs):
    mapping = {}
    used = set()
    components = profile.get("components", {})

    for registry_name in components:
        if registry_name in specs:
            mapping[registry_name] = registry_name
            used.add(registry_name)

    text_registry = []
    for registry_name, component in components.items():
        if registry_name in mapping:
            continue
        role = component.get("role", "")
        if role in ["large_text_encoder", "small_text_encoder"] or "text_encoder" in registry_name or registry_name in ["clip_l", "clip_g", "t5_encoder"]:
            text_registry.append(registry_name)

    available_text = [name for name in text_pipeline_names(specs) if name not in used]
    for registry_name, pipeline_name in zip(text_registry, available_text):
        mapping[registry_name] = pipeline_name
        used.add(pipeline_name)

    for registry_name, component in components.items():
        if registry_name in mapping:
            continue
        role = component.get("role", "")
        candidates = []
        if role == "vae" or "vae" in registry_name or "autoencoder" in registry_name:
            candidates = ["vae", "autoencoder"]
        elif "unet" in registry_name:
            candidates = ["unet"]
        elif role in ["image_dit", "video_dit"] or "transformer" in registry_name or registry_name == "denoiser":
            candidates = ["transformer", "unet"]
        elif role == "vision_encoder" or "vision" in registry_name or "image_encoder" in registry_name:
            candidates = ["image_encoder", "vision_encoder"]
        elif role == "llm_decoder" or "vision_language" in registry_name:
            candidates = ["vision_language_encoder", "text_encoder", "image_encoder"]

        for pipeline_name in candidates:
            if pipeline_name in specs and pipeline_name not in used:
                mapping[registry_name] = pipeline_name
                used.add(pipeline_name)
                break
    return mapping


def component_is_shardable(component):
    sharding = str(component.get("sharding", ""))
    return "block" in sharding or "device_map" in sharding or "shard" in sharding


def load_staged_diffusers_pipeline(pipeline_class, model_id, portable_base_kwargs=None, plan=None):
    plan = plan or get_active_plan()
    profile = profile_for_plan(plan)
    if not profile:
        return pipeline_class.from_pretrained(model_id, **(portable_base_kwargs or {}))

    specs = pipeline_component_specs(pipeline_class, model_id)
    mapping = registry_pipeline_component_map(profile, specs)
    manager = StagedComponentManager(model_id, plan, profile)
    kwargs = deepcopy(portable_base_kwargs or {})
    kwargs["torch_dtype"] = torch_dtype(plan)

    for registry_name, pipeline_name in mapping.items():
        if pipeline_name not in specs:
            continue
        component = profile.get("components", {}).get(registry_name, {})
        role = component.get("role", "")
        if role in ["scheduler", "tokenizer", "processor"]:
            continue
        component_class = resolve_component_class(specs[pipeline_name])
        config_values = component_config_values(model_id, pipeline_name)
        proxy = LazyStagedComponent(
            manager,
            registry_name,
            pipeline_name,
            component_class,
            model_id,
            config_values,
            shardable=component_is_shardable(component),
        )
        manager.add_proxy(registry_name, proxy)
        kwargs[pipeline_name] = proxy

    pipe = pipeline_class.from_pretrained(model_id, **kwargs)
    pipe._staged_component_manager = manager
    pipe.prepare_next_generation = manager.release_all
    return pipe


def pipeline_quantization_config(plan=None, component_mapping=None):
    plan = plan or get_active_plan()
    if is_exact_fast_path(plan):
        return None

    profile = profile_for_plan(plan)
    if not profile:
        return None
    if profile.get("checkpoint", {}).get("already_quantized"):
        return None

    mapping = {}
    for registry_name in profile.get("components", {}):
        precision = component_precision(registry_name, plan)
        role = component_role(registry_name, plan)
        if precision == "native" or role in ["vae", "small_text_encoder", "scheduler", "tokenizer", "processor"]:
            continue

        pipeline_name = (component_mapping or {}).get(registry_name) or pipeline_component_name(registry_name)
        if role in ["large_text_encoder", "small_text_encoder", "vision_encoder", "vision_language_encoder", "llm_decoder"]:
            config = transformers_quantization_config(registry_name, plan)
        else:
            config = diffusers_quantization_config(registry_name, plan)
        if config is not None:
            mapping[pipeline_name] = config

    if not mapping:
        return None

    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError as error:
        raise RuntimeError(
            "This quantized pipeline plan requires a Diffusers release with PipelineQuantizationConfig support."
        ) from error
    return PipelineQuantizationConfig(quant_mapping=mapping)


def portable_pipeline_kwargs(plan=None, component_mapping=None):
    plan = plan or get_active_plan()
    placement = plan_placement(plan)
    kwargs = {"torch_dtype": torch_dtype(plan)}
    quantization = pipeline_quantization_config(plan, component_mapping)
    if quantization is not None:
        kwargs["quantization_config"] = quantization

    if placement in ["single_gpu_resident", "multi_gpu_device_map", "device_map_with_cpu_overflow"]:
        kwargs["device_map"] = plan.get("device_map") or ("balanced" if logical_gpu_count(plan) > 1 else "cuda")
        kwargs["max_memory"] = plan_max_memory(plan)
    elif placement == "mps_resident":
        kwargs["device_map"] = "mps"
    elif placement in ["single_gpu_model_specific_staging", "multi_gpu_model_specific_staging"]:
        kwargs["device_map"] = "balanced" if logical_gpu_count(plan) > 1 else "cuda"
        kwargs["max_memory"] = plan_max_memory(plan)
    return kwargs


def apply_pipeline_plan(pipe, plan=None):
    plan = plan or get_active_plan()
    if is_exact_fast_path(plan):
        return pipe

    placement = plan_placement(plan)
    if placement == "diffusers_model_cpu_offload":
        pipe.enable_model_cpu_offload(gpu_id=0)
    elif placement == "diffusers_sequential_cpu_offload":
        pipe.enable_sequential_cpu_offload(gpu_id=0)
    elif placement in ["mps_resident", "mps_resident_or_unified_memory"]:
        if placement == "mps_resident_or_unified_memory":
            pipe.to("mps")
    elif placement == "cpu_only":
        pipe.to("cpu")
    return pipe


def load_diffusers_pipeline(pipeline_class, model_id, current_kwargs=None, portable_base_kwargs=None, plan=None):
    plan = plan or get_active_plan()
    current_kwargs = deepcopy(current_kwargs or {})
    portable_base_kwargs = deepcopy(portable_base_kwargs or {})

    if not plan or is_exact_fast_path(plan):
        return pipeline_class.from_pretrained(model_id, **current_kwargs)

    if "model_specific_staging" in plan_placement(plan):
        return load_staged_diffusers_pipeline(pipeline_class, model_id, portable_base_kwargs, plan)

    profile = profile_for_plan(plan)
    component_mapping = None
    if profile:
        specs = pipeline_component_specs(pipeline_class, model_id)
        component_mapping = registry_pipeline_component_map(profile, specs)

    kwargs = portable_base_kwargs
    kwargs.update(portable_pipeline_kwargs(plan, component_mapping))
    pipe = pipeline_class.from_pretrained(model_id, **kwargs)
    return apply_pipeline_plan(pipe, plan)


def component_target_device(component_name, plan=None):
    plan = plan or get_active_plan()
    role = component_role(component_name, plan)
    if plan.get("execution_backend", plan.get("hardware_backend")) != "cuda":
        return execution_device(plan)
    if role == "vae":
        return preview_device(plan)
    if role in ["large_text_encoder", "small_text_encoder", "vision_encoder", "vision_language_encoder"]:
        return execution_device(plan)
    return execution_device(plan)


def component_device_map(component_name, plan=None, shardable=False):
    plan = plan or get_active_plan()
    execution_backend = plan.get("execution_backend", plan.get("hardware_backend"))
    if execution_backend == "mps":
        return {"": "mps"}
    if execution_backend != "cuda":
        return None

    placement = plan_placement(plan)
    if placement == "device_map_with_cpu_overflow":
        return "auto"
    if placement in ["diffusers_model_cpu_offload", "diffusers_sequential_cpu_offload"]:
        return {"": "cpu"}
    if shardable and logical_gpu_count(plan) > 1:
        return "balanced"

    target = component_target_device(component_name, plan)
    if target.startswith("cuda:"):
        return {"": int(target.split(":", 1)[1])}
    return None


def load_component(component_class, model_id, component_name, current_kwargs=None, portable_base_kwargs=None, shardable=False, plan=None, portable_device_map=None):
    plan = plan or get_active_plan()
    current_kwargs = deepcopy(current_kwargs or {})
    portable_base_kwargs = deepcopy(portable_base_kwargs or {})

    if not plan or is_exact_fast_path(plan):
        return component_class.from_pretrained(model_id, **current_kwargs)

    kwargs = portable_base_kwargs
    kwargs["torch_dtype"] = component_torch_dtype(component_name, plan)
    role = component_role(component_name, plan)
    if role in ["large_text_encoder", "small_text_encoder", "vision_encoder", "vision_language_encoder", "llm_decoder"]:
        quantization = transformers_quantization_config(component_name, plan)
    else:
        quantization = diffusers_quantization_config(component_name, plan)
    if quantization is not None:
        kwargs["quantization_config"] = quantization

    device_map = portable_device_map
    if device_map is None:
        device_map = component_device_map(component_name, plan, shardable)
    if device_map is not None:
        kwargs["device_map"] = device_map
        max_memory = plan_max_memory(plan)
        if max_memory:
            kwargs["max_memory"] = max_memory

    component = component_class.from_pretrained(model_id, **kwargs)
    if device_map is None and plan.get("execution_backend", plan.get("hardware_backend")) in ["mps", "cpu"]:
        component.to(component_target_device(component_name, plan))
    return component


def build_block_device_map(block_prefix, block_count, fixed_first=None, fixed_last=None, plan=None):
    plan = plan or get_active_plan()
    gpu_count = max(1, logical_gpu_count(plan))
    max_memory = plan_max_memory(plan)
    budgets = []
    for index in range(gpu_count):
        text = str(max_memory.get(index, "1GiB")).lower().replace("gib", "").replace("gb", "")
        budgets.append(max(0.1, float(text)))

    total_budget = sum(budgets)
    raw_counts = [block_count * budget / total_budget for budget in budgets]
    counts = [int(value) for value in raw_counts]
    remaining = block_count - sum(counts)
    remainders = [raw_counts[index] - counts[index] for index in range(gpu_count)]

    for index in sorted(range(gpu_count), key=lambda item: remainders[item], reverse=True)[:remaining]:
        counts[index] += 1

    device_map = {}
    first_device = 0
    last_device = gpu_count - 1
    for module in fixed_first or []:
        device_map[module] = first_device
    for module in fixed_last or []:
        device_map[module] = last_device

    block_index = 0
    for device_index, count in enumerate(counts):
        for _ in range(count):
            device_map[f"{block_prefix}.{block_index}"] = device_index
            block_index += 1
    return device_map


def transformers_model_kwargs(plan=None):
    plan = plan or get_active_plan()
    kwargs = {"dtype": torch_dtype(plan)}
    placement = plan_placement(plan)
    precision = component_precision("language_model", plan)

    if precision != "native":
        kwargs["quantization_config"] = transformers_quantization_config("language_model", plan)

    execution_backend = plan.get("execution_backend", plan.get("hardware_backend"))
    if placement == "cpu_only" or execution_backend == "cpu":
        kwargs["device_map"] = {"": "cpu"}
        kwargs["dtype"] = torch_dtype(plan)
    elif execution_backend == "mps":
        kwargs["device_map"] = {"": "mps"}
    else:
        if placement == "single_gpu_resident":
            kwargs["device_map"] = {"": 0}
        elif placement in ["multi_gpu_device_map", "device_map_with_cpu_overflow"]:
            kwargs["device_map"] = "balanced" if placement == "multi_gpu_device_map" else "auto"
        else:
            kwargs["device_map"] = "auto"
        kwargs["max_memory"] = plan_max_memory(plan)
    return kwargs


def vllm_engine_overrides(plan=None):
    plan = plan or get_active_plan()
    gpu_count = max(1, logical_gpu_count(plan))
    budgets = []
    for index, value in plan_max_memory(plan).items():
        if isinstance(index, int):
            budgets.append(float(str(value).lower().replace("gib", "").replace("gb", "")))

    utilization = 0.90
    if budgets:
        utilization = min(0.95, max(0.50, min(budgets) / (min(budgets) + 2.0)))
    return {
        "tensor_parallel_size": gpu_count,
        "gpu_memory_utilization": utilization,
    }


def clear_accelerator_cache():
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except RuntimeError:
                    pass
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def workload_from_diffusion_gui(gui):
    keys = [
        "width", "height", "batch_size", "num_images_per_prompt", "num_videos_per_prompt",
        "num_frames", "num_inference_steps", "max_sequence_length", "guidance_scale",
    ]
    return {key: getattr(gui, key) for key in keys if hasattr(gui, key)}


def workload_from_chat_gui(gui):
    return {
        "batch_size": 1,
        "prompt_tokens": int(getattr(gui, "max_context_messages", 16)) * 512,
        "max_new_tokens": int(getattr(gui, "max_new_tokens", 256)),
    }



def set_runtime_phase(plan, phase):
    if plan is not None:
        plan["runtime_phase"] = phase


def run_guarded(plan, phase, workload, function):
    try:
        return function()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        failure_phase = (plan or {}).get("runtime_phase", phase)
        if is_memory_error(error):
            if plan:
                exit_for_memory_retry(plan, error, workload, failure_phase)
            raise
        if plan:
            exit_for_fatal_failure(plan, error, workload, failure_phase)
        raise


def mark_success(plan=None, workload=None):
    plan = plan or get_active_plan()
    if plan:
        report_success(plan, workload)


def is_cuda_plan(plan=None):
    plan = plan or get_active_plan()
    return plan.get("execution_backend", plan.get("hardware_backend")) == "cuda" and execution_device(plan).startswith("cuda")


def should_load_preview_vae(plan=None):
    plan = plan or get_active_plan()
    if not plan or is_exact_fast_path(plan):
        return True
    placement = plan_placement(plan)
    return is_cuda_plan(plan) and logical_gpu_count(plan) > 1 and placement in [
        "multi_gpu_device_map",
        "multi_gpu_model_specific_staging",
        "single_gpu_resident",
    ]


def prepare_preview_vae(component_class, model_id, current_kwargs, plan=None):
    plan = plan or get_active_plan()
    if not should_load_preview_vae(plan):
        return None

    if not plan or is_exact_fast_path(plan):
        vae = component_class.from_pretrained(model_id, **current_kwargs)
        return vae.to("cuda:1")

    portable_kwargs = dict(current_kwargs)
    portable_kwargs.pop("torch_dtype", None)
    vae = load_component(
        component_class,
        model_id,
        "vae",
        current_kwargs=current_kwargs,
        portable_base_kwargs=portable_kwargs,
        shardable=False,
        plan=plan,
    )
    return vae.to(preview_device(plan))
