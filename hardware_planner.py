import argparse
import itertools
import json
import os
import tempfile
import time
import uuid
from copy import deepcopy

from app_config import config_dir, load_user_config
from hardware_detection import detect_hardware, hardware_summary
from model_registry import MODEL_PROFILES, PLAN_TEMPLATES, candidate_memory_estimate, get_model_profile, make_candidate_plan, resolve_model_key
from plan_history import failed_attempt_ids, successful_plan


SPEED_ORDER = {
    "excellent": 0,
    "very_good": 1,
    "good": 2,
    "usable": 3,
    "hardware_dependent": 4,
    "slow": 5,
    "very_slow": 6,
    "emergency": 7,
}


QUALITY_ORDER = {
    "model_intended": 0,
    "checkpoint_native": 0,
    "native": 0,
    "near_native": 1,
    "reduced": 2,
}


def normalize_precision(value):
    if value in ["int8", "8bit", "bnb_int8", "native_fp8"]:
        return "int8"
    if value in ["int4", "4bit", "bnb_nf4", "gguf_q4_k_m", "native_mxfp4", "checkpoint_native_nf4"]:
        return "int4"
    if value in ["checkpoint_native", "bfloat16", "float16", "float32"]:
        return "native"
    return value


def precision_requires_int8(plan):
    return any(normalize_precision(value) == "int8" for value in plan.get("component_precision", {}).values())


def precision_requires_int4(plan):
    return any(normalize_precision(value) == "int4" for value in plan.get("component_precision", {}).values())


def precision_requires_dynamic_int8(plan):
    return any(value in ["int8", "bnb_int8"] for value in plan.get("component_precision", {}).values())


def precision_requires_dynamic_int4(plan):
    return any(value in ["int4", "bnb_nf4"] for value in plan.get("component_precision", {}).values())


def cpu_plan_required_ram_gib(candidate, profile):
    estimated = candidate.get("estimated_memory", {})
    minimum = float(estimated.get("minimum_system_ram_gib", 0.0))
    resident_weight = float(estimated.get("fully_resident_weight_gib", 0.0))
    headroom = float(profile.get("runtime_memory", {}).get("default_runtime_headroom_gib", 2.0))
    native_dtype = str(profile.get("checkpoint", {}).get("native_dtype", ""))
    precision_values = {
        normalize_precision(value)
        for value in candidate.get("component_precision", {}).values()
    }

    if "int8" in precision_values or "int4" in precision_values:
        return max(minimum, resident_weight + headroom)

    dtype_factor = 1.0
    if "float32" not in native_dtype:
        dtype_factor = 2.0
    return max(minimum, resident_weight * dtype_factor + headroom)


def mps_plan_required_ram_gib(candidate):
    estimated = candidate.get("estimated_memory", {})
    return float(estimated.get("required_total_usable_vram_gib", 0.0))


def selected_gpus_support_quantization(selected_gpus, key):
    if not selected_gpus:
        return True
    return all(gpu.get(key, True) for gpu in selected_gpus)


def candidate_backend_allowed(candidate, profile, hardware):
    backend = hardware["backend"]
    placement = candidate.get("placement")
    requirements = candidate.get("requirements", {})
    required_backend = requirements.get("backend")

    if placement in ["mps_resident", "mps_model_specific_staging", "mps_resident_or_unified_memory"]:
        return backend == "mps", "requires Apple MPS"
    if placement == "cpu_only":
        cpu_support = str(profile.get("backend_support", {}).get("cpu", ""))
        return backend == "cpu" or bool(cpu_support), "CPU backend is unsupported"
    if placement == "comfyui_managed":
        return True, ""
    if placement == "vllm_tensor_parallel":
        if backend != "cuda":
            return False, "vLLM plan requires CUDA"
        if not hardware.get("supports_vllm"):
            return False, "vLLM is not installed"
        return True, ""
    if required_backend == "mps":
        return backend == "mps", "plan requires Apple MPS"
    if required_backend == "cuda_or_cpu":
        return backend in ["cuda", "cpu"], "plan requires CUDA or CPU"
    if required_backend == "comfyui":
        return True, ""
    if required_backend == "cuda" and backend != "cuda":
        return False, "plan requires CUDA"
    if backend == "cuda":
        return True, ""
    return False, f"plan is CUDA-specific and detected backend is {backend}"

def exact_fast_path_matches(candidate, hardware):
    match = candidate.get("hardware_match", {})
    if not match:
        return True
    expected_backend = match.get("backend")
    if expected_backend == "cuda_or_cpu" and hardware["backend"] not in ["cuda", "cpu"]:
        return False
    if expected_backend not in [None, "", "cuda_or_cpu", "comfyui"] and expected_backend != hardware["backend"]:
        return False

    gpu_count = int(match.get("gpu_count", 0))
    if len(hardware["gpus"]) < gpu_count:
        return False

    minimum_each = float(match.get("minimum_total_vram_gib_each", 0.0))
    if minimum_each:
        eligible = [gpu for gpu in hardware["gpus"] if float(gpu["total_vram_gib"]) >= minimum_each]
        if len(eligible) < gpu_count:
            return False

    return True


def gpu_order(hardware, preferred_gpu=None):
    gpus = list(hardware.get("gpus", []))
    gpus.sort(key=lambda gpu: float(gpu["usable_vram_gib"]), reverse=True)

    if preferred_gpu is not None:
        preferred = [gpu for gpu in gpus if int(gpu["index"]) == int(preferred_gpu)]
        others = [gpu for gpu in gpus if int(gpu["index"]) != int(preferred_gpu)]
        if preferred:
            return preferred + others
    return gpus


def choose_single_gpu(candidate, hardware, preferred_gpu=None):
    minimum = float(candidate.get("estimated_memory", {}).get("minimum_largest_gpu_usable_vram_gib", 0.0))
    ordered = gpu_order(hardware, preferred_gpu)
    for gpu in ordered:
        if float(gpu["usable_vram_gib"]) >= minimum:
            return [gpu]
    return []


def choose_multi_gpus(candidate, hardware, preferred_gpu=None):
    minimum_count = max(2, int(candidate.get("requirements", {}).get("minimum_gpu_count", 2)))
    required_total = float(candidate.get("estimated_memory", {}).get("required_total_usable_vram_gib", 0.0))
    minimum_largest = float(candidate.get("estimated_memory", {}).get("minimum_largest_gpu_usable_vram_gib", 0.0))
    if candidate.get("placement") == "vllm_tensor_parallel":
        checkpoint_native = any(
            value == "checkpoint_native"
            for value in candidate.get("component_precision", {}).values()
        )
        if checkpoint_native:
            required_total *= 0.70
            minimum_largest = min(minimum_largest, required_total / minimum_count)
    ordered = gpu_order(hardware, preferred_gpu)

    for count in range(minimum_count, len(ordered) + 1):
        for combination in itertools.combinations(ordered, count):
            total = sum(float(gpu["usable_vram_gib"]) for gpu in combination)
            largest = max(float(gpu["usable_vram_gib"]) for gpu in combination)
            if total >= required_total and largest >= minimum_largest:
                selected = list(combination)
                if preferred_gpu is not None:
                    selected.sort(key=lambda gpu: 0 if int(gpu["index"]) == int(preferred_gpu) else 1)
                return selected
    return []


def choose_exact_fast_path_gpus(candidate, hardware, preferred_gpu=None):
    match = candidate.get("hardware_match", {})
    gpu_count = int(match.get("gpu_count", candidate.get("requirements", {}).get("minimum_gpu_count", 0)))
    if gpu_count == 0 and match.get("backend") == "cuda_or_cpu" and hardware.get("backend") == "cuda":
        gpu_count = 1
    minimum_each = float(match.get("minimum_total_vram_gib_each", 0.0))
    eligible = [
        gpu
        for gpu in hardware.get("gpus", [])
        if float(gpu.get("total_vram_gib", 0.0)) >= minimum_each
    ]
    eligible.sort(key=lambda gpu: float(gpu.get("usable_vram_gib", 0.0)), reverse=True)

    if preferred_gpu is not None:
        preferred = [gpu for gpu in eligible if int(gpu["index"]) == int(preferred_gpu)]
        others = [gpu for gpu in eligible if int(gpu["index"]) != int(preferred_gpu)]
        eligible = preferred + others

    return eligible[:gpu_count]


def choose_gpus(candidate, hardware, preferred_gpu=None):
    placement = candidate.get("placement")
    if placement in ["cpu_only", "mps_resident", "mps_model_specific_staging", "mps_resident_or_unified_memory", "comfyui_managed"]:
        return []
    if candidate.get("template_id") == "current_exact_fast_path":
        return choose_exact_fast_path_gpus(candidate, hardware, preferred_gpu)

    minimum_gpu_count = int(candidate.get("requirements", {}).get("minimum_gpu_count", 0))
    if (
        placement in ["multi_gpu_device_map", "multi_gpu_model_specific_staging", "vllm_tensor_parallel"]
        or minimum_gpu_count >= 2
    ):
        return choose_multi_gpus(candidate, hardware, preferred_gpu)

    return choose_single_gpu(candidate, hardware, preferred_gpu)


def candidate_filter(candidate, profile, hardware, preferred_gpu=None):
    placement = candidate.get("placement")
    checkpoint_format = profile.get("checkpoint", {}).get("format")
    if checkpoint_format == "gguf_q4_k_m" and placement not in ["existing_code_path", "vllm_tensor_parallel"]:
        return False, "the GGUF artifact is currently implemented through the vLLM executor", []

    if "model_specific_staging" in str(placement) and not profile.get("placement_support", {}).get("custom_staging"):
        return False, "the model profile does not support staged execution", []

    allowed, reason = candidate_backend_allowed(candidate, profile, hardware)
    if not allowed:
        return False, reason, []

    if candidate.get("template_id") == "current_exact_fast_path" and not exact_fast_path_matches(candidate, hardware):
        return False, "exact hardware fast path does not match", []

    if (
        candidate.get("template_id") == "current_exact_fast_path"
        and profile.get("current_code", {}).get("loader") == "vllm_async_engine"
        and not hardware.get("supports_vllm")
    ):
        return False, "the existing fast path requires vLLM, which is not installed", []

    if precision_requires_dynamic_int8(candidate):
        if not hardware.get("supports_bitsandbytes_int8") and candidate.get("template_id") != "current_exact_fast_path":
            return False, "BitsAndBytes INT8 is unavailable for the detected backend", []

    if precision_requires_dynamic_int4(candidate):
        if not hardware.get("supports_bitsandbytes_nf4"):
            checkpoint = profile.get("checkpoint", {})
            if not checkpoint.get("already_quantized"):
                return False, "BitsAndBytes NF4 is unavailable for the detected backend", []

    minimum_ram = float(candidate.get("estimated_memory", {}).get("minimum_system_ram_gib", 0.0))
    if placement == "cpu_only":
        minimum_ram = cpu_plan_required_ram_gib(candidate, profile)
    elif placement in ["mps_resident", "mps_model_specific_staging", "mps_resident_or_unified_memory"]:
        minimum_ram = mps_plan_required_ram_gib(candidate)

    memory_tolerance = 0.1 if hardware.get("backend") == "mps" else 0.0
    if minimum_ram > float(hardware.get("usable_system_ram_gib", 0.0)) + memory_tolerance:
        return False, f"requires at least {minimum_ram:.1f} GiB usable system RAM", []

    selected_gpus = choose_gpus(candidate, hardware, preferred_gpu)
    minimum_gpu_count = int(candidate.get("requirements", {}).get("minimum_gpu_count", 0))
    if hardware["backend"] == "cuda" and minimum_gpu_count and len(selected_gpus) < minimum_gpu_count:
        return False, "GPU capacity or GPU count is insufficient", []

    if precision_requires_dynamic_int8(candidate) and not selected_gpus_support_quantization(
        selected_gpus,
        "supports_bitsandbytes_int8",
    ):
        return False, "one or more selected GPUs do not support BitsAndBytes INT8", []

    if precision_requires_dynamic_int4(candidate) and not selected_gpus_support_quantization(
        selected_gpus,
        "supports_bitsandbytes_nf4",
    ):
        return False, "one or more selected GPUs do not support BitsAndBytes NF4", []

    return True, "", selected_gpus


def budget_steps_for(candidate, config):
    placement = candidate.get("placement")
    if candidate.get("template_id") == "current_exact_fast_path":
        return [0.0]
    if placement in ["cpu_only", "mps_resident", "mps_model_specific_staging", "mps_resident_or_unified_memory", "comfyui_managed"]:
        return [0.0]
    if placement in ["single_gpu_resident"]:
        return [0.0, 0.5]
    return [float(value) for value in config["planner"].get("budget_retry_steps_gib", [0.0, 0.5, 1.0, 2.0])]


def logical_gpu_map(selected_gpus):
    return {
        int(gpu["index"]): logical_index
        for logical_index, gpu in enumerate(selected_gpus)
    }


def create_max_memory(selected_gpus, budget_reduction, candidate, hardware):
    max_memory = {}
    for logical_index, gpu in enumerate(selected_gpus):
        budget = max(0.5, float(gpu["usable_vram_gib"]) - float(budget_reduction))
        max_memory[logical_index] = f"{budget:.2f}GiB"

    if candidate.get("allows_cpu_overflow") or candidate.get("requirements", {}).get("requires_system_ram_for_weights"):
        max_memory["cpu"] = f"{max(1.0, float(hardware['usable_system_ram_gib'])):.2f}GiB"
    return max_memory


def plan_dtype(profile, hardware, candidate=None):
    native_dtype = profile.get("checkpoint", {}).get("native_dtype", "bfloat16")
    placement = (candidate or {}).get("placement")
    if placement == "cpu_only" or hardware["backend"] == "cpu":
        if native_dtype == "float32":
            return "float32"
        return "float32"
    if hardware["backend"] == "mps":
        if hardware.get("supports_bfloat16") and "bfloat16" in native_dtype:
            return "bfloat16"
        return "float16"
    if hardware.get("supports_bfloat16") and "bfloat16" in native_dtype:
        return "bfloat16"
    if native_dtype == "float32":
        return "float32"
    return "float16"


def workload_variant_tag(requested_workload, effective_workload):
    parts = []
    labels = {
        "num_images_per_prompt": "images",
        "num_videos_per_prompt": "videos",
    }
    for key, label in labels.items():
        requested = (requested_workload or {}).get(key)
        effective = (effective_workload or {}).get(key)
        if requested is not None and effective is not None and requested != effective:
            parts.append(f"{label}_{effective}")
    return "__".join(parts)


def workload_variants(profile, workload):
    requested = deepcopy(workload or profile.get("default_workload", {}))
    variants = [requested]
    adjustable = profile.get("workload_policy", {}).get("adjustable", {})

    for key, policy in adjustable.items():
        current = requested.get(key)
        minimum = int(policy.get("minimum", 1))
        if not isinstance(current, (int, float)) or int(current) <= minimum:
            continue
        for value in range(int(current) - 1, minimum - 1, -1):
            variant = deepcopy(requested)
            variant[key] = value
            if variant not in variants:
                variants.append(variant)

    return variants


def candidate_for_workload(candidate, profile, workload):
    result = deepcopy(candidate)
    result["estimated_memory"] = candidate_memory_estimate(
        profile,
        result.get("component_precision", {}),
        result.get("placement"),
        workload,
    )
    return result


def mps_candidate(candidate, profile):
    placement_map = {
        "single_gpu_resident": "mps_resident",
        "single_gpu_model_specific_staging": "mps_model_specific_staging",
    }
    placement = candidate.get("placement")
    if placement not in placement_map:
        return None

    result = deepcopy(candidate)
    result["placement"] = placement_map[placement]
    result["plan_id"] = f"{candidate['plan_id']}__mps"
    result["requirements"] = deepcopy(result.get("requirements", {}))
    result["requirements"]["backend"] = "mps"
    result["requirements"]["minimum_gpu_count"] = 0
    result["requirements"]["requires_system_ram_for_weights"] = True
    result["estimated_memory"] = candidate_memory_estimate(
        profile,
        result.get("component_precision", {}),
        result["placement"],
        profile.get("default_workload", {}),
    )
    result["notes"] = list(result.get("notes", [])) + ["Apple MPS unified-memory variant of the portable plan."]
    return result


def build_status_text(candidate, selected_gpus):
    precision_values = {normalize_precision(value) for value in candidate.get("component_precision", {}).values()}
    precision_values.discard("native")
    precision = "+".join(sorted(precision_values)) if precision_values else "native"
    placement = candidate.get("placement", "unknown").replace("_", " ")
    if selected_gpus:
        devices = ", ".join(f"GPU {gpu['index']}" for gpu in selected_gpus)
        return f"{precision.upper()} · {placement} · {devices}"
    return f"{precision.upper()} · {placement}"


def build_concrete_attempt(candidate, profile, hardware, selected_gpus, budget_reduction, model_reference, workload, requested_workload, candidate_index, result_file):
    physical_ids = [int(gpu["index"]) for gpu in selected_gpus]
    mapping = logical_gpu_map(selected_gpus)
    attempt_suffix = str(budget_reduction).replace(".", "p")
    workload_tag = workload_variant_tag(requested_workload, workload)
    attempt_id = f"{candidate['plan_id']}__reserve_{attempt_suffix}"
    if workload_tag:
        attempt_id = f"{attempt_id}__{workload_tag}"

    plan = deepcopy(candidate)
    placement = candidate.get("placement")
    if placement == "cpu_only":
        execution_backend = "cpu"
    elif placement in ["mps_resident", "mps_model_specific_staging", "mps_resident_or_unified_memory"]:
        execution_backend = "mps"
    else:
        execution_backend = hardware["backend"]

    if selected_gpus:
        primary_device = "cuda:0"
        secondary = "cuda:1" if len(selected_gpus) > 1 else "cuda:0"
    elif execution_backend == "mps":
        primary_device = "mps"
        secondary = "mps"
    else:
        primary_device = "cpu"
        secondary = "cpu"

    plan.update({
        "model_key": profile["key"],
        "model_reference": model_reference,
        "runtime_model_id": profile.get("runtime_model_id", profile.get("model_id")),
        "hardware_signature": hardware["signature"],
        "hardware_backend": hardware["backend"],
        "execution_backend": execution_backend,
        "hardware_summary": hardware_summary(hardware),
        "candidate_index": candidate_index,
        "attempt_id": attempt_id,
        "budget_reduction_gib": budget_reduction,
        "selected_physical_gpu_ids": physical_ids,
        "physical_to_logical_gpu": mapping,
        "visible_device_order": physical_ids,
        "logical_gpu_count": len(selected_gpus),
        "max_memory": create_max_memory(selected_gpus, budget_reduction, candidate, hardware),
        "execution_device": primary_device,
        "secondary_device": secondary,
        "preview_device": secondary,
        "torch_dtype": plan_dtype(profile, hardware, candidate),
        "workload": workload or deepcopy(profile.get("default_workload", {})),
        "requested_workload": requested_workload or deepcopy(profile.get("default_workload", {})),
        "workload_adjusted": bool(workload_tag),
        "result_file": result_file,
        "status_text": build_status_text(candidate, selected_gpus),
        "profile_summary": {
            "category": profile.get("category"),
            "task": profile.get("task"),
            "backend_support": profile.get("backend_support", {}),
            "placement_support": profile.get("placement_support", {}),
            "quantization_policy": profile.get("quantization_policy", {}),
            "components": {
                name: {
                    "role": component.get("role"),
                    "skip_modules": component.get("skip_modules", []),
                    "sharding": component.get("sharding"),
                    "quantization_support": component.get("quantization_support", {}),
                }
                for name, component in profile.get("components", {}).items()
            },
        },
    })

    if len(selected_gpus) > 1:
        plan["device_map"] = "balanced"
    elif selected_gpus:
        plan["device_map"] = "cuda"
    else:
        plan["device_map"] = None

    placement = plan.get("placement")
    if placement == "device_map_with_cpu_overflow":
        plan["device_map"] = "auto"
    if placement in ["diffusers_model_cpu_offload", "diffusers_sequential_cpu_offload"]:
        plan["device_map"] = None
    if placement == "existing_code_path":
        plan["preserve_exact_loader"] = True

    return plan


def add_backend_fallback_candidates(profile, hardware, candidates):
    if hardware["backend"] == "mps":
        support = str(profile.get("backend_support", {}).get("mps", ""))
        result = []
        if "unsupported" not in support and "not_supported" not in support:
            for candidate in candidates:
                converted = mps_candidate(candidate, profile)
                if converted is not None:
                    result.append(converted)
                elif candidate.get("placement") in ["cpu_only", "comfyui_managed"]:
                    result.append(deepcopy(candidate))
        return result

    result = list(candidates)
    placements = {candidate.get("placement") for candidate in result}

    if hardware["backend"] == "cpu" and "cpu_only" not in placements:
        support = str(profile.get("backend_support", {}).get("cpu", ""))
        if "supported" in support or "practical" in support:
            result.append(make_candidate_plan(profile, "cpu_native", "native"))

    return result


def plan_attempts(model_reference, workload=None, preferred_gpu=None, hardware=None):
    hardware = hardware or detect_hardware()
    profile = get_model_profile(model_reference)
    if profile is None:
        return {
            "status": "cannot_run",
            "reason": f"No model registry profile exists for {model_reference}",
            "attempts": [],
            "rejections": [],
            "hardware": hardware,
        }

    config = load_user_config()
    candidates = add_backend_fallback_candidates(profile, hardware, profile.get("candidate_plans", []))
    requested_workload = deepcopy(workload or profile.get("default_workload", {}))
    variants = workload_variants(profile, requested_workload)
    failed = failed_attempt_ids(hardware["signature"], profile["key"], requested_workload)
    validated = successful_plan(hardware["signature"], profile["key"], requested_workload)
    attempts = []
    rejections = []
    result_dir = os.path.join(config_dir(), "planner_results")
    os.makedirs(result_dir, exist_ok=True)

    for effective_workload in variants:
        for candidate_index, source_candidate in enumerate(candidates):
            candidate = candidate_for_workload(source_candidate, profile, effective_workload)
            allowed, reason, selected_gpus = candidate_filter(candidate, profile, hardware, preferred_gpu)
            if not allowed:
                tag = workload_variant_tag(requested_workload, effective_workload)
                rejection_id = candidate.get("plan_id") if not tag else f"{candidate.get('plan_id')}__{tag}"
                rejections.append({"plan_id": rejection_id, "reason": reason})
                continue

            for budget_reduction in budget_steps_for(candidate, config):
                result_file = os.path.join(result_dir, f"{uuid.uuid4().hex}.json")
                attempt = build_concrete_attempt(
                    candidate,
                    profile,
                    hardware,
                    selected_gpus,
                    budget_reduction,
                    model_reference,
                    effective_workload,
                    requested_workload,
                    candidate_index,
                    result_file,
                )
                default_workload = profile.get("default_workload", {})
                attempt["restore_workload"] = workload is not None or effective_workload != default_workload

                if attempt["attempt_id"] in failed:
                    rejections.append({"plan_id": attempt["attempt_id"], "reason": "previously failed on this hardware and workload"})
                    continue
                attempts.append(attempt)

    if validated:
        matching = [attempt for attempt in attempts if attempt["attempt_id"] == validated.get("attempt_id")]
        other = [attempt for attempt in attempts if attempt["attempt_id"] != validated.get("attempt_id")]
        attempts = matching + other

    if not attempts:
        return {
            "status": "cannot_run",
            "reason": "No supported loading plan fits the detected backend and physical memory constraints.",
            "model_key": profile["key"],
            "attempts": [],
            "rejections": rejections,
            "hardware": hardware,
        }

    return {
        "status": "ready",
        "model_key": profile["key"],
        "model_reference": model_reference,
        "attempts": attempts,
        "rejections": rejections,
        "hardware": hardware,
        "validated_plan": validated,
    }

def save_plan_file(plan):
    os.makedirs(os.path.join(config_dir(), "active_plans"), exist_ok=True)
    path = os.path.join(config_dir(), "active_plans", f"{uuid.uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2, ensure_ascii=False)
    return path


def explain_plan(result):
    lines = []
    lines.append(f"Hardware: {hardware_summary(result['hardware'])}")
    lines.append(f"Status: {result['status']}")
    if result["status"] == "cannot_run":
        lines.append(f"Reason: {result['reason']}")
    else:
        for index, attempt in enumerate(result["attempts"][:20], start=1):
            lines.append(f"{index}. {attempt['attempt_id']}: {attempt['status_text']}")
            lines.append(f"   max_memory={attempt['max_memory']}")
    if result.get("rejections"):
        lines.append("Rejected candidates:")
        for rejection in result["rejections"][:30]:
            lines.append(f"- {rejection['plan_id']}: {rejection['reason']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Explain the hardware-aware loading plan for a model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--preferred-gpu", type=int)
    parser.add_argument("--workload-json", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workload = json.loads(args.workload_json) if args.workload_json else None
    result = plan_attempts(args.model, workload=workload, preferred_gpu=args.preferred_gpu)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(explain_plan(result))


if __name__ == "__main__":
    main()
