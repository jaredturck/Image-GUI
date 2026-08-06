import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

from app_config import load_user_config

GIB = 1024 ** 3


def module_available(name):
    return importlib.util.find_spec(name) is not None


def total_system_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass

    if hasattr(os, "sysconf"):
        names = os.sysconf_names
        if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))

    return 0


def version_of(package_name):
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return "unavailable"


def cuda_driver_available_without_torch():
    command = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []

    if process.returncode != 0:
        return []

    gpus = []
    for index, line in enumerate(process.stdout.splitlines()):
        if not line.strip() or "," not in line:
            continue
        name, memory_mib = line.rsplit(",", 1)
        gpus.append({
            "index": index,
            "name": name.strip(),
            "total_vram_gib": round(float(memory_mib.strip()) / 1024.0, 3),
        })
    return gpus


def reserve_for_gpu(total_gib, config):
    planner = config["planner"]
    if total_gib <= float(planner["small_gpu_threshold_gib"]):
        return float(planner["small_gpu_reserve_gib"])
    return float(planner["cuda_gpu_reserve_gib"])


def detect_with_torch(config):
    try:
        import torch
    except ImportError:
        return None

    profile = {
        "backend": "cpu",
        "gpus": [],
        "system_ram_gib": round(total_system_ram_bytes() / GIB, 3),
        "usable_system_ram_gib": 0.0,
        "supports_bfloat16": False,
        "supports_float16": True,
        "supports_bitsandbytes_int8": False,
        "supports_bitsandbytes_nf4": False,
        "supports_vllm": module_available("vllm"),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }

    if torch.cuda.is_available():
        profile["backend"] = "cuda"
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_gib = float(properties.total_memory) / GIB
            reserve_gib = reserve_for_gpu(total_gib, config)
            capability = list(torch.cuda.get_device_capability(index))

            supports_bfloat16 = False
            try:
                with torch.cuda.device(index):
                    supports_bfloat16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                supports_bfloat16 = capability[0] >= 8

            bitsandbytes_available = module_available("bitsandbytes")
            capability_value = capability[0] + capability[1] / 10.0
            profile["gpus"].append({
                "index": index,
                "name": properties.name,
                "total_vram_gib": round(total_gib, 3),
                "reserve_gib": round(reserve_gib, 3),
                "usable_vram_gib": round(max(0.0, total_gib - reserve_gib), 3),
                "compute_capability": capability,
                "supports_bfloat16": supports_bfloat16,
                "supports_float16": True,
                "supports_bitsandbytes_int8": bitsandbytes_available and capability_value >= 7.5,
                "supports_bitsandbytes_nf4": bitsandbytes_available and capability_value >= 6.0,
            })

        profile["supports_bfloat16"] = all(gpu["supports_bfloat16"] for gpu in profile["gpus"])
        profile["supports_bitsandbytes_int8"] = any(
            gpu["supports_bitsandbytes_int8"]
            for gpu in profile["gpus"]
        )
        profile["supports_bitsandbytes_nf4"] = any(
            gpu["supports_bitsandbytes_nf4"]
            for gpu in profile["gpus"]
        )

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        profile["backend"] = "mps"
        profile["supports_bfloat16"] = False
        profile["supports_bitsandbytes_int8"] = False
        profile["supports_bitsandbytes_nf4"] = False
    else:
        bitsandbytes_available = module_available("bitsandbytes")
        profile["supports_bitsandbytes_int8"] = bitsandbytes_available
        profile["supports_bitsandbytes_nf4"] = bitsandbytes_available

    reserve = float(config["planner"]["system_ram_reserve_gib"])
    profile["usable_system_ram_gib"] = round(max(0.0, profile["system_ram_gib"] - reserve), 3)
    return profile


def detect_hardware(allow_torch=True):
    config = load_user_config()
    profile = detect_with_torch(config) if allow_torch else None

    if profile is None:
        gpus = cuda_driver_available_without_torch()
        for gpu in gpus:
            reserve = reserve_for_gpu(gpu["total_vram_gib"], config)
            gpu["reserve_gib"] = reserve
            gpu["usable_vram_gib"] = round(max(0.0, gpu["total_vram_gib"] - reserve), 3)
            gpu["supports_bfloat16"] = None
            gpu["supports_float16"] = True
            gpu["supports_bitsandbytes_int8"] = module_available("bitsandbytes")
            gpu["supports_bitsandbytes_nf4"] = module_available("bitsandbytes")

        backend = "cuda" if gpus else ("mps" if platform.system() == "Darwin" else "cpu")
        system_ram_gib = round(total_system_ram_bytes() / GIB, 3)
        profile = {
            "backend": backend,
            "gpus": gpus,
            "system_ram_gib": system_ram_gib,
            "usable_system_ram_gib": round(max(0.0, system_ram_gib - float(config["planner"]["system_ram_reserve_gib"])), 3),
            "supports_bfloat16": None,
            "supports_float16": True,
            "supports_bitsandbytes_int8": module_available("bitsandbytes"),
            "supports_bitsandbytes_nf4": module_available("bitsandbytes"),
            "supports_vllm": module_available("vllm"),
            "torch_version": version_of("torch"),
        }

    profile["platform"] = platform.system().lower()
    profile["platform_release"] = platform.release()
    profile["python_version"] = platform.python_version()
    profile["software"] = {
        "torch": version_of("torch"),
        "transformers": version_of("transformers"),
        "diffusers": version_of("diffusers"),
        "accelerate": version_of("accelerate"),
        "bitsandbytes": version_of("bitsandbytes"),
        "vllm": version_of("vllm"),
    }

    signature_data = {
        "backend": profile["backend"],
        "gpus": [
            {
                "name": gpu["name"],
                "total_vram_gib": round(float(gpu["total_vram_gib"]), 1),
                "compute_capability": gpu.get("compute_capability"),
            }
            for gpu in profile["gpus"]
        ],
        "system_ram_gib": round(float(profile["system_ram_gib"]), 0),
        "platform": profile["platform"],
        "software": profile["software"],
        "planner_reserves": {
            "cuda_gpu_reserve_gib": config["planner"]["cuda_gpu_reserve_gib"],
            "small_gpu_reserve_gib": config["planner"]["small_gpu_reserve_gib"],
            "small_gpu_threshold_gib": config["planner"]["small_gpu_threshold_gib"],
            "system_ram_reserve_gib": config["planner"]["system_ram_reserve_gib"],
        },
    }
    encoded = json.dumps(signature_data, sort_keys=True).encode("utf-8")
    profile["signature"] = hashlib.sha256(encoded).hexdigest()[:20]
    return profile


def hardware_summary(profile):
    if profile["backend"] == "cuda":
        gpu_text = ", ".join(
            f"GPU {gpu['index']} {gpu['name']} ({gpu['total_vram_gib']:.1f} GiB, {gpu['usable_vram_gib']:.1f} GiB budget)"
            for gpu in profile["gpus"]
        )
        return f"CUDA: {gpu_text}; RAM {profile['system_ram_gib']:.1f} GiB"

    if profile["backend"] == "mps":
        return f"Apple MPS unified memory; RAM {profile['system_ram_gib']:.1f} GiB"

    return f"CPU only; RAM {profile['system_ram_gib']:.1f} GiB"


if __name__ == "__main__":
    print(json.dumps(detect_hardware(), indent=2))
