import json
import os
import platform
import tempfile
from copy import deepcopy

CONFIG_DIR_NAME = "config"
CONFIG_FILE_NAME = "user_config.json"
DEFAULT_MEDIA_DIR_NAME = "media"

OUTPUT_DIRECTORY_MAP = {
    "z_image_turbo": ("images", "z_image_turbo"),
    "kandinsky": ("images", "kandinsky_5"),
    "pixart_sigma": ("images", "pixart_sigma"),
    "anima": ("images", "anima"),
    "stable_diffusion3.5": ("images", "stable_diffusion_3_5"),
    "black_forest": ("images", "flux_1"),
    "glm_image": ("images", "glm_image"),
    "qwen_image": ("images", "qwen_image"),
    "black_forest_flux_2": ("images", "flux_2"),
    "kandinsky_i2i": ("images", "kandinsky_5_i2i"),
    "chronoedit": ("images", "chronoedit"),
    "qwen_image_edit": ("images", "qwen_image_edit"),
    "rmbg_1_4": ("images", "rmbg_1_4"),
    "real_esrgan": ("images", "real_esrgan"),
    "sd_x4_upscaler": ("images", "sd_x4_upscaler"),
    "skyreels_v2": ("videos", "skyreels_v2"),
    "kandinsky_tv2_lite": ("videos", "kandinsky_5_t2v"),
    "nvidia_cosmos": ("videos", "nvidia_cosmos"),
    "allegro": ("videos", "allegro"),
    "cogvideox": ("videos", "cogvideox"),
    "hunyuan_video": ("videos", "hunyuan_video_1_5"),
    "hunyuan_video_i2v": ("videos", "hunyuan_video_1_5_i2v"),
    "kandinsky_tv2_distilled": ("videos", "kandinsky_5_t2v_pro"),
    "kandinsky_tv2_pro": ("videos", "kandinsky_5_t2v_pro_sft"),
    "kandinsky_i2v": ("videos", "kandinsky_5_i2v"),
    "kandinsky_i2v_pro": ("videos", "kandinsky_5_i2v_pro_sft"),
}

DEFAULT_CONFIG = {
    "schema_version": 1,
    "paths": {
        "huggingface_cache_dir": "",
        "comfyui_dir": "",
        "output_root": "",
    },
    "planner": {
        "cuda_gpu_reserve_gib": 2.0,
        "small_gpu_reserve_gib": 1.0,
        "small_gpu_threshold_gib": 12.0,
        "system_ram_reserve_gib": 6.0,
        "budget_retry_steps_gib": [0.0, 0.5, 1.0, 2.0],
        "automatic_retry": True,
    },
    "installer": {
        "install_vllm": True,
        "install_bitsandbytes": True,
    },
}


def project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def config_dir():
    return os.path.join(project_dir(), CONFIG_DIR_NAME)


def config_path():
    return os.path.join(config_dir(), CONFIG_FILE_NAME)


def deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def jared_pc_defaults():
    home = os.path.expanduser("~")
    is_jared = home == "/home/jared" or os.path.isdir("/home/jared")
    defaults = {}

    if is_jared and os.path.isdir("/mnt/8TB_HDD"):
        defaults["huggingface_cache_dir"] = "/mnt/8TB_HDD/hf_cache"

    if is_jared and os.path.isdir("/home/jared/comfy/ComfyUI"):
        defaults["comfyui_dir"] = "/home/jared/comfy/ComfyUI"

    return defaults


def load_user_config():
    config = deepcopy(DEFAULT_CONFIG)
    path = config_path()

    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                stored = json.load(file)
            if isinstance(stored, dict):
                config = deep_merge(config, stored)
        except (OSError, json.JSONDecodeError):
            pass

    defaults = jared_pc_defaults()
    for key, value in defaults.items():
        if not config["paths"].get(key):
            config["paths"][key] = value

    return config


def save_user_config(config):
    os.makedirs(config_dir(), exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(prefix="user_config_", suffix=".json", dir=config_dir())
    os.close(handle)

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    os.replace(temporary_path, config_path())


def get_path(name, fallback=""):
    value = load_user_config().get("paths", {}).get(name, "")
    return value or fallback


def default_output_root():
    return os.path.join(project_dir(), DEFAULT_MEDIA_DIR_NAME)


def output_root():
    configured = get_path("output_root").strip()
    if not configured:
        return default_output_root()

    configured = os.path.expanduser(configured)
    if not os.path.isabs(configured):
        configured = os.path.join(project_dir(), configured)
    return os.path.abspath(configured)


def resolve_output_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path

    normalized = os.path.normpath(relative_path).replace("\\", "/").strip("./")
    category_and_name = OUTPUT_DIRECTORY_MAP.get(normalized)
    if category_and_name:
        return os.path.join(output_root(), *category_and_name)
    return os.path.join(output_root(), normalized)


def apply_runtime_environment(config=None):
    config = config or load_user_config()
    cache_dir = config.get("paths", {}).get("huggingface_cache_dir", "").strip()

    if not cache_dir and os.environ.get("USE_HHD", "False") == "True":
        cache_dir = "/mnt/8TB_HDD/hf_cache"

    if cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "hub")

    comfyui_dir = config.get("paths", {}).get("comfyui_dir", "").strip()
    if comfyui_dir:
        os.environ["AI_WORKSTATION_COMFYUI_DIR"] = os.path.abspath(os.path.expanduser(comfyui_dir))

    os.environ["AI_WORKSTATION_CONFIG"] = config_path()
    return config


def platform_name():
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


apply_runtime_environment()
