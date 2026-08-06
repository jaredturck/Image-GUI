# Local AI Workstation

A desktop launcher for local image generation, video generation, image utilities, and chat models. The application includes a hardware-aware loading planner that chooses a model-specific placement strategy for the detected CUDA, Apple MPS, or CPU hardware.

The planner preserves the existing tuned dual-RTX-3090 loading paths and adds portable alternatives for other systems. It uses physical GPU capacity minus a fixed reserve, rather than an instantaneous free-memory reading, and can retry progressively safer plans after a genuine memory-allocation failure.

## Main features

- Image, video, utility, and chat model launcher
- Model-specific hardware and quantization metadata in `model_registry.py`
- CUDA, MPS, and CPU hardware detection
- Single-GPU and unequal multi-GPU planning
- Native, BitsAndBytes INT8, and permitted NF4 INT4 component precision
- Automatic device maps and `max_memory` budgets
- Exact hand-written fast paths preserved for FLUX.2, GLM Image, Qwen Image, Qwen Image Edit, ChronoEdit, and Kandinsky I2V Pro
- Portable custom staging for FLUX.2, Qwen Image Edit, and Kandinsky I2V Pro, with official pipeline fallbacks for the other custom models
- Diffusers model CPU offload and sequential CPU offload fallbacks
- Fresh-process retries after CUDA, MPS, or CPU allocation failures
- Per-machine plan history in `config/plan_history.json`
- Current loading plan shown inside model GUIs
- Persistent user paths for the Hugging Face cache, ComfyUI, and output root
- GUI-based Python dependency installer

## Installation

### 1. Install Python

Use a current 64-bit Python 3.10 or newer release supported by PyTorch and the model libraries. Python, pip, and Tkinter must already be available before running the installer. Tkinter normally ships with Windows and macOS Python installers; some Linux distributions provide it separately as a package such as `python3-tk`.

A virtual environment is strongly recommended:

```bash
python -m venv .venv
```

Activate it before continuing.

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the GUI installer

```bash
python install.py
```

The installer detects the operating system and whether NVIDIA hardware is visible, then installs:

- one of `requirements-cuda.txt`, `requirements-cpu.txt`, or `requirements-macos.txt`
- `requirements.txt`
- optionally `requirements-vllm.txt` on CUDA Linux

The installer only installs Python packages through pip. It does not install operating-system packages, GPU drivers, CUDA drivers, FFmpeg, PortAudio, or ComfyUI.

### 3. Install external prerequisites

Install these separately where relevant:

- **FFmpeg** for video encoding, decoding, and preview operations
- **PortAudio** for microphone capture through SoundDevice
- **NVIDIA drivers** for CUDA systems
- **ComfyUI and comfy-script support** for the Anima workflow

Use the package manager appropriate for the operating system. The installer reports external prerequisites it cannot detect, but does not modify the system automatically.

### 4. Configure the environment

Copy `.env.example` to `.env` and provide the required chat-history key:

```bash
cp .env.example .env
```

`CHAT_HISTORY_KEY_B64` must contain a Base64-encoded 32-byte key. One can be generated with:

```bash
python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
```

### 5. Start the application

```bash
python gui.py
```

On first launch, the application asks for optional paths:

- Hugging Face cache directory
- ComfyUI directory
- Output root

These settings are stored in `config/user_config.json` and can be changed later with the **Settings** button. On Jared's existing Linux workstation, the known `/mnt/8TB_HDD/hf_cache` and `/home/jared/comfy/ComfyUI` paths are detected automatically when present.

## Hardware planning

The launcher calls `hardware_planner.py` before starting a model process. The planner combines:

1. The detected hardware profile
2. The model profile from `model_registry.py`
3. The model's default workload
4. Successful and failed attempts previously recorded for the same machine

The initial GPU budget is based on physical capacity:

```text
usable GPU capacity = total VRAM - configured reserve
```

The default reserve is 2 GiB per GPU, or 1 GiB for GPUs at or below 12 GiB. These values can be edited in `config/user_config.json`.

The planner does not use an instantaneous free-VRAM reading to decide whether to lower model precision. Static memory estimates are used to reject clearly impossible configurations and order credible plans; actual model loading and inference remain the final test.

### Typical plan families

- Existing exact optimized path
- Native single-GPU residency
- Native multi-GPU device map
- INT8 single- or multi-GPU residency
- CPU-overflow device map
- Model-specific staged loading
- Diffusers model CPU offload
- Permitted INT4 residency or staging
- Diffusers sequential CPU offload
- CPU or MPS execution where supported and practical
- A clear `cannot run` result when no supported plan is credible

Quantization is component-specific. VAEs remain at their intended floating-point precision; large transformer and text-encoder components may use INT8 or permitted INT4 configurations according to the registry.

## Retry behaviour

A plan is considered validated only after the model loads and completes one real inference. On a recognized memory failure, the model process records the failed plan and the exact workload that failed, then exits with the planner retry code. The launcher replans for that workload and starts a fresh process using the next safer plan. Explicit image dimensions, frame counts, context settings, and related workload values are restored in the retry process.

The retry ladder may:

- reduce declared GPU `max_memory` to leave more runtime headroom
- move more weights to CPU
- use another supported placement strategy
- change eligible components from native precision to INT8
- use INT4 only where the model registry permits it
- fall back to sequential offload or CPU where meaningful

Authentication errors, missing packages, unsupported operators, corrupted checkpoints, and ordinary programming errors do not trigger the memory fallback ladder.

## Plan diagnostics

Explain the plans for a model without loading it:

```bash
python hardware_planner.py --model flux_2
```

JSON output:

```bash
python hardware_planner.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --json
```

The output includes selected candidates, GPU budgets, and rejection reasons.

Validate the registry, compile the project files, and run the planner unit tests:

```bash
python validate_project.py
```

## Model access and downloads

Model weights are not included in this repository. Hugging Face models download when first selected. Some repositories are gated and require:

- a Hugging Face account
- acceptance of the model license
- a configured Hugging Face token

A repository authentication failure is reported as an ordinary model error, not as insufficient hardware.

## Anima and ComfyUI

The Anima workflow continues to use the existing ComfyUI integration and expects the configured ComfyUI installation and model files to be available. The application does not download or install ComfyUI automatically.

The workflow currently expects the existing component names used by the application:

- `anima-preview.safetensors`
- `qwen_3_06b_base.safetensors`
- `qwen_image_vae.safetensors`

Set the ComfyUI directory from the launcher Settings window.

## Requirements files

- `requirements.txt` — shared application and model-library dependencies
- `requirements-cuda.txt` — PyTorch, BitsAndBytes, and NVML support
- `requirements-cpu.txt` — CPU PyTorch packages and the CPU-capable BitsAndBytes runtime
- `requirements-macos.txt` — macOS/MPS PyTorch packages
- `requirements-vllm.txt` — optional vLLM runtime

The model catalogue uses recently added model classes. When a stable package release lacks a required class, update Transformers and Diffusers before treating the error as a planner failure.

## Configuration and generated state

Runtime state is created under `config/`:

- `user_config.json` — paths and reserve settings
- `plan_history.json` — successful and failed plan records
- `active_plans/` — concrete launch plans
- `planner_results/` — child-process result records

These files are excluded by `.gitignore`.

## Important files

- `gui.py` — main launcher and retry owner
- `base_gui.py` — shared image/video GUI runtime
- `model_gui.py` — image, video, and utility model implementations
- `model_loading.py` — custom staged and multi-GPU loaders
- `chat_gui.py` — local chat application
- `model_registry.py` — comprehensive model metadata and candidate definitions
- `MODEL_REGISTRY_NOTES.md` — evidence, confidence, quantization, and maintenance notes
- `hardware_detection.py` — stable hardware capability profile
- `hardware_planner.py` — candidate filtering and concrete plan creation
- `planner_runtime.py` — plan-aware loading helpers
- `planner_protocol.py` — child-process result and OOM protocol
- `plan_history.py` — validated and failed plan persistence
- `app_config.py` — paths, reserves, and environment configuration
- `install.py` — GUI pip installer

## Troubleshooting

### The model immediately reports `cannot run`

Run the planner explanation command and review the rejected candidates. Common causes are insufficient system RAM for an offload plan, no supported backend, unavailable BitsAndBytes support, or a model whose smallest supported plan remains larger than the machine.

### CUDA model loading retries repeatedly

Each retry is intentionally performed in a fresh process. The planner may first reduce the device-map budget before switching precision or placement strategy. Failed attempts are remembered for the same hardware and workload.

### A model fails with a missing class or import

Update the relevant Python model libraries. This is a dependency/API issue rather than a VRAM planning result.

### FFmpeg or audio errors

Install FFmpeg or PortAudio through the operating system's package manager. The Python installer does not install system software.

### Anima cannot find ComfyUI

Open launcher Settings and select the ComfyUI installation directory. Confirm the workflow model files exist in the locations expected by that installation.

### Reset learned plans

Close the application and remove `config/plan_history.json`. The planner will recreate it and evaluate candidates from the beginning.
