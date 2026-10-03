# Image GUI

Image GUI is a desktop launcher for local image, video, utility, and chat models. It supports Apple Silicon through PyTorch MPS and preserves the existing Linux/CUDA loading paths, including the tuned dual-RTX-3090 strategies.

Model weights are never included in the application or DMG. Hugging Face libraries download them into the normal Hugging Face cache.

## macOS release

The release target is Apple Silicon (M1 or newer) running macOS 14 or newer. Build the unsigned DMG on an Apple Silicon Mac with:

```bash
make macos-dmg
```

The artifact is written to:

```text
dist/Image-GUI-0.1.0-macOS-Apple-Silicon.dmg
```

The DMG contains Image GUI, its source, the hash-locked macOS dependency manifest, and the official Python.org 3.13.16 installer. It does not contain model weights or the Python dependency wheels.

End-user flow:

1. Open the DMG and drag **Image GUI** to **Applications**.
2. Open the application. Because this release is deliberately unsigned, macOS may require **Privacy & Security → Open Anyway**.
3. If Python 3.13 is not installed, complete the included official Python.org installer.
4. Image GUI creates one venv and performs one hash-verified pip installation, then starts.

Persistent files use conventional locations:

- Application: `/Applications/Image GUI.app`
- Venv and application settings: `~/Library/Application Support/Image GUI/`
- Model weights: `${HF_HUB_CACHE}`, `${HF_HOME}`, or `~/.cache/huggingface`
- Generated media: `~/Pictures/Image GUI/` unless changed in Settings

There is no Image GUI cache directory and no persistent application log directory.

Run the packaging preflight on Linux or macOS:

```bash
make macos-check
```

The actual DMG step uses Apple’s `hdiutil`, so it intentionally refuses to run anywhere except an Apple Silicon Mac.

## Source installation

Run:

```bash
python install.py
```

This creates one project-local `.venv` and installs the shared requirements plus the detected platform requirements. On the Linux CUDA development machine, the installer can also install vLLM for the retained Qwen3.6 multi-GPU fast path.

Launch later with:

```bash
./.venv/bin/python gui.py
```

## Dependency layout

- `requirements.txt` — cross-platform UI, Hugging Face, image, video, audio, and utility libraries
- `requirements-macos.txt` — PyTorch, TorchVision, TorchAudio, and stable Apple Silicon BitsAndBytes
- `requirements-macos-constraints.txt` — reviewed top-level macOS versions
- `requirements-lock-macos.txt` — complete 71-package hash lock used by the release
- `requirements-cuda.txt` — Linux CUDA PyTorch, BitsAndBytes, and NVML
- `requirements-vllm.txt` — optional Linux-only vLLM path
- `requirements-cpu.txt` — source-install CPU fallback

Anima, ComfyUI, Qwen3 Coder, and Real-ESRGAN have been removed completely. ComfyUI is no longer a dependency and there is no second environment.

## Model loading

`hardware_planner.py` selects a model-specific plan before a child model process starts. The priority is:

1. Preserve an exact hand-tuned fast path when its hardware matches.
2. Use portable native or quantized residency.
3. Use model-specific staging or supported offload.
4. Retry a safer plan in a fresh process after a genuine memory allocation failure.

CUDA and MPS are separate execution backends. Apple hardware uses the single MPS device and unified memory; CUDA-specific device maps, `cuda:0`/`cuda:1` placement, NVML, and vLLM are not used on macOS.

## Validation

Run the repository checks with:

```bash
python validate_project.py
make macos-check
```

The detailed dependency tree, release rationale, evidence, and remaining native-Mac acceptance tests are in [docs/README.md](docs/README.md).

## Important files

- `gui.py` — launcher and retry owner
- `model_gui.py`, `chat_gui.py` — model user interfaces
- `model_loading.py` — retained custom CUDA loaders and portable staged loaders
- `model_registry.py` — model facts, placement policy, and candidate plans
- `hardware_detection.py`, `hardware_planner.py` — CUDA/MPS/CPU detection and planning
- `planner_runtime.py` — plan-aware loading primitives
- `app_config.py` — platform paths and runtime configuration
- `install.py` — one-venv installer
- `scripts/build_macos.sh` — reproducible DMG builder
