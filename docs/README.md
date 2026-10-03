# macOS Apple Silicon dependency and release report

## Verdict

The codebase can use a conventional release architecture: an unsigned DMG containing the application and an official Python.org installer, followed by one standard venv and one hash-locked `pip install`. Model weights remain external and use the standard Hugging Face cache.

The minimum release target is macOS 14 on native `arm64`. That floor is not arbitrary: the stable Apple Silicon BitsAndBytes wheel requires macOS 14, and Apple’s current PyTorch MPS guidance also lists macOS 14 for the pinned stable PyTorch line.

No complete macOS runtime claim should be made until the physical-device matrix at the end of this report passes. Dependency resolution and platform routing are now proved; native MPS execution of every retained model still requires real Macs and real inference.

## Final runtime tree

```text
Image-GUI-<version>-macOS-Apple-Silicon.dmg
└── Image GUI.app
    ├── shell launcher (arm64 and install-location checks)
    ├── official Python.org 3.13.16 installer package
    └── application source + requirements-lock-macos.txt
        │
        ├── Python absent → Apple Installer installs official CPython 3.13
        └── Python present
            └── ~/Library/Application Support/Image GUI/venv
                └── pip --require-hashes --only-binary=:all:
                    ├── UI: tkinter, CustomTkinter, Pillow, Matplotlib
                    ├── compute: PyTorch, TorchVision, TorchAudio, BitsAndBytes
                    ├── models: Transformers, Diffusers, Accelerate, PEFT
                    ├── storage: huggingface-hub, Safetensors, SentencePiece
                    ├── image/video: NumPy, SciPy, OpenCV contrib, PyAV,
                    │                ImageIO, bundled ImageIO FFmpeg
                    ├── audio: SoundDevice
                    └── application: GUI, planner, registry, model loaders
                         │
                         ├── model weights → standard Hugging Face cache
                         ├── settings/state → Application Support/Image GUI
                         └── outputs → Pictures/Image GUI by default
```

There is one runtime environment. Anima, ComfyUI, Qwen3 Coder, Real-ESRGAN, BasicSR, and the former ComfyUI environment/source checkout are absent.

TensorFlow is not used anywhere in the retained codebase and is not installed. CUDA is not bundled into the macOS release; Apple execution uses the PyTorch MPS backend. The Linux requirements still retain CUDA-capable PyTorch, BitsAndBytes, NVML, and optional vLLM.

## Dependency policy

The release uses stable artifacts from their normal publishers:

- CPython comes from `python.org` and its package SHA-256 is pinned in the build script.
- Python dependencies come from PyPI, are exact-version locked transitively, require binary wheels, and require hashes.
- PyTorch uses its normal stable PyPI Apple Silicon wheel; no nightly index is used.
- BitsAndBytes uses its stable PyPI `macosx_14_0_arm64` wheel; no preview wheel or GitHub artifact is used.
- No dependency is built from source on an end-user machine.
- vLLM and NVML are Linux/CUDA-only dependencies and are not in the macOS lock.

The lock currently contains 71 packages. The principal reviewed versions are:

| Layer | Version |
|---|---:|
| CPython | 3.13.16 |
| PyTorch / TorchVision / TorchAudio | 2.11.0 / 0.26.0 / 2.11.0 |
| BitsAndBytes | 0.50.0 |
| Transformers | 5.14.1 |
| Diffusers | 0.39.0 |
| Accelerate | 1.14.0 |
| OpenCV contrib | 5.0.0.93 |

PyTorch MPS remains labelled beta by PyTorch. This project does not substitute an unofficial backend; MPS is the official PyTorch route to Apple’s Metal Performance Shaders. The release must therefore test the exact pinned stack rather than assuming CUDA behavior transfers to Metal.

## Apple Silicon scope

Apple’s hardware scope is one native architecture and one GPU API:

- CPU architecture: `arm64`
- GPU execution: one logical PyTorch `mps` device backed by unified memory
- supported chip generations: M1, M2, M3, M4, M5, and later Apple Silicon
- product families: MacBook Air, MacBook Pro, iMac, Mac mini, Mac Studio, and Apple Silicon Mac Pro
- minimum OS: macOS 14

The application does not need a chip-specific Python or PyTorch build for M1 versus M5. The same arm64 wheel executes across those machines; the meaningful product differences are unified-memory capacity, memory bandwidth, GPU generation, thermal envelope, and OS version. The planner therefore keys macOS decisions to available unified memory and supported operations, not to marketing chip names.

Every M1 Mac can run macOS 14, so the minimum OS does not exclude the first Apple Silicon generation. Later chips ship with later compatible macOS releases and satisfy the same `>=14` application floor.

Primary references:

- [Apple’s Apple Silicon Mac inventory](https://support.apple.com/en-us/116943)
- [Apple’s macOS 14 compatibility list](https://support.apple.com/en-ie/105113)
- [Apple’s PyTorch/Metal guidance](https://developer.apple.com/metal/pytorch/)
- [Stable BitsAndBytes 0.49+ Apple Silicon wheel and support table](https://pypi.org/project/bitsandbytes/0.49.0/)
- [Python 3.13.16 official release](https://www.python.org/downloads/release/python-31316/)
- [Apple’s programmatic DMG guidance](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution)

## Loading strategy

The existing optimized Linux code is retained. Exact dual-RTX-3090 paths remain the first planner candidates when the detected hardware matches two CUDA GPUs with the expected capacity. This includes the custom component/layer placement used by the largest diffusion models; the release work does not flatten those paths into a generic loader.

For Apple Silicon:

1. Hardware detection reports `mps`, no CUDA GPU list, and physical unified memory.
2. CUDA-only exact paths fail their hardware match and are skipped without being modified.
3. The planner converts eligible portable candidates to MPS residency or MPS model-specific staging.
4. Native precision is preferred when unified memory permits it.
5. Stable BitsAndBytes INT8/NF4 candidates are available when a model profile permits quantization.
6. Custom staged loaders move one model component or execution phase at a time through MPS and release the previous phase.
7. A real MPS out-of-memory result is recorded and retried in a fresh process with the next credible plan.

This uses the advantage of unified memory without pretending it is unlimited VRAM. Large models still require high-memory Macs. “All models are present and supported” does not mean an 8 GB M1 can physically execute a 32B model.

The retained custom loader families are FLUX.2, GLM Image, Qwen Image, Qwen Image Edit, ChronoEdit, and Kandinsky 5 I2V. All other image/video paths use their official Diffusers pipeline classes with planner-controlled placement. Chat models use Transformers except the retained Qwen3.6 dual-GPU Linux fast path, which uses optional vLLM.

## Filesystem behavior

The packaged application writes only where a user expects persistent state:

```text
/Applications/Image GUI.app
~/Library/Application Support/Image GUI/
├── venv/
├── user_config.json
├── chat_history.enc
├── cachelight-settings.json
├── active_plans/
├── planner_results/
└── plan_history.json

~/.cache/huggingface/                 # unless HF_HOME/HF_HUB_CACHE overrides it
~/Pictures/Image GUI/                 # generated media, user-configurable
```

There is no app-specific directory under `~/Library/Caches` and no persistent logs directory. Pip’s first-run transaction uses temporary system storage and `--no-cache-dir`. Hugging Face itself owns model download staging, resume, deduplication, and final cache placement.

## DMG implementation

`make macos-dmg` calls `scripts/build_macos.sh`. The script:

1. Refuses non-Darwin or non-arm64 build hosts.
2. Validates the explicit application and packaging input manifest.
3. Downloads Python only from the pinned `python.org` HTTPS URL when it is not already in the build cache.
4. Verifies the official package against the pinned SHA-256.
5. Creates a normal `.app` bundle with an arm64 priority and macOS 14 minimum.
6. Copies an explicit source manifest, so local venvs, models, caches, histories, and generated media cannot leak into the release.
7. Adds the conventional `/Applications` link to the DMG.
8. Calls Apple’s `hdiutil` to create a compressed read-only disk image.

The build deliberately performs no Developer ID signing and no notarization. Users should expect the normal Gatekeeper warning for an unsigned internet download.

## What has been verified

- Registry validation passes after removing all three model stacks and their auxiliary profiles.
- All 47 retained launcher entries receive a valid plan on the dual-24-GiB CUDA test profile.
- FLUX.2’s exact dual-3090 plan, GPU selection, and memory map remain covered by tests.
- The synthetic 64-GiB MPS planner profile produces a plan for all 47 retained launcher models; the 32-GiB profile produces a plan for 46 of 47, with only Qwen3.6 35B rejected on capacity.
- CPython 3.13 Apple Silicon dependency resolution succeeds without source distributions.
- The generated macOS lock contains 71 exact packages with hashes.
- The macOS build-input preflight and both shell scripts pass on Linux.
- Core configuration, planner, registry, and model-GUI modules import successfully in the existing dependency environment.

These checks prove dependency availability and code routing. They do not prove Metal operator coverage, peak memory, output correctness, or DMG behavior on physical Apple hardware.

## Native acceptance matrix

Before a public release, run the generated DMG on at least these machines:

| Lane | Purpose |
|---|---|
| M1, 8–16 GB | oldest chip, low-memory failure quality, installer and UI |
| M1/M2 Max or Ultra, 64–128 GB | first-generation high-memory large-model execution |
| M3/M4 Pro or Max, 36–64 GB | current laptop GPU/operator and thermal behavior |
| M4/M5 high-memory system | current OS, current Metal generation, largest feasible models |

For each retained model: launch from a clean application install, allow Hugging Face to obtain required weights, load the selected plan, complete one default inference, save/open the output, unload, and relaunch from the cached weights. Gated models also need authentication-error and authenticated-success cases.

Release acceptance requires:

- DMG drag/install/first-run flow succeeds without Homebrew or Xcode.
- Official Python installation and the one-venv pip transaction succeed.
- `torch.backends.mps.is_available()` is true and a real MPS tensor operation succeeds.
- No model attempts a CUDA device on macOS.
- Memory failure produces a clear `cannot run` result or a successful safer retry, not a crash loop.
- Existing dual-3090 Linux tests remain green.

## Maintenance

Refresh the lock only when intentionally updating the reviewed stack:

```bash
make macos-lock
python validate_project.py
make macos-check
```

Then build and test the DMG on an Apple Silicon Mac. A dependency update is not accepted merely because it resolves; at least the smoke model set and one representative model from each custom loader family must complete real inference.
