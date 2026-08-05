Kandinsky 5 Image-to-Image GUI
==============================

Files
-----
kandinsky5_i2i_gui.py  - the complete standalone GUI and model loader
requirements.txt       - required Python packages

Requirements
------------
- Python 3.10 or newer. Python 3.11 or 3.12 is recommended.
- Apple Silicon Mac: macOS 14 or newer and an arm64 Python installation.
- NVIDIA computer: a working NVIDIA driver and CUDA-compatible PyTorch installation.
- Around 36 GB of free disk space for the model download, plus extra cache/output space.

Important Apple memory note
---------------------------
This exact checkpoint is very large. A normal M1 MacBook with 8 GB or 16 GB of
unified memory is not expected to have enough memory. A 32 GB machine may still
be marginal. A 64 GB Apple Silicon Mac is the realistic target for this model.
The program uses MPS and model CPU offloading, but CPU and GPU still share the
same physical unified memory on Apple Silicon.

Install
-------
Open Terminal in this folder and run:

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt

Run
---
With the virtual environment active:

    python3 kandinsky5_i2i_gui.py

The first launch downloads the model. Generated images are saved in the
"kandinsky_i2i" folder beside the script.

Backend selection
-----------------
- Apple Silicon: MPS with float16 and model CPU offloading.
- NVIDIA: CUDA with bfloat16 when supported, otherwise float16, using the same
  model CPU-offloading approach as the original application.
- CPU: float32 only when neither MPS nor CUDA is available.

If tkinter is missing on macOS
------------------------------
Use a current arm64 installer from python.org, which includes Tk support. Verify
it with:

    python3 -c "import tkinter; print('Tkinter OK')"

Troubleshooting
---------------
- "MPS not available": update macOS, confirm Python is arm64, and reinstall
  PyTorch inside the virtual environment.
- Out-of-memory errors: reduce width and height. This can reduce generation
  memory, but it cannot solve insufficient memory for the model weights.
- The model download resumes automatically if interrupted.
