import argparse
import base64
import binascii
import hashlib
import json
import os
import platform
import queue
import secrets
import subprocess
import sys
import threading
import venv
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

from app_config import (
    apply_runtime_environment,
    is_packaged_macos,
    load_user_config,
    runtime_venv_dir,
    save_user_config,
)


APP_NAME = "Image GUI"
SOURCE_DIR = Path(__file__).resolve().parent
MACOS_LOCK = SOURCE_DIR / "requirements-lock-macos.txt"
SHARED_REQUIREMENTS = SOURCE_DIR / "requirements.txt"
INSTALL_MARKER = ".image-gui-install.json"


def venv_python(venv_dir):
    scripts = "Scripts" if platform.system() == "Windows" else "bin"
    executable = "python.exe" if platform.system() == "Windows" else "python"
    return Path(venv_dir) / scripts / executable


def requirements_for_current_platform(include_vllm=False, packaged=False):
    if packaged:
        if not MACOS_LOCK.is_file():
            raise FileNotFoundError("The packaged macOS dependency lock is missing.")
        return [MACOS_LOCK]

    system = platform.system()
    files = [SHARED_REQUIREMENTS]
    if system == "Darwin":
        if platform.machine() != "arm64":
            raise RuntimeError("The macOS release supports Apple Silicon only.")
        files.append(SOURCE_DIR / "requirements-macos.txt")
        files.append(SOURCE_DIR / "requirements-macos-constraints.txt")
    elif system == "Linux" and _nvidia_hardware_present():
        files.append(SOURCE_DIR / "requirements-cuda.txt")
        if include_vllm:
            files.append(SOURCE_DIR / "requirements-vllm.txt")
    else:
        files.append(SOURCE_DIR / "requirements-cpu.txt")
    return files


def _nvidia_hardware_present():
    if Path("/proc/driver/nvidia/version").exists():
        return True
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def dependency_fingerprint(requirement_files):
    digest = hashlib.sha256()
    for path in requirement_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def marker_path(venv_dir):
    return Path(venv_dir) / INSTALL_MARKER


def environment_is_current(venv_dir, requirement_files):
    python = venv_python(venv_dir)
    marker = marker_path(venv_dir)
    if not python.is_file() or not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("dependency_fingerprint") == dependency_fingerprint(requirement_files)


def ensure_application_settings():
    config = load_user_config()
    secrets_config = config.setdefault("secrets", {})
    encoded_key = secrets_config.get("chat_history_key_b64", "") or os.environ.get("CHAT_HISTORY_KEY_B64", "")
    legacy_env = SOURCE_DIR / ".env"
    if not encoded_key and legacy_env.is_file():
        try:
            for line in legacy_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("CHAT_HISTORY_KEY_B64="):
                    encoded_key = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass
    try:
        key_is_valid = len(base64.b64decode(encoded_key, validate=True)) == 32
    except (ValueError, TypeError, binascii.Error):
        key_is_valid = False
    if not key_is_valid:
        encoded_key = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    if secrets_config.get("chat_history_key_b64") != encoded_key:
        secrets_config["chat_history_key_b64"] = encoded_key
        save_user_config(config)
    apply_runtime_environment(config)


def launch_application(venv_dir):
    ensure_application_settings()
    os.chdir(SOURCE_DIR)
    os.execv(str(venv_python(venv_dir)), [str(venv_python(venv_dir)), str(SOURCE_DIR / "gui.py")])


class Installer:
    def __init__(self, auto_start=False):
        self.packaged = is_packaged_macos()
        self.venv_dir = Path(runtime_venv_dir())
        self.events = queue.Queue()
        self.installing = False

        self.root = tk.Tk()
        self.include_vllm = tk.BooleanVar(
            master=self.root,
            value=platform.system() == "Linux" and _nvidia_hardware_present(),
        )
        self.root.title(f"{APP_NAME} Setup")
        self.root.geometry("760x540")
        self.root.minsize(680, 480)
        self._build_ui()
        self.root.after(100, self._drain_events)
        if auto_start:
            self.root.after(250, self.start_install)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=f"Set up {APP_NAME}", font=("TkDefaultFont", 20, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "This creates one standard Python virtual environment and installs the application dependencies. "
                "Model weights are not included; supported models use the normal Hugging Face cache when first loaded."
            ),
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(8, 14))

        details = ttk.LabelFrame(outer, text="Installation", padding=12)
        details.pack(fill="x")
        location = str(self.venv_dir)
        ttk.Label(details, text=f"Virtual environment: {location}", wraplength=680).pack(anchor="w")
        ttk.Label(details, text="Model cache: standard Hugging Face cache", wraplength=680).pack(anchor="w", pady=(5, 0))

        if platform.system() == "Linux" and _nvidia_hardware_present() and not self.packaged:
            ttk.Checkbutton(
                details,
                text="Install vLLM for the existing multi-GPU Qwen3.6 fast path",
                variable=self.include_vllm,
            ).pack(anchor="w", pady=(8, 0))

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(16, 8))
        self.status = ttk.Label(outer, text="Ready to install")
        self.status.pack(anchor="w")

        output_frame = ttk.LabelFrame(outer, text="Setup output", padding=8)
        output_frame.pack(fill="both", expand=True, pady=(12, 12))
        self.output = tk.Text(output_frame, height=12, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.install_button = ttk.Button(controls, text="Install", command=self.start_install)
        self.install_button.pack(side="right")
        ttk.Button(controls, text="Quit", command=self.root.destroy).pack(side="right", padx=(0, 8))

    def append_output(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def start_install(self):
        if self.installing:
            return
        self.installing = True
        self.install_vllm_requested = self.include_vllm.get()
        self.install_button.configure(state="disabled")
        self.status.configure(text="Preparing the virtual environment…")
        self.progress.start(12)
        threading.Thread(target=self._install, daemon=True).start()

    def _emit(self, kind, value):
        self.events.put((kind, value))

    def _run(self, command):
        self._emit("output", f"$ {' '.join(map(str, command))}\n")
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=SOURCE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            self._emit("output", line)
        return process.wait() == 0

    def _install(self):
        try:
            requirement_files = requirements_for_current_platform(
                include_vllm=self.install_vllm_requested,
                packaged=self.packaged,
            )
            self.venv_dir.parent.mkdir(parents=True, exist_ok=True)
            if not environment_is_current(self.venv_dir, requirement_files):
                self._emit("status", "Creating the Python virtual environment…")
                venv.EnvBuilder(with_pip=True, clear=True).create(self.venv_dir)

            python = venv_python(self.venv_dir)
            self._emit("status", "Installing application dependencies…")
            command = [python, "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:"]
            if self.packaged:
                command.append("--require-hashes")
            for requirements in requirement_files:
                option = "-c" if requirements.name.endswith("-constraints.txt") else "-r"
                command.extend([option, requirements])
            if not self._run(command):
                raise RuntimeError("pip could not install the application dependencies.")

            self._emit("status", "Checking the installed environment…")
            validation = (
                "import cv2, customtkinter, diffusers, torch, transformers; "
                "print('torch', torch.__version__); "
                "print('accelerator', 'mps' if torch.backends.mps.is_available() else "
                "('cuda' if torch.cuda.is_available() else 'cpu'))"
            )
            if not self._run([python, "-c", validation]):
                raise RuntimeError("The installed environment failed its import check.")

            ensure_application_settings()
            marker_path(self.venv_dir).write_text(
                json.dumps(
                    {
                        "dependency_fingerprint": dependency_fingerprint(requirement_files),
                        "python": platform.python_version(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._emit("complete", None)
        except Exception as error:
            self._emit("error", str(error))

    def _drain_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "output":
                    self.append_output(value)
                elif kind == "status":
                    self.status.configure(text=value)
                elif kind == "complete":
                    self.progress.stop()
                    self.status.configure(text="Setup complete. Starting Image GUI…")
                    self.root.after(300, lambda: launch_application(self.venv_dir))
                elif kind == "error":
                    self.progress.stop()
                    self.installing = False
                    self.install_button.configure(state="normal", text="Retry")
                    self.status.configure(text="Setup failed")
                    messagebox.showerror(f"{APP_NAME} Setup", value, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description=f"Install and launch {APP_NAME}.")
    parser.add_argument("--app-launch", action="store_true", help="Launch from the macOS application bundle.")
    parser.add_argument("--install", action="store_true", help="Begin installation immediately.")
    return parser.parse_args()


def main():
    args = parse_args()
    packaged = is_packaged_macos()
    requirement_files = requirements_for_current_platform(packaged=packaged)
    target_venv = Path(runtime_venv_dir())

    if args.app_launch and environment_is_current(target_venv, requirement_files):
        launch_application(target_venv)

    Installer(auto_start=args.app_launch or args.install).run()


if __name__ == "__main__":
    main()
