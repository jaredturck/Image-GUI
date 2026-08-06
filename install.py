import base64
import ctypes.util
import json
import os
import platform
import queue
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import urllib.request
import zipfile
from tkinter import messagebox, ttk


COMFYUI_VERSION = "v0.28.0"
COMFYUI_ARCHIVE_URL = f"https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/{COMFYUI_VERSION}.zip"
COMFY_SCRIPT_SPEC = "comfy-script[default]"


class InstallerApp:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_venv_dir = os.path.join(self.base_dir, ".venv")
        self.comfy_venv_dir = os.path.join(self.base_dir, ".comfy_venv")
        self.runtime_dir = os.path.join(self.base_dir, "runtime")
        self.comfyui_dir = os.path.join(self.runtime_dir, "ComfyUI")
        self.config_dir = os.path.join(self.base_dir, "config")
        self.user_config_path = os.path.join(self.config_dir, "user_config.json")
        self.installation_state_path = os.path.join(self.config_dir, "installation.json")

        self.backend = self.detect_backend()
        self.output_queue = queue.Queue()
        self.current_process = None
        self.active_stage = None
        self.main_ready = False
        self.comfy_ready = False
        self.comfy_skipped = False
        self.missing_prerequisites = self.detect_external_prerequisites()

        self.root = tk.Tk()
        self.root.title("AI Workstation Installer")
        self.root.geometry("900x700")
        self.root.minsize(780, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close_installer)

        self.install_vllm = tk.BooleanVar(
            value=self.backend == "cuda" and platform.system() == "Linux"
        )

        self.container = ttk.Frame(self.root)
        self.container.pack(fill="both", expand=True)

        self.create_welcome_page()
        self.create_main_install_page()
        self.create_comfy_install_page()
        self.create_completion_page()
        self.show_page(self.welcome_page)
        self.root.after(100, self.poll_output)

    def detect_backend(self):
        if platform.system() == "Darwin":
            return "macos"
        if shutil.which("nvidia-smi"):
            return "cuda"
        return "cpu"

    def detect_external_prerequisites(self):
        missing = []
        if not shutil.which("ffmpeg"):
            missing.append("FFmpeg")
        if not ctypes.util.find_library("portaudio"):
            missing.append("PortAudio (required by SoundDevice microphone input)")
        return missing

    def base_python(self):
        return getattr(sys, "_base_executable", None) or sys.executable

    def environment_python(self, environment_dir):
        if os.name == "nt":
            return os.path.join(environment_dir, "Scripts", "python.exe")
        return os.path.join(environment_dir, "bin", "python")

    def main_python(self):
        return self.environment_python(self.main_venv_dir)

    def comfy_python(self):
        return self.environment_python(self.comfy_venv_dir)

    def show_page(self, page):
        for child in self.container.winfo_children():
            child.pack_forget()
        page.pack(fill="both", expand=True)

    def create_header(self, parent, title, description):
        ttk.Label(
            parent,
            text=title,
            font=("TkDefaultFont", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            parent,
            text=description,
            wraplength=840,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(10, 18))

    def create_welcome_page(self):
        self.welcome_page = ttk.Frame(self.container, padding=22)
        self.welcome_page.columnconfigure(0, weight=1)
        self.welcome_page.rowconfigure(4, weight=1)

        self.create_header(
            self.welcome_page,
            "AI Workstation Installer",
            (
                "The installer creates two private Python environments inside this project. "
                "The main application uses .venv, while the optional ComfyUI backend uses "
                ".comfy_venv. You do not need to create or activate either environment yourself."
            ),
        )

        info = ttk.LabelFrame(self.welcome_page, text="Detected environment", padding=14)
        info.grid(row=2, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)

        values = [
            ("Operating system", f"{platform.system()} {platform.release()}"),
            ("Bootstrap Python", sys.version.split()[0]),
            ("Selected backend", self.backend.upper()),
            ("Main environment", self.main_venv_dir),
            ("ComfyUI environment", self.comfy_venv_dir),
        ]

        for row, value in enumerate(values):
            ttk.Label(info, text=f"{value[0]}:").grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 14),
                pady=2,
            )
            ttk.Label(info, text=value[1], wraplength=610).grid(
                row=row,
                column=1,
                sticky="nw",
                pady=2,
            )

        options = ttk.LabelFrame(self.welcome_page, text="Application options", padding=14)
        options.grid(row=3, column=0, sticky="ew", pady=(14, 0))

        vllm = ttk.Checkbutton(
            options,
            text="Install optional vLLM support (CUDA Linux only)",
            variable=self.install_vllm,
        )
        vllm.pack(anchor="w")
        if self.backend != "cuda" or platform.system() != "Linux":
            vllm.state(["disabled"])

        prerequisites = ttk.LabelFrame(
            self.welcome_page,
            text="External system software",
            padding=14,
        )
        prerequisites.grid(row=4, column=0, sticky="nsew", pady=(14, 0))

        if self.missing_prerequisites:
            text = (
                "The following optional system software was not detected:\n\n"
                + "\n".join(f"• {item}" for item in self.missing_prerequisites)
                + "\n\nThe installer will not modify your operating system. Install these later "
                "with your normal package manager when you need the related features."
            )
        else:
            text = (
                "FFmpeg and PortAudio were detected. The installer will only install Python "
                "packages and project-local files."
            )

        ttk.Label(
            prerequisites,
            text=text,
            wraplength=820,
            justify="left",
        ).pack(anchor="nw")

        bottom = ttk.Frame(self.welcome_page)
        bottom.grid(row=5, column=0, sticky="ew", pady=(18, 0))
        bottom.columnconfigure(0, weight=1)

        ttk.Button(
            bottom,
            text="Install Application",
            command=self.begin_main_install,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Button(
            bottom,
            text="Close",
            command=self.close_installer,
        ).grid(row=0, column=2, padx=(8, 0))

    def create_main_install_page(self):
        self.main_page = ttk.Frame(self.container, padding=22)
        self.main_page.columnconfigure(0, weight=1)
        self.main_page.rowconfigure(3, weight=1)

        self.create_header(
            self.main_page,
            "Installing the AI Workstation",
            (
                "This stage creates or updates .venv and installs the application, model, "
                "planner, and selected hardware-backend dependencies."
            ),
        )

        self.main_status = ttk.Label(self.main_page, text="Waiting to start")
        self.main_status.grid(row=2, column=0, sticky="w", pady=(0, 8))

        log_frame = ttk.LabelFrame(self.main_page, text="Installation output", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.main_log = tk.Text(log_frame, wrap="word", state="disabled")
        self.main_log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.main_log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.main_log.configure(yscrollcommand=scrollbar.set)

        self.main_progress = ttk.Progressbar(self.main_page, mode="indeterminate")
        self.main_progress.grid(row=4, column=0, sticky="ew", pady=(14, 0))

        bottom = ttk.Frame(self.main_page)
        bottom.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        bottom.columnconfigure(0, weight=1)

        self.main_action_button = ttk.Button(
            bottom,
            text="Installing...",
            state="disabled",
        )
        self.main_action_button.grid(row=0, column=1, padx=(8, 0))

        ttk.Button(
            bottom,
            text="Close",
            command=self.close_installer,
        ).grid(row=0, column=2, padx=(8, 0))

    def create_comfy_install_page(self):
        self.comfy_page = ttk.Frame(self.container, padding=22)
        self.comfy_page.columnconfigure(0, weight=1)
        self.comfy_page.rowconfigure(4, weight=1)

        self.create_header(
            self.comfy_page,
            "Optional ComfyUI backend",
            (
                "ComfyUI is used by the Anima workflow. It is installed separately so a "
                "ComfyUI dependency problem cannot prevent the rest of the AI Workstation "
                "from being installed or launched."
            ),
        )

        details = ttk.LabelFrame(self.comfy_page, text="Managed installation", padding=12)
        details.grid(row=2, column=0, sticky="ew")
        details.columnconfigure(1, weight=1)

        detail_values = [
            ("ComfyUI source", self.comfyui_dir),
            ("Python environment", self.comfy_venv_dir),
            ("Pinned ComfyUI release", COMFYUI_VERSION),
        ]

        for row, value in enumerate(detail_values):
            ttk.Label(details, text=f"{value[0]}:").grid(
                row=row,
                column=0,
                sticky="nw",
                padx=(0, 12),
                pady=2,
            )
            ttk.Label(details, text=value[1], wraplength=620).grid(
                row=row,
                column=1,
                sticky="nw",
                pady=2,
            )

        self.comfy_status = ttk.Label(
            self.comfy_page,
            text="Ready to install the optional backend",
        )
        self.comfy_status.grid(row=3, column=0, sticky="w", pady=(12, 8))

        log_frame = ttk.LabelFrame(self.comfy_page, text="ComfyUI installation output", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.comfy_log = tk.Text(log_frame, wrap="word", state="disabled")
        self.comfy_log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.comfy_log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.comfy_log.configure(yscrollcommand=scrollbar.set)

        self.comfy_progress = ttk.Progressbar(self.comfy_page, mode="indeterminate")
        self.comfy_progress.grid(row=5, column=0, sticky="ew", pady=(14, 0))

        bottom = ttk.Frame(self.comfy_page)
        bottom.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        bottom.columnconfigure(0, weight=1)

        self.skip_comfy_button = ttk.Button(
            bottom,
            text="Continue Without ComfyUI",
            command=self.skip_comfy_install,
        )
        self.skip_comfy_button.grid(row=0, column=1, padx=(8, 0))

        self.comfy_action_button = ttk.Button(
            bottom,
            text="Install ComfyUI",
            command=self.begin_comfy_install,
        )
        self.comfy_action_button.grid(row=0, column=2, padx=(8, 0))

        ttk.Button(
            bottom,
            text="Close",
            command=self.close_installer,
        ).grid(row=0, column=3, padx=(8, 0))

    def create_completion_page(self):
        self.completion_page = ttk.Frame(self.container, padding=26)
        self.completion_page.columnconfigure(0, weight=1)
        self.completion_page.rowconfigure(3, weight=1)

        self.create_header(
            self.completion_page,
            "Installation complete",
            (
                "The installer has finished. You can launch the AI Workstation now or close "
                "this window and start it later with the project .venv Python interpreter."
            ),
        )

        self.completion_status = ttk.Label(
            self.completion_page,
            text="",
            font=("TkDefaultFont", 12, "bold"),
        )
        self.completion_status.grid(row=2, column=0, sticky="w")

        summary = ttk.LabelFrame(self.completion_page, text="Installation summary", padding=16)
        summary.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        summary.columnconfigure(0, weight=1)

        self.completion_summary = ttk.Label(
            summary,
            text="",
            wraplength=810,
            justify="left",
        )
        self.completion_summary.grid(row=0, column=0, sticky="nw")

        bottom = ttk.Frame(self.completion_page)
        bottom.grid(row=4, column=0, sticky="ew", pady=(18, 0))
        bottom.columnconfigure(0, weight=1)

        self.launch_status = ttk.Label(bottom, text="Ready")
        self.launch_status.grid(row=0, column=0, sticky="w")

        self.launch_button = ttk.Button(
            bottom,
            text="Launch AI Workstation",
            command=self.launch_application,
        )
        self.launch_button.grid(row=0, column=1, padx=(8, 0))

        ttk.Button(
            bottom,
            text="Close",
            command=self.close_installer,
        ).grid(row=0, column=2, padx=(8, 0))

    def append_text(self, widget, text):
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def command_text(self, command):
        if os.name == "nt":
            return subprocess.list2cmdline(command)
        return shlex.join(command)

    def requirements_files(self):
        if self.backend == "cuda":
            files = ["requirements-cuda.txt"]
        elif self.backend == "macos":
            files = ["requirements-macos.txt"]
        else:
            files = ["requirements-cpu.txt"]

        files.append("requirements.txt")
        if self.install_vllm.get() and self.backend == "cuda" and platform.system() == "Linux":
            files.append("requirements-vllm.txt")
        return files

    def backend_requirements_file(self):
        if self.backend == "cuda":
            return "requirements-cuda.txt"
        if self.backend == "macos":
            return "requirements-macos.txt"
        return "requirements-cpu.txt"

    def missing_requirement_files(self):
        names = set(self.requirements_files())
        names.add(self.backend_requirements_file())
        return [
            name
            for name in sorted(names)
            if not os.path.isfile(os.path.join(self.base_dir, name))
        ]

    def begin_main_install(self):
        if self.active_stage:
            return

        if sys.version_info < (3, 10):
            messagebox.showerror(
                "Python version",
                "Python 3.10 or newer is required.",
                parent=self.root,
            )
            return

        missing_files = self.missing_requirement_files()
        if missing_files:
            messagebox.showerror(
                "Missing requirements files",
                "The following files are missing:\n\n" + "\n".join(missing_files),
                parent=self.root,
            )
            return

        self.show_page(self.main_page)
        self.main_action_button.configure(text="Installing...", state="disabled")
        self.main_status.configure(text="Preparing the application environment...")
        self.main_progress.configure(mode="indeterminate")
        self.main_progress.start(12)
        self.active_stage = "main"

        thread = threading.Thread(target=self.main_install_worker, daemon=True)
        thread.start()

    def main_install_worker(self):
        success = False
        message = "The application dependency installation failed."

        try:
            if not self.ensure_environment(self.main_venv_dir, "main"):
                self.output_queue.put(("main_finished", False, message))
                return

            main_python = self.main_python()
            if not self.run_pip_upgrade(main_python, "main"):
                self.output_queue.put(("main_finished", False, message))
                return

            requirement_files = self.requirements_files()
            total = len(requirement_files)

            for index, requirement_file in enumerate(requirement_files, start=1):
                self.output_queue.put(
                    (
                        "main_status",
                        f"Installing {requirement_file} ({index} of {total})...",
                    )
                )
                path = os.path.join(self.base_dir, requirement_file)
                command = [
                    main_python,
                    "-m",
                    "pip",
                    "install",
                    "--progress-bar",
                    "on",
                    "-r",
                    path,
                ]
                if not self.run_command(command, "main", self.base_dir):
                    self.output_queue.put(("main_finished", False, message))
                    return

            self.output_queue.put(("main_status", "Validating the application environment..."))
            verification = (
                "import customtkinter; import torch; import transformers; import diffusers; "
                "import accelerate; "
                "print('Application environment validation passed')"
            )
            if not self.run_command(
                [main_python, "-c", verification],
                "main",
                self.base_dir,
            ):
                message = (
                    "Dependencies were installed, but the application environment validation failed."
                )
                self.output_queue.put(("main_finished", False, message))
                return

            self.ensure_env_file()
            self.write_installation_state("main", "ready")
            success = True
            message = "The main AI Workstation environment is ready."
        except Exception as error:
            self.output_queue.put(("main_log", f"\nUnexpected installer error: {error}\n"))
            message = str(error)

        self.output_queue.put(("main_finished", success, message))

    def ensure_environment(self, environment_dir, target):
        python_path = self.environment_python(environment_dir)
        if os.path.isfile(python_path):
            self.output_queue.put(
                (f"{target}_log", f"Reusing existing environment: {environment_dir}\n")
            )
            return True

        if os.path.exists(environment_dir):
            current_python = os.path.realpath(sys.executable)
            target_python = os.path.realpath(python_path)
            if current_python == target_python:
                self.output_queue.put(
                    (
                        f"{target}_log",
                        "The active environment is incomplete and cannot be rebuilt while it is running.\n",
                    )
                )
                return False
            shutil.rmtree(environment_dir)

        os.makedirs(os.path.dirname(environment_dir), exist_ok=True)
        self.output_queue.put(
            (f"{target}_status", f"Creating {os.path.basename(environment_dir)}...")
        )
        command = [self.base_python(), "-m", "venv", environment_dir]
        return self.run_command(command, target, self.base_dir)

    def run_pip_upgrade(self, python_path, target):
        self.output_queue.put((f"{target}_status", "Updating pip tooling..."))
        command = [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--progress-bar",
            "on",
            "pip",
            "setuptools",
            "wheel",
        ]
        return self.run_command(command, target, self.base_dir)

    def run_command(self, command, target, cwd=None):
        self.output_queue.put(
            (f"{target}_log", f"\n$ {self.command_text(command)}\n")
        )

        environment = os.environ.copy()
        environment["PIP_PROGRESS_BAR"] = "on"
        environment["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            command,
            cwd=cwd or self.base_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        self.current_process = process

        for line in process.stdout:
            self.output_queue.put((f"{target}_log", line))

        return_code = process.wait()
        self.current_process = None

        if return_code != 0:
            self.output_queue.put(
                (f"{target}_log", f"\nCommand exited with code {return_code}.\n")
            )
            return False
        return True

    def ensure_env_file(self):
        env_path = os.path.join(self.base_dir, ".env")
        if os.path.isfile(env_path):
            self.output_queue.put(("main_log", "Existing .env file preserved.\n"))
            return

        key = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
        content = f"CHAT_HISTORY_KEY_B64={key}\nUSE_HHD=False\n"
        with open(env_path, "w", encoding="utf-8") as file:
            file.write(content)
        self.output_queue.put(("main_log", "Created .env with a secure chat-history key.\n"))

    def continue_to_comfy_page(self):
        self.show_page(self.comfy_page)

    def begin_comfy_install(self):
        if self.active_stage:
            return

        self.comfy_action_button.configure(text="Installing...", state="disabled")
        self.skip_comfy_button.configure(state="disabled")
        self.comfy_status.configure(text="Preparing the ComfyUI backend...")
        self.comfy_progress.configure(mode="indeterminate", value=0)
        self.comfy_progress.start(12)
        self.active_stage = "comfy"

        thread = threading.Thread(target=self.comfy_install_worker, daemon=True)
        thread.start()

    def comfy_install_worker(self):
        success = False
        message = "The optional ComfyUI backend could not be installed."

        try:
            if not self.ensure_comfyui_source():
                self.output_queue.put(("comfy_finished", False, message))
                return

            if not self.ensure_environment(self.comfy_venv_dir, "comfy"):
                self.output_queue.put(("comfy_finished", False, message))
                return

            comfy_python = self.comfy_python()
            if not self.run_pip_upgrade(comfy_python, "comfy"):
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.output_queue.put(
                ("comfy_status", "Installing the ComfyUI hardware backend...")
            )
            backend_path = os.path.join(self.base_dir, self.backend_requirements_file())
            backend_command = [
                comfy_python,
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "on",
                "-r",
                backend_path,
            ]
            if not self.run_command(backend_command, "comfy", self.base_dir):
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.output_queue.put(("comfy_status", "Installing ComfyUI dependencies..."))
            comfy_requirements = os.path.join(self.comfyui_dir, "requirements.txt")
            comfy_command = [
                comfy_python,
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "on",
                "-r",
                comfy_requirements,
            ]
            if not self.run_command(comfy_command, "comfy", self.comfyui_dir):
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.output_queue.put(("comfy_status", "Installing ComfyScript support..."))
            comfy_script_command = [
                comfy_python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--progress-bar",
                "on",
                COMFY_SCRIPT_SPEC,
            ]
            if not self.run_command(comfy_script_command, "comfy", self.comfyui_dir):
                self.output_queue.put(("comfy_finished", False, message))
                return

            main_comfy_script_command = [
                self.main_python(),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--progress-bar",
                "on",
                COMFY_SCRIPT_SPEC,
            ]
            if not self.run_command(main_comfy_script_command, "comfy", self.base_dir):
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.output_queue.put(("comfy_status", "Validating ComfyScript..."))
            verification = (
                "import nest_asyncio2; from comfy_script.runtime import Workflow, util; "
                "print('ComfyScript validation passed')"
            )
            if not self.run_command(
                [comfy_python, "-c", verification],
                "comfy",
                self.comfyui_dir,
            ):
                message = "ComfyScript installed but did not pass its backend import validation."
                self.output_queue.put(("comfy_finished", False, message))
                return

            if not self.run_command(
                [self.main_python(), "-c", verification],
                "comfy",
                self.base_dir,
            ):
                message = "ComfyScript installed but did not pass its application import validation."
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.output_queue.put(("comfy_status", "Running the ComfyUI startup check..."))
            quick_test = [
                comfy_python,
                os.path.join(self.comfyui_dir, "main.py"),
                "--quick-test-for-ci",
                "--disable-all-custom-nodes",
            ]
            if not self.run_command(quick_test, "comfy", self.comfyui_dir):
                message = "ComfyUI installed but failed its startup validation."
                self.output_queue.put(("comfy_finished", False, message))
                return

            self.save_managed_comfyui_path()
            self.write_installation_state("comfyui", "ready")
            success = True
            message = "The optional ComfyUI backend is ready."
        except Exception as error:
            self.output_queue.put(("comfy_log", f"\nUnexpected installer error: {error}\n"))
            message = str(error)

        self.output_queue.put(("comfy_finished", success, message))

    def ensure_comfyui_source(self):
        main_path = os.path.join(self.comfyui_dir, "main.py")
        requirements_path = os.path.join(self.comfyui_dir, "requirements.txt")

        if os.path.isfile(main_path) and os.path.isfile(requirements_path):
            self.output_queue.put(
                ("comfy_log", f"Reusing existing managed ComfyUI source: {self.comfyui_dir}\n")
            )
            return True

        os.makedirs(self.runtime_dir, exist_ok=True)
        archive_path = os.path.join(self.runtime_dir, f"ComfyUI-{COMFYUI_VERSION}.zip")
        extract_dir = os.path.join(self.runtime_dir, ".comfyui_extract")

        if os.path.isdir(self.comfyui_dir):
            shutil.rmtree(self.comfyui_dir)
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.isfile(archive_path):
            os.remove(archive_path)

        self.output_queue.put(
            ("comfy_status", f"Downloading ComfyUI {COMFYUI_VERSION}...")
        )
        if not self.download_file(COMFYUI_ARCHIVE_URL, archive_path):
            return False

        self.output_queue.put(("comfy_status", "Extracting ComfyUI..."))
        self.output_queue.put(("comfy_progress_mode", "indeterminate"))
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)

        source_dir = self.find_comfyui_source(extract_dir)
        if not source_dir:
            self.output_queue.put(
                ("comfy_log", "Downloaded archive did not contain a valid ComfyUI source directory.\n")
            )
            return False

        shutil.move(source_dir, self.comfyui_dir)
        marker_path = os.path.join(self.comfyui_dir, ".ai_workstation_version")
        with open(marker_path, "w", encoding="utf-8") as file:
            file.write(COMFYUI_VERSION + "\n")

        shutil.rmtree(extract_dir, ignore_errors=True)
        if os.path.isfile(archive_path):
            os.remove(archive_path)

        self.output_queue.put(
            ("comfy_log", f"Installed ComfyUI source at {self.comfyui_dir}\n")
        )
        return True

    def find_comfyui_source(self, extract_dir):
        for name in os.listdir(extract_dir):
            candidate = os.path.join(extract_dir, name)
            if not os.path.isdir(candidate):
                continue
            if os.path.isfile(os.path.join(candidate, "main.py")) and os.path.isfile(
                os.path.join(candidate, "requirements.txt")
            ):
                return candidate
        return None

    def download_file(self, url, destination):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "AI-Workstation-Installer/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0") or "0")
                received = 0
                self.output_queue.put(("comfy_progress_mode", "determinate" if total else "indeterminate"))

                with open(destination, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = min(100.0, received * 100.0 / total)
                            self.output_queue.put(("comfy_progress_value", percent))
                            self.output_queue.put(
                                (
                                    "comfy_status",
                                    f"Downloading ComfyUI {COMFYUI_VERSION}: {percent:.0f}%",
                                )
                            )
            return True
        except Exception as error:
            self.output_queue.put(("comfy_log", f"ComfyUI download failed: {error}\n"))
            return False

    def save_managed_comfyui_path(self):
        config = {}
        if os.path.isfile(self.user_config_path):
            try:
                with open(self.user_config_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, dict):
                    config = loaded
            except (OSError, json.JSONDecodeError):
                config = {}

        paths = config.setdefault("paths", {})
        paths["comfyui_dir"] = self.comfyui_dir
        installer = config.setdefault("installer", {})
        installer["managed_comfyui"] = True
        installer["comfyui_environment"] = self.comfy_venv_dir

        self.write_json(self.user_config_path, config)
        self.output_queue.put(
            ("comfy_log", "Saved the managed ComfyUI path in config/user_config.json.\n")
        )

    def write_installation_state(self, component, status):
        state = {}
        if os.path.isfile(self.installation_state_path):
            try:
                with open(self.installation_state_path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                if isinstance(loaded, dict):
                    state = loaded
            except (OSError, json.JSONDecodeError):
                state = {}

        state["backend"] = self.backend
        state["main_environment"] = self.main_venv_dir
        state["comfyui_environment"] = self.comfy_venv_dir
        state["comfyui_directory"] = self.comfyui_dir
        state["comfyui_version"] = COMFYUI_VERSION
        state[component] = status
        self.write_json(self.installation_state_path, state)

    def write_json(self, path, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle, temporary_path = tempfile.mkstemp(
            prefix="installer_",
            suffix=".json",
            dir=os.path.dirname(path),
        )
        os.close(handle)

        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        os.replace(temporary_path, path)

    def skip_comfy_install(self):
        if self.active_stage:
            return
        self.comfy_skipped = True
        self.comfy_ready = False
        self.write_installation_state("comfyui", "skipped")
        self.show_completion()

    def show_completion(self):
        self.show_page(self.completion_page)

        main_text = "Ready" if self.main_ready else "Unavailable"
        if self.comfy_ready:
            comfy_text = f"Ready ({COMFYUI_VERSION})"
        elif self.comfy_skipped:
            comfy_text = "Skipped"
        else:
            comfy_text = "Unavailable"

        prerequisites = "Detected"
        if self.missing_prerequisites:
            prerequisites = "Not detected: " + ", ".join(self.missing_prerequisites)

        launch_command = self.command_text([self.main_python(), "gui.py"])
        summary = (
            f"Main application environment: {main_text}\n"
            f"ComfyUI backend: {comfy_text}\n"
            f"External system software: {prerequisites}\n\n"
            f"Main environment: {self.main_venv_dir}\n"
            f"ComfyUI environment: {self.comfy_venv_dir}\n\n"
            "To launch the application later, run:\n"
            f"{launch_command}"
        )

        self.completion_status.configure(text="The AI Workstation is ready to launch.")
        self.completion_summary.configure(text=summary)
        self.launch_button.configure(state="normal" if self.main_ready else "disabled")

    def launch_application(self):
        python_path = self.main_python()
        gui_path = os.path.join(self.base_dir, "gui.py")

        if not os.path.isfile(python_path):
            messagebox.showerror(
                "Application environment not found",
                ".venv is missing or incomplete. Run the installer again.",
                parent=self.root,
            )
            return

        if not os.path.isfile(gui_path):
            messagebox.showerror(
                "Application not found",
                "gui.py was not found in the project directory.",
                parent=self.root,
            )
            return

        command = [python_path, gui_path]
        if os.name == "nt":
            subprocess.Popen(
                command,
                cwd=self.base_dir,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            subprocess.Popen(
                command,
                cwd=self.base_dir,
                start_new_session=True,
            )

        self.launch_status.configure(
            text="AI Workstation launched. The installer will remain open until you close it."
        )

    def close_installer(self):
        if self.active_stage and self.current_process is not None:
            should_close = messagebox.askyesno(
                "Installation in progress",
                "An installation command is still running. Stop it and close the installer?",
                parent=self.root,
            )
            if not should_close:
                return
            self.current_process.terminate()

        self.root.destroy()

    def handle_main_finished(self, success, message):
        self.active_stage = None
        self.main_progress.stop()

        if success:
            self.main_ready = True
            self.main_status.configure(text="Main application installation complete")
            self.append_text(self.main_log, f"\n{message}\n")
            self.main_action_button.configure(
                text="Next: ComfyUI",
                command=self.continue_to_comfy_page,
                state="normal",
            )
        else:
            self.main_ready = False
            self.main_status.configure(text="Application installation failed")
            self.append_text(self.main_log, f"\n{message}\n")
            self.main_action_button.configure(
                text="Retry",
                command=self.begin_main_install,
                state="normal",
            )
            messagebox.showerror(
                "Installation failed",
                "The main environment could not be installed. Review the output and retry.",
                parent=self.root,
            )

    def handle_comfy_finished(self, success, message):
        self.active_stage = None
        self.comfy_progress.stop()
        self.comfy_progress.configure(mode="indeterminate", value=0)

        if success:
            self.comfy_ready = True
            self.comfy_skipped = False
            self.comfy_status.configure(text="ComfyUI installation complete")
            self.append_text(self.comfy_log, f"\n{message}\n")
            self.comfy_action_button.configure(
                text="Continue",
                command=self.show_completion,
                state="normal",
            )
            self.skip_comfy_button.configure(state="disabled")
        else:
            self.comfy_ready = False
            self.comfy_status.configure(text="ComfyUI installation failed")
            self.append_text(
                self.comfy_log,
                f"\n{message}\nThe main AI Workstation remains available.\n",
            )
            self.comfy_action_button.configure(
                text="Retry ComfyUI",
                command=self.begin_comfy_install,
                state="normal",
            )
            self.skip_comfy_button.configure(state="normal")
            messagebox.showwarning(
                "Optional backend unavailable",
                (
                    "ComfyUI could not be installed. You can retry or continue without it. "
                    "The rest of the AI Workstation is unaffected."
                ),
                parent=self.root,
            )

    def poll_output(self):
        while True:
            try:
                event = self.output_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "main_log":
                self.append_text(self.main_log, event[1])
            elif event_type == "main_status":
                self.main_status.configure(text=event[1])
            elif event_type == "main_finished":
                self.handle_main_finished(event[1], event[2])
            elif event_type == "comfy_log":
                self.append_text(self.comfy_log, event[1])
            elif event_type == "comfy_status":
                self.comfy_status.configure(text=event[1])
            elif event_type == "comfy_finished":
                self.handle_comfy_finished(event[1], event[2])
            elif event_type == "comfy_progress_mode":
                self.comfy_progress.stop()
                self.comfy_progress.configure(mode=event[1], value=0)
                if event[1] == "indeterminate":
                    self.comfy_progress.start(12)
            elif event_type == "comfy_progress_value":
                self.comfy_progress.configure(value=event[1])

        self.root.after(100, self.poll_output)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    InstallerApp().run()
