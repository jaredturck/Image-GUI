import ctypes.util
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk


class InstallerApp:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.backend = self.detect_backend()
        self.output_queue = queue.Queue()
        self.installing = False
        self.missing_prerequisites = []

        self.root = tk.Tk()
        self.root.title("AI Workstation Installer")
        self.root.geometry("820x620")
        self.root.minsize(720, 520)

        self.install_vllm = tk.BooleanVar(value=self.backend == "cuda" and platform.system() == "Linux")
        self.create_ui()
        self.root.after(100, self.poll_output)

    def detect_backend(self):
        if platform.system() == "Darwin":
            return "macos"
        if shutil.which("nvidia-smi"):
            return "cuda"
        return "cpu"

    def create_ui(self):
        self.install_page = ttk.Frame(self.root, padding=18)
        self.install_page.pack(fill="both", expand=True)
        self.install_page.columnconfigure(0, weight=1)
        self.install_page.rowconfigure(4, weight=1)

        title = ttk.Label(self.install_page, text="AI Workstation Installer", font=("TkDefaultFont", 18, "bold"))
        title.grid(row=0, column=0, sticky="w")

        description = ttk.Label(
            self.install_page,
            text=(
                "This installer selects the Python requirements for the detected backend and installs them with pip. "
                "It does not install GPU drivers, FFmpeg, PortAudio, or other operating-system packages."
            ),
            wraplength=760,
            justify="left",
        )
        description.grid(row=1, column=0, sticky="ew", pady=(8, 12))

        info = ttk.LabelFrame(self.install_page, text="Detected environment", padding=12)
        info.grid(row=2, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)

        ttk.Label(info, text="Operating system:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, text=f"{platform.system()} {platform.release()}").grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="Python:").grid(row=1, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, text=sys.version.split()[0]).grid(row=1, column=1, sticky="w")
        ttk.Label(info, text="Selected backend:").grid(row=2, column=0, sticky="w", padx=(0, 10))
        ttk.Label(info, text=self.backend.upper()).grid(row=2, column=1, sticky="w")

        options = ttk.Frame(self.install_page)
        options.grid(row=3, column=0, sticky="ew", pady=12)
        vllm = ttk.Checkbutton(
            options,
            text="Install optional vLLM support (CUDA Linux only)",
            variable=self.install_vllm,
        )
        vllm.pack(side="left")
        if self.backend != "cuda" or platform.system() != "Linux":
            vllm.state(["disabled"])

        log_frame = ttk.LabelFrame(self.install_page, text="Installation output", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        self.progress = ttk.Progressbar(self.install_page, mode="indeterminate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        bottom = ttk.Frame(self.install_page)
        bottom.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        bottom.columnconfigure(0, weight=1)

        self.status = ttk.Label(bottom, text="Ready")
        self.status.grid(row=0, column=0, sticky="w")

        self.install_button = ttk.Button(bottom, text="Install", command=self.start_install)
        self.install_button.grid(row=0, column=1, padx=(8, 0))

        self.close_button = ttk.Button(bottom, text="Close", command=self.root.destroy)
        self.close_button.grid(row=0, column=2, padx=(8, 0))

        self.create_completion_page()
        self.show_prerequisite_status()

    def create_completion_page(self):
        self.completion_page = ttk.Frame(self.root, padding=24)
        self.completion_page.columnconfigure(0, weight=1)
        self.completion_page.rowconfigure(2, weight=1)

        title = ttk.Label(
            self.completion_page,
            text="Installation complete",
            font=("TkDefaultFont", 20, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        description = ttk.Label(
            self.completion_page,
            text=(
                "The Python dependencies have finished installing. You can now launch the AI Workstation. "
                "The installer will remain open until you close it."
            ),
            wraplength=740,
            justify="left",
        )
        description.grid(row=1, column=0, sticky="ew", pady=(12, 18))

        summary_frame = ttk.LabelFrame(self.completion_page, text="Installation summary", padding=14)
        summary_frame.grid(row=2, column=0, sticky="nsew")
        summary_frame.columnconfigure(0, weight=1)

        self.completion_summary = ttk.Label(summary_frame, text="", wraplength=700, justify="left")
        self.completion_summary.grid(row=0, column=0, sticky="nw")

        bottom = ttk.Frame(self.completion_page)
        bottom.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        bottom.columnconfigure(0, weight=1)

        self.launch_status = ttk.Label(bottom, text="Ready to launch")
        self.launch_status.grid(row=0, column=0, sticky="w")

        self.launch_button = ttk.Button(bottom, text="Launch AI Workstation", command=self.launch_application)
        self.launch_button.grid(row=0, column=1, padx=(8, 0))

        close = ttk.Button(bottom, text="Close", command=self.root.destroy)
        close.grid(row=0, column=2, padx=(8, 0))

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def show_prerequisite_status(self):
        missing = []
        if not shutil.which("ffmpeg"):
            missing.append("FFmpeg")
        if not ctypes.util.find_library("portaudio"):
            missing.append("PortAudio (required by SoundDevice)")
        if self.backend == "cuda" and not shutil.which("nvidia-smi"):
            missing.append("NVIDIA driver tools")

        self.missing_prerequisites = missing
        if missing:
            self.append_log("External prerequisites not detected:\n")
            for item in missing:
                self.append_log(f"  - {item}\n")
            self.append_log("Install these with your operating system's package manager. The Python installation can continue.\n\n")
        else:
            self.append_log("External prerequisite checks passed.\n\n")

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

    def start_install(self):
        if self.installing:
            return

        if sys.version_info < (3, 10):
            messagebox.showerror(
                "Python version",
                "Python 3.10 or newer is required by the current model and quantization libraries.",
                parent=self.root,
            )
            return

        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
        )
        if pip_check.returncode != 0:
            messagebox.showerror(
                "pip unavailable",
                "pip is not available for this Python installation. Install or enable pip, then run the installer again.",
                parent=self.root,
            )
            return

        missing_files = [name for name in self.requirements_files() if not os.path.isfile(os.path.join(self.base_dir, name))]
        if missing_files:
            messagebox.showerror("Missing requirements", "Missing files:\n" + "\n".join(missing_files), parent=self.root)
            return

        self.installing = True
        self.install_button.configure(state="disabled")
        self.status.configure(text="Preparing installation...")
        self.progress.start(12)
        thread = threading.Thread(target=self.install_worker, daemon=True)
        thread.start()

    def install_worker(self):
        success = True
        requirement_files = self.requirements_files()
        total_files = len(requirement_files)

        for index, requirement_file in enumerate(requirement_files, start=1):
            self.output_queue.put(("status", f"Installing {requirement_file} ({index} of {total_files})..."))
            path = os.path.join(self.base_dir, requirement_file)
            command = [sys.executable, "-m", "pip", "install", "-r", path]
            self.output_queue.put(("log", f"\n$ {' '.join(command)}\n"))

            process = subprocess.Popen(
                command,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                self.output_queue.put(("log", line))
            return_code = process.wait()
            if return_code != 0:
                success = False
                self.output_queue.put(("log", f"Installation stopped: {requirement_file} returned {return_code}.\n"))
                break

        self.output_queue.put(("finished", success))

    def show_completion_page(self):
        self.install_page.pack_forget()

        if self.missing_prerequisites:
            missing_text = "\n".join(f"• {item}" for item in self.missing_prerequisites)
            summary = (
                "Python dependency installation completed successfully.\n\n"
                "The following optional system dependencies were not detected:\n"
                f"{missing_text}\n\n"
                "Install them with your operating system's package manager when you need the related features."
            )
        else:
            summary = (
                "Python dependency installation completed successfully.\n\n"
                "All checked external prerequisites were detected."
            )

        self.completion_summary.configure(text=summary)
        self.completion_page.pack(fill="both", expand=True)

    def launch_application(self):
        gui_path = os.path.join(self.base_dir, "gui.py")
        if not os.path.isfile(gui_path):
            messagebox.showerror(
                "Application not found",
                "gui.py was not found in the project directory.",
                parent=self.root,
            )
            return

        command = [sys.executable, gui_path]
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

        self.launch_status.configure(text="AI Workstation launched. You may close the installer.")

    def poll_output(self):
        while True:
            try:
                event = self.output_queue.get_nowait()
            except queue.Empty:
                break

            if event[0] == "log":
                self.append_log(event[1])
            elif event[0] == "status":
                self.status.configure(text=event[1])
            elif event[0] == "finished":
                self.installing = False
                self.progress.stop()
                if event[1]:
                    self.status.configure(text="Installation complete")
                    self.append_log("\nPython dependency installation completed. Click Continue to finish.\n")
                    self.install_button.configure(
                        text="Continue",
                        command=self.show_completion_page,
                        state="normal",
                    )
                else:
                    self.status.configure(text="Installation failed")
                    self.install_button.configure(text="Install", command=self.start_install, state="normal")
                    messagebox.showerror(
                        "Installation failed",
                        "pip reported an error. Review the installation output for details.",
                        parent=self.root,
                    )

        self.root.after(100, self.poll_output)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    InstallerApp().run()
