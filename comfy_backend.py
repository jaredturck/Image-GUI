import atexit
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

_comfy_process = None
_comfy_log_file = None
_comfy_url = None

def project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def comfy_python_path():
    environment_dir = os.path.join(project_dir(), ".comfy_venv")
    if os.name == "nt":
        return os.path.join(environment_dir, "Scripts", "python.exe")
    return os.path.join(environment_dir, "bin", "python")


def available_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def server_is_ready(url):
    try:
        with urllib.request.urlopen(f"{url}system_stats", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_server(url, process, timeout_seconds=180):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        if server_is_ready(url):
            return True
        time.sleep(1)
    return False



def load_runtime_nodes(url):
    try:
        from comfy_script.runtime import load, nodes
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ComfyScript is not installed in the main application environment. Run install.py and complete the ComfyUI stage."
        ) from error

    load(url)
    return nodes

def connect_comfy_backend(comfyui_dir):
    global _comfy_process
    global _comfy_log_file
    global _comfy_url

    if _comfy_url and _comfy_process and _comfy_process.poll() is None:
        return load_runtime_nodes(_comfy_url)

    comfyui_dir = os.path.abspath(os.path.expanduser(comfyui_dir or ""))
    main_path = os.path.join(comfyui_dir, "main.py")
    python_path = comfy_python_path()

    if not os.path.isfile(main_path):
        raise RuntimeError(
            "The managed ComfyUI installation is missing. Run install.py and complete the ComfyUI stage."
        )

    if not os.path.isfile(python_path):
        raise RuntimeError(
            "The managed ComfyUI Python environment is missing. Run install.py and complete the ComfyUI stage."
        )

    port = available_port()
    _comfy_url = f"http://127.0.0.1:{port}/"

    config_dir = os.path.join(project_dir(), "config")
    os.makedirs(config_dir, exist_ok=True)
    log_path = os.path.join(config_dir, "comfyui_server.log")
    _comfy_log_file = open(log_path, "a", encoding="utf-8")

    command = [
        python_path,
        main_path,
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-auto-launch",
        "--disable-all-custom-nodes",
    ]

    _comfy_process = subprocess.Popen(
        command,
        cwd=comfyui_dir,
        stdout=_comfy_log_file,
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(_comfy_url, _comfy_process):
        shutdown_comfy_backend()
        raise RuntimeError(
            f"ComfyUI did not start successfully. Review {log_path} for the startup output."
        )

    try:
        return load_runtime_nodes(_comfy_url)
    except RuntimeError:
        shutdown_comfy_backend()
        raise


def shutdown_comfy_backend():
    global _comfy_process
    global _comfy_log_file
    global _comfy_url

    if _comfy_process and _comfy_process.poll() is None:
        _comfy_process.terminate()
        try:
            _comfy_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _comfy_process.kill()
            _comfy_process.wait(timeout=5)

    if _comfy_log_file:
        _comfy_log_file.close()

    _comfy_process = None
    _comfy_log_file = None
    _comfy_url = None


atexit.register(shutdown_comfy_backend)
