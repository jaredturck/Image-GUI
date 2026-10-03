import os
import platform
import shutil
import subprocess


def open_path(path, reveal=False):
    path = os.path.abspath(os.path.expanduser(path))
    system = platform.system()

    if system == "Darwin":
        command = ["open", "-R", path] if reveal else ["open", path]
        return subprocess.Popen(command)
    if system == "Windows":
        target = os.path.dirname(path) if reveal and os.path.isfile(path) else path
        os.startfile(target)
        return None

    opener = shutil.which("xdg-open")
    if opener is None:
        raise RuntimeError("No supported file manager command was found.")
    target = os.path.dirname(path) if reveal and os.path.isfile(path) else path
    return subprocess.Popen([opener, target])


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
