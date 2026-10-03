#!/bin/bash
set -euo pipefail

APP_CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
APP_BUNDLE="$(dirname "$APP_CONTENTS")"
RESOURCES="$APP_CONTENTS/Resources"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13"

show_error() {
    /usr/bin/osascript -e "display dialog \"$1\" with title \"Image GUI\" buttons {\"OK\"} default button \"OK\" with icon stop"
}

if [ "$(/usr/bin/uname -m)" != "arm64" ]; then
    show_error "This release requires an Apple Silicon Mac."
    exit 1
fi

case "$APP_BUNDLE" in
    /Volumes/*)
        show_error "Drag Image GUI to Applications, eject the disk image, then open Image GUI from Applications."
        exit 1
        ;;
esac

if [ ! -x "$PYTHON_BIN" ]; then
    /usr/bin/open -W -a Installer "$RESOURCES/python-3.13.16-macos11.pkg"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    show_error "Python was not installed. Reopen Image GUI and complete the official Python installer."
    exit 1
fi

export IMAGE_GUI_PACKAGED=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
unset PYTHONPATH

exec "$PYTHON_BIN" "$RESOURCES/app/install.py" --app-launch
