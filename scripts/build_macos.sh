#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${IMAGE_GUI_VERSION:-0.1.0}"
PYTHON_VERSION="3.13.16"
PYTHON_PACKAGE="python-${PYTHON_VERSION}-macos11.pkg"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_PACKAGE}"
PYTHON_SHA256="30666509020b4da0dd8bc2e773255f34d76b7bb80b66960a928d5f6daa0192d7"
BUILD_DIR="$ROOT_DIR/build/macos"
CACHE_DIR="$BUILD_DIR/cache"
STAGE_DIR="$BUILD_DIR/dmg-root"
APP_DIR="$STAGE_DIR/Image GUI.app"
CONTENTS_DIR="$APP_DIR/Contents"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
APP_SOURCE_DIR="$RESOURCES_DIR/app"
DIST_DIR="$ROOT_DIR/dist"
DMG_PATH="$DIST_DIR/Image-GUI-${VERSION}-macOS-Apple-Silicon.dmg"

SOURCE_FILES=(
    app_config.py
    base_gui.py
    cachelight.py
    chat_gui.py
    gui.py
    hardware_detection.py
    hardware_planner.py
    img_editor.py
    install.py
    model_gui.py
    model_loading.py
    model_registry.py
    plan_history.py
    planner_protocol.py
    planner_runtime.py
    platform_utils.py
    requirements-lock-macos.txt
    vram_estimation.py
    CACHELITE_LICENSE
)

check_inputs() {
    local missing=0
    for relative_path in "${SOURCE_FILES[@]}" packaging/macos/Info.plist packaging/macos/launcher.sh; do
        if [ ! -f "$ROOT_DIR/$relative_path" ]; then
            echo "Missing build input: $relative_path" >&2
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        exit 1
    fi
}

check_inputs

if [ "${1:-}" = "--check" ]; then
    echo "macOS packaging inputs are complete."
    exit 0
fi

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "The DMG must be built on an Apple Silicon Mac." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR" "$DIST_DIR"
if [ ! -f "$CACHE_DIR/$PYTHON_PACKAGE" ]; then
    /usr/bin/curl --fail --location --proto '=https' --tlsv1.2 \
        "$PYTHON_URL" --output "$CACHE_DIR/$PYTHON_PACKAGE"
fi

ACTUAL_SHA256="$(/usr/bin/shasum -a 256 "$CACHE_DIR/$PYTHON_PACKAGE" | /usr/bin/awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$PYTHON_SHA256" ]; then
    echo "Official Python package checksum mismatch." >&2
    exit 1
fi

rm -rf "$STAGE_DIR"
mkdir -p "$CONTENTS_DIR/MacOS" "$APP_SOURCE_DIR"

/usr/bin/sed "s/__VERSION__/$VERSION/g" \
    "$ROOT_DIR/packaging/macos/Info.plist" > "$CONTENTS_DIR/Info.plist"
/usr/bin/ditto "$ROOT_DIR/packaging/macos/launcher.sh" "$CONTENTS_DIR/MacOS/Image GUI"
/bin/chmod 755 "$CONTENTS_DIR/MacOS/Image GUI"

for relative_path in "${SOURCE_FILES[@]}"; do
    /usr/bin/ditto "$ROOT_DIR/$relative_path" "$APP_SOURCE_DIR/$relative_path"
done
/usr/bin/ditto "$CACHE_DIR/$PYTHON_PACKAGE" "$RESOURCES_DIR/$PYTHON_PACKAGE"
/bin/ln -s /Applications "$STAGE_DIR/Applications"

/usr/bin/hdiutil create \
    -volname "Image GUI" \
    -srcfolder "$STAGE_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

echo "Created $DMG_PATH"
