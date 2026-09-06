#!/usr/bin/env bash
# Build the Linux Vulkan layer (.so) + Windows DLSSNR helper (.exe).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/layer

# Vendored SDK headers (proven with MinGW by standalone_runner); /usr/include only
# as a last-resort search dir (-idirafter) so glibc stdint.h can't shadow the
# MinGW CRT (LP64 vs LLP64 uintptr_t conflict + VkPipelineStageFlagBits2 truncation).
VK_INC="-I standalone_runner/third_party"

echo "[1/5] Linux layer .so"
g++ -O2 -std=c++17 -shared -fPIC -Wall $VK_INC -I layer_linux/src \
    layer_linux/src/layer.cpp \
    layer_linux/src/shader_vk.cpp \
    layer_linux/src/dlssnr_pass.cpp \
    layer_linux/src/composition.cpp \
    -o build/layer/libVkLayer_NV_dlssnr.so -lpthread
cp layer_linux/manifest/VK_LAYER_NV_dlssnr.json build/layer/

echo "[2/5] Windows helper .exe"
x86_64-w64-mingw32-g++ -O2 -std=c++17 -static -mwindows -Wall $VK_INC -idirafter /usr/include -I core \
    helper/main.cpp core/ngx_snippet.cpp core/guard.cpp -o build/dlssnr_helper.exe

echo "[3/5] smoke .exe"
x86_64-w64-mingw32-g++ -O2 -std=c++17 -static $VK_INC -idirafter /usr/include \
    test_layer/smoke.cpp -o build/smoke.exe

echo "[4/5] runner probe + shm control"
g++ -O2 -std=c++17 -Wall common/runner_discovery.cpp tools/runner_probe.cpp -o build/runner_probe
g++ -O2 -std=c++17 -Wall tools/shmctl.cpp -o build/dlssnr-shmctl

echo "[5/5] Qt helper GUI"
if command -v qmake6 >/dev/null 2>&1; then
    mkdir -p build/gui
    (cd build/gui && qmake6 ../../gui/dlssnr_gui.pro && make -j"$(nproc)")
else
    echo "  skipped: qmake6 not found"
fi

# DXVK (which backs wine's vulkan-1.dll) needs a dxvk.conf in the helper's CWD
# advertising the NVIDIA GPU, otherwise the snippet's NvAPI display enumeration
# returns 0 displays and VULKAN_CreateFeature(18) fails with 0xbad00001.
GPU_ID="$(lspci -d 10de: -n 2>/dev/null | awk '/0300/{for(i=1;i<=NF;i++) if($i ~ /^[0-9a-f]{4}:[0-9a-f]{4}$/){print $i; exit}}')"
VEN="${GPU_ID%%:*}"; DEV="${GPU_ID##*:}"
if [ -n "$VEN" ] && [ -n "$DEV" ]; then
    printf 'dxgi.customVendorId = %s\ndxgi.customDeviceId = %s\n' "$VEN" "$DEV" > build/dxvk.conf
else
    : > build/dxvk.conf
    echo "warning: no NVIDIA GPU detected; build/dxvk.conf left empty" >&2
fi

# Install the layer as an implicit layer so the loader (native + winevulkan)
# auto-loads it when VKLayer_DLSS5=1. winevulkan ignores VK_LAYER_PATH, so the
# manifest must live in the implicit-layer dir with an ABSOLUTE library_path.
if [ "${DLSSNR_SKIP_MANIFEST_INSTALL:-0}" != "1" ]; then
    IMPLICIT_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/vulkan/implicit_layer.d"
    mkdir -p "$IMPLICIT_DIR"
    sed "s#./libVkLayer_NV_dlssnr.so#$PWD/build/layer/libVkLayer_NV_dlssnr.so#" \
        layer_linux/manifest/VK_LAYER_NV_dlssnr.json > "$IMPLICIT_DIR/VK_LAYER_NV_dlssnr.json"
fi

echo "built:"
echo "  build/layer/libVkLayer_NV_dlssnr.so (+ installed implicit manifest)"
echo "  build/dlssnr_helper.exe"
echo "  build/smoke.exe"
echo "  build/runner_probe"
echo "  build/dlssnr-shmctl"
echo "  build/gui/dlssnr_gui (if Qt6/qmake6 is available)"
echo
echo "use it:"
echo "  GUI: ./build/gui/dlssnr_gui          # start/stop helper + live settings"
echo "  CLI: ./run_helper.sh                 # start the neural helper"
echo "  game: launch with VKLayer_DLSS5=1"
echo "     e.g.  VKLayer_DLSS5=1 %command%   (Steam launch options)"
echo "     or    VKLayer_DLSS5=1 ./your_native_game"
echo "  CLI stop helper: ./run_helper.sh stop"
echo "  optional: DLSSNR_Passes=2 ./run_helper.sh"
