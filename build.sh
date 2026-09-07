#!/usr/bin/env bash
# Build the Linux Vulkan layer (.so) + Windows DLSSNR helper (.exe).
#
# Packaging is opt-in:
#   ./build.sh --tar    build everything, then stage the .tar.gz tarballs (public + personal)
#   ./build.sh --rpm    build everything, then build the RPMs (public + personal)
#   ./build.sh --dist   build everything, then tarballs and RPMs
# With no flag nothing is packaged, exactly as before.
set -euo pipefail
cd "$(dirname "$0")"

DO_TAR=0
DO_RPM=0
for arg in "$@"; do
    case "$arg" in
        --tar)  DO_TAR=1 ;;
        --rpm)  DO_RPM=1 ;;
        --dist) DO_TAR=1; DO_RPM=1 ;;
        -h|--help)
            echo "usage: $0 [--tar|--rpm|--dist]"
            echo "  --tar   build + .tar.gz tarballs only"
            echo "  --rpm   build + RPMs only"
            echo "  --dist  build + tarballs + RPMs"
            echo "  (no flag: build only, package nothing)"
            exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
    esac
done

mkdir -p build/layer

# Vendored SDK headers (proven with MinGW by standalone_runner); /usr/include only
# as a last-resort search dir (-idirafter) so glibc stdint.h can't shadow the
# MinGW CRT (LP64 vs LLP64 uintptr_t conflict + VkPipelineStageFlagBits2 truncation).
VK_INC="-I standalone_runner/third_party"

LAYER_SRC="layer_linux/src/layer.cpp
    layer_linux/src/shader_vk.cpp
    layer_linux/src/dlssnr_pass.cpp
    layer_linux/src/composition.cpp
    layer_linux/src/capture.cpp
    layer_linux/src/scaler_vk.cpp
    layer_linux/src/hotkey.cpp"

echo "[1/5] Linux layer .so (64-bit)"
# Keep the loader's entry points, and nothing else, visible -- and bind our own references to them
# locally, so a game that links libvulkan.so.1 cannot preempt them with the loader's copies. Without
# this the layer hands the loader the loader's own vkGetInstanceProcAddr and the chain calls itself.
LAYER_LDFLAGS="-Wl,--version-script=layer_linux/dlssnr.map -Wl,-Bsymbolic"

g++ -O2 -std=c++17 -shared -fPIC -Wall $VK_INC -I layer_linux/src \
    $LAYER_SRC -o build/layer/libVkLayer_NV_dlssnr.so $LAYER_LDFLAGS -lpthread

# A 32-bit layer as well, for 32-bit Vulkan games.
#
# The Vulkan loader can only load a layer of the process's own word size, so a 32-bit game -- an
# OpenGL title running on zink, or a Windows game under a Proton that is not in WoW64 mode -- sees
# nothing at all unless there is a 32-bit build of the layer for it to load. The shared-memory header
# is laid out identically under both ABIs, so the 32-bit layer talks to the same 64-bit helper.
#
# Skipped rather than fatal when there is no 32-bit toolchain: most people do not need it, and a
# missing multilib compiler should not stop the 64-bit build.
mkdir -p build/layer32
if echo 'int main(){return 0;}' | g++ -m32 -x c++ - -o /dev/null 2>/dev/null; then
    echo "[1/5] Linux layer .so (32-bit)"
    g++ -m32 -O2 -std=c++17 -shared -fPIC -Wall -DDLSSNR_LAYER_32 $VK_INC -I layer_linux/src \
        $LAYER_SRC -o build/layer32/libVkLayer_NV_dlssnr.so $LAYER_LDFLAGS -lpthread
else
    echo "[1/5] 32-bit layer skipped: no 'g++ -m32' (install a multilib toolchain for 32-bit games)"
    rm -f build/layer32/libVkLayer_NV_dlssnr.so
fi

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
    # Distro Qt builds disagree on the visibility of the meta-object data symbols
    # (QSpinBox::staticMetaObject and friends): Fedora's are default, CachyOS/Arch's are
    # protected. GCC bakes copy-relocation references against them even under -fPIE, and
    # glibc 2.41+ refuses to copy-relocate a protected symbol -- the binary dies at exec
    # with GNU_PROPERTY_1_NEEDED_INDIRECT_EXTERN_ACCESS. clang always reaches those symbols
    # through the GOT, and -z nocopyreloc pins the GOT to the library's definition, so the
    # GUI loads against either flavour of Qt. Fall back to g++ when clang is missing or too
    # old for the linker flags; the GUI then only loads on default-visibility Qt.
    GUI_CXX=""
    GUI_QMAKE_FLAGS=()
    GUI_LINK_FLAGS=()
    if command -v clang++ >/dev/null 2>&1 &&
       echo 'int main(){}' | clang++ -x c++ - -pie -Wl,-z,nocopyreloc -Wl,-z,indirect-extern-access -o /dev/null 2>/dev/null; then
        GUI_CXX=clang++
        GUI_QMAKE_FLAGS=("QMAKE_CXXFLAGS+=-fPIE" "QMAKE_LFLAGS+=-pie -Wl,-z,nocopyreloc -Wl,-z,indirect-extern-access")
        GUI_LINK_FLAGS=(-pie -Wl,-z,nocopyreloc -Wl,-z,indirect-extern-access)
    else
        echo "  warning: no clang++ with '-z nocopyreloc'; the GUI will carry copy relocations" >&2
        echo "           and may not load on distros whose Qt uses protected visibility" >&2
    fi

    mkdir -p build/gui
    # qmake does not notice a compiler swap, so drop the objects when the chosen compiler
    # or flags change -- g++ objects carry the PC32 references nocopyreloc refuses.
    GUI_STAMP="$GUI_CXX ${GUI_QMAKE_FLAGS[*]-}"
    if [ "$(cat build/gui/.cxx-stamp 2>/dev/null)" != "$GUI_STAMP" ]; then
        rm -f build/gui/*.o build/gui/dlssnr_gui
        printf '%s' "$GUI_STAMP" > build/gui/.cxx-stamp
    fi
    (cd build/gui && qmake6 ../../gui/dlssnr_gui.pro ${GUI_CXX:+QMAKE_CXX=$GUI_CXX} "${GUI_QMAKE_FLAGS[@]-}" && make -j"$(nproc)")

    # The binder's regression test. Offscreen, so it needs no display; run it with
    #   QT_QPA_PLATFORM=offscreen ./build/binder_test
    if pkg-config --exists Qt6Widgets; then
        # Qt6 dropped QT_INSTALL_LIBEXECDIR, so probe PATH and the distro layouts instead.
        MOC="$(command -v moc 2>/dev/null || true)"
        if [ ! -x "$MOC" ]; then
            for c in /usr/lib64/qt6/libexec/moc /usr/lib/qt6/libexec/moc /usr/lib/qt6/moc; do
                if [ -x "$c" ]; then MOC="$c"; break; fi
            done
        fi
        if [ -x "$MOC" ]; then
            "$MOC" -I gui -I common gui/shm_binder.h -o build/gui/moc_binder_test.cpp
            "${GUI_CXX:-g++}" -O2 -std=c++17 -fPIC -I gui -I common $(pkg-config --cflags Qt6Widgets) \
                test_gui/binder_test.cpp gui/shm_binder.cpp build/gui/moc_binder_test.cpp \
                $(pkg-config --libs Qt6Widgets) "${GUI_LINK_FLAGS[@]-}" -o build/binder_test
        else
            echo "  binder_test skipped: moc not found"
        fi
    fi
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

    # One manifest per architecture, each naming its own library. The loader reads every manifest in
    # the directory and skips the one whose library is the wrong word size, which is exactly how Steam
    # ships its own overlay (steamoverlay_i386.json beside steamoverlay_x86_64.json).
    # library_arch lets the loader skip the manifest for the other word size by reading it, rather
    # than by trying to dlopen the library and reporting "wrong ELF class" for every process.
    sed -e "s#./libVkLayer_NV_dlssnr.so#$PWD/build/layer/libVkLayer_NV_dlssnr.so#" \
        -e 's#"implementation_version"#"library_arch": "64",\n    "implementation_version"#' \
        layer_linux/manifest/VK_LAYER_NV_dlssnr.json > "$IMPLICIT_DIR/VK_LAYER_NV_dlssnr.x86_64.json"

    if [ -f build/layer32/libVkLayer_NV_dlssnr.so ]; then
        sed -e "s#./libVkLayer_NV_dlssnr.so#$PWD/build/layer32/libVkLayer_NV_dlssnr.so#" \
            -e 's#"VK_LAYER_NV_dlssnr"#"VK_LAYER_NV_dlssnr_32"#' \
            -e 's#"implementation_version"#"library_arch": "32",\n    "implementation_version"#' \
            layer_linux/manifest/VK_LAYER_NV_dlssnr.json > "$IMPLICIT_DIR/VK_LAYER_NV_dlssnr.i686.json"
    else
        rm -f "$IMPLICIT_DIR/VK_LAYER_NV_dlssnr.i686.json"
    fi

    # The single-architecture manifest earlier builds installed. Left in place it would load the
    # 64-bit layer a second time under a different name, which the layer now refuses but still logs.
    rm -f "$IMPLICIT_DIR/VK_LAYER_NV_dlssnr.json"
fi

echo "built:"
echo "  build/layer/libVkLayer_NV_dlssnr.so   (64-bit)"
echo "  build/layer32/libVkLayer_NV_dlssnr.so (32-bit, if a multilib toolchain is present)"
echo "  build/dlssnr_helper.exe"
echo "  build/smoke.exe"
echo "  build/runner_probe"
echo "  build/dlssnr-shmctl"
echo "  build/gui/dlssnr_gui (if Qt6/qmake6 is available)"
echo "  build/binder_test    (QT_QPA_PLATFORM=offscreen ./build/binder_test)"

if [ "$DO_TAR" = 1 ] || [ "$DO_RPM" = 1 ]; then
    echo
    if [ "$DO_TAR" = 1 ] && [ "$DO_RPM" = 1 ]; then MODE=both
    elif [ "$DO_TAR" = 1 ]; then MODE=tar
    else MODE=rpm; fi
    ./packaging/make-dist.sh "$MODE"
fi

echo
echo "use it:"
echo "  GUI: ./build/gui/dlssnr_gui          # start/stop helper + live settings"
echo "  CLI: ./run_helper.sh                 # start the neural helper"
echo "  game: launch with VKLayer_DLSS5=1"
echo "     e.g.  VKLayer_DLSS5=1 %command%   (Steam launch options)"
echo "     or    VKLayer_DLSS5=1 ./your_native_game"
echo "  CLI stop helper: ./run_helper.sh stop"
echo "  optional: DLSSNR_Passes=2 ./run_helper.sh"
