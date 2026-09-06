#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="${DLSSNR_VERSION:-0.1.0}"
DIST="dist"
BUILD="${DLSSNR_BUILD_DIR:-build}"
export DLSSNR_SKIP_MANIFEST_INSTALL=1

[ -f "$BUILD/dlssnr_helper.exe" ] || ./build.sh
[ -f "$BUILD/runner_probe" ] || ./build.sh
[ -f "$BUILD/gui/dlssnr_gui" ] || ./build.sh

mkdir -p "$DIST"

stage_variant() {
  local variant="$1"
  local pkg_name="$2"
  local pkg_dir="$DIST/$pkg_name-$VERSION-linux-x86_64"
  local root="$pkg_dir/root"

  rm -rf "$pkg_dir"
  mkdir -p \
    "$root/usr/lib64/dlssnr/layer" \
    "$root/usr/lib64/dlssnr/helper" \
    "$root/usr/lib64/dlssnr/bin" \
    "$root/usr/lib64/dlssnr/dxvk/2.7.1" \
    "$root/usr/bin" \
    "$root/usr/share/vulkan/implicit_layer.d" \
    "$root/usr/share/applications" \
    "$root/usr/share/doc/dlssnr"

  cp "$BUILD/layer/libVkLayer_NV_dlssnr.so" "$root/usr/lib64/dlssnr/layer/"
  cp "$BUILD/dlssnr_helper.exe" "$root/usr/lib64/dlssnr/helper/"
  cp "$BUILD/runner_probe" "$root/usr/lib64/dlssnr/bin/"
  cp "$BUILD/gui/dlssnr_gui" "$root/usr/bin/dlssnr-gui"
  cp dlssnr-helper "$root/usr/bin/dlssnr-helper"
  ln -sf ../lib64/dlssnr/bin/runner_probe "$root/usr/bin/dlssnr-runner-probe"
  cp third_party/dxvk/2.7.1/x64/vulkan-1.dll "$root/usr/lib64/dlssnr/dxvk/2.7.1/"
  cp third_party/dxvk/2.7.1/LICENSE.txt "$root/usr/share/doc/dlssnr/dxvk-license.txt"
  cp packaging/dlssnr.desktop "$root/usr/share/applications/"
  cp packaging/install.sh packaging/uninstall.sh "$pkg_dir/"

  sed "s#./libVkLayer_NV_dlssnr.so#/usr/lib64/dlssnr/layer/libVkLayer_NV_dlssnr.so#" \
    layer_linux/manifest/VK_LAYER_NV_dlssnr.json \
    > "$root/usr/share/vulkan/implicit_layer.d/VK_LAYER_NV_dlssnr.json"

  chmod 755 "$pkg_dir/install.sh" "$pkg_dir/uninstall.sh"
  chmod 755 "$root/usr/bin/dlssnr-helper" "$root/usr/bin/dlssnr-gui"
  chmod 755 "$root/usr/lib64/dlssnr/bin/runner_probe"
  chmod 755 "$root/usr/lib64/dlssnr/layer/libVkLayer_NV_dlssnr.so"
  chmod 755 "$root/usr/lib64/dlssnr/helper/dlssnr_helper.exe"

  if [ "$variant" = "personal" ]; then
    mkdir -p "$root/usr/lib64/dlssnr/helper/binaries"
    cp binaries/*.dll "$root/usr/lib64/dlssnr/helper/binaries/" 2>/dev/null || true
    cp binaries/*.license.txt "$root/usr/lib64/dlssnr/helper/binaries/" 2>/dev/null || true
    chmod 644 "$root/usr/lib64/dlssnr/helper/binaries/"* 2>/dev/null || true
  fi

  tar -C "$DIST" -czf "$DIST/$pkg_name-$VERSION-linux-x86_64.tar.gz" "$pkg_name-$VERSION-linux-x86_64"
  echo "built $DIST/$pkg_name-$VERSION-linux-x86_64.tar.gz"
}

build_rpm() {
  local spec="$1"
  local topdir="$PWD/$DIST/rpmbuild"
  mkdir -p "$topdir"
  rpmbuild --define "_topdir $topdir" --define "_sourcedir $PWD/$DIST" -bb "$spec"
  cp "$topdir"/RPMS/x86_64/*.rpm "$DIST/" 2>/dev/null || true
}

stage_variant public dlssnr
stage_variant personal dlssnr-personal

build_rpm packaging/dlssnr.spec
build_rpm packaging/dlssnr-personal.spec

echo
echo "artifacts:"
ls -1 "$DIST"/dlssnr-*.tar.gz "$DIST"/dlssnr-*.rpm 2>/dev/null || true