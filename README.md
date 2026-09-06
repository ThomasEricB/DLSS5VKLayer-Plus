# DLSS5VKLayer

DLSS5VKLayer is a Linux Vulkan layer plus helper service that forwards presented frames to a Windows NGX neural-rendering helper running under Wine or a custom Proton compatibility tool.

This project is experimental. It is intended for local testing and research.

## Features

- Vulkan implicit layer for native Linux games and Proton games.
- Fail-open design: if the helper is missing or neural initialization fails, games continue presenting original frames.
- Windows NGX helper runs under:
  - custom Steam compatibility tools such as Proton-CachyOS, Proton-GE, Wine-GE, and similar tools
  - system Wine with a managed prefix and vendored DXVK 2.7.1
- Qt GUI for live controls:
  - enable/disable neural processing
  - multi-pass rendering
  - intensity, tone, structure, skin structure, and sharpness
  - per-pass overrides
- CLI helper manager:
  - runner discovery
  - start/stop/status
  - diagnostics
  - NVIDIA NGX binary import
- XDG-aware paths:
  - config: `$XDG_CONFIG_HOME/dlssnr/config.ini`
  - runtime/SHM/PID: `$XDG_RUNTIME_DIR/dlssnr/`, fallback `/tmp/dlssnr-$UID/`
  - logs/state: `$XDG_STATE_HOME/dlssnr/`, fallback `~/.local/state/dlssnr/`
  - managed prefix: `~/.local/share/dlssnr/prefix/`

## Requirements

- NVIDIA GPU and NVIDIA driver.
- Vulkan loader.
- Qt 6 for the GUI.
- One of:
  - a custom Steam compatibility tool that bundles DXVK-NVAPI, such as Proton-CachyOS or Proton-GE
  - system Wine for the fallback path

Valve's official Proton releases and Proton Experimental are not targeted as primary runners because they do not bundle the required DXVK-NVAPI stack.

## NVIDIA NGX DLLs

The public package does not include NVIDIA proprietary NGX DLLs.

You must provide the required DLLs yourself, for example:

- `nvngx_dlssnr.dll`
- `nvngx.dll`
- `nvapi64.dll`
- `sl.*.dll`

Import them with:

```bash
dlssnr-helper import-binaries /path/to/dlls
```

or use the GUI import flow if available.

The personal package variant includes these DLLs. Only redistribute the personal variant if you have the rights to do so.

## Install From RPM

Public package:

```bash
sudo rpm -Uvh dist/dlssnr-0.1.0-1.fc44.x86_64.rpm
```

Personal package:

```bash
sudo rpm -Uvh dist/dlssnr-personal-0.1.0-1.fc44.x86_64.rpm
```

`wine` is a recommended package, not a hard dependency, so Proton-only users are not forced to install host Wine.

## Install From Tarball

Extract the tarball:

```bash
tar -xzf dist/dlssnr-0.1.0-linux-x86_64.tar.gz
cd dlssnr-0.1.0-linux-x86_64
```

User install, no root required:

```bash
./install.sh --user
```

System install:

```bash
sudo ./install.sh --system
```

For user installs, make sure `~/.local/bin` is in your `PATH`.

## First Run

Initialize configuration:

```bash
dlssnr-helper init
dlssnr-helper doctor
```

Start the helper:

```bash
dlssnr-helper start
```

Check status:

```bash
dlssnr-helper status
```

Stop the helper:

```bash
dlssnr-helper stop
```

GUI:

```bash
dlssnr-gui
```

## Game Usage

Launch a game with the layer enabled:

```bash
VKLayer_DLSS5=1 ./your_native_game
```

For Steam:

```text
VKLayer_DLSS5=1 %command%
```

The layer is disabled unless `VKLayer_DLSS5=1` is present.

## Runner Discovery

Custom compatibility tools are discovered from:

```text
$XDG_DATA_HOME/Steam/compatibilitytools.d
~/.var/app/com.valvesoftware.Steam/data/Steam/compatibilitytools.d
~/snap/steam/common/.local/share/Steam/compatibilitytools.d
```

List discovered runners:

```bash
dlssnr-helper runners
```

You can also set a custom runner path manually in:

```text
~/.config/dlssnr/config.ini
```

Example:

```ini
runner_type=proton
runner_path=/path/to/compatibilitytools.d/Proton-CachyOS/proton
```

## Diagnostics

Useful commands:

```bash
dlssnr-helper doctor
dlssnr-helper runners
dlssnr-helper status
dlssnr-helper config
```

If you previously used an older build, remove stale shared-memory files:

```bash
rm -f /tmp/dlssnr_shm.bin
rm -f "${XDG_RUNTIME_DIR:-/tmp/dlssnr-$UID}/dlssnr/shm.bin"
```

Current builds use the XDG runtime path by default, so games and the helper should not need `DLSSNR_SHM` set manually.

Logs are written to the XDG state directory:

```text
$XDG_STATE_HOME/dlssnr/helper.log
```

Fallback:

```text
~/.local/state/dlssnr/helper.log
```

## Building From Source

Install build dependencies:

- `gcc-c++`
- `mingw64-gcc-c++`
- `qt6-qtbase-devel`
- `vulkan-loader-devel` or equivalent Vulkan headers, though Vulkan headers are vendored

Build:

```bash
./build.sh
```

Outputs:

```text
build/layer/libVkLayer_NV_dlssnr.so
build/dlssnr_helper.exe
build/runner_probe
build/gui/dlssnr_gui
```

## Packaging

Build public and personal tarballs plus RPMs:

```bash
./packaging/make-dist.sh
```

Artifacts are written to `dist/`.

The public package does not include NVIDIA DLLs. The personal package does.

## Uninstall

RPM:

```bash
sudo dnf remove dlssnr
```

or:

```bash
sudo dnf remove dlssnr-personal
```

Tarball user install:

```bash
./uninstall.sh --user
```

Tarball system install:

```bash
sudo ./uninstall.sh --system
```

Add `--purge` to also remove user config, state, runtime data, and the managed prefix.

## Important Notes

- The helper and game run in separate processes, so frames cross a GPU/CPU/GPU path. This is not a zero-copy integration.
- The layer currently assumes a present-time swapchain layout that works for the tested games and emulators. Some games may need layout handling work.
- Steam runtime issues may require per-game or per-runtime debugging.
- Do not use the personal package publicly unless you are certain you may redistribute the bundled NVIDIA binaries.