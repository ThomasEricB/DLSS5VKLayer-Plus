%global debug_package %{nil}
%global _enable_debug_packages 0
%global _include_debuginfo_sources 0
%global pkg_release 4

Name:           dlssnr
Version:        0.2.1
Release:        %{pkg_release}%{?dist}
Summary:        DLSS5 Neural Rendering Vulkan layer and helper
License:        MIT
Source0:        %{name}-%{version}-%{pkg_release}-linux-x86_64.tar.gz

BuildArch:      x86_64

Requires:       bash
Requires:       vulkan-loader
Requires:       qt6-qtbase
Requires:       pciutils
Recommends:     wine

%description
DLSS5VKLayer installs a Vulkan implicit layer that forwards presented frames
to a Windows NGX helper running under Wine or a custom Proton compatibility
tool. This public package does not include NVIDIA proprietary NGX DLLs.

%prep
%setup -q -n %{name}-%{version}-%{pkg_release}-linux-x86_64

%build
# Prebuilt binary payload.

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}
cp -a root/usr %{buildroot}/usr

%files
%defattr(-,root,root,-)
%{_libdir}/dlssnr
%{_bindir}/dlssnr-helper
%{_bindir}/dlssnr-gui
%{_bindir}/dlssnr-runner-probe
%{_datadir}/vulkan/implicit_layer.d/VK_LAYER_NV_dlssnr.json
%{_datadir}/applications/dlssnr.desktop
%doc %{_datadir}/doc/dlssnr/dxvk-license.txt

%changelog
* Sun Sep 06 2026 DLSS5VKLayer - 0.2.1-4
- Offload NVOF MVec post-processing to the GPU: compute pass decodes SFIXED5 flow, applies a deadzone and upscales the grid into `DLSSNR.MVec` with no host readback.
- Add `DLSSNR_MVEC_DEADZONE` (default 0.5 px) and `DLSSNR_MVEC_COMPUTE=0` kill switch; static geometry now yields exactly zero motion vectors.
- Overlap prep/flow/post command buffers across graphics and optical-flow queues via fence ring + semaphores; fix fence-ring slot reuse that caused device loss.
- Read NVOF output in-place through a `R16G16_UINT` storage view; image-to-image copies out of SFIXED5 flow output fault the driver.
- Set `TOP_OF_PIPE` semaphore wait stages (ALL_COMMANDS is invalid on the optical-flow-only queue).
- Add HDR/SDR tonemap hint, pre-exposure parameters and DoSharpening/AutoExposure feature flags at create; route Sharpness per evaluate.

* Sun Sep 06 2026 DLSS5VKLayer - 0.2.1-3
- Fix runtime synthetic motion-vector re-enable so OFF -> ON resets NVOF history and DLSSNR temporal state immediately.
- Clear stale MVec buffers and re-prime optical-flow history on motion-vector reactivation.
- Force `DLSSNR.Reset` on the reactivation frame and log reset/reactivation transitions.

* Sun Sep 06 2026 DLSS5VKLayer - 0.2.1-2
- Fix present synchronization and layout so processed frames are not presented as raw alternating frames.
- Add shared-memory `seq_ok` and layer processed/raw counters to verify helper success before presenting.
- Fix NVOF current-to-previous session binding and history copy behavior.
- Force NVOF grid to prefer 4x4 and disable cost/hint inputs to avoid Quality-mode 1x1 collapse.
- Add GPU timestamp logging for the optical-flow window and reset/scene-cut diagnostics.
- Bind a zero-filled depth resource and log `DLSSNR.Reset` slot 11 readback.

* Sun Sep 06 2026 DLSS5VKLayer - 0.1.2-1
- Fix synthetic motion vectors to encode `DLSSNR.MVec` as `R16G16_SFLOAT` half-floats.
- Default motion-vector scale to source pixels and flow direction to current-to-previous.
- Add hybrid low-res CPU conversion plus GPU float upscale when direct GPU flow conversion is invalid.
- Make Fast/Balanced/Quality motion-vector modes use different NVOF grids and performance levels.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.1-1
- Add GUI settings for synthetic motion vectors and DLSS5 presets.
- Add global and per-pass DLSS5 Native/Natural/Cinematic preset controls.
- Add motion-vector enable, scale mode, and quality controls.
- Document Steam pressure-vessel shared-memory visibility fix.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-4
- Detect bundled binaries and runner defaults in doctor/start when config is empty.
- Log stale helper quit flags and explicit helper quit reasons.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-3
- Fix Steam overlay conflict by preserving the loader device create-info chain.
- Clear stale helper quit state on helper startup.
- Sync stale shared-memory sequence counters when the helper attaches.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-2
- Use XDG runtime shared-memory path by default.
- Add helper heartbeat recovery.
- Build helper as a GUI-subsystem executable to avoid Proton console windows.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-1
- Initial packaged build.