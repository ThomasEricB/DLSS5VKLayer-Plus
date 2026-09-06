%global debug_package %{nil}
%global _enable_debug_packages 0
%global _include_debuginfo_sources 0
%global pkg_release 4

Name:           dlssnr-personal
Version:        0.1.0
Release:        %{pkg_release}%{?dist}
Summary:        DLSS5 Neural Rendering Vulkan layer and helper with bundled NGX DLLs
License:        Proprietary
Source0:        %{name}-%{version}-%{pkg_release}-linux-x86_64.tar.gz

BuildArch:      x86_64

Provides:       dlssnr = %{version}-%{release}
Conflicts:      dlssnr

Requires:       bash
Requires:       vulkan-loader
Requires:       qt6-qtbase
Requires:       pciutils
Recommends:     wine

%description
DLSS5VKLayer installs a Vulkan implicit layer that forwards presented frames
to a Windows NGX helper running under Wine or a custom Proton compatibility
tool. This personal package includes bundled NVIDIA NGX DLLs and should only
be redistributed if you have the rights to do so.

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
- Initial packaged build with bundled NGX DLLs.