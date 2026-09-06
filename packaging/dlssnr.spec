%global debug_package %{nil}
%global _enable_debug_packages 0
%global _include_debuginfo_sources 0

Name:           dlssnr
Version:        0.1.0
Release:        2%{?dist}
Summary:        DLSS5 Neural Rendering Vulkan layer and helper
License:        MIT
Source0:        %{name}-%{version}-linux-x86_64.tar.gz

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
%setup -q -n %{name}-%{version}-linux-x86_64

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
* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-2
- Use XDG runtime shared-memory path by default.
- Add helper heartbeat recovery.
- Build helper as a GUI-subsystem executable to avoid Proton console windows.

* Sat Sep 05 2026 DLSS5VKLayer - 0.1.0-1
- Initial packaged build.