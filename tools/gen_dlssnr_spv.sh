#!/usr/bin/env bash
# Regenerate the composition shader's SPIR-V and its embedded header.
#
# dlssnr.hlsl is compiled with dxc, exactly as upstream does it: the checked-in module was built by
# dxc 1.9 and recompiling it with the same version reproduces it byte for byte, so a regen that only
# changes the cbuffer tail can be reviewed as a real diff rather than a toolchain reshuffle.
#
# Needs dxc. Set DXC=/path/to/dxc, or drop the linux release binary at build/dxc
# (https://github.com/microsoft/DirectXShaderCompiler/releases -- unpack, bin/dxc is the file).
#
# The .h is regenerated from the .spv by hand rather than with xxd -i because the file's shape --
# twelve bytes a line, the array named dlssnr_spv -- is what the layer includes.
set -euo pipefail
cd "$(dirname "$0")/.."

DXC="${DXC:-build/dxc}"
if [ ! -x "$DXC" ]; then
    DXC="$(command -v dxc || true)"
fi
if [ -z "${DXC:-}" ] || [ ! -x "$DXC" ]; then
    echo "dxc not found (set DXC=/path/to/dxc or place it at build/dxc)" >&2
    exit 1
fi

SRC=layer_linux/src/dlssnr/dlssnr.hlsl
SPV=layer_linux/src/dlssnr/DlssNr_Shader_Vk.spv
HDR=layer_linux/src/dlssnr/DlssNr_Shader_Vk.h

"$DXC" -spirv -D VK_MODE -T cs_6_0 -E CSMain -Fo "$SPV" "$SRC"

python3 - "$SPV" "$HDR" <<'EOF'
import sys
data = open(sys.argv[1], 'rb').read()
if len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
    sys.exit('not a SPIR-V module')
lines = []
for i in range(0, len(data), 12):
    chunk = data[i:i + 12]
    lines.append('    ' + ', '.join('0x%02x' % b for b in chunk) + ',')
body = '\n'.join(lines).rstrip(',')
open(sys.argv[2], 'w').write(
    '#pragma once\n\n'
    'inline static const unsigned char dlssnr_spv[] = {\n' + body + '\n};\n')
print('wrote %s (%d bytes SPIR-V)' % (sys.argv[2], len(data)))
EOF