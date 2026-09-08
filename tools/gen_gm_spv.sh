#!/usr/bin/env bash
# Regenerate layer_linux/src/motion/GlobalMotion_Shaders.h from globalmotion.comp.
#
# One source, five entry points, selected by -D: the reduction to a working
# size, the block search, the pick that turns costs into a displacement, the
# gradient refinement, and the temporal filter. See the shader for what each
# one is for and why the pair search+gradients is split that way.
#
# Needs glslang (distro package 'glslang'), or tools/glsl2spv, which build.sh
# builds and which dlopens libshaderc_shared to fill the same role. The
# generated header is committed, so an ordinary build needs neither.
set -euo pipefail
cd "$(dirname "$0")/.."

GLSLANG="${GLSLANG:-$(command -v glslang || command -v glslangValidator || true)}"
if [ -n "$GLSLANG" ]; then
    compile() { "$GLSLANG" "$1" -V -D"$2" -o "$3" >/dev/null; }
elif [ -x build/glsl2spv ]; then
    compile() { build/glsl2spv "$1" "$3" -D "$2" >/dev/null; }
else
    echo "neither glslang nor build/glsl2spv found (build glsl2spv first or set GLSLANG=/path)" >&2
    exit 1
fi

src="layer_linux/src/motion/globalmotion.comp"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

emit_array() { # emit_array <spv> <array-name>
    echo "inline static const unsigned char $2[] = {"
    od -An -tu1 -v "$1" | tr -s ' ' '\n' | grep -v '^$' | \
        awk '{printf "0x%02x, ", $0; if (++n % 12 == 0) printf "\n"} END {if (n % 12) printf "\n"}' | \
        sed 's/^/    /; s/, *$/,/'
    echo "};"
}

{
    echo "#pragma once"
    echo "// Generated from globalmotion.comp by tools/gen_gm_spv.sh; see that file."
    for v in REDUCE:reduce MATCH:match PICK:pick LK:lk SMOOTH:smooth; do
        d="PASS_${v%%:*}"; n="${v##*:}"
        compile "$src" "$d" "$out/$n.spv"
        spirv-val "$out/$n.spv" >/dev/null
        echo
        emit_array "$out/$n.spv" "gm_${n}_spv"
    done
} > layer_linux/src/motion/GlobalMotion_Shaders.h
echo "wrote GlobalMotion_Shaders.h ($(for f in "$out"/*.spv; do printf '%s ' "$(basename "$f") $(wc -c < "$f")B"; done))"
