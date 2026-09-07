// glsl2spv — the regen scripts' stand-in for glslangValidator.
//
// tools/gen_mvec_spv.sh and tools/gen_meter_spv.sh regenerate the committed
// SPIR-V headers, and they ask for glslang by name. Most distros ship the
// library (libshaderc_shared, which wraps the same glslang front end) without
// the CLI, and installing the CLI needs root. This tool dlopens whatever
// libshaderc_shared it can find and speaks its C API, so the regen workflow
// runs on a machine where nothing can be installed. The API is stable C with
// external linkage; the declarations below mirror shaderc.h.
//
// usage: glsl2spv input.comp output.spv [-D NAME[=VALUE]]...
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>

namespace {

using Compiler = void*;
using Options = void*;
using Result = void*;

// shaderc_shader_kind: compute is the third enumerator.
constexpr int kShaderKindCompute = 2;
// shaderc_compilation_status: success is zero.
constexpr int kStatusSuccess = 0;
// shaderc_optimization_level: performance.
constexpr int kOptimizationPerformance = 2;

using FnCompilerInit = Compiler (*)();
using FnOptionsInit = Options (*)();
using FnOptionsRelease = void (*)(Options);
using FnAddMacro = void (*)(Options, const char*, size_t, const char*, size_t);
using FnSetOptLevel = void (*)(Options, int);
using FnCompile = Result (*)(Compiler, const char*, size_t, int, const char*, const char*, Options);
using FnSetSourceLang = void (*)(Options, int);
using FnStatus = int (*)(Result);
using FnLength = size_t (*)(Result);
using FnGetBytes = const unsigned char* (*)(Result);
using FnErrorMessage = const char* (*)(Result);
using FnResultRelease = void (*)(Result);

struct Api {
    void* lib = nullptr;
    FnCompilerInit compilerInit = nullptr;
    FnOptionsInit optionsInit = nullptr;
    FnOptionsRelease optionsRelease = nullptr;
    FnAddMacro addMacro = nullptr;
    FnSetOptLevel setOptLevel = nullptr;
    FnCompile compile = nullptr;
    FnSetSourceLang setSourceLang = nullptr;
    FnStatus status = nullptr;
    FnLength length = nullptr;
    FnGetBytes getBytes = nullptr;
    FnErrorMessage errorMessage = nullptr;
    FnResultRelease resultRelease = nullptr;

    bool Load() {
        const char* candidates[] = {
            "libshaderc_shared.so.1", "libshaderc_shared.so",
            "/usr/lib64/libshaderc_shared.so.1", "/usr/lib/x86_64-linux-gnu/libshaderc_shared.so.1",
        };
        for (const char* name : candidates) {
            lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib) { fprintf(stderr, "glsl2spv: %s\n", dlerror()); return false; }
#define SYM(field, name)                                                        \
    *reinterpret_cast<void**>(&field) = dlsym(lib, name);                       \
    if (!field) { fprintf(stderr, "glsl2spv: missing %s\n", name); return false; }
        SYM(compilerInit, "shaderc_compiler_initialize")
        SYM(optionsInit, "shaderc_compile_options_initialize")
        SYM(optionsRelease, "shaderc_compile_options_release")
        SYM(addMacro, "shaderc_compile_options_add_macro_definition")
        SYM(setOptLevel, "shaderc_compile_options_set_optimization_level")
        SYM(compile, "shaderc_compile_into_spv")
        SYM(setSourceLang, "shaderc_compile_options_set_source_language")
        SYM(status, "shaderc_result_get_compilation_status")
        SYM(length, "shaderc_result_get_length")
        SYM(getBytes, "shaderc_result_get_bytes")
        SYM(errorMessage, "shaderc_result_get_error_message")
        SYM(resultRelease, "shaderc_result_release")
#undef SYM
        return true;
    }
};

std::vector<char> ReadFile(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "glsl2spv: cannot open %s\n", path); return {}; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data(n > 0 ? size_t(n) : 0);
    if (n > 0 && fread(data.data(), 1, size_t(n), f) != size_t(n)) {
        fprintf(stderr, "glsl2spv: short read on %s\n", path);
        data.clear();
    }
    fclose(f);
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: glsl2spv input output.spv [-D NAME[=VALUE]]... [-hlsl ENTRY]\n");
        return 1;
    }
    const char* inPath = argv[1];
    const char* outPath = argv[2];
    std::vector<std::string> macros;
    const char* entry = "main";
    bool hlsl = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "-D") == 0 && i + 1 < argc) macros.emplace_back(argv[++i]);
        else if (std::strncmp(argv[i], "-D", 2) == 0 && argv[i][2]) macros.emplace_back(argv[i] + 2);
        else if (std::strcmp(argv[i], "-hlsl") == 0 && i + 1 < argc) { hlsl = true; entry = argv[++i]; }
        else { fprintf(stderr, "glsl2spv: unknown argument %s\n", argv[i]); return 1; }
    }

    std::vector<char> source = ReadFile(inPath);
    if (source.empty()) return 1;

    Api api;
    if (!api.Load()) return 1;

    Compiler compiler = api.compilerInit();
    Options options = api.optionsInit();
    api.setOptLevel(options, kOptimizationPerformance);
    if (hlsl && api.setSourceLang) api.setSourceLang(options, 1);
    for (const std::string& m : macros) {
        const size_t eq = m.find('=');
        const std::string name = m.substr(0, eq);
        const std::string value = eq == std::string::npos ? "1" : m.substr(eq + 1);
        api.addMacro(options, name.c_str(), name.size(), value.c_str(), value.size());
    }

    Result result = api.compile(compiler, source.data(), source.size(), kShaderKindCompute,
                                inPath, entry, options);
    const int status = api.status(result);
    if (status != kStatusSuccess) {
        fprintf(stderr, "glsl2spv: %s\n", api.errorMessage(result));
        api.resultRelease(result);
        api.optionsRelease(options);
        return 2;
    }

    const unsigned char* bytes = api.getBytes(result);
    const size_t len = api.length(result);
    FILE* out = fopen(outPath, "wb");
    if (!out) { fprintf(stderr, "glsl2spv: cannot write %s\n", outPath); return 1; }
    const bool ok = len && fwrite(bytes, 1, len, out) == len;
    fclose(out);
    api.resultRelease(result);
    api.optionsRelease(options);
    if (!ok) { fprintf(stderr, "glsl2spv: empty compilation output\n"); return 1; }
    fprintf(stderr, "glsl2spv: wrote %s (%zu bytes SPIR-V)\n", outPath, len);
    return 0;
}