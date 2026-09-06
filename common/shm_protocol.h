#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>

static constexpr uint32_t kShmMagic = 0x524E5346;  // 'FNSR'
static constexpr uint32_t kMaxW = 4096, kMaxH = 2160;
static constexpr size_t kMaxFrame = size_t(kMaxW) * kMaxH * 4;
static constexpr uint32_t kMaxPasses = 8;

struct PassControl {
    std::atomic<uint32_t> enabled;
    std::atomic<uint32_t> intensityBits;
    std::atomic<uint32_t> localToneBits;
    std::atomic<uint32_t> localStructureBits;
    std::atomic<uint32_t> skinStructureBits;
    std::atomic<uint32_t> sharpnessBits;
};

struct PassStrength {
    float intensity = 1.0f;
    float localTone = 1.0f;
    float localStructure = 1.0f;
    float skinStructure = -1.0f;
    float sharpness = 0.0f;
};

struct ShmHeader {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> seq_req;
    std::atomic<uint32_t> seq_resp;
    std::atomic<uint32_t> width;
    std::atomic<uint32_t> height;
    std::atomic<uint32_t> quit;
    std::atomic<uint32_t> format;  // 0 = BGRA byte order, 1 = RGBA byte order
    std::atomic<uint32_t> passes;
    std::atomic<uint32_t> enabled;
    std::atomic<uint32_t> intensityBits;
    std::atomic<uint32_t> localToneBits;
    std::atomic<uint32_t> localStructureBits;
    std::atomic<uint32_t> skinStructureBits;
    std::atomic<uint32_t> sharpnessBits;
    std::atomic<uint32_t> controlSeq;
    PassControl pass[kMaxPasses];
    uint32_t reserved[1];
};

inline uint32_t FloatToBits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float BitsToFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline void ShmInitDefaults(ShmHeader* h) {
    h->magic.store(kShmMagic);
    h->seq_req.store(0);
    h->seq_resp.store(0);
    h->width.store(0);
    h->height.store(0);
    h->quit.store(0);
    h->format.store(0);
    h->passes.store(1);
    h->enabled.store(1);
    h->intensityBits.store(FloatToBits(1.0f));
    h->localToneBits.store(FloatToBits(1.0f));
    h->localStructureBits.store(FloatToBits(1.0f));
    h->skinStructureBits.store(FloatToBits(-1.0f));
    h->sharpnessBits.store(FloatToBits(0.0f));
    h->controlSeq.store(0);
    for (uint32_t i = 0; i < kMaxPasses; ++i) {
        h->pass[i].enabled.store(0);
        h->pass[i].intensityBits.store(FloatToBits(1.0f));
        h->pass[i].localToneBits.store(FloatToBits(1.0f));
        h->pass[i].localStructureBits.store(FloatToBits(1.0f));
        h->pass[i].skinStructureBits.store(FloatToBits(-1.0f));
        h->pass[i].sharpnessBits.store(FloatToBits(0.0f));
    }
}

inline uint32_t ShmPasses(const ShmHeader* h) {
    uint32_t p = h->passes.load();
    if (p == 0 || p > kMaxPasses) return 1;
    return p;
}

inline bool ShmNeuralEnabled(const ShmHeader* h) {
    return h->enabled.load() != 0;
}

inline PassStrength ShmGetPassStrength(const ShmHeader* h, uint32_t pass) {
    PassStrength s;
    if (pass < kMaxPasses && h->pass[pass].enabled.load()) {
        s.intensity = BitsToFloat(h->pass[pass].intensityBits.load());
        s.localTone = BitsToFloat(h->pass[pass].localToneBits.load());
        s.localStructure = BitsToFloat(h->pass[pass].localStructureBits.load());
        s.skinStructure = BitsToFloat(h->pass[pass].skinStructureBits.load());
        s.sharpness = BitsToFloat(h->pass[pass].sharpnessBits.load());
    } else {
        s.intensity = BitsToFloat(h->intensityBits.load());
        s.localTone = BitsToFloat(h->localToneBits.load());
        s.localStructure = BitsToFloat(h->localStructureBits.load());
        s.skinStructure = BitsToFloat(h->skinStructureBits.load());
        s.sharpness = BitsToFloat(h->sharpnessBits.load());
    }
    return s;
}