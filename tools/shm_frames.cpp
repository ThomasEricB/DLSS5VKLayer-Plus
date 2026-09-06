// Push synthetic frames through the helper SHM queue (execution check for the
// GPU MVec deadzone pass + async pipeline). Not part of the shipped layer.
#include "../common/shm_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void FillFrame(uint8_t* px, uint32_t w, uint32_t h, int frame, bool cut) {
    for (uint32_t y = 0; y < h; ++y) {
        uint32_t* row = (uint32_t*)px + (size_t)y * w;
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t r = cut ? (x * 7 + y * 13) & 0xFF : (x + frame * 3) & 0xFF;
            uint32_t g = cut ? (x * 31) & 0xFF : (y + frame * 2) & 0xFF;
            uint32_t b = cut ? (y * 17) & 0xFF : 128;
            row[x] = (b << 16) | (g << 8) | r;  // BGRA byte order
        }
    }
    if (!cut) {  // moving block drives real optical flow
        uint32_t bx = (uint32_t)(frame * 24) % (w - 128);
        uint32_t by = h / 2 - 64;
        for (uint32_t y = by; y < by + 128; ++y) {
            uint32_t* row = (uint32_t*)px + (size_t)y * w;
            for (uint32_t x = bx; x < bx + 128; ++x) row[x] = 0x00FFFFFFu;
        }
    }
}

int main(int argc, char** argv) {
    const uint32_t w = argc > 1 ? (uint32_t)atoi(argv[1]) : 1280;
    const uint32_t h = argc > 2 ? (uint32_t)atoi(argv[2]) : 720;
    const uint32_t frames = argc > 3 ? (uint32_t)atoi(argv[3]) : 6;
    std::string path = ShmDefaultPath();
    if (const char* e = getenv("DLSSNR_SHM"); e && *e) path = e;
    int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) { printf("[client] open %s failed\n", path.c_str()); return 1; }
    size_t total = 4096 + kMaxFrame * 2;
    struct stat st{};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < total) ftruncate(fd, (off_t)total);
    void* m = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { printf("[client] mmap failed\n"); return 1; }
    ShmHeader* hdr = (ShmHeader*)m;
    uint8_t* inPx = (uint8_t*)m + 4096;
    uint8_t* outPx = inPx + kMaxFrame;
    if (hdr->magic.load() != kShmMagic || hdr->passes.load() == 0) ShmInitDefaults(hdr);
    hdr->quit.store(0);
    hdr->width.store(w);
    hdr->height.store(h);
    hdr->format.store(0);
    printf("[client] shm=%s %ux%u frames=%u\n", path.c_str(), w, h, frames);
    bool cut = false;
    const bool repeat = getenv("DLSSNR_CLIENT_STATIC") != nullptr;  // identical frames
    for (uint32_t i = 0; i < frames; ++i) {
        cut = !repeat && i + 1 == frames;  // last frame: scene cut
        FillFrame(inPx, w, h, repeat ? 0 : (int)i, cut);
        const uint32_t req = hdr->seq_req.load() + 1;
        hdr->seq_req.store(req);
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = false;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 60.0) {
            if (hdr->seq_resp.load() >= req) { ok = hdr->seq_ok.load() >= req; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        uint64_t sum = 0;
        for (size_t p = 0; p < (size_t)w * h * 4; p += 997) sum += outPx[p];
        printf("[client] frame %u (%s) seq=%u ok=%d outsum=%llu\n", i, cut ? "cut" : "motion",
               req, int(ok), (unsigned long long)sum);
        if (!ok) return 2;
    }
    printf("[client] PASS\n");
    return 0;
}