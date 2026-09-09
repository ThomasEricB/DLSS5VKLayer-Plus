// VK_LAYER_NV_dlssnr — Linux-side Vulkan layer (loaded by the host loader,
// including the one inside Wine/winevulkan). Hooks the swapchain lifecycle;
// on present, ships the frame to the Windows DLSSNR helper over shared
// memory and presents the neural-processed result.
//
// Enabled implicitly via enable_environment DLSSNR_ENABLE=1 (winevulkan
// rejects explicitly-named layers, so implicit enable is required under Wine).
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <unordered_map>
#include <vector>

#include "../../common/shm_protocol.h"
#include "composition.h"
#include "hotkey.h"
#include "vk_table.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/syscall.h>

// The layer's name has to differ per architecture.
//
// The loader keys implicit layers by name, so two manifests claiming the same name are one layer to
// it: it keeps whichever it read first and then rejects it for the process's word size, reporting
// only "Requested layer VK_LAYER_NV_dlssnr was wrong bit-type" -- with the manifest that would have
// worked sitting unread beside it. Steam hit the same wall and answered it the same way, which is why
// its overlay is VK_LAYER_VALVE_steam_overlay_32 next to _64 rather than one name twice.
#ifdef DLSSNR_LAYER_32
#define VK_LAYER_NAME "VK_LAYER_NV_dlssnr_32"
#else
#define VK_LAYER_NAME "VK_LAYER_NV_dlssnr"
#endif

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
static void Log(const char* fmt, ...) {
    static FILE* f = [] {
        const char* p = getenv("DLSSNR_LOG");
        return p && *p ? fopen(p, "a") : stderr;
    }();
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // One mutex for the sink, not one per line: the previous form allocated a fresh
    // std::mutex on every call and leaked it, which at present rates is a leak per frame.
    static std::mutex sinkMutex;
    std::lock_guard<std::mutex> lk(sinkMutex);
    fprintf(f, "[dlssnr-layer] %s\n", buf);
    fflush(f);
}

static bool TimeEnabled() {
    static const bool v = [] {
        const char* p = getenv("DLSSNR_TIME");
        return p && p[0] == '1';
    }();
    return v;
}

static int TimeInterval() {
    static const int v = [] {
        const char* p = getenv("DLSSNR_TIME_EVERY");
        return p && *p ? atoi(p) : 30;
    }();
    return v > 0 ? v : 30;
}

static bool VerboseEnabled() {
    static const bool v = [] {
        const char* p = getenv("DLSSNR_VERBOSE");
        return p && p[0] == '1';
    }();
    return v;
}

static inline double NowMs() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline void CpuYield() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// ---------------------------------------------------------------------------
// Shared memory transport
// ---------------------------------------------------------------------------
struct ShmMap {
    // Three mappings rather than one.
    //
    // The file is a header followed by two pixel regions each large enough for the biggest frame the
    // protocol allows, which is a quarter of a gigabyte in total. Mapping all of it was fine on
    // 64-bit and is not on 32-bit: a 32-bit game has about 3 GB of address space and would be handing
    // a tenth of it to a reservation it never touches. Both region offsets are page-aligned by
    // construction, so each can be mapped on its own at the size actually in use -- a few megabytes
    // for a real frame instead of 265.
    int fd = -1;
    ShmHeader* hdr = nullptr;
    uint8_t* inPixels = nullptr;
    uint8_t* outPixels = nullptr;
    size_t mappedFrameBytes = 0;
    uint32_t seq = 0;
    uint32_t timeouts = 0;
    uint32_t pendingReq = 0;  // pipelined: the request the helper has not answered yet

    // Liveness, so a game is never made to wait on a helper that is not there.
    uint32_t firstHeartbeat = 0;
    bool everAnswered = false;
    double retryAfterMs = 0.0;
    uint32_t lastControlSeq = 0;
    uint32_t lastHeartbeat = 0;

    // The two pixel regions, imported as device memory so the GPU reads and writes them where they
    // already are. Null when the device could not offer VK_EXT_external_memory_host, in which case
    // the transport copies through host buffers as it always did.
    VkDevice zcDevice = VK_NULL_HANDLE;
    VkDeviceMemory inMem = VK_NULL_HANDLE, outMem = VK_NULL_HANDLE;
    VkBuffer inBuf = VK_NULL_HANDLE, outBuf = VK_NULL_HANDLE;
    VkDeviceMemory motionMem = VK_NULL_HANDLE;
    VkBuffer motionBuf = VK_NULL_HANDLE;
    uint8_t* motionPixels = nullptr;
    uint32_t motionSeqSeen = 0;

    // How old, in presented frames, the answer being composed actually is.
    //
    // Instrumented because the whole pipelined design rests on the edit being "one frame behind" and
    // that was an assumption, never a measurement. The reprojection corrects a single frame-interval
    // of motion, so if an answer is really several frames old the correction is a fraction of the
    // displacement and the edit lands short of its own content -- which is a ghost, and one no
    // threshold can gate away because the edit is genuinely misplaced rather than merely doubtful.
    uint64_t frames = 0;
    uint64_t publishedAtFrame = 0;
    uint32_t lastAge = 1;  // presented frames the last collected answer spent in flight
    uint64_t prevPublishedAtFrame = 0;
    // How much of one round trip of motion the stale edit actually has to be moved by.
    //
    // The helper estimates flow between the two frames it saw, which are one round trip apart, so the
    // field describes exactly one round trip of motion -- but the *previous* one, and the answer being
    // composed is not necessarily a round trip old. Age was measured at 16 frames on average and 27 at
    // worst, so a field applied whole under- or over-corrects by whatever that frame's gap happens to
    // be. This is the ratio that makes the correction the right length.
    float reprojScale = 1.0f;
    uint64_t ageSum = 0;
    uint32_t ageCount = 0;
    uint32_t ageMax = 0;

    // Where the round trip goes. See the attribution note in shm_protocol.h.
    double recordMs = 0.0;      // when this frame's proxy pixels were recorded
    double publishMs = 0.0;     // when the request was handed to the helper
    double rtSum = 0.0, helperSum = 0.0, wakeSum = 0.0, returnSum = 0.0, deferSum = 0.0;
    uint32_t rtCount = 0;
    bool clocksComparable = true;
    size_t importedBytes = 0;
    bool zeroCopy = false;
    bool dead = false;
    // The file path, kept so the dma-buf socket can be named beside it.
    std::string path;
};

// The directory now lives under /tmp, which is world-writable, so it is worth checking that what we
// are about to open really is ours: a directory, owned by this uid, with nothing granted to anyone
// else. Anything else and we refuse rather than create the file inside it.
static bool EnsureParentDir(const std::string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return true;
    std::string dir = path.substr(0, slash);
    size_t pos = 1;
    while ((pos = dir.find('/', pos)) != std::string::npos) {
        mkdir(dir.substr(0, pos).c_str(), 0700);
        pos += 1;
    }
    mkdir(dir.c_str(), 0700);

    struct stat st{};
    if (lstat(dir.c_str(), &st) != 0) { Log("[shm] %s is missing", dir.c_str()); return false; }
    if (!S_ISDIR(st.st_mode) || st.st_uid != getuid() || (st.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        Log("[shm] refusing %s: it is not a private directory owned by this user", dir.c_str());
        return false;
    }
    return true;
}

// Maps a file range at a hint address, trying successive 2 MiB-aligned slots until one is free.
// The hint is 64 KiB-aligned and MAP_FIXED_NOREPLACE either takes that exact address or fails, so
// a success here is a pointer the driver will accept for VK_EXT_external_memory_host -- NVIDIA
// demands minImportedHostPointerAlignment, which is 64 KiB. If every slot is taken the plain map
// still works; the composition checks the pointer's alignment and falls back to staging.
static void* ShmMapAligned(int fd, off_t offset, size_t want, uintptr_t hint) {
#if defined(MAP_FIXED_NOREPLACE) && UINTPTR_MAX > 0xFFFFFFFFull
    for (int i = 0; i < 128; ++i) {
        void* p = mmap((void*) (hint + size_t(i) * (2u << 20)), want, PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED_NOREPLACE, fd, offset);
        if (p != MAP_FAILED) return p;
    }
#else
    (void) hint;
#endif
    return mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
}



static bool ShmMapFrames(ShmMap& s, size_t bytes) {
    if (s.mappedFrameBytes >= bytes && s.inPixels && s.outPixels) return true;
    if (bytes > kMaxFrame) return false;

    // Round up so a small change in resolution does not remap every frame, and to a 64 KiB multiple so
    // a host-pointer import of the whole region (whose allocation size must be a multiple of the
    // driver's import alignment) stays inside the mapping.
    const size_t kImportAlign = 65536;
    const size_t want = ((bytes + kImportAlign - 1) / kImportAlign) * kImportAlign;

    if (s.inPixels) munmap(s.inPixels, s.mappedFrameBytes);
    if (s.outPixels) munmap(s.outPixels, s.mappedFrameBytes);
    if (s.motionPixels) munmap(s.motionPixels, s.mappedFrameBytes);
    s.inPixels = s.outPixels = s.motionPixels = nullptr;
    s.mappedFrameBytes = 0;

    // The file offsets of both regions are already 64 KiB multiples (v7 of the protocol moved the
    // header to 64 KiB and kMaxFrame is an exact multiple); what the kernel adds is the address.
#if UINTPTR_MAX > 0xFFFFFFFFull
    const uintptr_t kHint = UINT64_C(0x200000000000);
    const uintptr_t outHint = kHint + size_t(130) * (2u << 20) + ((want + ((2u << 20) - 1)) & ~size_t((2u << 20) - 1));
#else
    const uintptr_t kHint = 0, outHint = 0;
#endif

    void* in = ShmMapAligned(s.fd, (off_t) kHeaderBytes, want, kHint);
    if (in == MAP_FAILED) { Log("[shm] could not map the input region (%zu bytes)", want); return false; }

    void* motion = ShmMapAligned(s.fd, (off_t) ShmMotionOffset(), want, 0);
    if (motion == MAP_FAILED) {
        munmap(in, want);
        Log("[shm] could not map the motion region (%zu bytes)", want);
        return false;
    }

    void* out = ShmMapAligned(s.fd, (off_t) (kHeaderBytes + kMaxFrame), want, outHint);
    if (out == MAP_FAILED) {
        munmap(in, want);
        munmap(motion, want);
        Log("[shm] could not map the output region (%zu bytes)", want);
        return false;
    }

    s.inPixels = (uint8_t*) in;
    s.outPixels = (uint8_t*) out;
    s.motionPixels = (uint8_t*) motion;
    s.mappedFrameBytes = want;
    return true;
}

static bool ShmOpen(ShmMap& s) {
    if (s.hdr) return true;
    const char* path = getenv("DLSSNR_SHM");
    std::string p = (path && *path) ? path : ShmDefaultPath();
    if (!EnsureParentDir(p)) return false;
    int fd = open(p.c_str(), O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
    if (fd < 0) { Log("[shm] open %s failed", p.c_str()); return false; }
    // The file still spans the whole protocol -- the offsets are fixed and both sides agree on them --
    // but it is sparse, so the size on disk is what has actually been written.
    size_t total = ShmTotalBytes();
    struct stat st{};
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < total) {
        if (ftruncate(fd, (off_t)total) != 0) { close(fd); return false; }
    }

    void* m = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { Log("[shm] mmap of the header failed"); close(fd); return false; }
    s.fd = fd;
    s.hdr = (ShmHeader*)m;
    // A mapping left by an older build has a different magic, a different version, or a header
    // laid out differently; re-initialising is the only safe reading of any of those.
    //
    // But say so, loudly. A live process on the other side of the mismatch keeps re-initialising the
    // other way, and the two then silently reset each other's settings forever -- the layer keeps
    // composing with its old field set and every setting the newer side writes is invisible. That is
    // indistinguishable from "the new feature does nothing", which is how a stale layer reads until
    // someone checks the log.
    if (s.hdr->magic.load() != kShmMagic || s.hdr->version.load() != kShmVersion ||
        s.hdr->passes.load() == 0) {
        if (s.hdr->magic.load() == kShmMagic && s.hdr->version.load() != kShmVersion)
            Log("[shm] header is version %u but this layer is v%u -- another process is out of date, "
                "re-initialising it; update the layer, the helper and the GUI together",
                s.hdr->version.load(), kShmVersion);
        ShmInitDefaults(s.hdr);
    }
    s.lastHeartbeat = s.hdr->heartbeat.load();
    s.firstHeartbeat = s.lastHeartbeat;
    s.path = p;
    Log("[shm] attached %s seq_req=%u seq_resp=%u", p.c_str(),
        s.hdr->seq_req.load(), s.hdr->seq_resp.load());
    return true;
}

static bool ShmNeuralEnabled(ShmMap& s) {
    if (!ShmOpen(s)) return true;
    if (s.hdr->quit.load()) { s.dead = true; return false; }
    const uint32_t ctrl = s.hdr->controlSeq.load();
    if (ctrl != s.lastControlSeq) {
        s.lastControlSeq = ctrl;
        if (s.dead && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            Log("[shm] control changed, re-enabling");
        }
    }
    const uint32_t hb = s.hdr->heartbeat.load();
    if (hb != s.lastHeartbeat) {
        s.lastHeartbeat = hb;
        // A heartbeat alone is not a reason to try again immediately. The helper ticks it while it
        // sits idle, so a helper that is up but not answering used to re-enable the layer the moment
        // it had given up -- which cost the game another round of full-length waits, over and over.
        // That is the stutter: recover, stall, give up, recover.
        if (s.dead && NowMs() >= s.retryAfterMs && ::ShmNeuralEnabled(s.hdr)) {
            s.dead = false;
            s.timeouts = 0;
            Log("[shm] helper heartbeat, trying again");
        }
    }
    if (s.dead) return false;
    return ::ShmNeuralEnabled(s.hdr);
}

// The pipelined transport: publish, and do not wait.
//
// The blocking round trip below is exact and it is also the whole frame cost -- the game's GPU idles
// while the model runs and the model's idles while the game draws, so the two times add rather than
// overlap. Publishing without waiting lets them overlap, at the price of the answer being a frame
// old when it lands.
//
// That price is only payable because the shader has an additive path for it: the frame under the
// edit is always the one being presented and only what is added to it is behind. The composition
// still differences a matched pair, because the layer keeps the proxy that went with each answer.
//
// Whether the input region may be written again is not guessed: seq_resp says the helper has
// finished reading the last one, and a frame that cannot publish simply does not.
static bool ShmAnswerReady(ShmMap& s) {
    return s.hdr && s.pendingReq != 0 && s.hdr->seq_resp.load() >= s.pendingReq;
}

static bool ShmInputFree(ShmMap& s) {
    return s.hdr && (s.pendingReq == 0 || s.hdr->seq_resp.load() >= s.pendingReq);
}

// Take delivery of an answer that has already arrived. Never blocks.
// The gap the user asked for between answers, in presented frames. See publishStride in the
// protocol. DLSSNR_PUBLISH_STRIDE overrides the setting, so a cadence can be swept from a launch
// option without going through the GUI. Bounded, because a stride longer than the estimator can warp
// across is a worse picture rather than a smoother one.
static uint32_t ShmPublishStride(ShmMap& s) {
    static const int forced = [] {
        const char* v = getenv("DLSSNR_PUBLISH_STRIDE");
        return v && *v ? atoi(v) : -1;
    }();
    if (forced >= 0) return uint32_t(forced > 64 ? 64 : forced);
    if (!s.hdr) return 0;
    const uint32_t v = s.hdr->publishStride.load();
    return v > 64 ? 64 : v;
}

// How many presented frames passed between the request going out and its answer being taken up.
static void ShmNoteAge(ShmMap& s) {
    const uint64_t age = s.frames - s.publishedAtFrame;
    const uint64_t interval = s.publishedAtFrame - s.prevPublishedAtFrame;
    if (interval > 0 && age > 0) {
        // Bounded: a first frame, a resize or a hitch can make either number nonsense, and a wild
        // scale would fling the edit somewhere arbitrary rather than merely leave it where it was.
        const float r = float(double(age) / double(interval));
        s.reprojScale = r < 0.25f ? 0.25f : (r > 3.0f ? 3.0f : r);
    }
    {
        const double collect = NowMs();
        const double detect = BitsDouble(s.hdr->dbgDetectMs.load());
        const double written = BitsDouble(s.hdr->dbgWrittenMs.load());
        const double rt = collect - s.publishMs;
        const double helper = written - detect;
        // The two spans that need no cross-clock comparison, and are therefore always believable.
        if (rt > 0.0 && helper >= 0.0 && helper <= rt) {
            s.rtSum += rt;
            s.helperSum += helper;
            s.deferSum += s.publishMs - s.recordMs;
            // The split into wake and return is only meaningful if the two clocks share an epoch.
            const double wake = detect - s.publishMs;
            const double back = collect - written;
            if (wake >= 0.0 && back >= 0.0 && wake + back <= rt + 1.0) {
                s.wakeSum += wake;
                s.returnSum += back;
            } else {
                s.clocksComparable = false;
            }
            ++s.rtCount;
        }
    }
    s.lastAge = uint32_t(age);
    s.ageSum += age;
    s.ageMax = age > s.ageMax ? uint32_t(age) : s.ageMax;
    if (++s.ageCount >= 300) {
        Log("[pipe] answers are %.1f frames old on average, worst %u; reprojection scaled by %.2f; "
            "cadence gap %u (0 = as soon as ready)",
            double(s.ageSum) / double(s.ageCount), s.ageMax, s.reprojScale, ShmPublishStride(s));
        if (s.rtCount) {
            const double n = double(s.rtCount);
            const double rt = s.rtSum / n, helper = s.helperSum / n;
            if (s.clocksComparable) {
                Log("[pipe] round trip %.2f ms = wake %.2f + helper %.2f + return %.2f; "
                    "the proxy was recorded %.2f ms before it was even published",
                    rt, s.wakeSum / n, helper, s.returnSum / n, s.deferSum / n);
            } else {
                Log("[pipe] round trip %.2f ms = helper %.2f + handshake %.2f (the two clocks do not "
                    "share an epoch, so wake and return cannot be separated); the proxy was recorded "
                    "%.2f ms before it was even published",
                    rt, helper, rt - helper, s.deferSum / n);
            }
            s.rtSum = s.helperSum = s.wakeSum = s.returnSum = s.deferSum = 0.0;
            s.rtCount = 0;
        }
        s.ageSum = 0;
        s.ageCount = 0;
        s.ageMax = 0;
    }
}

static bool ShmCollect(ShmMap& s, size_t bytes, void* modelOut) {
    if (!ShmAnswerReady(s)) return false;
    const bool ok = s.hdr->seq_ok.load() >= s.pendingReq;
    ShmNoteAge(s);
    s.pendingReq = 0;
    if (!ok) return false;
    s.everAnswered = true;
    if (modelOut) std::memcpy(modelOut, s.outPixels, bytes);
    return true;
}

// Hand over a frame whose pixels the GPU has already finished writing. Never blocks.
static bool ShmPublish(ShmMap& s, uint32_t w, uint32_t h, const void* proxy) {
    if (s.dead || !s.hdr || w > kMaxW || h > kMaxH) return false;
    const size_t bytes = size_t(w) * h * 4;
    if (!ShmMapFrames(s, bytes)) { s.dead = true; return false; }
    if (proxy) std::memcpy(s.inPixels, proxy, bytes);
    s.hdr->width.store(w);
    s.hdr->height.store(h);
    s.hdr->format.store(1u);
    const uint32_t req = s.hdr->seq_req.load() + 1;
    s.hdr->seq_req.store(req);
    s.pendingReq = req;
    s.prevPublishedAtFrame = s.publishedAtFrame;
    s.publishedAtFrame = s.frames;
    s.publishMs = NowMs();
    s.hdr->dbgRecordMs.store(DoubleBits(s.recordMs));
    s.hdr->dbgPublishMs.store(DoubleBits(s.publishMs));
    return true;
}

// One round trip: publish the proxy, wait for the model's answer, copy it back.
//
// What crosses is the proxy at the model's own resolution, always R8G8B8A8_UNORM and always
// display-referred, because the encode has already done that work on the GPU. The helper therefore
// never has to know what format the game presents in, and the working scale reduces this copy
// quadratically -- which on this transport is the difference the setting actually buys.
// Take this process's own reference on a dma-buf another process exported: procfs opens a fresh
// descriptor for the same buffer. Same uid and a permissive yama setting are what make the open
// work; where it does not, the caller falls back and nothing else changes.
// Adopt a descriptor the helper exported. The primary route is pidfd_getfd, which duplicates a
// descriptor straight out of the helper's table -- the only way to receive a dma-buf, since those
// live on an anonymous filesystem that cannot be reopened by path. The old route (opening
// /proc/<pid>/fd/<n>) is kept as a fallback for kernels predating pidfd; it works for ordinary
// files even though it cannot see dma-bufs. Both need the same uid and a permissive yama.
static int AdoptPeerFd(uint32_t pid, uint32_t fd) {
    if (!pid || fd == 0 || fd > 1000000u) return -1;
#ifdef __NR_pidfd_open
    const int pidfd = (int)syscall(__NR_pidfd_open, pid, 0);
    if (pidfd >= 0) {
        const int dup = (int)syscall(__NR_pidfd_getfd, pidfd, fd, 0);
        close(pidfd);
        if (dup >= 0) return dup;
    }
#endif
    char p[64];
    snprintf(p, sizeof(p), "/proc/%u/fd/%u", pid, fd);
    return open(p, O_RDWR | O_CLOEXEC);
}

// The exchange is on unless the environment says off; the helper and driver get the final say by
// what they publish.
static bool DmaBufEnabled() {
    static const bool on = [] {
        const char* v = getenv("DLSSNR_DMABUF");
        return !(v && !strcmp(v, "0"));
    }();
    return on;
}

static bool ShmProcessFrame(ShmMap& s, uint32_t w, uint32_t h, size_t bytes, const void* proxy,
                          void* modelOut, bool proxyInRegion, bool answerFromFd, bool hdrEncode) {
    if (s.dead) return false;
    if (!ShmOpen(s)) { s.dead = true; return false; }
    if (w > kMaxW || h > kMaxH) return false;
    if (bytes != size_t(w) * h * 4 && bytes != size_t(w) * h * 8) return false;
    if (s.hdr->quit.load()) { s.dead = true; return false; }

    const bool time = TimeEnabled();
    const double t0 = NowMs();
    if (!ShmMapFrames(s, bytes)) { s.dead = true; return false; }
    // When the transport buffer IS this region (the imported case), or the proxy crossed as a
    // dma-buf instead, the GPU already wrote the bytes where they belong and there is nothing to
    // copy.
    if (!proxyInRegion && proxy != (const void*) s.inPixels) std::memcpy(s.inPixels, proxy, bytes);
    const double tCopy = NowMs();
    s.hdr->width.store(w);
    s.hdr->height.store(h);
    s.hdr->format.store(1u);  // RGBA byte order either way; the float path keeps the same swizzle
    // Say what the bytes ARE before announcing them: the helper sizes its read by this, never by
    // what it hopes the layer has switched to. The release fence below covers it like the pixels.
    s.hdr->hdrEncode.store(hdrEncode ? 1u : 0u);
    uint32_t req = s.hdr->seq_req.load() + 1;
    // The release pairs with the helper's acquire on seq_resp: everything this process wrote --
    // the proxy, whether by the GPU into the imported region or by the memcpy above -- is visible
    // to the helper before it sees the new request number. (The GPU's own write is fenced earlier,
    // by leg 1's vkWaitForFences; this fence covers the host-visible ordering across processes.)
    std::atomic_thread_fence(std::memory_order_release);
    s.hdr->seq_req.store(req);

    // How long this frame may wait, which is a question about whether anyone is listening.
    //
    // A live helper needs real time: the model is milliseconds of work and building its feature on the
    // first frame is far more than that. A helper that is not running needs none at all, and the old
    // fixed second-per-frame budget meant a game whose helper was simply not started froze for eight
    // seconds before the layer gave up. That is what this is for.
    // Is anything listening? The helper says so itself, from the moment it attaches until it exits,
    // which is the only signal that stays true while it is busy. Heartbeats do not: it stops ticking
    // them precisely while it is building the model's feature.
    const bool helperPresent = s.hdr->helperState.load() != kHelperStopped;

    // The first frame of a size is not like the others. It makes the helper load the model and build
    // a feature -- measured at 194 ms for a small frame and more for a large one -- against about 4 ms
    // once it is warm. Timing that out and giving up is how a working helper gets abandoned before it
    // has answered once.
    const bool warmingUp = !s.everAnswered;
    const double budgetMs = !helperPresent ? 20.0 : (warmingUp ? 10000.0 : 1000.0);

    // Wait for the helper (fail-open: present the original frame on timeout).
    const double tSignal = NowMs();
    for (;;) {
        if (s.hdr->seq_resp.load() >= req) {
            s.timeouts = 0;
            s.everAnswered = true;
            // The helper's GPU wrote the answer into this region (or the memcpy below reads the
            // staging copy of it); the acquire pairs with the helper's release before seq_resp.
            std::atomic_thread_fence(std::memory_order_acquire);
            // The helper answers even when it could not use the frame. seq_ok says whether the
            // answer is worth composing; when it is not, the game's own frame is what to present.
            // The echo says the answer was made for this raster: another swapchain (the Steam
            // overlay, or this one's predecessor mid-resize) may have had its request answered in
            // the meantime, and seq_resp only counts. Composing that answer here would copy a
            // different number of bytes into these surfaces -- the row-shifted colour garbage this
            // check exists to refuse.
            const bool ok = s.hdr->seq_ok.load() >= req && s.hdr->answeredW.load() == w &&
                            s.hdr->answeredH.load() == h;
            if (!ok) Log("[shm] helper could not use frame %u (ok=%u)", req, s.hdr->seq_ok.load());
            if (ok && !answerFromFd && modelOut != (void*) s.outPixels) std::memcpy(modelOut, s.outPixels, bytes);
            if (time) {
                static int frameNo = 0;
                if (++frameNo % TimeInterval() == 0) {
                    const double tDone = NowMs();
                    Log("[time] shm copy=%.2f signal=%.2f wait=%.2f total=%.2f ms",
                        tCopy - t0, tSignal - tCopy, tDone - tSignal, tDone - t0);
                }
            }
            return ok;
        }
        if (s.hdr->quit.load()) { s.dead = true; return false; }
        const double elapsed = NowMs() - tSignal;
        if (elapsed >= budgetMs) break;
        if (elapsed < 2.0) CpuYield();
        else std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    // Four rather than eight, and with a pause before the next attempt, so giving up costs a
    // fraction of a second and retrying costs that again only every few seconds.
    if (++s.timeouts >= 4) {
        s.dead = true;
        s.retryAfterMs = NowMs() + 5000.0;
        Log("[shm] no answer in %.0f ms x4 (helper %s); passing frames through, retrying in 5s "
            "(seq_req=%u seq_resp=%u heartbeat=%u)",
            budgetMs, helperPresent ? "is present but silent" : "not running",
            s.hdr->seq_req.load(), s.hdr->seq_resp.load(), s.hdr->heartbeat.load());
    }
    return false;
}

// ---------------------------------------------------------------------------
// Dispatch chains
// ---------------------------------------------------------------------------
struct InstanceChain {
    PFN_vkGetInstanceProcAddr next_gipa = nullptr;

    // The instance-level entry points the composition needs, resolved once. Kept here rather than on
    // the device chain because this is where the VkInstance handle is in scope.
    dlssnr::InstanceTable table;

    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
};

#define DEVICE_FN_LIST(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkGetDeviceQueue2) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) \
    X(vkGetSwapchainImagesKHR) X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkGetFenceStatus) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) \
    X(vkFreeMemory) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) X(vkCmdPipelineBarrier) X(vkDeviceWaitIdle) \
    X(vkAcquireNextImageKHR) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkCmdBlitImage) \
    X(vkReleaseSwapchainImagesEXT)

struct SwapchainState {
    std::vector<VkImage> images;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // HdrKind: what this swapchain's format and colour space say the frame carries. The float
    // swapchain holds linear light; a 10-bit one with a PQ colour space holds ST 2084 code.
    uint32_t hdrKind = kHdrNone;
    uint32_t width = 0, height = 0;
    bool ready = false;
    bool passThrough = false;
    VkCommandPool pool = VK_NULL_HANDLE;

    // Two fences for the synchronous path. Leg 1's must be waited on before the proxy is handed to
    // the helper -- the sequence number is the helper's only ordering signal, and it may not be
    // bumped ahead of the write it announces. Leg 2's needs no wait in its own frame: the present
    // follows it on the same queue, so the GPU orders them without the CPU, and the wait moves to
    // the start of the next present where the surfaces are reused.
    VkFence fenceLeg1 = VK_NULL_HANDLE;
    VkFence fenceLeg2 = VK_NULL_HANDLE;
    bool leg2Pending = false;

    // A ring for the pipelined path, so the present hook never waits for the work it just submitted.
    //
    // Waiting was almost the entire cost of the hook -- at 200% the hook measured 1.76 ms of which
    // 1.67 ms was the fence -- and the cost is the smaller half of the harm. Blocking inside
    // vkQueuePresentKHR stops the game's render thread until this pass's GPU work is done, so the
    // game cannot queue the next frame while this one finishes. Three slots: one being recorded, one
    // in flight, one spare, so the only wait left is on a slot three frames old.
    static constexpr uint32_t kSlots = 3;
    VkCommandBuffer cb[kSlots] = {};
    VkFence fence[kSlots] = {};
    bool submitted[kSlots] = {};
    uint32_t slot = 0;

    // The send recorded into a slot, published once that slot's GPU work has actually landed. The
    // helper reads those pages the moment it sees the sequence number, so the publish cannot happen
    // before the copy into them has completed -- but it need not happen on the same frame.
    uint32_t pendingSlot = kSlots;
    uint32_t pendingW = 0, pendingH = 0;

    // The pass. Owns every surface it needs, including the transport pair -- the shared-memory
    // regions themselves when the driver will import them, host-visible staging when it will not.
    std::unique_ptr<dlssnr::Composition> comp;

    // What the game asked the presentation engine for. Frame generation cares because it is what
    // decides whether an extra present is paced or merely queued; see FrameGen.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Frame generation: an extra present the game did not make, carrying the frame it did make
    // forward along the measured displacement.
    //
    // Nothing is interpolated and nothing is delayed. Interpolation needs the frame after the one it
    // sits between, so a real frame has to be held back to have two -- that is where the latency in
    // frame generation usually comes from. Here the generated frame is an extrapolation of the frame
    // being presented right now, so the real frame goes out first and unchanged, and the generated
    // one follows it into the gap before the next.
    //
    // The whole thing is guarded by an acquire with a zero timeout. If the swapchain has no free
    // image the generated frame is simply not made -- counted as missed and forgotten -- and the game
    // never waits on this path for anything.
    struct FrameGen {
        static constexpr uint32_t kSlots = 4;
        VkCommandBuffer cb[kSlots] = {};
        VkFence fence[kSlots] = {};
        // One acquire semaphore per slot, signalled by vkAcquireNextImageKHR and waited on by the
        // blit; one done semaphore per slot, signalled by the blit and waited on by the present.
        VkSemaphore acquired[kSlots] = {};
        // A present's semaphore may not be reused until that present has finished, and core Vulkan
        // provides no way to learn when that is.
        //
        // Both of the rules broken here say the same thing from different sides. Validation put it
        // plainly: "swapchain image 4 was presented but was not re-acquired, so the semaphore may
        // still be in use and cannot safely be reused". Reuse was being timed against a fence -- which
        // says the submit finished -- or against an image being acquired again, which is a guess that
        // holds for a simple application and does not hold for a translation layer like zink.
        //
        // VK_EXT_swapchain_maintenance1 answers the question directly: a fence attached to the
        // present, signalled when the present is done. So each present takes a pair from this ring
        // and gives it back only when its own fence says so. Nothing here is inferred any more.
        struct PresentPair {
            VkSemaphore sem = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            bool inFlight = false;
            // Which present this pair was last handed to, for the fallback below.
            uint64_t serial = 0;
        };
        // Presents this layer has made on this swapchain, real and generated.
        uint64_t presentSerial = 0;
        // Wide enough that presents in flight are never the limit.
        //
        // Eight was a guess and the wrong one: a frame takes one pair for the real present and one
        // for each generated frame, so at three per frame the ring turns over every two frames --
        // far faster than a present fence signals, which happens a vertical blank or more later. The
        // log said so plainly once it was asked, "no present pair has come back yet" a thousand times
        // against a few hundred frames generated. A semaphore and a fence are cheap; being unable to
        // generate is not.
        static constexpr uint32_t kRing = 32;
        std::vector<PresentPair> ring;

        // A slot's life: free, then holding an image acquired ahead of time, then submitted and
        // waiting on its fence to come back round to free.
        //
        // The image is acquired a frame early because it cannot be acquired late. Under FIFO an
        // image is released when the presentation engine is finished displaying it, which happens at
        // a vertical blank -- so at the instant the game's present returns there is frequently
        // nothing free, and a zero-timeout acquire there fails every time however many images the
        // swapchain has. Acquiring at the top of the next present instead gives that release a whole
        // frame to happen in, and by the time the generated frame is wanted the image is already in
        // hand.
        enum State : uint8_t { kFree, kHeld, kSubmitted };
        State state[kSlots] = {};
        uint32_t index[kSlots] = {};
        // Which slot's fence covers this one. All the frames in a batch are recorded into one command
        // buffer and submitted once, so only the lead slot's fence is ever submitted -- the others
        // would wait forever on a fence nobody signals, and never be reused.
        uint32_t guardedBy[kSlots] = {};
        // Held slots in the order they were acquired, so generated frames are presented in that
        // order too.
        uint32_t queue[kSlots] = {};
        uint32_t queued = 0;
        uint32_t wantFrames = 0;
        // How long the first acquire of a batch may wait, in microseconds. Not a setting and never
        // was a good one: the right value is a property of what the display is doing, which nobody
        // can know in advance and the layer can measure. See MfgDynamicFactor.
        uint32_t waitUs = 0;

        bool ready = false;
        bool unavailable = false;
        uint64_t generated = 0;
        uint64_t missed = 0;
        float lastDx = 0.0f, lastDy = 0.0f;
        bool motionValid = false;

        // The climb that decides how many frames actually fit in the gap. See MfgDynamicFactor.
        uint32_t active = 0;          // what is being generated per real frame right now
        uint32_t windowFrames = 0;    // real presents counted into this measurement window
        uint32_t windowMissed = 0;    // acquires that found nothing, this window
        double windowStartMs = 0.0;
        double baseRate = 0.0;        // the game's real frame rate with nothing generated
        uint32_t windowsSinceBase = 0;
        uint32_t coolWindows = 0;    // after a step down, how long before trying again

    } fg;
};


// Frame pacing, which is not the same question as frame cost.
//
// A pass that costs 1 ms on average and 40 ms once a second reports a fine average and feels like a
// slideshow, because what anyone sees is the spread between presents rather than their mean. So this
// keeps the distribution: the interval the game actually achieved, and how long the present hook held
// it up, both as percentiles over a window.
struct Pacing {
    static constexpr size_t kWindow = 1024;
    double interval[kWindow] = {};
    double hook[kWindow] = {};
    double fence[kWindow] = {};
    size_t n = 0;
    double lastPresent = 0.0;

    void Add(double intervalMs, double hookMs, double fenceMs) {
        const size_t i = n % kWindow;
        interval[i] = intervalMs;
        hook[i] = hookMs;
        fence[i] = fenceMs;
        ++n;
    }

    static double Pct(double* v, size_t count, double p) {
        if (!count) return 0.0;
        std::vector<double> c(v, v + count);
        std::sort(c.begin(), c.end());
        size_t idx = size_t(p * (count - 1) + 0.5);
        return c[idx];
    }

    void Report() {
        const size_t count = n < kWindow ? n : kWindow;
        if (count < 16) return;
        // The number that matches the complaint: how often a frame took more than twice the usual.
        // A max alone can be one hitch in a thousand; this says whether it is one or a hundred.
        const double p50 = Pct(interval, count, 0.50);
        size_t hitches = 0;
        for (size_t i = 0; i < count; ++i)
            if (interval[i] > p50 * 2.0) ++hitches;
        Log("[pace] %zu of %zu frames took over twice the usual interval", hitches, count);
        Log("[pace] over %zu frames -- present interval p50=%.2f p95=%.2f p99=%.2f max=%.2f ms; "
            "hook p50=%.2f p95=%.2f p99=%.2f max=%.2f ms; fence wait p50=%.2f p99=%.2f max=%.2f ms",
            count,
            Pct(interval, count, 0.50), Pct(interval, count, 0.95), Pct(interval, count, 0.99),
            Pct(interval, count, 1.0),
            Pct(hook, count, 0.50), Pct(hook, count, 0.95), Pct(hook, count, 0.99),
            Pct(hook, count, 1.0),
            Pct(fence, count, 0.50), Pct(fence, count, 0.99), Pct(fence, count, 1.0));
    }
};

struct DeviceChain {
    InstanceChain* instance = nullptr;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice self = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr next_dpa = nullptr;

    // The same entry points again, in the form the composition takes them.
    dlssnr::DeviceTable table;

    // Whether this device was created with VK_EXT_external_memory_host, which is what decides
    // between the zero-copy transport and copying through staging.
    bool hostImport = false;
    // Whether a fence may be attached to a present, which is the only way to learn that a present
    // has finished and therefore the only way to recycle what it was waiting on. Frame generation
    // does not run without it.
    bool presentFence = false;

    // The loader's hook for installing a dispatch table on a dispatchable object a layer creates.
    // Handed to every layer in its own VkLayerDeviceCreateInfo node; see Hook_CreateDevice.
    PFN_vkSetDeviceLoaderData setDeviceLoaderData = nullptr;
#define X(name) PFN_##name name = nullptr;
    DEVICE_FN_LIST(X)
#undef X
    std::atomic<bool> inert{false};
    std::mutex lock;
    std::unordered_map<VkSwapchainKHR, SwapchainState> swapchains;
    std::unordered_map<VkQueue, uint32_t> queueFamilies;
    ShmMap shm;
    uint64_t framesComposed = 0;
    uint64_t framesPassedThrough = 0;
    double fenceWaitMs = 0.0;   // this frame's total, reset by the present hook
    Pacing pace;
    // Phase 5: the dma-buf exchange. The export sequences last imported; a new sequence means the
    // image behind the descriptor changed and the reference is taken again.
    uint32_t proxySeqSeen = 0;
    uint32_t answerSeqSeen = 0;
};

static std::unordered_map<VkInstance, InstanceChain> g_instances;
static std::unordered_map<VkPhysicalDevice, InstanceChain*> g_phys;
static std::unordered_map<VkDevice, DeviceChain*> g_devices;
static std::mutex g_stateMutex;

// The one swapchain allowed to drive the neural round trip, chosen as the largest in the process.
//
// The shared-memory channel carries a single raster at a time, but a process can present more than
// one swapchain -- the game window and the Steam overlay, or, mid-resize, the old and new windows at
// once. Feeding all of them through one channel makes the helper rebuild its model on every size
// switch and lets one swapchain be handed another's answer. The largest is the game; the rest present
// raw. The record is global rather than per-device because the overlay builds its own VkDevice.
struct PrimarySwap {
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint64_t area = 0;
};
static PrimarySwap g_primary;
// Its own mutex, never nested with dc->lock or g_stateMutex, so the lock order in the present hook
// cannot invert against the device hooks.
static std::mutex g_primaryMutex;

// Adopts a larger swapchain; a present from anything else passes through untouched.
static bool ClaimPrimary(VkDevice device, VkSwapchainKHR swapchain, uint32_t w, uint32_t h) {
    std::lock_guard<std::mutex> lk(g_primaryMutex);
    const uint64_t area = uint64_t(w) * h;
    if (g_primary.swapchain == swapchain && g_primary.device == device) return true;
    if (g_primary.swapchain != VK_NULL_HANDLE && area <= g_primary.area) return false;
    g_primary.device = device;
    g_primary.swapchain = swapchain;
    g_primary.area = area;
    return true;
}

static void ReleasePrimary(VkDevice device, VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lk(g_primaryMutex);
    if (g_primary.swapchain == swapchain && g_primary.device == device) g_primary = PrimarySwap{};
}

// Where this copy of the layer was loaded from, for the duplicate check below.
static std::string LayerObjectPath() {
    Dl_info info{};
    if (dladdr((const void*)&LayerObjectPath, &info) && info.dli_fname && *info.dli_fname)
        return info.dli_fname;
    return std::string();
}

// True when a *different* copy of this layer is already in the chain.
//
// build.sh installs an implicit-layer manifest pointing at the build tree while install.sh installs
// another pointing at the install prefix, and the loader honours both: two copies of the layer, two
// present hooks, two full round trips, and a single shared-memory file with two writers racing on
// one sequence number. Only the first copy stays live; the rest declare themselves inert and pass
// everything through, which turns a corrupted picture or a hang into one warning line.
//
// The claim is the object's own path rather than a bare flag, so a second call into the same copy --
// which is legal, the loader may negotiate more than once -- is told apart from a second copy.
static bool DuplicateLayerCopy() {
    static const bool dup = [] {
        const std::string self = LayerObjectPath();
        const char* claimed = getenv("DLSSNR_LAYER_OBJECT");
        if (claimed && *claimed) {
            if (self.empty() || self == claimed) return false;
            Log("[layer] another copy is already loaded from %s; this copy (%s) stays inert. "
                "Remove one of the implicit-layer manifests.", claimed, self.c_str());
            return true;
        }
        if (!self.empty()) setenv("DLSSNR_LAYER_OBJECT", self.c_str(), 0);
        return false;
    }();
    return dup;
}

// One set of keyboards for the process, however many devices the game creates.
static dlssnr::Hotkeys g_hotkeys;

// The key to watch, from the header if the interface has set one and from the environment otherwise,
// so it can be bound in a launch option without the interface being involved.
static uint32_t ToggleKey(const ShmHeader* hdr) {
    static const uint32_t fromEnv = [] {
        const char* v = getenv("DLSSNR_TOGGLE_KEY");
        return v && *v ? dlssnr::KeyCodeFromName(v) : 0u;
    }();
    if (fromEnv) return fromEnv;
    return hdr ? hdr->toggleKey.load() : 0u;
}

// Polled before anything asks whether the pass is enabled, because asking first would make turning it
// off a one-way door: the early return would skip the very code that reads the key to turn it back on.
static void PollHotkeys(DeviceChain* dc) {
    if (!ShmOpen(dc->shm) || !dc->shm.hdr) return;
    const uint32_t key = ToggleKey(dc->shm.hdr);
    if (!key || !g_hotkeys.Pressed(key)) return;

    const bool wasOn = dc->shm.hdr->enabled.load() != 0;
    dc->shm.hdr->enabled.store(wasOn ? 0u : 1u);
    dc->shm.hdr->controlSeq.fetch_add(1);
    Log("[hotkey] %s -> neural rendering %s", dlssnr::KeyNameFromCode(key), wasOn ? "off" : "on");
}

static bool LayerEnabled() {
    static const bool e = [] {
        if (DuplicateLayerCopy()) return false;
        const char* v = getenv("VKLayer_DLSS5");
        if (v && v[0] == '1') return true;
        const char* o = getenv("DLSSNR_ENABLE");
        return o && o[0] == '1';
    }();
    return e;
}

// ---------------------------------------------------------------------------
// Instance hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateInstance(
    const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance) {
    auto* link = const_cast<VkLayerInstanceCreateInfo*>((const VkLayerInstanceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerInstanceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto create = (PFN_vkCreateInstance)next_gipa(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // Ask for the two instance extensions the zero-copy transport's device extension depends on.
    //
    // Below Vulkan 1.1 the external-memory family is not core, and VK_KHR_external_memory -- which
    // the device hook adds -- needs VK_KHR_external_memory_capabilities here. That one in turn needs
    // VK_KHR_get_physical_device_properties2, so it is both or neither.
    //
    // Without them the driver cannot be asked whether an imported host pointer is a valid buffer
    // handle type, and every such question is answered "no" by default -- which is what made the
    // transport look invalid when it is not.
    //
    // Tried rather than enumerated: a layer cannot reliably enumerate instance extensions through the
    // chain's own vkGetInstanceProcAddr, and an instance that refuses them simply gets created the
    // way the game asked. Nothing here may turn a working instance into a failed one.
    const uint32_t api =
        pCreateInfo->pApplicationInfo ? pCreateInfo->pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
    std::vector<const char*> instExts(pCreateInfo->ppEnabledExtensionNames,
                                      pCreateInfo->ppEnabledExtensionNames +
                                          pCreateInfo->enabledExtensionCount);
    const bool preVulkan11 = VK_API_VERSION_MAJOR(api) == 1 && VK_API_VERSION_MINOR(api) < 1;
    const auto want = [&](const char* name) {
        for (const char* e : instExts)
            if (e && !std::strcmp(e, name)) return;
        instExts.push_back(name);
    };
    if (preVulkan11) {
        want(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        want(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    }
    // What frame generation needs to know when a present has finished.
    //
    // Recycling a semaphore that a present is waiting on requires knowing when that present is done,
    // and core Vulkan offers no way to ask. Swapchain maintenance 1 adds a fence to the present,
    // which answers exactly that question; its device extension needs these two on the instance.
    want(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
    want(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
    const bool addedInstExts = instExts.size() > pCreateInfo->enabledExtensionCount;


    // Documented pattern: keep the link node in pNext (layers below need it)
    // and advance u.pLayerInfo so the next layer resolves its own chain entry.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    VkInstanceCreateInfo ici = *pCreateInfo;
    ici.enabledExtensionCount = uint32_t(instExts.size());
    ici.ppEnabledExtensionNames = instExts.empty() ? nullptr : instExts.data();

    VkResult res = create(addedInstExts ? &ici : pCreateInfo, pAllocator, pInstance);
    if (res != VK_SUCCESS && addedInstExts) {
        // Never turn a working instance into a failed one for the sake of an optimisation.
        res = create(pCreateInfo, pAllocator, pInstance);
    }
    if (res != VK_SUCCESS) return res;

    InstanceChain chain{};
    chain.next_gipa = next_gipa;
    chain.vkDestroyInstance = (PFN_vkDestroyInstance)next_gipa(*pInstance, "vkDestroyInstance");
    chain.vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)next_gipa(*pInstance, "vkEnumeratePhysicalDevices");
    chain.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)next_gipa(*pInstance, "vkGetPhysicalDeviceProperties");
    chain.vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)next_gipa(*pInstance, "vkEnumerateDeviceExtensionProperties");
    chain.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)next_gipa(*pInstance, "vkGetPhysicalDeviceMemoryProperties");

    chain.table.next_gipa = next_gipa;
    chain.table.Load(*pInstance);

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_instances[*pInstance] = chain;
    Log("[layer] vkCreateInstance -> %p", (void*)*pInstance);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyInstance(VkInstance instance,
                                                       const VkAllocationCallbacks* pAllocator) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_instances.find(instance);
    if (it == g_instances.end()) return;
    auto destroy = it->second.vkDestroyInstance;
    InstanceChain* chain = &it->second;
    g_instances.erase(it);
    for (auto pit = g_phys.begin(); pit != g_phys.end();)
        pit = (pit->second == chain) ? g_phys.erase(pit) : std::next(pit);
    if (destroy) destroy(instance, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_EnumeratePhysicalDevices(
    VkInstance instance, uint32_t* pCount, VkPhysicalDevice* pPhysicalDevices) {
    InstanceChain* chain = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end()) chain = &it->second;
    }
    if (!chain || !chain->vkEnumeratePhysicalDevices) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult res = chain->vkEnumeratePhysicalDevices(instance, pCount, pPhysicalDevices);
    if (res == VK_SUCCESS && pPhysicalDevices) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        for (uint32_t i = 0; i < *pCount; ++i) g_phys[pPhysicalDevices[i]] = chain;
    }
    return res;
}

// ---------------------------------------------------------------------------
// Device hooks
// ---------------------------------------------------------------------------
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDevice(
    VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    auto* link = const_cast<VkLayerDeviceCreateInfo*>((const VkLayerDeviceCreateInfo*)pCreateInfo->pNext);
    while (link && !(link->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                     link->function == VK_LAYER_LINK_INFO))
        link = (VkLayerDeviceCreateInfo*)link->pNext;
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr next_gipa = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr next_dpa = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    auto create = (PFN_vkCreateDevice)next_gipa(VK_NULL_HANDLE, "vkCreateDevice");
    if (!create) return VK_ERROR_INITIALIZATION_FAILED;

    // A second node in the same chain carries vkSetDeviceLoaderData. Every dispatchable object this
    // layer allocates has to be passed through it; see SetLoaderData below for why.
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
    for (const auto* n = (const VkLayerDeviceCreateInfo*)pCreateInfo->pNext; n;
         n = (const VkLayerDeviceCreateInfo*)n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
            n->function == VK_LOADER_DATA_CALLBACK) {
            setLoaderData = n->u.pfnSetDeviceLoaderData;
            break;
        }
    }

    InstanceChain* ic = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_phys.find(physicalDevice);
        if (it != g_phys.end()) ic = it->second;
    }

    // Ask for one extension the game did not.
    //
    // VK_EXT_external_memory_host is what lets the proxy be handed to the helper without a copy: both
    // processes import their own mapping of the same pages as VkDeviceMemory and the GPU reads and
    // writes them in place. The game has no reason to enable it, so the layer adds it -- which is
    // allowed, and is what layers that need a device feature do. If the device does not offer it the
    // list is left exactly as the game wrote it and the transport keeps copying.
    // VK_EXT_external_memory_host is what lets the transport buffers BE the shared-memory regions,
    // so the proxy and the model's answer never pass through a private staging copy. The two fd
    // extensions do the same job across the process boundary: VK_KHR_external_memory_fd is what
    // vkGetMemoryFdKHR and the fd imports need, and VK_EXT_external_memory_dma_buf names the handle
    // type the images are shared as. They are device extensions and the application decides what
    // the device enables, but a layer may add to that list on the way down -- and does, when the
    // pass is on, the device offers them, and the app did not already enable them. If any of that
    // is false the composition falls back to the next transport down.
    static const char* const kWantExts[] = { VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
                                             VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                             VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                                             // Last on purpose: the three above are the transport's
                                             // and decide hostImport, this one is generation's.
                                             VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME };
    constexpr size_t kWantCount = sizeof(kWantExts) / sizeof(kWantExts[0]);
    const VkDeviceCreateInfo* effective = pCreateInfo;
    VkDeviceCreateInfo modified = *pCreateInfo;
    std::vector<const char*> enabledExts;
    bool addedTransport = false, addedSwapchainMaint = false;
    if (LayerEnabled() && ic && ic->vkEnumerateDeviceExtensionProperties) {
        bool have[kWantCount] = {};
        uint32_t n = 0;
        ic->vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> avail(n);
        if (n && ic->vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &n, avail.data()) == VK_SUCCESS) {
            for (uint32_t i = 0; i < n; ++i)
                for (size_t k = 0; k < kWantCount; ++k)
                    if (!std::strcmp(avail[i].extensionName, kWantExts[k])) have[k] = true;
        }
        for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
            for (size_t k = 0; k < kWantCount; ++k)
                if (!std::strcmp(pCreateInfo->ppEnabledExtensionNames[i], kWantExts[k])) have[k] = false;
        for (size_t k = 0; k < kWantCount; ++k) {
            if (!have[k]) continue;
            if (enabledExts.empty()) {
                enabledExts.reserve(pCreateInfo->enabledExtensionCount + kWantCount);
                for (uint32_t i = 0; i < pCreateInfo->enabledExtensionCount; ++i)
                    enabledExts.push_back(pCreateInfo->ppEnabledExtensionNames[i]);
            }
            enabledExts.push_back(kWantExts[k]);
            if (k + 1 == kWantCount) addedSwapchainMaint = true;
            else addedTransport = true;
        }
        if (!enabledExts.empty()) {
            modified.enabledExtensionCount = uint32_t(enabledExts.size());
            modified.ppEnabledExtensionNames = enabledExts.data();
            effective = &modified;
        }
    }

    // Ask for the one feature the composition shader needs.
    //
    // Its storage images are declared with no format, because one binding serves surfaces of three
    // different formats and no single operand is right for all of them. That is what the pass has
    // always needed; declaring a format it does not bind is undefined behaviour, and undefined values
    // in a channel is what a magenta or blue pixel is.
    //
    // Written wherever the game put its features: into VkPhysicalDeviceFeatures2 in the pNext chain
    // if it used that, into a copy of pEnabledFeatures otherwise, and into a fresh one if it asked
    // for no features at all. Only ever setting a bit, never clearing one.
    VkPhysicalDeviceFeatures ownFeatures{};
    if (pCreateInfo->pEnabledFeatures) ownFeatures = *pCreateInfo->pEnabledFeatures;
    VkPhysicalDeviceFeatures2* chained = nullptr;
    for (auto* n = (VkBaseOutStructure*) const_cast<void*>(pCreateInfo->pNext); n; n = n->pNext) {
        if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
            chained = (VkPhysicalDeviceFeatures2*) n;
            break;
        }
    }
    if (chained) {
        chained->features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    } else {
        ownFeatures.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    }

    // The features go into whichever create info is actually handed down. `modified` is a copy of the
    // game's, so writing into it is safe whether or not any extension was added.
    modified.pNext = pCreateInfo->pNext;
    // The extension does nothing unless its feature is asked for.
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapMaint{};
    if (addedSwapchainMaint) {
        swapMaint.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
        swapMaint.swapchainMaintenance1 = VK_TRUE;
        swapMaint.pNext = const_cast<void*>(modified.pNext);
        modified.pNext = &swapMaint;
    }
    if (!chained) modified.pEnabledFeatures = &ownFeatures;
    if (effective == pCreateInfo) {
        modified.enabledExtensionCount = pCreateInfo->enabledExtensionCount;
        modified.ppEnabledExtensionNames = pCreateInfo->ppEnabledExtensionNames;
    }
    effective = &modified;

    // The chain link the next layer reads. Saved because a retry has to hand the rest of the chain
    // the same starting point; the layers below advance it themselves as they call down.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    auto* const nextLayerInfo = link->u.pLayerInfo;
    VkResult res = create(physicalDevice, effective, pAllocator, pDevice);
    if (res != VK_SUCCESS && !enabledExts.empty()) {
        // The game's own list was fine; ours was not. Never turn a working device into a failed one
        // for the sake of an optimisation.
        Log("[layer] vkCreateDevice refused the added extensions (%d); retrying with the game's list",
            (int) res);
        link->u.pLayerInfo = nextLayerInfo;
        VkDeviceCreateInfo plain = *pCreateInfo;
        plain.pNext = pCreateInfo->pNext;
        if (!chained) plain.pEnabledFeatures = &ownFeatures;
        enabledExts.clear();
        addedTransport = addedSwapchainMaint = false;
        res = create(physicalDevice, &plain, pAllocator, pDevice);
    }
    if (res != VK_SUCCESS) return res;
    const bool wantHostImport = addedTransport;

    DeviceChain* dc = new DeviceChain();
    dc->hostImport = wantHostImport;
    dc->presentFence = addedSwapchainMaint;
    Log("[mfg] present fences %s (VK_EXT_swapchain_maintenance1 %s)",
        addedSwapchainMaint ? "available" : "not available",
        addedSwapchainMaint ? "enabled by the layer" : "absent or already the game's");
    dc->instance = ic;
    dc->physical = physicalDevice;
    dc->self = *pDevice;
    dc->next_dpa = next_dpa;
    dc->setDeviceLoaderData = setLoaderData;
#define X(name) dc->name = (PFN_##name)next_dpa(*pDevice, #name);
    DEVICE_FN_LIST(X)
#undef X
    dc->table.next_dpa = next_dpa;
    dc->table.Load(*pDevice);
    if (!dc->vkQueuePresentKHR || !dc->vkCreateSwapchainKHR || !ic) dc->inert = true;

    // Neural Rendering is an NGX feature and the helper only ever creates its own device on an
    // NVIDIA GPU, so on anything else there is nothing for this layer to do but cost a round trip.
    // Hybrid machines are the case that matters: an implicit layer is loaded for every device the
    // loader builds, including the integrated one a game may well be running on.
    char deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = "?";
    if (ic && ic->vkGetPhysicalDeviceProperties) {
        VkPhysicalDeviceProperties props{};
        ic->vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::snprintf(deviceName, sizeof(deviceName), "%s", props.deviceName);
        if (props.vendorID != 0x10DE) {
            dc->inert = true;
            Log("[layer] inert on non-NVIDIA device (vendor %#x): %s", props.vendorID, deviceName);
        }
    }

    std::lock_guard<std::mutex> lk(g_stateMutex);
    g_devices[*pDevice] = dc;
    Log("[layer] vkCreateDevice -> %p on %s (inert=%d enabled=%d)", (void*)*pDevice, deviceName,
        (int) dc->inert.load(), (int) LayerEnabled());
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDevice(VkDevice device,
                                                     const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) { dc = it->second; g_devices.erase(it); }
    }
    if (!dc) return;
    {
        std::lock_guard<std::mutex> lk(dc->lock);
        for (auto& kv : dc->swapchains) ReleasePrimary(device, kv.first);
    }
    if (dc->vkDeviceWaitIdle) dc->vkDeviceWaitIdle(device);
    {
        std::lock_guard<std::mutex> lk(dc->lock);
        for (auto& kv : dc->swapchains) {
            SwapchainState& sc = kv.second;
            sc.comp.reset();
            for (VkFence f : sc.fence)
                if (f) dc->vkDestroyFence(device, f, nullptr);
            if (sc.fenceLeg1) dc->vkDestroyFence(device, sc.fenceLeg1, nullptr);
            if (sc.fenceLeg2) dc->vkDestroyFence(device, sc.fenceLeg2, nullptr);
            if (sc.pool) dc->vkDestroyCommandPool(device, sc.pool, nullptr);
        }
        dc->swapchains.clear();
    }
    if (dc->vkDestroyDevice) dc->vkDestroyDevice(device, pAllocator);
    delete dc;
}

static DeviceChain* FindDevice(VkDevice device) {
    std::lock_guard<std::mutex> lk(g_stateMutex);
    auto it = g_devices.find(device);
    return it == g_devices.end() ? nullptr : it->second;
}

static void RememberQueue(DeviceChain* dc, VkQueue queue, uint32_t family) {
    if (!queue) return;
    std::lock_guard<std::mutex> lk(dc->lock);
    dc->queueFamilies[queue] = family;
}

static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue(VkDevice device, uint32_t family,
                                                      uint32_t index, VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->vkGetDeviceQueue) return;
    dc->vkGetDeviceQueue(device, family, index, pQueue);
    RememberQueue(dc, *pQueue, family);
}

// The 1.1 way of asking for a queue, and the only way to reach one created with
// VkDeviceQueueCreateFlags. A game that uses it never registered its queue through the hook above,
// so the present path could not tell which family the queue belonged to and fell back to family
// zero -- which is the family the command pool was then created on, and need not be the queue's.
static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue2(VkDevice device,
                                                       const VkDeviceQueueInfo2* pQueueInfo,
                                                       VkQueue* pQueue) {
    DeviceChain* dc = FindDevice(device);
    if (!dc || !dc->vkGetDeviceQueue2) return;
    dc->vkGetDeviceQueue2(device, pQueueInfo, pQueue);
    if (pQueueInfo) RememberQueue(dc, *pQueue, pQueueInfo->queueFamilyIndex);
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------
// Which swapchain formats the pass can work in. Every one of them has a UNORM twin the composition
// uses internally; the ten-bit and float entries are new here, and are what lets an HDR game reach
// the model at all -- the encode is exactly the step that turns open-ended light into the kind of
// picture the model was trained on.
static bool SupportedFormat(VkFormat f) {
    return dlssnr::CompositionFormat(f) != VK_FORMAT_UNDEFINED;
}

// What a swapchain's format and colour space together say about the light in the frame.
//
// A float swapchain is the easy case: games hand over linear light and the HDR path divides it by
// the white point and hands the model the result. The ten-bit formats are the ones worth the colour
// space: on a desktop set to HDR10 they carry ST 2084 code -- absolute nits, which is why the old
// display-referred reading of them (tone map as if it were SDR) banding-crushed them to eight bits
// on the way to the model. A ten-bit swapchain in an SDR colour space is just a bit more precision
// on a tone-mapped frame, and stays on the SDR path.
static uint32_t DetectHdrKind(VkFormat f, VkColorSpaceKHR cs) {
    if (f == VK_FORMAT_R16G16B16A16_SFLOAT) return kHdrLinearFp16;
    const bool tenBit = f == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
                        f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
                        f == VkFormat(1000452000) /* R12G12B12A16_UNORM_PACK32 */;
    const bool pq = cs == VK_COLOR_SPACE_HDR10_ST2084_EXT ||
                    cs == VkColorSpaceKHR(1000459000) /* HDR10_ST2084_COMPATIBLE */;
    if (tenBit && pq) return kHdrPq10;
    // A float swapchain in a linear BT.2020 space is still linear light; nothing else here is HDR.
    return kHdrNone;
}

// Finding the game's depth buffer.
//
// A layer is not confined to the present hook -- it sees every call the game makes, including the
// ones that create and use render targets. That is how the reference project finds a HUD-less colour
// buffer in games that do not tag one: track resource creation and identify the right target by what
// it looks like. The same is available here, and saying otherwise was wrong.
//
// This is the first step of that: watch every image the game creates and report the depth ones. If
// the depth buffer a scene is rendered with turns out to be identifiable and stable -- one image, at
// the swapchain's size, created once and reused -- then fetching it is a matter of copying it at the
// right moment. If instead there are thirty of them at every size, the heuristic has to be much
// cleverer and that is worth knowing before writing it.
//
// Reporting only. Nothing is copied and nothing is kept. DLSSNR_SCAN=1.
static bool ScanEnabled() {
    static const bool v = [] {
        const char* e = getenv("DLSSNR_SCAN");
        return e && e[0] == '1';
    }();
    return v;
}

static bool IsDepthFormat(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImage(VkDevice device, const VkImageCreateInfo* ci,
                                                       const VkAllocationCallbacks* alloc,
                                                       VkImage* out) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    // The cached pointer, resolved once at device creation like every other entry point here.
    //
    // The first version of this resolved it through the loader on every call and, when the device
    // lookup missed, returned VK_ERROR_INITIALIZATION_FAILED -- an error it invented. Every image the
    // game made then failed and it died on launch. A probe that reports must never be able to refuse:
    // if there is nothing to call through to, the honest thing is to say so once and stop watching,
    // not to break image creation.
    // A chain-wide fallback, remembered the first time any device resolves one.
    //
    // The pointer below us is a property of the layer chain, not of one device: whatever sits next
    // dispatches on the device handle it is given, so a pointer learned from one device is the right
    // one to call for another. That matters because the alternative, when this device is not in the
    // map, is to fail the call -- and failing vkCreateImage kills the process. A probe must never be
    // able to refuse. If the map misses, the fallback still creates the image and we simply do not
    // record that one.
    static std::atomic<PFN_vkCreateImage> g_anyCreateImage{nullptr};

    PFN_vkCreateImage next = dc ? dc->vkCreateImage : nullptr;
    if (!next && dc && dc->next_dpa) next = (PFN_vkCreateImage)dc->next_dpa(device, "vkCreateImage");
    if (next) {
        g_anyCreateImage.store(next, std::memory_order_relaxed);
    } else {
        next = g_anyCreateImage.load(std::memory_order_relaxed);
        static std::once_flag said;
        std::call_once(said, [] { Log("[scan] device not in the map; calling through the chain pointer"); });
    }
    if (!next) {
        // Unreachable in practice: the loader only hands this pointer out for a device we made, so
        // the fallback is always already set. Kept because the one thing this must never do is
        // invent a failure, and there is nothing left to call.
        static std::once_flag none;
        std::call_once(none, [] { Log("[scan] no vkCreateImage anywhere in the chain; images unhooked"); });
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkResult r = next(device, ci, alloc, out);
    if (r == VK_SUCCESS && ci && ScanEnabled() && IsDepthFormat(ci->format)) {
        // Counted per shape rather than logged per image: a game makes thousands and the useful
        // question is which shapes exist, not how many times each was made.
        static std::mutex m;
        static std::map<uint64_t, uint32_t> seen;
        const uint64_t key = (uint64_t(ci->extent.width) << 40) ^ (uint64_t(ci->extent.height) << 16) ^
                             uint64_t(ci->format);
        std::lock_guard<std::mutex> lk(m);
        const uint32_t n = ++seen[key];
        if (n == 1 || n == 10 || n == 100 || n == 1000)
            Log("[scan] depth image %ux%u fmt=%d samples=%d usage=%#x tiling=%d  (seen %u)",
                ci->extent.width, ci->extent.height, (int)ci->format, (int)ci->samples,
                (unsigned)ci->usage, (int)ci->tiling, n);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc || !dc->vkCreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;

    VkSwapchainCreateInfoKHR m = *pCreateInfo;
    if (!dc->inert && LayerEnabled() && SupportedFormat(pCreateInfo->imageFormat))
        m.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // Frame generation presents images the game never asked for, so they have to be asked for here.
    //
    // Without this the acquire always times out: a game sizes its swapchain for its own acquire and
    // present cycle and every image is either held by it or held by the presentation engine, which is
    // not a bug -- there is genuinely nothing spare. The extra images are requested at creation
    // because that is the only time they can be, which does mean turning generation on takes effect
    // at the next swapchain the game makes rather than instantly.
    //
    // Asked for by trying rather than by reading the surface capabilities: maxImageCount may be zero
    // for "no limit" and the driver is the authority either way, so the bumped count is attempted and
    // the game's own count used if it is refused. A game that would have got a swapchain always gets
    // one.
    // The image count is left exactly as the game asked for it.
    //
    // Extra images were requested so that a generated frame would have one to go into. They are not
    // free: a swapchain is a contract between the application and the presentation engine about how
    // many pictures are in flight, and widening it changes when the application's own acquire
    // returns. With the holding removed there is nothing to hold them for anyway -- an image is taken
    // and presented inside one call or not taken at all -- and this was the last thing that still
    // behaved differently merely because generation was switched on, in runs that generated nothing
    // and stalled regardless.
    //
    // Generation is rarer for it. That is the correct trade against a game that does not run.
    const bool bumped = false;
    VkResult res = dc->vkCreateSwapchainKHR(device, &m, pAllocator, pSwapchain);
    if (res != VK_SUCCESS && bumped) {
        Log("[mfg] %u swapchain images refused; falling back to the game's %u (generation will not run)",
            m.minImageCount, pCreateInfo->minImageCount);
        m.minImageCount = pCreateInfo->minImageCount;
        res = dc->vkCreateSwapchainKHR(device, &m, pAllocator, pSwapchain);
    }
    if (res != VK_SUCCESS || dc->inert || !LayerEnabled()) return res;

    uint32_t count = 0;
    dc->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, nullptr);
    std::vector<VkImage> images(count);
    dc->vkGetSwapchainImagesKHR(device, *pSwapchain, &count, images.data());

    SwapchainState sc{};
    sc.images = std::move(images);
    sc.format = pCreateInfo->imageFormat;
    sc.hdrKind = DetectHdrKind(sc.format, pCreateInfo->imageColorSpace);
    sc.width = pCreateInfo->imageExtent.width;
    sc.height = pCreateInfo->imageExtent.height;
    sc.passThrough = !SupportedFormat(sc.format) || sc.width > kMaxW || sc.height > kMaxH;
    sc.presentMode = pCreateInfo->presentMode;

    std::lock_guard<std::mutex> lk(dc->lock);
    // Tell the helper what the game presents in.
    //
    // It has never needed to know: the proxy crosses in whatever the encode chose. Frame generation
    // is the first thing on that side that is told a backbuffer format, and inventing one there meant
    // claiming R8G8B8A8 while the game presents B8G8R8A8 -- the same bytes in the other order.
    if (dc->shm.hdr) {
        dc->shm.hdr->swapchainFormat.store((uint32_t)sc.format);
        dc->shm.hdr->swapchainImageCount.store(count);
    }

    if (bumped)
        Log("[mfg] asked for %u swapchain images (the game asked for %u); got %u",
            m.minImageCount, pCreateInfo->minImageCount, count);
    Log("[layer] swapchain %p %ux%u fmt=%d hdr=%u passThrough=%d%s", (void*)*pSwapchain,
        pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height,
        (int)pCreateInfo->imageFormat, sc.hdrKind, (int)sc.passThrough,
        sc.passThrough ? (SupportedFormat(sc.format) ? " (too large)" : " (unsupported format)") : "");
    dc->swapchains[*pSwapchain] = std::move(sc);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroySwapchainKHR(VkDevice device,
    VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end()) dc = it->second;
    }
    if (!dc) return;
    ReleasePrimary(device, swapchain);
    std::unique_lock<std::mutex> lk(dc->lock);
    auto it = dc->swapchains.find(swapchain);
    if (it != dc->swapchains.end()) {
        lk.unlock();
        if (dc->vkDeviceWaitIdle) dc->vkDeviceWaitIdle(device);
        lk.lock();
        SwapchainState& sc = it->second;

        // The swapchain goes first, and the order is load-bearing.
        //
        // Frame generation leaves two kinds of operation outstanding that vkDeviceWaitIdle does not
        // cover, because neither is queue work: a present waiting on this swapchain's semaphore, and
        // an acquire that has signalled one for an image still held. Destroying those semaphores
        // while the presentation engine still refers to them is what it sounds like. It never showed
        // up on vkcube because vkcube keeps one swapchain for its whole life; Half-Life builds three
        // during startup and throws two away, and crashed there.
        //
        // vkDestroySwapchainKHR is what actually retires those references. After it returns, nothing
        // of the presentation engine's can be pointing at anything below.
        std::vector<VkSemaphore> retiredSemaphores;
        std::vector<VkFence> retiredFences;
        for (VkSemaphore sem : sc.fg.acquired)
            if (sem) retiredSemaphores.push_back(sem);
        for (auto& pp : sc.fg.ring) {
            if (pp.sem) retiredSemaphores.push_back(pp.sem);
            if (pp.fence) retiredFences.push_back(pp.fence);
        }
        sc.fg.ring.clear();

        if (dc->vkDestroySwapchainKHR) dc->vkDestroySwapchainKHR(device, swapchain, pAllocator);

        sc.comp.reset();
        for (VkFence f : sc.fence)
            if (f) dc->vkDestroyFence(device, f, nullptr);
        for (VkFence f : sc.fg.fence)
            if (f) dc->vkDestroyFence(device, f, nullptr);
        for (VkSemaphore sem : retiredSemaphores) dc->vkDestroySemaphore(device, sem, nullptr);
        for (VkFence f : retiredFences) dc->vkDestroyFence(device, f, nullptr);
        if (sc.fenceLeg1) dc->vkDestroyFence(device, sc.fenceLeg1, nullptr);
        if (sc.fenceLeg2) dc->vkDestroyFence(device, sc.fenceLeg2, nullptr);
        if (sc.pool) dc->vkDestroyCommandPool(device, sc.pool, nullptr);
        dc->swapchains.erase(it);
        lk.unlock();
        return;
    }
    lk.unlock();
    if (dc->vkDestroySwapchainKHR) dc->vkDestroySwapchainKHR(device, swapchain, pAllocator);
}


// ---------------------------------------------------------------------------
// Frame generation
// ---------------------------------------------------------------------------
static bool NoteVk(DeviceChain* dc, VkResult r, const char* what);

// Why a gap went unfilled, said a few times and then not again. A generated frame that never happens
// is not an error and must not fill a log, but "generated 0, missed 3016" with no reason attached is
// not something anyone can act on either.
static void MfgMiss(const char* why) {
    static std::mutex m;
    static std::map<std::string, uint32_t> seen;
    std::lock_guard<std::mutex> lk(m);
    const uint32_t n = ++seen[why];
    if (n <= 3 || n == 100 || n == 1000) Log("[mfg] no frame generated -- %s (%u)", why, n);
}
// Whether to generate on this swapchain this frame, and how many.
//
// Two of these conditions are refusals rather than settings. A swapchain of fewer than three images
// has no image to spare -- taking one would be taking it from the game -- and under a present mode
// that does not pace, an extra present is not placed in the gap so much as raced into it: FIFO shows
// queued presents one vblank apart, which is what puts the generated frame where it belongs, while
// MAILBOX may discard it and IMMEDIATE may show it at once and make the pair a stutter rather than a
// smoothing. Mode 1 lifts the second refusal for anyone who wants to see it anyway.
static uint32_t MfgFactorFor(DeviceChain* dc, const SwapchainState& sc, uint32_t* stateOut) {
    uint32_t state = 0;  // off
    uint32_t factor = 0;
    ShmHeader* h = dc->shm.hdr;
    if (h && h->mfgEnabled.load(std::memory_order_relaxed)) {
        state = 1;  // on, but not generating
        const bool pacedMode = sc.presentMode == VK_PRESENT_MODE_FIFO_KHR ||
                               sc.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
        if (!dc->presentFence && !pacedMode) {
            // Without a fence on the present, the only thing left is the ordering argument in
            // TakePresentPair, and that argument is about a queue the engine retires in order. Under
            // a present mode that may drop or reorder, it does not hold and there is nothing else to
            // fall back to.
            static std::once_flag said;
            std::call_once(said, [] {
                Log("[mfg] no VK_EXT_swapchain_maintenance1 and no paced present mode: nothing here "
                    "can tell when a present has finished, so generation will not run");
            });
            state = 3;
        } else if (sc.fg.unavailable || !sc.fg.ready || !sc.comp) {
            state = 3;  // unavailable on this swapchain
        } else if (sc.images.size() < 3) {
            state = 3;
        } else if (!h->pipeline.load(std::memory_order_relaxed)) {
            // Generation lives on the pipelined path and cannot be lifted off it.
            //
            // What it needs is a measurement of how far the picture moved, and the estimator makes
            // that from the proxy just encoded against the proxy kept aside when the outstanding
            // request went out. The second of those exists only while an answer is in flight, which
            // is what the pipelined path is. Waiting for the model instead means there is never a
            // frame in flight to measure against, so there is no displacement to carry a frame
            // forward along and nothing to generate from.
            state = 4;  // on, but the pipelined path is off
        } else {
            const uint32_t mode = h->mfgMode.load(std::memory_order_relaxed);
            const bool paced = sc.presentMode == VK_PRESENT_MODE_FIFO_KHR ||
                               sc.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;
            if (mode != 0 || paced) {
                const uint32_t f = h->mfgFactor.load(std::memory_order_relaxed);
                factor = f < 1 ? 1u : (f > 3 ? 3u : f);
                state = 2;  // generating
            }
        }
    }
    if (stateOut) *stateOut = state;
    return factor;
}


// How many frames to generate this present, found by measurement rather than set.
//
// The question "how many generated frames fit between two real ones" has no answer a layer can look
// up. It depends on the refresh rate, on how far below it the game is running, and on how readily the
// presentation engine lets an image go -- and the first two are exactly what a game changes from one
// scene to the next. What a layer can see is the consequence, so that is what this measures: add a
// frame, and watch whether the game's own frame rate survives it.
//
// The distinction being drawn is between generation and displacement, and it is not academic. On a
// display already receiving a new frame every vertical blank there is no gap, and a generated frame
// takes a real frame's slot: measured on a 144 Hz display, adding one took the game from 2866 real
// frames to 1475 over twenty seconds. The total presented was unchanged. That is not frame
// generation and no fixed count can tell it apart from the case where it works -- but the real frame
// rate falling by a third says it plainly, one window later.
//
// So: measure the baseline with nothing generated, then climb while the real rate holds and step back
// when it does not. The baseline is re-measured every so often, because a scene that gets cheaper
// moves the answer and a number decided once would sit there being wrong.
static uint32_t MfgDynamicFactor(DeviceChain* dc, SwapchainState& sc, uint32_t ceiling) {
    ShmHeader* h = dc->shm.hdr;
    if (!h) return ceiling;
    const bool autoCount = h->mfgAuto.load(std::memory_order_relaxed) != 0;

    // The window runs whether or not the count is being measured.
    //
    // It used to return early when the count was pinned, and the wait is measured in here -- so
    // pinning the count silently switched off the only thing that finds a wait, the wait stayed at
    // zero, and almost every gap went unfilled. Two settings that look independent were not: 64
    // frames generated against 4419 missed, with "waiting 0 us" underneath a count the user had
    // deliberately fixed at three.
    static constexpr uint32_t kWindow = 120;

    const double now = NowMs();
    if (sc.fg.windowStartMs == 0.0) sc.fg.windowStartMs = now;
    ++sc.fg.windowFrames;
    if (sc.fg.windowFrames < kWindow) {
        const uint32_t use = autoCount ? sc.fg.active : ceiling;
        return use > ceiling ? ceiling : use;
    }

    const double elapsed = now - sc.fg.windowStartMs;
    const double rate = elapsed > 0.0 ? double(sc.fg.windowFrames) * 1000.0 / elapsed : 0.0;

    // The reference is the best rate seen lately rather than a rate measured with generation
    // switched off.
    //
    // Measuring a baseline meant forcing the count to zero every so often, which stopped generation
    // for a whole window to re-learn something it already knew, and on a game that is merely
    // expensive it settled at zero and stayed there. The best recent rate answers the same question
    // -- "is this costing the game anything" -- without giving up a window to ask it, and it decays
    // slowly so a scene that gets genuinely heavier moves the reference instead of looking like a
    // regression forever.
    if (rate > sc.fg.baseRate) sc.fg.baseRate = rate;
    else sc.fg.baseRate *= 0.995;
    const bool healthy = sc.fg.baseRate <= 0.0 || rate >= sc.fg.baseRate * 0.92;

    // The wait, always. Its ceiling is a quarter of the frame the game is actually achieving: not a
    // number anyone chose, but the only budget there is -- spend more than that inside a present and
    // the frame rate is being paid out of rather than filled in.
    const uint32_t budgetUs = rate > 1.0 ? uint32_t((1000000.0 / rate) * 0.25) : 0u;
    const bool missingALot = sc.fg.windowMissed > sc.fg.windowFrames / 4;
    if (!healthy) {
        if (sc.fg.waitUs > 0) {
            sc.fg.waitUs /= 2;
            Log("[mfg] %.0f fps against %.0f: waiting %u us for an image", rate, sc.fg.baseRate,
                sc.fg.waitUs);
        }
    } else if (missingALot && sc.fg.waitUs < budgetUs) {
        const uint32_t step = budgetUs / 8 ? budgetUs / 8 : 1u;
        sc.fg.waitUs = sc.fg.waitUs + step > budgetUs ? budgetUs : sc.fg.waitUs + step;
        Log("[mfg] %u of %u presents found nothing free: waiting %u us of a %u us budget",
            sc.fg.windowMissed, sc.fg.windowFrames, sc.fg.waitUs, budgetUs);
    }

    // The count, only when it is being measured. The wait is given up before the count is, because
    // waiting is the part that can cost the game anything.
    if (autoCount) {
        if (!healthy && sc.fg.waitUs == 0) {
            if (sc.fg.active > 0) {
                --sc.fg.active;
                Log("[mfg] %.0f fps against %.0f: generating %u per frame", rate, sc.fg.baseRate,
                    sc.fg.active);
            }
        } else if (healthy && !missingALot && sc.fg.active < ceiling) {
            ++sc.fg.active;
            Log("[mfg] room for another: generating %u per frame at %.0f fps", sc.fg.active, rate);
        }
        if (sc.fg.active > ceiling) sc.fg.active = ceiling;
    } else {
        sc.fg.active = ceiling;
    }

    sc.fg.windowFrames = 0;
    sc.fg.windowMissed = 0;
    sc.fg.windowStartMs = now;
    h->mfgActiveFactor.store(sc.fg.active, std::memory_order_relaxed);
    h->mfgAcquireWaitUs.store(sc.fg.waitUs, std::memory_order_relaxed);
    return sc.fg.active;
}

// Claimed on the way out, not at the end of the caller's loop.
//
// It used to be marked in flight only once the whole batch had been recorded, so two frames in the
// same batch were handed the same pair: two presents waiting on one semaphore that is signalled once.
// The second waits for a signal that never comes, and the same semaphore appears twice in one
// submit's signal list, which is the VUID-00067 validation caught. Two generated frames per real one
// was all it took, and that is the count the climb reaches first.
static uint32_t TakePresentPair(DeviceChain* dc, SwapchainState& sc) {
    for (uint32_t i = 0; i < sc.fg.ring.size(); ++i) {
        auto& pp = sc.fg.ring[i];
        if (!pp.sem) continue;
        if (!pp.inFlight) { pp.inFlight = true; return i; }

        if (dc->presentFence && pp.fence) {
            if (dc->vkGetFenceStatus(dc->self, pp.fence) != VK_SUCCESS) continue;
            dc->vkResetFences(dc->self, 1, &pp.fence);
            return i;
        }

        // The fallback, for a driver without swapchain maintenance 1.
        //
        // Counting presents rather than asking. Under a paced present mode the engine retires
        // presents in the order they were queued, and every present needs an image the engine has
        // finished with -- so once this swapchain has taken as many further presents as it has
        // images, the one this pair was used for has necessarily completed. The margin is the
        // image count again, so the answer holds even if a present is dropped rather than displayed.
        //
        // This is an argument about ordering, not a guess about timing, which is the difference
        // between it and what was here before. It is still weaker than a fence, so the ring is
        // wider than it needs to be and generation simply stops when nothing is old enough.
        const uint64_t margin = uint64_t(sc.images.size()) * 2 + 2;
        if (sc.fg.presentSerial < pp.serial + margin) continue;
        return i;
    }
    return UINT32_MAX;
}

// A pair claimed and then not used goes straight back.
//
// Claiming on the way out fixed one bug and created this one: the claim happens before the acquire,
// and every acquire that finds nothing free abandoned its pair still marked in flight, with no fence
// ever attached to clear it. The ring drained within a few frames however wide it was -- widening it
// from eight to thirty-two changed nothing, which is what said the pairs were being lost rather than
// merely being slow to come back.
static void ReturnPresentPair(SwapchainState& sc, uint32_t pair) {
    if (pair < sc.fg.ring.size()) sc.fg.ring[pair].inFlight = false;
}

// Retire finished slots, decide whether a generated frame is wanted, and take an image for one.
//
// The acquire does not happen here, and that is the whole point. It was measured: a zero-timeout
// acquire on the game's own thread succeeds about once in three hundred, because an image is released
// when the presentation engine finishes with it -- at a vertical blank -- and the instant just after
// the game's present is the worst moment in the frame to ask. Waiting 12 ms instead made every
// acquire succeed and cost the game the 12 ms, inside vkQueuePresentKHR, which is not a trade worth
// making. So the waiting moved to a thread of its own that asks a thousand times a second with a zero
// timeout, and catches an image in the moment it comes free.
static void MfgAcquireAhead(DeviceChain* dc, SwapchainState& sc, VkSwapchainKHR swapchain,
                            uint32_t factor) {
    if (!sc.fg.ready || sc.fg.unavailable || !dc->vkAcquireNextImageKHR) return;

    for (uint32_t i = 0; i < SwapchainState::FrameGen::kSlots; ++i) {
        if (sc.fg.state[i] != SwapchainState::FrameGen::kSubmitted) continue;
        const uint32_t guard = sc.fg.guardedBy[i];
        if (dc->vkGetFenceStatus(dc->self, sc.fg.fence[guard]) != VK_SUCCESS) continue;
        sc.fg.state[i] = SwapchainState::FrameGen::kFree;
        // Reset the fence only with the slot that owns it, and only once every slot it covers has
        // been retired -- resetting it while a peer still needs to read it would strand that peer.
        if (i == guard) {
            bool others = false;
            for (uint32_t j = 0; j < SwapchainState::FrameGen::kSlots; ++j)
                if (j != i && sc.fg.state[j] == SwapchainState::FrameGen::kSubmitted &&
                    sc.fg.guardedBy[j] == guard) { others = true; break; }
            if (!others) dc->vkResetFences(dc->self, 1, &sc.fg.fence[guard]);
        }
    }
    // A fence whose owner retired before its peers is reset once they are all done.
    for (uint32_t i = 0; i < SwapchainState::FrameGen::kSlots; ++i) {
        if (sc.fg.state[i] != SwapchainState::FrameGen::kFree) continue;
        bool guards = false;
        for (uint32_t j = 0; j < SwapchainState::FrameGen::kSlots; ++j)
            if (sc.fg.state[j] == SwapchainState::FrameGen::kSubmitted && sc.fg.guardedBy[j] == i)
                { guards = true; break; }
        if (!guards && dc->vkGetFenceStatus(dc->self, sc.fg.fence[i]) == VK_SUCCESS)
            dc->vkResetFences(dc->self, 1, &sc.fg.fence[i]);
    }

    // The gate belongs before an image is taken, not after.
    //
    // It was the other way round and that was the whole reason almost nothing was generated: an image
    // was acquired, the displacement then turned out to be unusable, and the image stayed held --
    // legal, since it is released by presenting it and a generated frame is the only thing that will,
    // but it meant the next present found the slot occupied and never asked again. Reading the
    // displacement first costs nothing: the estimate is a few frames old either way, so it is exactly
    // as good an answer at the top of the present as at the bottom.
    float dx = 0.0f, dy = 0.0f, conf = 0.0f;
    bool want = true;
    if (!sc.comp || !sc.comp->ReadGlobalMotion(dx, dy, conf)) {
        MfgMiss("no displacement measured yet");
        want = false;
    } else {
        // What the estimator says its answer is worth, against what this needs it to be worth.
        // Tunable because the right number is a property of the estimator rather than of frame
        // generation, and it is the one number here that can only be set by looking at real values.
        // A spinning cube reads 0.05 to 0.10 and should be refused: rotation is not translation.
        static const float kMinConfidence = [] {
            const char* v = getenv("DLSSNR_MFG_CONFIDENCE");
            return v && *v ? (float)atof(v) : 0.15f;
        }();
        {
            static uint32_t n = 0;
            if ((n++ % 600) == 0)
                Log("[mfg] displacement %.2f, %.2f px at confidence %.3f (needs %.2f)", dx, dy, conf,
                    kMinConfidence);
        }
        if (conf < kMinConfidence) { MfgMiss("displacement not confident enough"); want = false; }
        // Below a pixel of travel there is nothing to carry forward and the generated frame would be
        // a second copy of the real one. Compared squared, so no square root is needed to ask it.
        else if (dx * dx + dy * dy < 1.0f) { MfgMiss("the picture is not moving"); want = false; }
    }
    sc.fg.wantFrames = 0;
    if (want) {
        sc.fg.lastDx = dx;
        sc.fg.lastDy = dy;
        sc.fg.motionValid = true;
        sc.fg.wantFrames = factor;
    }

    // No image is taken here, and that is the correction.
    //
    // This used to acquire "a present early" so that a zero-timeout acquire would have had a vertical
    // blank in which to succeed. It worked, and it starved the application: an image held across
    // frames is one the client cannot have, and a client that expects to acquire freely -- zink, and
    // every translation layer like it -- stops dead. Measured at one image held: the test app stalled
    // almost immediately, having generated exactly one frame and missed nothing, because the single
    // image it took never came back.
    //
    // So the image is taken at the moment it is used, in MfgRecordFrames, and presented in the same
    // call. Nothing is held between frames, and there is nothing the client can be starved of.
    (void)factor;
}

// Record the generated frames and submit them, before the real frame is presented.
//
// The order is the whole correctness argument. Generation reads the image the game is about to
// present, and once vkQueuePresentKHR has been called on an image the presentation engine owns it --
// reading it after that is a race, which synchronisation validation names exactly:
//
//   WRITE_AFTER_PRESENT: vkCmdPipelineBarrier writes to VkImage ..., which was previously written
//   by vkQueuePresentKHR
//
// It is not a theoretical race. It corrupted the driver badly enough to take Half-Life down with a
// jump through a junk pointer, within a second of the count reaching two generated frames per real
// one, when the reads doubled.
//
// So the read happens here, while the image is still the layer's, and the real present is made to
// wait on the semaphore this submit signals. One command buffer and one submit for all of them: they
// all read the same source, so recording them together is both cheaper and easier to reason about
// than n submits racing over one image.
//
// Why this and not DLSS-G: frame generation through nvngx_dlssg.dll cannot be created from a layer at
// all. CreateFeature(11) answers InvalidParameter, and it is right to -- the feature is built from a
// contract only the game can fill, camera matrices and depth and motion vectors and a HUD-less colour
// buffer, none of which exist below a swapchain. The project this was modelled on looks like it calls
// DLSS-G and does not: it implements the NGX entry points, receives that contract from a game that
// already speaks them, and answers with FSR. It is upstream of the constraint, not inside it.
//
// What is available below a swapchain is the frame itself and how far the picture moved to reach it,
// and a global displacement is exactly the part of frame generation a translation can express. So the
// warp is a blit with its source rectangle offset: linear filtering resamples, and there is no
// shader, no history and nothing to get out of step.
// Give back images that were acquired and will not be presented.
//
// An acquired image is released by presenting it and by nothing else, so every path that acquires and
// then gives up leaks one -- and a swapchain leaks its way to a stall one image at a time. There were
// four such paths in the recording loop below: a degenerate blit rectangle, an index out of range, a
// command buffer that would not end, and a submit that failed. Each of them returned without
// presenting, and the client was left one image poorer for good.
//
// vkReleaseSwapchainImagesEXT is the only thing that undoes an acquire. It requires the acquire's
// semaphore to have been waited on first, so an empty batch does that before the release. Without the
// extension there is nothing that can undo it, and the honest response is to stop generating on this
// swapchain rather than to keep leaking.
static void MfgReleaseAcquired(DeviceChain* dc, SwapchainState& sc, VkQueue queue,
                               VkSwapchainKHR swapchain, const uint32_t* slots,
                               const uint32_t* indices, uint32_t count) {
    if (!count) return;
    if (!dc->vkReleaseSwapchainImagesEXT || !dc->presentFence) {
        if (!sc.fg.unavailable) {
            sc.fg.unavailable = true;
            Log("[mfg] %u image(s) acquired and unusable, and nothing here can give them back; "
                "generation stops on this swapchain", count);
        }
        return;
    }
    VkSemaphore sems[SwapchainState::FrameGen::kSlots] = {};
    VkPipelineStageFlags stages[SwapchainState::FrameGen::kSlots] = {};
    uint32_t n = 0;
    for (uint32_t i = 0; i < count && i < SwapchainState::FrameGen::kSlots; ++i) {
        sems[n] = sc.fg.acquired[slots[i]];
        stages[n] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        ++n;
    }
    if (n) {
        VkSubmitInfo drain{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        drain.waitSemaphoreCount = n;
        drain.pWaitSemaphores = sems;
        drain.pWaitDstStageMask = stages;
        if (dc->vkQueueSubmit(queue, 1, &drain, VK_NULL_HANDLE) != VK_SUCCESS) return;
    }
    VkReleaseSwapchainImagesInfoEXT info{ VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT };
    info.swapchain = swapchain;
    info.imageIndexCount = count;
    info.pImageIndices = indices;
    NoteVk(dc, dc->vkReleaseSwapchainImagesEXT(dc->self, &info), "vkReleaseSwapchainImagesEXT");
}

struct MfgPending {
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    uint32_t count = 0;
    uint32_t index[SwapchainState::FrameGen::kSlots] = {};
    uint32_t slot[SwapchainState::FrameGen::kSlots] = {};
    uint32_t ring[SwapchainState::FrameGen::kSlots] = {};
    VkSemaphore wait[SwapchainState::FrameGen::kSlots] = {};
    VkFence fence[SwapchainState::FrameGen::kSlots] = {};
    VkSemaphore realWait = VK_NULL_HANDLE;
    VkFence realFence = VK_NULL_HANDLE;
};

static uint32_t MfgRecordFrames(DeviceChain* dc, SwapchainState& sc, VkQueue queue,
                                VkSwapchainKHR swapchain, VkImage presented, uint32_t presentedIndex,
                                MfgPending* out) {
    if (!sc.fg.wantFrames || !sc.comp || !dc->vkCmdBlitImage || !dc->vkAcquireNextImageKHR) return 0;
    if (!sc.fg.motionValid) return 0;
    if (sc.fg.ring.empty()) { MfgMiss("no present ring"); return 0; }

    // The displacement the acquire already read and accepted. An image is only held because that
    // check passed, so there is nothing to decide here -- and re-reading would risk taking a
    // different answer than the one the image was taken for.
    const float dx = sc.fg.lastDx, dy = sc.fg.lastDy;

    // The slot whose command buffer and fence carry the whole batch.
    const uint32_t lead = 0;
    if (sc.fg.state[lead] != SwapchainState::FrameGen::kFree) {
        MfgMiss("the batch before this one has not finished");
        ++sc.fg.windowMissed;
        return 0;
    }
    VkCommandBuffer cb = sc.fg.cb[lead];
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (dc->vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) { ++sc.fg.missed; return 0; }

    auto barrier = [&](VkImage img, VkImageLayout from, VkImageLayout to,
                       VkAccessFlags srcA, VkAccessFlags dstA) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.srcAccessMask = srcA;
        b.dstAccessMask = dstA;
        b.oldLayout = from;
        b.newLayout = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        dc->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // The composition has just written this image and left it in PRESENT_SRC_KHR. It is read here and
    // put back before anyone else sees it.
    barrier(presented, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    const int32_t w = int32_t(sc.width), h = int32_t(sc.height);
    uint32_t made = 0;
    VkSemaphore waitSems[SwapchainState::FrameGen::kSlots] = {};
    uint32_t waitCount = 0;
    // How many to try for, and the wait each may spend.
    //
    // mfgwait is no longer a timeout handed to every acquire. Blocking while holding an image is what
    // deadlocked the client -- it waits inside the client's own present for something only the client
    // can release. Only the first acquire may wait, because at that moment this layer holds nothing:
    // the wait then costs the frame some latency and cannot starve anyone. Every acquire after it is
    // made with a zero timeout, since by then an image is held and waiting would be the deadlock
    // again.
    const uint32_t n = sc.fg.wantFrames;
    // Every image acquired in this loop that does not end up presented, so it can be given back.
    uint32_t strandedSlot[SwapchainState::FrameGen::kSlots] = {};
    uint32_t strandedIndex[SwapchainState::FrameGen::kSlots] = {};
    uint32_t stranded = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t slot = k < SwapchainState::FrameGen::kSlots ? k : 0;
        if (sc.fg.state[slot] != SwapchainState::FrameGen::kFree) {
            MfgMiss("a slot is still in flight");
            ++sc.fg.missed;
            ++sc.fg.windowMissed;
            continue;
        }
        // The rectangle is decided before anything is claimed. It depends only on the displacement and
        // the frame size, both known here, and a degenerate one used to be discovered after the
        // acquire -- at which point the image was already gone and nothing gave it back.
        const float tPre = float(k + 1) / float(n + 1);
        int32_t px0 = int32_t(-dx * tPre), py0 = int32_t(-dy * tPre);
        int32_t px1 = w + px0, py1 = h + py0;
        if (px0 < 0) { px1 -= px0; px0 = 0; }
        if (py0 < 0) { py1 -= py0; py0 = 0; }
        if (px1 > w) { px0 -= (px1 - w); px1 = w; }
        if (py1 > h) { py0 -= (py1 - h); py1 = h; }
        if (px0 < 0) px0 = 0;
        if (py0 < 0) py0 = 0;
        if (px1 <= px0 || py1 <= py0) { MfgMiss("the warp would be degenerate"); ++sc.fg.missed; continue; }

        const uint32_t pair = TakePresentPair(dc, sc);
        if (pair == UINT32_MAX) {
            MfgMiss("no present pair has come back yet");
            ++sc.fg.missed;
            ++sc.fg.windowMissed;
            continue;
        }

        const uint64_t waitNs = (k == 0) ? uint64_t(sc.fg.waitUs) * 1000ull : 0ull;
        uint32_t index = 0;
        const VkResult acq = dc->vkAcquireNextImageKHR(dc->self, swapchain, waitNs,
                                                       sc.fg.acquired[slot], VK_NULL_HANDLE, &index);
        if (acq != VK_SUCCESS) {
            MfgMiss(acq == VK_NOT_READY || acq == VK_TIMEOUT ? "nothing free to acquire"
                                                             : "acquire failed");
            ReturnPresentPair(sc, pair);
            ++sc.fg.missed;
            ++sc.fg.windowMissed;
            break;
        }
        if (index >= sc.images.size() || sc.images[index] == presented) {
            ReturnPresentPair(sc, pair);
            ++sc.fg.missed;
            if (index < sc.images.size()) {
                strandedSlot[stranded] = slot;
                strandedIndex[stranded] = index;
                ++stranded;
            }
            break;
        }
        sc.fg.index[slot] = index;

        VkImage dst = sc.images[index];
        barrier(dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT);

        // Shifting the source rectangle by -d*t moves the picture by +d*t, and the blit's linear
        // filter resamples the fractional part. Clamped to the image, because a source rectangle
        // outside it is not merely empty, it is invalid.
        const int32_t x0 = px0, y0 = py0, x1 = px1, y1 = py1;

        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcOffsets[0] = { x0, y0, 0 };
        blit.srcOffsets[1] = { x1, y1, 1 };
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { w, h, 1 };
        dc->vkCmdBlitImage(cb, presented, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        barrier(dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT);

        out->index[made] = index;
        out->slot[made] = slot;
        out->wait[made] = sc.fg.ring[pair].sem;
        out->ring[made] = pair;
        ++made;
        waitSems[waitCount++] = sc.fg.acquired[slot];
    }

    barrier(presented, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT);

    // Ended exactly once, whatever happens next. The first version of this ended it again when
    // nothing had been recorded, which is invalid use of a command buffer that had already ended.
    const bool ended = dc->vkEndCommandBuffer(cb) == VK_SUCCESS;
    if (!ended || made == 0) {
        for (uint32_t k = 0; k < made; ++k) ReturnPresentPair(sc, out->ring[k]);
        for (uint32_t k = 0; k < made; ++k) {
            strandedSlot[stranded] = out->slot[k];
            strandedIndex[stranded] = out->index[k];
            ++stranded;
        }
        MfgReleaseAcquired(dc, sc, queue, swapchain, strandedSlot, strandedIndex, stranded);
        return 0;
    }
    MfgReleaseAcquired(dc, sc, queue, swapchain, strandedSlot, strandedIndex, stranded);
    stranded = 0;

    // Signalled: one semaphore per generated image for its own present, and the presented image's own
    // semaphore for the real present, which is what stops the engine reading a frame this submit is
    // still reading.
    VkSemaphore signalSems[SwapchainState::FrameGen::kSlots + 1] = {};
    uint32_t signalCount = 0;
    for (uint32_t k = 0; k < made; ++k) signalSems[signalCount++] = out->wait[k];
    // And one for the real present, so it too waits on this submit rather than on nothing.
    const uint32_t realPair = TakePresentPair(dc, sc);
    if (realPair != UINT32_MAX) signalSems[signalCount++] = sc.fg.ring[realPair].sem;

    VkPipelineStageFlags waitStages[SwapchainState::FrameGen::kSlots];
    for (uint32_t i = 0; i < waitCount; ++i) waitStages[i] = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waitCount ? waitSems : nullptr;
    si.pWaitDstStageMask = waitCount ? waitStages : nullptr;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = signalCount;
    si.pSignalSemaphores = signalSems;
    if (!NoteVk(dc, dc->vkQueueSubmit(queue, 1, &si, sc.fg.fence[lead]), "vkQueueSubmit (mfg)")) {
        // Nothing was signalled, so these cannot be presented -- they have to be handed back.
        for (uint32_t k = 0; k < made; ++k) ReturnPresentPair(sc, out->ring[k]);
        if (realPair != UINT32_MAX) ReturnPresentPair(sc, realPair);
        sc.fg.missed += made;
        for (uint32_t k = 0; k < made; ++k) {
            strandedSlot[stranded] = out->slot[k];
            strandedIndex[stranded] = out->index[k];
            ++stranded;
        }
        MfgReleaseAcquired(dc, sc, queue, swapchain, strandedSlot, strandedIndex, stranded);
        return 0;
    }

    // Only the slots that actually contributed are in flight.
    //
    // Marking all of them was wrong in a way that took a while to see: a slot skipped in the loop
    // above still holds an acquired image, and the only thing that ever releases an acquired image is
    // presenting it. Marked submitted, it was eventually freed and reacquired while its old image
    // stayed acquired forever -- the swapchain drained one image at a time until the application's
    // own acquire had nothing left, which is the hang, and validation caught the same wound from the
    // other side as an image index presented that "was not acquired from the swapchain".
    for (uint32_t k = 0; k < made; ++k) {
        sc.fg.state[out->slot[k]] = SwapchainState::FrameGen::kSubmitted;
        sc.fg.guardedBy[out->slot[k]] = lead;
    }
    sc.fg.generated += made;
    out->count = made;
    if (realPair != UINT32_MAX) {
        out->realWait = sc.fg.ring[realPair].sem;
        out->realFence = sc.fg.ring[realPair].fence;
        sc.fg.ring[realPair].serial = ++sc.fg.presentSerial;
    }
    for (uint32_t k = 0; k < made; ++k) {
        out->fence[k] = sc.fg.ring[out->ring[k]].fence;
        sc.fg.ring[out->ring[k]].serial = ++sc.fg.presentSerial;
    }
    return made;
}

// ---------------------------------------------------------------------------
// Present-time neural round-trip
// ---------------------------------------------------------------------------
// Give a dispatchable object this layer allocated the dispatch table the loader expects on it.
//
// VkCommandBuffer and VkQueue are dispatchable: their first word points at a dispatch table, and
// every layer below reads it to find its own state for that object. The loader fills that word in
// for objects the application allocates through the trampoline -- but a layer that allocates one by
// calling straight down the chain bypasses the trampoline, so the loader never sees it and the word
// keeps whatever the ICD left there. The loader hands each layer vkSetDeviceLoaderData precisely so
// it can do that fill-in itself, and calling it is mandatory, not advisory.
//
// Skipping it is invisible with no other layer present: the next call goes straight to the driver,
// which does not read the word. Add any second layer -- Steam's overlay, MangoHud, validation -- and
// that layer reads the word, finds the ICD's loader magic instead of a table, and aborts. Validation
// says so out loud: 'The VkDevice dispatch handle was not found and Validation will crash.'
static bool SetLoaderData(DeviceChain* dc, void* object) {
    if (!dc->setDeviceLoaderData) return true;  // no loader in the chain; nothing to fill in
    return dc->setDeviceLoaderData(dc->self, object) == VK_SUCCESS;
}

static bool CreateResources(DeviceChain* dc, SwapchainState& sc, uint32_t family) {
    VkDevice d = dc->self;

    VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = family;
    if (dc->vkCreateCommandPool(d, &cpci, nullptr, &sc.pool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = sc.pool;
    cbai.commandBufferCount = SwapchainState::kSlots;
    if (dc->vkAllocateCommandBuffers(d, &cbai, sc.cb) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    for (uint32_t i = 0; i < SwapchainState::kSlots; ++i) {
        if (!SetLoaderData(dc, sc.cb[i])) {
            Log("[layer] vkSetDeviceLoaderData failed for the present command buffer");
            return false;
        }
        if (dc->vkCreateFence(d, &fci, nullptr, &sc.fence[i]) != VK_SUCCESS) return false;
    }
    if (dc->vkCreateFence(d, &fci, nullptr, &sc.fenceLeg1) != VK_SUCCESS) return false;
    if (dc->vkCreateFence(d, &fci, nullptr, &sc.fenceLeg2) != VK_SUCCESS) return false;

    // Frame generation's own command buffers, fences and semaphores.
    //
    // Separate from the present path's on purpose: a generated frame is submitted after the real
    // present has already been made, so it cannot share a slot with work the real frame is still
    // using, and a failure to build any of it must cost the feature rather than the frame.
    {
        VkCommandBufferAllocateInfo fgai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        fgai.commandPool = sc.pool;
        fgai.commandBufferCount = SwapchainState::FrameGen::kSlots;
        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        bool ok = dc->vkAllocateCommandBuffers(d, &fgai, sc.fg.cb) == VK_SUCCESS;
        for (uint32_t i = 0; ok && i < SwapchainState::FrameGen::kSlots; ++i) {
            ok = SetLoaderData(dc, sc.fg.cb[i]) &&
                 dc->vkCreateFence(d, &fci, nullptr, &sc.fg.fence[i]) == VK_SUCCESS &&
                 dc->vkCreateSemaphore(d, &sci, nullptr, &sc.fg.acquired[i]) == VK_SUCCESS;
        }
        sc.fg.ring.assign(SwapchainState::FrameGen::kRing, SwapchainState::FrameGen::PresentPair{});
        for (size_t i = 0; ok && i < sc.fg.ring.size(); ++i) {
            ok = dc->vkCreateSemaphore(d, &sci, nullptr, &sc.fg.ring[i].sem) == VK_SUCCESS &&
                 dc->vkCreateFence(d, &fci, nullptr, &sc.fg.ring[i].fence) == VK_SUCCESS;
        }
        sc.fg.ready = ok;
        sc.fg.unavailable = !ok;
        if (!ok) Log("[mfg] frame generation resources unavailable on this swapchain; it will not generate");
    }

    if (!dc->instance) return false;
    sc.comp = std::make_unique<dlssnr::Composition>(&dc->table, &dc->instance->table, d, dc->physical);
    if (!sc.comp->Usable()) {
        Log("[layer] composition unavailable: %s", sc.comp->Reason());
        sc.comp.reset();
        return false;
    }
    return true;
}

// Every Vulkan result on the present path, looked at rather than collapsed into a bool.
//
// A failure here was previously indistinguishable from 'nothing to do': the call returned false, the
// caller presented the original frame, and the next frame tried exactly the same thing again. That is
// the right answer for a transient failure and the wrong one for VK_ERROR_DEVICE_LOST, where the
// device is gone, every subsequent call will fail the same way, and the log fills with it.
//
// Losing the device also latches the layer inert, because after that point the fail-open path is the
// only correct one and it costs nothing to take it directly.
static bool NoteVk(DeviceChain* dc, VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    if (r == VK_ERROR_DEVICE_LOST) {
        if (!dc->inert.exchange(true)) Log("[layer] %s -> DEVICE_LOST; layer inert for this device", what);
        return false;
    }
    static std::atomic<uint32_t> reported{0};
    if (reported.fetch_add(1) < 8) Log("[layer] %s -> %d", what, (int) r);
    return false;
}

// Returns true if the swapchain image now holds the composed frame.
//
// Three steps around one round trip. The pass encodes a proxy of the frame on the GPU, that proxy
// crosses to the helper and comes back as the model's answer, and the pass composes the answer onto
// the frame. The proxy's crossing is fenced on the CPU because the model is in another process and
// there is nothing to wait on but a sequence number; the answer's return is not -- leg 2 and the
// present are ordered by the queue itself, and the fence that covers leg 2 is only waited on at the
// start of the NEXT frame, where the command buffer and the composed surfaces are reused.
//
// The caller's present semaphores are consumed by the first submit, because that submit is the first
// thing to touch the image. They are therefore unsignalled by the time this returns and must not be
// handed to vkQueuePresentKHR again; the caller presents with none.
//
// Every path out leaves the swapchain image in PRESENT_SRC_KHR, including the ones that give up.
static bool ProcessPresent(DeviceChain* dc, SwapchainState& sc, VkQueue queue,
                           VkImage swapchainImage, uint32_t waitCount,
                           const VkSemaphore* waitSemaphores, bool* consumedWaits) {
    if (!sc.comp) return false;
    VkDevice d = dc->self;
    // Take the next slot and make sure its previous submission has landed. Three frames back, so in
    // practice this returns at once; it is a correctness guard, not a stall.
    const uint32_t slot = sc.slot;
    sc.slot = (sc.slot + 1) % SwapchainState::kSlots;
    if (sc.submitted[slot]) {
        const double tW = NowMs();
        if (!NoteVk(dc, dc->vkWaitForFences(dc->self, 1, &sc.fence[slot], VK_TRUE, UINT64_MAX),
                    "vkWaitForFences"))
            return false;
        dc->fenceWaitMs += NowMs() - tW;
        dc->vkResetFences(dc->self, 1, &sc.fence[slot]);
        sc.submitted[slot] = false;
    }
    VkCommandBuffer cb = sc.cb[slot];
    const bool time = TimeEnabled();
    const double t0 = time ? NowMs() : 0.0;

    // The previous frame's compose, if it is still running, must finish before anything here
    // touches the surfaces it reads or the command buffer it was recorded into. Waiting here rather
    // than at the end of that frame keeps the game thread out of the GPU's way for the whole of the
    // helper's round trip. The capture pair that compose recorded lands with it.
    if (sc.leg2Pending) {
        if (!NoteVk(dc, dc->vkWaitForFences(d, 1, &sc.fenceLeg2, VK_TRUE, UINT64_MAX),
                    "vkWaitForFences(leg2)"))
            return false;
        dc->vkResetFences(d, 1, &sc.fenceLeg2);
        sc.leg2Pending = false;
        sc.comp->WriteCapturedFrame();
    }

    // Not const: running alongside the frame overrides the composition bypass below, and the
    // override has to be the one every later reader sees rather than a second variable they might
    // forget to consult.
    dlssnr::FrameSettings fs = dlssnr::FrameSettings::Read(dc->shm.hdr);

    // The HDR decision, made once per frame before anything is sized or encoded.
    //
    // hdrActive is what this process intends; proxyFormat is what the helper actually built, and the
    // float encode is only taken when both agree -- a model that refused the float input leaves the
    // frame on the 8-bit path it has always used. hdrActive doubles as the echo the helper reads, so
    // it builds the float images only for a layer that has said it will feed them.
    const uint32_t hdrMode = dc->shm.hdr ? dc->shm.hdr->hdrMode.load() : kHdrAuto;
    const bool hdrActive = hdrMode != kHdrOff && (hdrMode == kHdrForce || sc.hdrKind != kHdrNone);
    const bool hdrProxy = hdrActive && dc->shm.hdr &&
                          dc->shm.hdr->proxyFormat.load() == kProxyRgba16F;
    const uint32_t hdrTransfer = sc.hdrKind == kHdrPq10 ? 1u : 0u;

    const bool linearHdr =
        hdrProxy || dlssnr::ColourIsLinearHdr(sc.format, dc->shm.hdr ? dc->shm.hdr->colourMode.load() : kColourAuto);

    // Point the transport at the shared-memory regions before Prepare sizes anything, so the first
    // frame at a new raster imports the mapping instead of building staging that then has to be
    // thrown away. The regions are mapped at the model's own size, which is what the GPU copies
    // into and out of.
    if (dc->shm.hdr && ShmOpen(dc->shm)) {
        uint32_t mw = 0, mh = 0;
        dlssnr::Composition::ModelExtent(sc.width, sc.height, fs, mw, mh);
        if (ShmMapFrames(dc->shm, size_t(mw) * mh * (hdrProxy ? 8 : 4)))
            sc.comp->SetTransport(dc->shm.inPixels, dc->shm.outPixels, dc->shm.mappedFrameBytes);
        else
            sc.comp->SetTransport(nullptr, nullptr, 0);
    } else {
        sc.comp->SetTransport(nullptr, nullptr, 0);
    }

        // ---- Phase 5: the dma-buf exchange ----
    //
    // The helper owns both images that cross the boundary and names their exported dma-bufs in
    // the header; this process takes its own reference on each through pidfd_getfd (or /proc on older kernels).
    // Everything here is best-effort and restated every frame: a descriptor that cannot be opened
    // or imported leaves that direction on the shared-memory transport, which is the arrangement
    // that shipped before. The flags written below say which surfaces this frame's bytes travel
    // through, and the helper honours them on exactly the frame they were set for.
    sc.comp->SetDmaBuf(DmaBufEnabled());

    if (!sc.comp->Prepare(sc.width, sc.height, sc.format, fs, linearHdr, hdrProxy, hdrTransfer)) {
        Log("[layer] composition cannot run here: %s", sc.comp->Reason());
        return false;
    }

    if (dc->shm.hdr) {
        dc->shm.hdr->hdrDetected.store(sc.hdrKind);
        // The intent, not the format-gated decision: the helper builds the float images only for a
        // layer that has said it will feed them, and that handshake has to start while the proxy is
        // still 8-bit. Publishing HdrProxyActive() here would wait on proxyFormat, which waits on
        // this field, and neither would ever move.
        dc->shm.hdr->hdrActive.store(hdrActive ? 1u : 0u);
    }

    if (sc.comp->DmaBuf() && dc->shm.hdr) {
        ShmHeader* hdr = dc->shm.hdr;
        const uint32_t ps = hdr->proxyExportSeq.load();
        if (ps && ps != dc->proxySeqSeen) {
            const int fd = AdoptPeerFd(hdr->proxyPid.load(), hdr->proxyFd.load());
            if (fd >= 0 && sc.comp->ImportProxy(fd, sc.comp->ModelWidth(), sc.comp->ModelHeight()))
                dc->proxySeqSeen = ps;
        } else if (!ps) {
            dc->proxySeqSeen = 0;
        }
        const uint32_t as = hdr->answerExportSeq.load();
        if (as && as != dc->answerSeqSeen) {
            const int fd = AdoptPeerFd(hdr->answerPid.load(), hdr->answerFd.load());
            if (fd >= 0 && sc.comp->ImportAnswerFd(fd, sc.comp->ModelWidth(), sc.comp->ModelHeight()))
                dc->answerSeqSeen = as;
        } else if (!as) {
            dc->answerSeqSeen = 0;
        }
    }

    // One decision, made before the request goes out: the echo the helper reads and the surfaces
    // this frame writes and composes from are the same decision, not two that must agree. The
    // echo names the export sequence this process holds a reference at, so the helper reads the
    // fd path only for the very image the layer imported -- not a stale one behind a restart.
    const bool answerViaFd = sc.comp->AnswerViaFd();
    sc.comp->SetAnswerViaFd(answerViaFd);
    if (dc->shm.hdr) {
        dc->shm.hdr->layerProxySeq.store(sc.comp->ProxyActive() ? dc->proxySeqSeen : 0u);
        dc->shm.hdr->layerAnswerSeq.store(answerViaFd ? dc->answerSeqSeen : 0u);
    }

    // A capture is asked for by writing a frame count into the header; taking it clears the request,
    // so one press produces one run rather than one per frame for as long as nobody clears it.
    if (dc->shm.hdr) {
        if (const uint32_t frames = dc->shm.hdr->captureRequest.exchange(0); frames > 0)
            sc.comp->RequestCapture(std::min<uint32_t>(frames, 64));
    }

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    // Only the first submit waits: the later one is ordered behind it on the same queue and fenced
    // besides, and a binary semaphore may be waited on once per signal.
    std::vector<VkPipelineStageFlags> waitStages(waitCount, VK_PIPELINE_STAGE_TRANSFER_BIT);
    si.waitSemaphoreCount = waitCount;
    si.pWaitSemaphores = waitCount ? waitSemaphores : nullptr;
    si.pWaitDstStageMask = waitCount ? waitStages.data() : nullptr;
    const auto dropWaits = [&] {
        si.waitSemaphoreCount = 0;
        si.pWaitSemaphores = nullptr;
        si.pWaitDstStageMask = nullptr;
    };

    const auto runLeg = [&](bool waitForIt = true) {
        if (!NoteVk(dc, dc->vkEndCommandBuffer(cb), "vkEndCommandBuffer")) return false;
        if (si.waitSemaphoreCount && consumedWaits) *consumedWaits = true;
        if (!NoteVk(dc, dc->vkQueueSubmit(queue, 1, &si, sc.fence[slot]), "vkQueueSubmit")) return false;
        sc.submitted[slot] = true;
        if (!waitForIt) return true;

        // The blocking round trip still has to wait: it hands the proxy over on this frame and needs
        // the answer before it can compose, so there is nothing to overlap with.
        const double tFence = NowMs();
        if (!NoteVk(dc, dc->vkWaitForFences(d, 1, &sc.fence[slot], VK_TRUE, UINT64_MAX),
                    "vkWaitForFences"))
            return false;
        dc->fenceWaitMs += NowMs() - tFence;
        dc->vkResetFences(d, 1, &sc.fence[slot]);
        sc.submitted[slot] = false;
        return true;
    };

    // The synchronous path's own pair. It submits each leg against its own fence rather than the
    // ring's, because it waits on leg 1 in the same frame -- the sequence number is the helper's only
    // ordering signal and must not be bumped ahead of the write it announces.
    const auto endAndSubmit = [&](VkFence fence) {
        if (!NoteVk(dc, dc->vkEndCommandBuffer(cb), "vkEndCommandBuffer")) return false;
        // Whoever waits on the game's semaphores owns them. Submitting with them and not saying so
        // leaves the caller passing the same semaphores to the present, which then waits on a signal
        // that has already been consumed -- one complaint per session, on the first frame, because
        // that is the only frame where the semaphores are fresh.
        if (si.waitSemaphoreCount && consumedWaits) *consumedWaits = true;
        if (!NoteVk(dc, dc->vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit")) return false;
        return true;
    };
    const auto waitAndReset = [&](VkFence fence) {
        if (!NoteVk(dc, dc->vkWaitForFences(d, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences"))
            return false;
        dc->vkResetFences(d, 1, &fence);
        return true;
    };

    // Point the transport at the shared regions, mapped at this frame's size. The composition builds
    // its own buffers over them when the driver will import the pages and falls back to staging when
    // it will not, so this is the one decision and there is no second path to keep in step.
    if (dc->shm.hdr && ShmOpen(dc->shm)) {
        const size_t modelBytes = sc.comp->ModelBytes();
        if (ShmMapFrames(dc->shm, modelBytes))
            sc.comp->SetTransport(dc->shm.inPixels, dc->shm.outPixels, dc->shm.mappedFrameBytes);
        else
            sc.comp->SetTransport(nullptr, nullptr, 0);
    } else {
        sc.comp->SetTransport(nullptr, nullptr, 0);
    }

    // Bypassing the composition and running alongside used to be refused outright, and the reason was
    // sound at the time: bypass makes the model's answer the frame itself, so a pipelined one is a
    // whole stale frame rather than a stale edit on a current one -- and one answer arrives per round
    // trip, so the picture would freeze in steps while the game ran on.
    //
    // What removes that objection is the layer measuring its own displacement. The stale frame can be
    // warped to where the camera is now, every frame, including the frames between answers -- which is
    // asynchronous reprojection, and it is what makes a low answer rate look like continuous motion
    // rather than a slideshow. The resolve already samples the model at the reprojected position, so
    // the bypass path needed nothing: it was only ever being denied the chance to run.
    //
    // Still refused when there is no estimate to warp with, because then the original objection
    // stands exactly as it did.
    //
    // The two are now exclusive the other way round: running alongside the frame implies bypassing
    // the composition, and the composition is not offered while it is on. Composing a stale answer
    // means the composition's every judgement -- the luminance ratio, the highlight guard, the chroma
    // agreement -- is made between the current frame and a picture of an older one, and those
    // judgements are what decide how much of the model reaches the screen. They are hard enough to
    // get right on a matched pair. The GUI hides the composition controls and states why; this is
    // what makes that true rather than merely displayed, because the header can also be written by
    // the CLI and by a settings file.
    const bool canWarpBypass = sc.comp && sc.comp->HasGlobalMotion();
    if (fs.pipelined) fs.compositionBypass = true;
    const bool pipelined = fs.pipelined && canWarpBypass;
    if (fs.pipelined && !canWarpBypass) {
        static std::once_flag said;
        std::call_once(said, [] {
            Log("[layer] the composition is bypassed and there is no motion estimate to reproject the "
                "model's answer with, so the round trip runs in front of the frame instead");
        });
    }

    if (pipelined) {
        // One command buffer, one submit, and no waiting on the helper at all.
        //
        // Order: grab this frame, encode it, put the encode aside as what is being sent, then compose
        // the answer that arrived for an earlier frame onto this one. The encode runs before the
        // composition so the keep the resolve reads as "the untouched frame" is *this* frame -- that
        // is what the additive path lays the stale edit onto -- while the pair being differenced is
        // the one kept aside when it was sent.
        const size_t modelBytes = size_t(sc.comp->ModelWidth()) * sc.comp->ModelHeight() * 4;
        // Ask the helper for the motion field only when it will actually be read.
        //
        // Since the layer measures the displacement itself -- over the interval that matters, which
        // the helper's field cannot cover -- that field is dead weight whenever the estimate is
        // available. Publishing it costs the helper a frame-sized copy and a blocking submit on every
        // round trip, and the round trip is the edit's staleness, which is the ghost. Measured, the
        // readback it sits inside is 2.8 ms of a 13.8 ms round trip.
        const bool wantField = !sc.comp->HasGlobalMotion();
        if (dc->shm.hdr) dc->shm.hdr->wantMotion.store(wantField ? 1u : 0u);

        dc->shm.frames++;
        // The game's own present count, which is the sequence frame generation should be told about
        // rather than a count of the helper's answers.
        if (dc->shm.hdr) dc->shm.hdr->presentIndex.store(dc->shm.frames);
        // The cadence gate: the *request* goes out only on a frame that is a multiple of the stride.
        //
        // The gate is here rather than on the collect, which is where it was first put and which does
        // nothing. ShmInputFree asks only whether the helper has answered, not whether the layer ever
        // took the answer up, so gating the collect leaves the publish free to run at full rate --
        // it just overwrites pendingReq, dropping the answer nobody collected. Measured, that made
        // the helper work 264 times in 4000 frames to produce an edit that updated fewer than 60
        // times, which is the cost of the cadence with none of the benefit.
        //
        // Pacing the send instead paces the whole cycle. The answer is still collected the moment it
        // arrives, so it is as fresh as the round trip allows; what is pinned is the interval between
        // one update and the next, because each is the same round trip after an evenly spaced send.
        // A stride below 2 is the behaviour that has always been here.
        const uint32_t cadence = ShmPublishStride(dc->shm);
        const bool atCadence = cadence < 2 || (dc->shm.frames % cadence) == 0;
        const bool haveAnswer = ShmCollect(dc->shm, modelBytes, sc.comp->ModelPixels());
        // Free to send only when the helper has finished with the last request *and* the last send
        // has actually been handed over. The publish is deferred by a frame now, and recording a
        // second send before the first is published would overwrite the region it is about to be
        // told to read -- and, on the copying fallback, the staging the publish still reads from.
        const bool mayPublish = atCadence && ShmInputFree(dc->shm) && !dc->shm.dead &&
                                sc.pendingSlot >= SwapchainState::kSlots;
        if (haveAnswer) {
            // The proxy this answer was computed from becomes the matched one, before anything reads
            // the pair. Until now it was held aside as "in flight" precisely so the composition kept
            // differencing the previous answer against the picture *it* came from.
            sc.comp->AdoptSentProxy();
            sc.comp->MarkModelFrame();
        }
        const bool willCompose = sc.comp->HasModelFrame() && sc.comp->HasSentProxy();
        if (!NoteVk(dc, dc->vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer")) return false;
        // Grabbed and encoded every frame, whether or not the helper is ready for another request.
        // The encode makes the keep, and the keep is the frame the composition lays its edit onto --
        // skip it and the pass presents the previous picture, which at full frame rate reads as a
        // game running at a fraction of the rate it reports. Only the send is gated.
        if (!sc.comp->RecordGrab(cb, swapchainImage, fs) || !sc.comp->RecordEncode(cb, fs)) {
            dc->vkEndCommandBuffer(cb);
            return false;
        }
        // The motion field that came with this answer, if one did. It maps this frame back to the
        // frame the answer belongs to, so the edit can be sampled from under its own content rather
        // than left on the edges that content has moved off.
        // Never taken while the layer measures its own displacement, which it does whenever the
        // estimator built -- and the import that filled motionBuf went with the old transport, so
        // this is the shape of the fallback rather than a live path. Left in place because the
        // decision above is the one that matters and this states what it decides against.
        if (wantField && haveAnswer && dc->shm.hdr && dc->shm.motionBuf) {
            const uint32_t mseq = dc->shm.hdr->motionSeq.load();
            const uint32_t mw = dc->shm.hdr->motionW.load();
            const uint32_t mh = dc->shm.hdr->motionH.load();
            if (mseq != dc->shm.motionSeqSeen && mw && mh) {
                dc->shm.motionSeqSeen = mseq;
                sc.comp->RecordMotion(cb, dc->shm.motionBuf, mw, mh);
            }
        }

        // Composed before the encode's proxy is claimed as "sent", so the pair kept aside is still
        // the one this answer was computed from.
        sc.comp->SetReprojScale(dc->shm.reprojScale);
        sc.comp->SetAnswerAge(float(dc->shm.lastAge));
        if (willCompose && !sc.comp->RecordCompose(cb, swapchainImage, fs, haveAnswer)) {
            dc->vkEndCommandBuffer(cb);
            return false;
        }
        if (mayPublish && (!sc.comp->RecordSend(cb, fs) || !sc.comp->RecordKeepSent(cb))) {
            dc->vkEndCommandBuffer(cb);
            return false;
        }
        // Submitted and left to run. Nothing here waits for it: the game returns from present with
        // this pass's work still on the GPU, which is the whole point -- it can get on with the next
        // frame while the model works on this one.
        if (!runLeg(false)) return false;
        dropWaits();
        sc.comp->ConsumeMeter();

        // The one thing that does have to wait, and only when it was asked for: a capture reads back
        // what the copy produced, so it needs the copy to have happened. Debug path, taken on the
        // handful of frames someone asked to see, never on the ones they are playing.
        if (sc.comp->CaptureRecorded()) {
            if (NoteVk(dc, dc->vkWaitForFences(dc->self, 1, &sc.fence[slot], VK_TRUE, UINT64_MAX),
                       "vkWaitForFences")) {
                dc->vkResetFences(dc->self, 1, &sc.fence[slot]);
                sc.submitted[slot] = false;
            }
            sc.comp->WriteCapturedFrame();
        }

        // A send recorded on an earlier frame, published now that its slot has landed. The helper
        // reads those pages the moment it sees the sequence number, so the copy into them has to have
        // completed -- but waiting for that here is what stalled the game, and by the next present it
        // is done anyway. Checked, never waited on: a slot that is not ready yet is published on a
        // later frame instead.
        if (sc.pendingSlot < SwapchainState::kSlots) {
            const uint32_t p = sc.pendingSlot;
            if (!sc.submitted[p] ||
                dc->vkGetFenceStatus(dc->self, sc.fence[p]) == VK_SUCCESS) {
                ShmPublish(dc->shm, sc.pendingW, sc.pendingH, sc.comp->ProxyPixels());
                sc.pendingSlot = SwapchainState::kSlots;
            }
        }
        if (mayPublish) {
            // When these pixels were recorded, which is earlier than when they are published -- the
            // publish waits for the slot's fence. That gap is part of the edit's real age and was
            // missing from every staleness number measured so far.
            dc->shm.recordMs = NowMs();
            sc.pendingSlot = slot;
            sc.pendingW = sc.comp->ModelWidth();
            sc.pendingH = sc.comp->ModelHeight();
        }

        if (dc->shm.hdr) {
            ShmStore64(dc->shm.hdr->layerFramesLo, dc->shm.hdr->layerFramesHi, ++dc->framesComposed);
            dc->shm.hdr->layerWidth.store(sc.width);
            dc->shm.hdr->layerHeight.store(sc.height);
            dc->shm.hdr->layerFormat.store(uint32_t(sc.format));
            dc->shm.hdr->layerCompositionUp.store(1);
            dc->shm.hdr->layerMsBits.store(FloatToBits(float((time ? NowMs() : 0.0) - t0)));
            dc->shm.hdr->layerMeasuredWhiteBits.store(FloatToBits(sc.comp->MeasuredWhitePoint()));
            dc->shm.hdr->layerHeartbeat.fetch_add(1);
        }

        if (time) {
            static int frameNo = 0;
            if (++frameNo % TimeInterval() == 0) {
                Log("[time] alongside=%.2f ms (model %ux%u, answer %s)", NowMs() - t0,
                    sc.comp->ModelWidth(), sc.comp->ModelHeight(),
                    haveAnswer ? "new this frame" : "carried from an earlier one");
            }
        }
        return true;
    }

    if (dc->shm.hdr && !pipelined) dc->shm.hdr->wantMotion.store(0);

    // ---- leg 1: the frame the model is shown ----
    if (!NoteVk(dc, dc->vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer")) return false;
    if (!sc.comp->RecordCapture(cb, swapchainImage, fs)) {
        dc->vkEndCommandBuffer(cb);
        return false;
    }
    // This one fence is real: the proxy the helper is about to read is written by these commands,
    // and the sequence number must not outrun the pixels it announces.
    if (!endAndSubmit(sc.fenceLeg1)) return false;
    if (!waitAndReset(sc.fenceLeg1)) return false;
    dropWaits();
    sc.comp->ConsumeMeter();
    const double tCapture = time ? NowMs() : 0.0;

    // ---- the round trip ----
    if (!ShmProcessFrame(dc->shm, sc.comp->ModelWidth(), sc.comp->ModelHeight(), sc.comp->ModelBytes(),
                         sc.comp->ProxyPixels(), sc.comp->ModelPixels(), sc.comp->ProxyActive(),
                         answerViaFd, sc.comp->HdrProxyActive())) {
        // Fail-open. Leg 1 already put the image back in PRESENT_SRC_KHR, so the original frame is
        // what gets presented and nothing else is owed.
        return false;
    }
    sc.comp->MarkModelFrame();
    const double tHelper = time ? NowMs() : 0.0;

    // ---- leg 2: the answer, composed back ----
    if (!NoteVk(dc, dc->vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer")) return false;
    if (!sc.comp->RecordCompose(cb, swapchainImage, fs)) {
        dc->vkEndCommandBuffer(cb);
        return false;
    }
    // No wait. The present that follows runs on this same queue behind these commands, so the image
    // is composed before it is shown without the CPU ever parking here; the fence is collected at
    // the top of the next frame, where the reused surfaces actually need it.
    if (!endAndSubmit(sc.fenceLeg2)) return false;
    sc.leg2Pending = true;
    const double tReturn = time ? NowMs() : 0.0;

    if (dc->shm.hdr) {
        ShmStore64(dc->shm.hdr->layerFramesLo, dc->shm.hdr->layerFramesHi, ++dc->framesComposed);
        dc->shm.hdr->layerWidth.store(sc.width);
        dc->shm.hdr->layerHeight.store(sc.height);
        dc->shm.hdr->layerFormat.store(uint32_t(sc.format));
        dc->shm.hdr->layerCompositionUp.store(1);
        dc->shm.hdr->layerMsBits.store(FloatToBits(float(tReturn - t0)));
        dc->shm.hdr->layerMeasuredWhiteBits.store(FloatToBits(sc.comp->MeasuredWhitePoint()));
        dc->shm.hdr->layerHeartbeat.fetch_add(1);
    }

    if (time) {
        static int frameNo = 0;
        if (++frameNo % TimeInterval() == 0) {
            Log("[time] encode=%.2f helper=%.2f resolve=%.2f total=%.2f ms (model %ux%u)",
                tCapture - t0, tHelper - tCapture, tReturn - tHelper, tReturn - t0,
                sc.comp->ModelWidth(), sc.comp->ModelHeight());
        }
    }
    return true;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueuePresentKHR(VkQueue queue,
                                                           const VkPresentInfoKHR* pPresentInfo) {
    DeviceChain* dc = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        if (g_devices.size() == 1) dc = g_devices.begin()->second;
        else {
            for (auto& kv : g_devices) {
                std::lock_guard<std::mutex> dl(kv.second->lock);
                if (kv.second->queueFamilies.count(queue)) { dc = kv.second; break; }
            }
        }
    }
    if (!dc || !dc->vkQueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;

    // Whether this call's wait semaphores have already been consumed by a submit of ours. They are
    // handed to the first swapchain we actually process; every path after that presents with none,
    // because a semaphore signalled once may only be waited on once. Presenting with them a second
    // time is a wait that never completes -- which is what a second layer in the chain, Steam's
    // overlay among them, turns from a latent bug into a hang.
    bool waitsConsumed = false;

    // What to generate from, once the real present has been made. Generation cannot happen inside the
    // loop below: the frame it carries forward is the one being presented, so the present has to come
    // first -- the real frame is never held back to make a generated one.
    MfgPending fgPending;
    // A present fence has to be given per swapchain presented, so the simple case is the only one
    // taken. A game presenting two swapchains at once gets no generated frames and nothing else
    // changes for it.
    const bool singleSwapchain = pPresentInfo && pPresentInfo->swapchainCount == 1;

    if (!dc->inert && LayerEnabled()) {
        std::lock_guard<std::mutex> lk(dc->lock);
        PollHotkeys(dc);
        if (!ShmNeuralEnabled(dc->shm)) return dc->vkQueuePresentKHR(queue, pPresentInfo);
        uint32_t family = 0;
        auto qit = dc->queueFamilies.find(queue);
        if (qit != dc->queueFamilies.end()) family = qit->second;
        for (uint32_t i = 0; i < pPresentInfo->swapchainCount; ++i) {
            auto sit = dc->swapchains.find(pPresentInfo->pSwapchains[i]);
            if (sit == dc->swapchains.end()) continue;
            SwapchainState& sc = sit->second;
            if (sc.passThrough || pPresentInfo->pImageIndices[i] >= sc.images.size()) continue;
            // One swapchain drives the channel; the rest present raw. See ClaimPrimary.
            if (!ClaimPrimary(dc->self, pPresentInfo->pSwapchains[i], sc.width, sc.height)) continue;
            if (!sc.ready && !dc->shm.dead) {
                if (!CreateResources(dc, sc, family)) {
                    Log("[layer] staging resources failed for swapchain %p (%ux%u, family %u); "
                        "passing this swapchain through",
                        (void*)pPresentInfo->pSwapchains[i], sc.width, sc.height, family);
                    sc.passThrough = true;
                    // This swapchain claimed the primary role and just gave it up. Without the
                    // release the claim would sit on a swapchain that never drives the channel,
                    // and no peer of equal or smaller area could take it over.
                    ReleasePrimary(dc->self, pPresentInfo->pSwapchains[i]);
                    continue;
                }
                sc.ready = true;
            }
            if (!sc.ready || dc->shm.dead) {
                ReleasePrimary(dc->self, pPresentInfo->pSwapchains[i]);
                continue;
            }
            const uint32_t waitCount = waitsConsumed ? 0u : pPresentInfo->waitSemaphoreCount;
            const double tHookStart = NowMs();
            dc->fenceWaitMs = 0.0;
            bool consumed = false;
            // Decided before the pass records, because it is what asks the estimator to read its
            // displacement back to the CPU, and that has to be in the same command buffer.
            uint32_t mfgState = 0;
            uint32_t mfgFactor = MfgFactorFor(dc, sc, &mfgState);
            // The setting is the ceiling; how much of it is used is measured. A ceiling reached and
            // held is the answer "all of it fits", not a number nobody checked.
            if (mfgFactor) mfgFactor = MfgDynamicFactor(dc, sc, mfgFactor);
            sc.comp->SetMotionReadback(mfgFactor != 0);
            if (dc->shm.hdr) dc->shm.hdr->mfgState.store(mfgState, std::memory_order_relaxed);
            // A present early, so the image has a vertical blank in which to become free.
            if (mfgFactor && singleSwapchain) MfgAcquireAhead(dc, sc, pPresentInfo->pSwapchains[i], mfgFactor);
            const bool composed = ProcessPresent(dc, sc, queue, sc.images[pPresentInfo->pImageIndices[i]],
                                                 waitCount, pPresentInfo->pWaitSemaphores, &consumed);
            // Recorded before the real present, not after: generation reads the image the game is
            // about to present, and the presentation engine owns it the moment the present is made.
            //
            // Not conditional on the frame having been composed either, and that was a starvation bug
            // rather than a missed opportunity. The image is acquired a present early; if generation
            // is then skipped, that image is still held -- the only way to release one is to present
            // it. On the pipelined path a frame passes through uncomposed whenever no answer is
            // ready, which is most of them, so held images accumulated and the game ran out.
            // Composition has nothing to do with it: what is carried forward is whatever was
            // presented.
            if (mfgFactor && singleSwapchain) {
                fgPending.swapchain = pPresentInfo->pSwapchains[i];
                MfgRecordFrames(dc, sc, queue, pPresentInfo->pSwapchains[i],
                                sc.images[pPresentInfo->pImageIndices[i]],
                                pPresentInfo->pImageIndices[i], &fgPending);
                if (dc->shm.hdr) {
                    dc->shm.hdr->mfgGeneratedLo.store((uint32_t)(sc.fg.generated & 0xFFFFFFFFu),
                                                      std::memory_order_relaxed);
                    dc->shm.hdr->mfgGeneratedHi.store((uint32_t)(sc.fg.generated >> 32),
                                                      std::memory_order_relaxed);
                    dc->shm.hdr->mfgMissedLo.store((uint32_t)(sc.fg.missed & 0xFFFFFFFFu),
                                                   std::memory_order_relaxed);
                    dc->shm.hdr->mfgMissedHi.store((uint32_t)(sc.fg.missed >> 32),
                                                   std::memory_order_relaxed);
                    dc->shm.hdr->mfgMotionXBits.store(FloatToBits(sc.fg.lastDx),
                                                      std::memory_order_relaxed);
                    dc->shm.hdr->mfgMotionYBits.store(FloatToBits(sc.fg.lastDy),
                                                      std::memory_order_relaxed);
                }
            }
            // Claimed only when a submit actually waited on them. Giving up before that point and
            // still claiming them would leave the present with nothing to wait on, and the game's
            // render-complete semaphore signalled with no one to clear it.
            waitsConsumed = waitsConsumed || consumed;

            // What the hook cost this frame, and how far apart the presents actually landed.
            const double now = NowMs();
            const double gap = dc->pace.lastPresent > 0.0 ? now - dc->pace.lastPresent : 0.0;
            dc->pace.lastPresent = now;
            if (gap > 0.0) dc->pace.Add(gap, now - tHookStart, dc->fenceWaitMs);
            if (TimeEnabled() && dc->pace.n && dc->pace.n % (TimeInterval() * 4) == 0) dc->pace.Report();

            // Give the present something to wait on, so the display cannot read a frame the layer
            // is still writing.
            //
            // This layer composes into the swapchain image and then presents it having consumed the
            // game's own semaphores, so the present goes out with nothing to wait on at all and the
            // ordering rests on queue submission order -- which orders commands against each other
            // and says nothing about the presentation engine. Synchronisation validation calls it
            // PRESENT_AFTER_WRITE, and it is the one hazard that was already there before frame
            // generation existed.
            //
            // An empty batch is enough: a semaphore signal is ordered after everything already
            // submitted to the queue, so signalling one here signals it after the composition's work,
            // and the present waits on that. Exactly one thing signals it per present -- generation
            // does it when it ran, because its own submit is what touches the image last.
            const uint32_t imgIdx = pPresentInfo->pImageIndices[i];
            // Nothing is added to a present that has no generated frames behind it.
            //
            // This block existed to fix PRESENT_AFTER_WRITE, which is real but predates frame
            // generation: the layer composes into the swapchain image and presents it having consumed
            // the game's semaphores, so the present waits on nothing. Fixing it here meant an extra
            // submit and a present fence on every composed frame whether or not anything was
            // generated -- and Half-Life aborted itself, on its main thread, in a run where not one
            // frame was generated. The only thing that had changed for such a run was this.
            //
            // So it is scoped to what it is for. When frames are generated the read of the presented
            // image genuinely has to be waited on, and MfgRecordFrames signals for that. When none
            // are, the present goes out exactly as it did before, hazard and all -- which is the
            // behaviour that ran for forty-four minutes without complaint.
            (void)imgIdx;
            if (false && composed && singleSwapchain && !sc.fg.ring.empty() &&
                !sc.fg.unavailable && dc->presentFence) {
                const uint32_t pair = TakePresentPair(dc, sc);
                if (pair != UINT32_MAX) {
                    fgPending.swapchain = pPresentInfo->pSwapchains[i];
                    VkSubmitInfo ssi{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    ssi.signalSemaphoreCount = 1;
                    ssi.pSignalSemaphores = &sc.fg.ring[pair].sem;
                    if (dc->vkQueueSubmit(queue, 1, &ssi, VK_NULL_HANDLE) == VK_SUCCESS) {
                        fgPending.realWait = sc.fg.ring[pair].sem;
                        fgPending.realFence = sc.fg.ring[pair].fence;
                        sc.fg.ring[pair].inFlight = true;
                        sc.fg.ring[pair].serial = ++sc.fg.presentSerial;
                    }
                }
            }

            if (!composed) ++dc->framesPassedThrough;
            if (VerboseEnabled()) {
                Log("[present] swapchain=%p image=%u seq=%u composed=%d",
                    (void*)pPresentInfo->pSwapchains[i], pPresentInfo->pImageIndices[i],
                    dc->shm.hdr ? dc->shm.hdr->seq_req.load() : 0u, int(composed));
            }
            // On failure we simply present the original frame (fail-open).
        }
        if (TimeEnabled()) {
            static int frameNo = 0;
            if (++frameNo % TimeInterval() == 0) {
                Log("[layer] composed=%llu passed through=%llu",
                    (unsigned long long)dc->framesComposed,
                    (unsigned long long)dc->framesPassedThrough);
            }
        }
    }

    // The real frame goes out here, exactly as it would without this layer -- except that when
    // frames were generated from it, the present waits on the submit that read it. Without that wait
    // the engine may start displaying the frame while the blit is still reading it, which is the
    // WRITE_AFTER_PRESENT hazard that took the game down.
    VkResult res;
    {
        VkPresentInfoKHR pi = *pPresentInfo;
        if (waitsConsumed) {
            // pNext is carried through untouched: present ids, present timing and the rest belong to
            // the caller and none of them are about semaphores.
            pi.waitSemaphoreCount = 0;
            pi.pWaitSemaphores = nullptr;
        }
        VkSwapchainPresentFenceInfoEXT realFenceInfo{
            VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT };
        if (fgPending.realWait) {
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &fgPending.realWait;
            // The fence that says when this present is done, which is what lets its semaphore be
            // handed out again.
            if (dc->presentFence && fgPending.realFence) {
                realFenceInfo.swapchainCount = 1;
                realFenceInfo.pFences = &fgPending.realFence;
                realFenceInfo.pNext = pi.pNext;
                pi.pNext = &realFenceInfo;
            }
        }
        res = (waitsConsumed || fgPending.realWait) ? dc->vkQueuePresentKHR(queue, &pi)
                                                    : dc->vkQueuePresentKHR(queue, pPresentInfo);
    }

    // And the generated frames follow it into the gap, in the order they were recorded.
    //
    // Every semaphore signalled above has to be waited on exactly once, and a present is the only
    // thing that waits on these. A present that fails leaves its semaphore signalled with nothing to
    // consume it, and from then on every wait is answered by the previous frame's signal -- each
    // present handed a semaphore that was already consumed, which is the same undefined territory as
    // reusing one that is still pending.
    //
    // A failing present is not a rare case here either. It is what a resize produces, and the game
    // that keeps crashing resizes twice during startup while vkcube, which never does, has never
    // reproduced any of this.
    //
    // There is no way to put the signals back, and draining them by submitting a wait would hang if
    // the failed present did consume them -- the specification leaves that unsaid. So generation
    // stops on this swapchain instead. That costs nothing: a present failing with OUT_OF_DATE means
    // the swapchain is about to be replaced, and the replacement starts with fresh semaphores.
    // Presented whatever the real present did.
    //
    // These images are acquired and their semaphores are signalled; skipping the present leaves both
    // outstanding with nothing that can ever settle them. If the swapchain really is out of date the
    // present here fails too, harmlessly, and generation is switched off below.
    bool presentFailed = (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR);
    if (fgPending.count) {
        for (uint32_t k = 0; k < fgPending.count; ++k) {
            VkSwapchainPresentFenceInfoEXT genFence{
                VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT };
            genFence.swapchainCount = 1;
            genFence.pFences = &fgPending.fence[k];
            VkPresentInfoKHR gp{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
            if (dc->presentFence && fgPending.fence[k]) gp.pNext = &genFence;
            gp.waitSemaphoreCount = 1;
            gp.pWaitSemaphores = &fgPending.wait[k];
            gp.swapchainCount = 1;
            gp.pSwapchains = &fgPending.swapchain;
            gp.pImageIndices = &fgPending.index[k];
            const VkResult pr = dc->vkQueuePresentKHR(queue, &gp);
            if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
                NoteVk(dc, pr, "vkQueuePresentKHR (mfg)");
                presentFailed = true;
                break;
            }
        }
    }
    if (presentFailed && fgPending.swapchain != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lk(dc->lock);
        auto it = dc->swapchains.find(fgPending.swapchain);
        if (it != dc->swapchains.end() && !it->second.fg.unavailable) {
            it->second.fg.unavailable = true;
            Log("[mfg] a present failed; generation stops on this swapchain until it is rebuilt");
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Loader entry points
// ---------------------------------------------------------------------------
static PFN_vkVoidFunction LookupHook(const char* n) {
    if (!std::strcmp(n, "vkCreateInstance")) return (PFN_vkVoidFunction)Hook_CreateInstance;
    if (!std::strcmp(n, "vkDestroyInstance")) return (PFN_vkVoidFunction)Hook_DestroyInstance;
    if (!std::strcmp(n, "vkEnumeratePhysicalDevices")) return (PFN_vkVoidFunction)Hook_EnumeratePhysicalDevices;
    if (!std::strcmp(n, "vkCreateDevice")) return (PFN_vkVoidFunction)Hook_CreateDevice;
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    return nullptr;
}

static PFN_vkVoidFunction LookupDeviceHook(const char* n) {
    // Only offered while scanning. An ordinary run never gets this hook in its chain at all, which
    // is the difference between a probe that is off and a probe that is on and doing nothing.
    //
    // It belongs on the device side, not the instance side. vkCreateImage is a device function: the
    // loader builds its device dispatch table from vkGetDeviceProcAddr, and an application resolves
    // it the same way, so a registration in LookupHook is never in the chain and never runs.
    //
    // Worth remembering when this layer is blamed for a game that will not start: if the layer's own
    // log has no new bytes in it, the layer did not run and nothing in this file can be the cause.
    // Two such reports turned out to be mangled Steam launch options -- a U+00A0 no-break space, and
    // a %command% that had lost its closing percent sign -- neither of which reaches Vulkan at all.
    if (ScanEnabled() && !std::strcmp(n, "vkCreateImage")) return (PFN_vkVoidFunction)Hook_CreateImage;
    if (!std::strcmp(n, "vkDestroyDevice")) return (PFN_vkVoidFunction)Hook_DestroyDevice;
    if (!std::strcmp(n, "vkGetDeviceQueue")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue;
    if (!std::strcmp(n, "vkGetDeviceQueue2")) return (PFN_vkVoidFunction)Hook_GetDeviceQueue2;
    if (!std::strcmp(n, "vkCreateSwapchainKHR")) return (PFN_vkVoidFunction)Hook_CreateSwapchainKHR;
    if (!std::strcmp(n, "vkDestroySwapchainKHR")) return (PFN_vkVoidFunction)Hook_DestroySwapchainKHR;
    if (!std::strcmp(n, "vkQueuePresentKHR")) return (PFN_vkVoidFunction)Hook_QueuePresentKHR;
    return nullptr;
}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR VkResult VKAPI_CALL
vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
    if (!v || v->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (v->loaderLayerInterfaceVersion > 7) v->loaderLayerInterfaceVersion = 7;
    if (v->loaderLayerInterfaceVersion >= 2) {
        v->pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        v->pfnGetDeviceProcAddr = vkGetDeviceProcAddr;
        v->pfnGetPhysicalDeviceProcAddr = nullptr;
    }
    // Once per process, not once per negotiate.
    //
    // The loader re-enumerates the implicit layer directory many times during a single instance
    // creation -- 628 times for one 32-bit vkCreateInstance here, and the same for every other
    // manifest in the directory -- and loads this library on each pass. That is the loader's
    // business, but announcing it each time turned one line into 627 in the user's log. Every other
    // layer stays quiet because none of them log from here.
    static std::once_flag announced;
    std::call_once(announced, [] {
        const char* v = getenv("VKLayer_DLSS5");
        Log("=== %s loaded (VKLayer_DLSS5=%s) ===", VK_LAYER_NAME, v ? v : "(unset)");
    });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t* pCount,
                                                                  VkLayerProperties* pProperties) {
    if (!pCount) return VK_SUCCESS;
    if (!pProperties) { *pCount = 1; return VK_SUCCESS; }
    if (*pCount < 1) { *pCount = 1; return VK_INCOMPLETE; }
    std::memset(pProperties, 0, sizeof(*pProperties));
    std::strncpy(pProperties->layerName, VK_LAYER_NAME, VK_MAX_EXTENSION_NAME_SIZE - 1);
    std::strncpy(pProperties->description, "DLSS 5 Neural Rendering injection layer",
                 VK_MAX_DESCRIPTION_SIZE - 1);
    pProperties->specVersion = VK_MAKE_VERSION(1, 3, 0);
    pProperties->implementationVersion = 1;
    *pCount = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char*, uint32_t* pCount,
                                                                      VkExtensionProperties*) {
    if (pCount) *pCount = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (!std::strcmp(pName, "vkNegotiateLoaderLayerInterfaceVersion"))
        return (PFN_vkVoidFunction)vkNegotiateLoaderLayerInterfaceVersion;
    if (!std::strcmp(pName, "vkEnumerateInstanceLayerProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceLayerProperties;
    if (!std::strcmp(pName, "vkEnumerateInstanceExtensionProperties"))
        return (PFN_vkVoidFunction)vkEnumerateInstanceExtensionProperties;
    if (auto fn = LookupHook(pName)) return fn;
    if (instance) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_instances.find(instance);
        if (it != g_instances.end() && it->second.next_gipa) return it->second.next_gipa(instance, pName);
    }
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    if (!pName) return nullptr;
    if (!std::strcmp(pName, "vkGetDeviceProcAddr")) return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (auto fn = LookupDeviceHook(pName)) return fn;
    if (device) {
        std::lock_guard<std::mutex> lk(g_stateMutex);
        auto it = g_devices.find(device);
        if (it != g_devices.end() && it->second->next_dpa) return it->second->next_dpa(device, pName);
    }
    return nullptr;
}

}  // extern "C"