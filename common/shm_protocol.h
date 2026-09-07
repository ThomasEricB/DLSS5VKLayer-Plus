#pragma once
// Shared-memory contract between the three processes that make up DLSSNR:
//
//   the Linux Vulkan layer   captures the frame, runs the composition, presents the result
//   the Windows helper       owns nvngx_dlssnr.dll and runs the model
//   the Qt GUI               writes settings and reads status
//
// Everything here is plain atomics in a file mapping, so no side needs the others' toolchain and a
// process dying leaves the others reading a consistent -- if stale -- picture.
//
// Layout of the mapping:
//
//   [0, kHeaderBytes)                       ShmHeader
//   [kHeaderBytes, +kMaxFrame)              the proxy the layer encoded, for the model
//   [kHeaderBytes + kMaxFrame, +kMaxFrame)  the model's answer, for the composition
//
// The proxy is always R8G8B8A8_UNORM, display-referred: the encode has already scaled and sRGB-encoded
// whatever the swapchain held, so the helper never has to know what format the game presents in and
// there is no channel swizzle left to get wrong. `format` is kept for older helpers and is always 1.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif

// 'GNR2'. Bumped from the v1 magic on purpose: a stale v1 mapping left in XDG_RUNTIME_DIR must be
// re-initialised rather than half-read, because the header grew and every offset moved.
static constexpr uint32_t kShmMagic = 0x32524E47;
// v10: settlePercent, the rate the pipelined edit walks toward a new answer.
// v8: two sides grew the header at once -- compositionBypass and rebuildSettleMs upstream, pipeline
// here -- so neither side's number describes this layout.
// v9: a third region for the motion field, so the header, the file size and the offsets all moved.
static constexpr uint32_t kShmVersion = 10;

static constexpr uint32_t kMaxW = 7680, kMaxH = 4320;
static constexpr size_t kMaxFrame = size_t(kMaxW) * kMaxH * 4;
static constexpr size_t kHeaderBytes = 8192;

// The ceiling on how many times the model runs over one frame, and what the slider offers unless the
// ceiling is lifted. Both are OptiScaler's numbers (DlssNr::kMaxPasses / kDefaultMaxPasses) and the
// arrays here are sized for the first.
static constexpr uint32_t kMaxPasses = 30;
static constexpr uint32_t kDefaultMaxPasses = 5;

static constexpr size_t kReasonBytes = 192;
static constexpr size_t kNameBytes = 128;

// Which fields a per-pass entry actually overrides. Sparse by design: a pass with no entry, or an
// entry that does not name a field, follows the global setting -- which is what makes "pass 3 is
// gentler" expressible without restating everything else about pass 3.
enum PassOverrideBit : uint32_t {
    kOverrideIntensity = 1u << 0,
    kOverrideLocalStructure = 1u << 1,
    kOverrideLocalTone = 1u << 2,
    kOverrideSkinStructure = 1u << 3,
    kOverrideStyle = 1u << 4,
    kOverridePreset = 1u << 5,
    kOverrideAutoMask = 1u << 6,
    kOverrideSharpness = 1u << 7,
};

// Where the white point comes from. The layer has no game exposure texture to read -- it sees a
// finished swapchain image and nothing else -- so OptiScaler's source 1 has no counterpart here and
// the choice is between the slider and the frame's own measurement.
enum WhitePointSource : uint32_t {
    kWhitePointManual = 0,   // the slider, and nothing else
    kWhitePointMeasured = 1, // the calibration grid, measured off the untouched copy
};

// What the swapchain holds. Getting this wrong encodes an encoded frame a second time, which looks
// washed out and banded -- so the default is to decide it from the format rather than to guess.
enum ColourMode : uint32_t {
    kColourAuto = 0,      // 8-bit: display-referred. 10-bit and float: linear HDR.
    kColourDisplay = 1,   // force display-referred; the encode becomes a pass-through
    kColourLinearHdr = 2, // force linear HDR; the encode scales by the white point and sRGB-encodes
};

// The RenoDX reversible proxy. Which encode the model is shown, and how its answer comes back.
//
//   0  soft knee + our composition            the shipped behaviour, and byte-identical to it
//   1  unclipped Neutwo proxy + composition
//   2  Neutwo proxy + pure-inverse replace    the model's answer straight back, no composition
//   3  hybrid proxy + composition             identity midtones, unclipped highlights
//   4  hybrid proxy + replace
//
// From Dagherbou/OptiScaler_DLSSNR; the proxy itself is RenoDX's (clshortfuse). Modes 2 and 4 are
// noted upstream as flashing on bright lights, which is why 0 is the default rather than a taste.
enum ReversibleMode : uint32_t {
    kReversibleKnee = 0,
    kReversibleNeutwo = 1,
    kReversibleNeutwoReplace = 2,
    kReversibleHybrid = 3,
    kReversibleHybridReplace = 4,
    kReversibleModeCount = 5,
};

// The filter that brings the model's answer back down when it ran above native resolution. Only
// consulted when the working scale is above 1.0.
//
// These are OptiScaler's own Scaler numbers, kept identical so a value copied from an OptiScaler
// profile means the same thing here. FSR1 keeps slot 0 for that reason even though this pass cannot
// use it -- it wants a different constant block, and it is an upscaler rather than the averaging
// filter the down-leg needs. A header asking for it falls back to Lanczos3.
enum Downscaler : uint32_t {
    kDownscaleFsr1 = 0,  // unsupported here; reserved so the numbering matches upstream
    kDownscaleBicubic = 1,
    kDownscaleCatmullRom = 2,
    kDownscaleLanczos2 = 3,
    kDownscaleLanczos3 = 4,  // upstream's default: the sharp one
    kDownscaleKaiser2 = 5,
    kDownscaleKaiser3 = 6,
    kDownscaleMagic = 7,
    kDownscalerCount = 8,
};

// What the helper has managed to do, for the GUI and for the layer's fail-open decision.
// The layer reads this to decide whether anything is listening, so "nobody" has to be the value a
// freshly initialised header holds -- not a state that also means "starting".
enum HelperState : uint32_t {
    kHelperStarting = 0,
    kHelperNoVulkan = 1,   // no NVIDIA device with the NVX extensions
    kHelperNoBinaries = 2, // nvngx_dlssnr.dll not found
    kHelperModelFailed = 3,
    kHelperRunning = 4,
    kHelperStopped = 5,
};

// How the motion field the helper hands the model is scaled. From bmitch87's motion-vector work.
enum MVecScaleMode : uint32_t {
    kMVecNormalized = 0,
    kMVecPixels = 1,
    kMVecUv01 = 2,
};

// What the optical-flow engine is asked for. Higher costs more of the frame's budget.
enum MVecQuality : uint32_t {
    kMVecFast = 0,
    kMVecBalanced = 1,
    kMVecQuality = 2,
};

// Where the mapping lives.
//
// It has to name the same file in every process that touches it, and a Steam game does not share a
// mount namespace with the helper: pressure-vessel gives the container a private tmpfs at
// $XDG_RUNTIME_DIR, so a mapping put there is simply absent inside the game. The layer then creates
// its own empty one at a path that reads identically in the log and waits forever for a helper that
// is answering on the other file -- the "attached ... seq_req=0 / helper not running" case. /tmp is
// bind-mounted from the host into the container, so both sides land on one file; it is also what a
// Wine prefix exposes as Z:\tmp\..., which is how the helper opens it.
inline std::string ShmRuntimeDir() {
    const char* uid = std::getenv("DLSSNR_UID");
    if (uid && *uid) return std::string("/tmp/dlssnr-") + uid;
#ifdef _WIN32
    // The helper is always handed DLSSNR_SHM by the launcher, so this is only ever a last resort.
    return "/tmp/dlssnr";
#else
    return "/tmp/dlssnr-" + std::to_string((unsigned) getuid());
#endif
}

inline std::string ShmDefaultPath() { return ShmRuntimeDir() + "/shm.bin"; }

// Three regions now: the proxy going out, the answer coming back, and the motion field that says
// where each of this frame's pixels was in the frame the answer belongs to. Sparse on disk, so the
// third costs what is written rather than what is reserved.
inline size_t ShmTotalBytes() { return kHeaderBytes + kMaxFrame * 3; }

// Where the helper writes the motion field and the layer reads it.
inline size_t ShmMotionOffset() { return kHeaderBytes + kMaxFrame * 2; }

// Both pixel regions start on a page boundary, and that is load-bearing rather than tidy.
//
// VK_EXT_external_memory_host imports an ordinary host pointer as VkDeviceMemory, which is how the
// layer and the helper come to share one allocation across the Linux/Wine boundary -- winevulkan
// exposes only the Win32 handle types, so an fd could never cross, but a mapped pointer can. The
// driver requires the imported pointer to be aligned to minImportedHostPointerAlignment, which is a
// page on every implementation that offers the extension at all. If either offset stops being a
// multiple of a page, the import fails and the transport silently falls back to copying.
static constexpr size_t kHostImportAlignment = 4096;
static_assert(kHeaderBytes % kHostImportAlignment == 0,
              "the input region must start on a page boundary to be importable");
static_assert(kMaxFrame % kHostImportAlignment == 0,
              "the output region must start on a page boundary to be importable");


// One pass's overrides. Every field is present; `overrideMask` says which of them mean anything.
struct PassControl {
    std::atomic<uint32_t> overrideMask;
    std::atomic<uint32_t> intensityBits;
    std::atomic<uint32_t> localToneBits;
    std::atomic<uint32_t> localStructureBits;
    std::atomic<uint32_t> skinStructureBits;
    std::atomic<uint32_t> sharpnessBits;
    std::atomic<uint32_t> style;
    std::atomic<uint32_t> preset;
    std::atomic<uint32_t> autoMask;
};

// A pass's settings after the global values and its own overrides have been merged. Plain floats:
// this is the resolved answer, not shared state.
struct PassTuning {
    float intensity = 1.0f;
    float localTone = 1.0f;
    float localStructure = 1.0f;
    float skinStructure = -1.0f;  // -1 follows local structure; it is not a strength of zero
    float sharpness = 0.0f;
    uint32_t style = 0;
    uint32_t preset = 0;
    uint32_t autoMask = 1;

    bool SameCreateParams(const PassTuning& o) const {
        // Everything the model latches when its feature is built. Sharpness is absent because it is
        // read at evaluate, and so is the only one of these a running feature will actually follow.
        return intensity == o.intensity && localTone == o.localTone && localStructure == o.localStructure &&
               skinStructure == o.skinStructure && style == o.style && preset == o.preset && autoMask == o.autoMask;
    }
};

struct ShmHeader {
    std::atomic<uint32_t> magic;
    std::atomic<uint32_t> version;

    // The frame handshake. The layer bumps seq_req after writing a proxy; the helper answers by
    // storing the same number into seq_resp once the model's answer is in the output region.
    std::atomic<uint32_t> seq_req;
    std::atomic<uint32_t> seq_resp;
    std::atomic<uint32_t> width;
    std::atomic<uint32_t> height;
    std::atomic<uint32_t> format;  // always 1 (RGBA byte order); kept so a v1 helper is not silently wrong
    std::atomic<uint32_t> quit;
    std::atomic<uint32_t> heartbeat;

    // Bumped by whoever writes a setting. The layer and the helper watch it rather than re-reading
    // thirty values every frame.
    std::atomic<uint32_t> controlSeq;

    // Bumped only when something the model latches at feature creation changes. The helper rebuilds
    // its features on this and debounces the rebuild; bumping it every frame exhausts the driver's
    // latches and the model stops responding until the process restarts.
    std::atomic<uint32_t> tuningSeq;

    // --- the model ---------------------------------------------------------------------------
    std::atomic<uint32_t> enabled;
    std::atomic<uint32_t> passes;
    std::atomic<uint32_t> unlockPasses;
    std::atomic<uint32_t> preset;
    std::atomic<uint32_t> style;
    std::atomic<uint32_t> autoMask;
    std::atomic<uint32_t> intensityBits;
    std::atomic<uint32_t> localToneBits;
    std::atomic<uint32_t> localStructureBits;
    std::atomic<uint32_t> skinStructureBits;
    std::atomic<uint32_t> sharpnessBits;

    // --- the composition ---------------------------------------------------------------------
    // How much of the model's edit reaches the frame, and how much of it is allowed to be colour
    // rather than luminance. Separating the two is what keeps saturated highlights from shifting hue.
    std::atomic<uint32_t> transferStrengthBits;
    std::atomic<uint32_t> colourStrengthBits;
    // The most the pass may multiply or divide a pixel by. The transfer is a ratio, and a ratio
    // against a near-black proxy pixel is unbounded without one.
    std::atomic<uint32_t> maxRatioBits;
    // How a model that worked below the frame's size is brought back. 0 classic, 1 matched residual,
    // 2 native + edit -- the frame's own pixels with only the model's difference added, so what the
    // model left alone never passes through the enlargement.
    std::atomic<uint32_t> transfer;
    // 0 off, 1 the picture the model was shown, 2 its raw answer, 3 what it changed, amplified.
    std::atomic<uint32_t> debugView;
    std::atomic<uint32_t> debugScaleBits;
    std::atomic<uint32_t> whitePointBits;
    std::atomic<uint32_t> whitePointScaleBits;
    std::atomic<uint32_t> whitePointSource;
    std::atomic<uint32_t> whitePointTrimBits;
    // What fraction of the frame's resolution the model works at. The frame itself is never reduced:
    // only the model's contribution is computed at this scale and resized, so the picture underneath
    // is untouched whatever this is.
    //
    // Above 1.0 is supersampling -- the model runs above native and its answer is brought back down
    // by scalingDownscaler. Upstream allows up to 2.0. Below 1.0 it also cuts what crosses the shared
    // memory, quadratically, which on this transport matters more than it does upstream.
    std::atomic<uint32_t> workingScaleBits;
    // 0 off, 1 side by side, 2 a wipe.
    std::atomic<uint32_t> compareMode;
    std::atomic<uint32_t> compareSplitBits;
    std::atomic<uint32_t> compareZoomBits;
    std::atomic<uint32_t> compareSwap;
    std::atomic<uint32_t> colourMode;
    // Writes one set of matched before/after frames per session when the layer next presents.
    std::atomic<uint32_t> captureRequest;

    // A Linux KEY_ code the layer watches to toggle the pass, or 0 for none. Unbound by default,
    // because a key that does something unexpected is worse than a key that does nothing.
    //
    // Only useful where the layer can read the keyboard at all: an X11 or XWayland session, inside
    // gamescope, or anywhere the user is in the 'input' group. A game presenting through winewayland
    // is a Wayland client whose keys never reach this process, and keyboards get no uaccess ACL, so
    // there the answer is a desktop shortcut bound to 'dlssnr-shmctl toggle enabled' instead.
    std::atomic<uint32_t> toggleKey;

    // Which proxy the model is shown, and whether its answer is composed or substituted. See
    // ReversibleMode. Default 0 keeps the picture identical to the pre-import behaviour.
    std::atomic<uint32_t> reversibleMode;

    // Whether the model's edit is applied at all. Off keeps the whole pass running -- the capture,
    // the round trip, the encode -- and simply presents the clean frame, which is what makes an
    // honest A/B possible: the cost is unchanged, so only the picture differs.
    std::atomic<uint32_t> applyModel;

    // Freeze the frame the pass works on, so changing a setting re-runs the composition over the
    // SAME picture instead of over whatever the game has drawn since. The only clean way to compare
    // two settings, and a live testing control rather than a saved preference.
    //
    // In this architecture it is cheaper than upstream: the layer already holds the captured proxy
    // and the model's last answer, so holding means not re-capturing rather than keeping a frame
    // alive somewhere it would not otherwise be.
    std::atomic<uint32_t> holdFrame;

    // The filter for the supersampling down-leg. See Downscaler; only read when workingScale > 1.
    std::atomic<uint32_t> scalingDownscaler;

    // --- status, written by the helper --------------------------------------------------------
    std::atomic<uint32_t> helperState;
    std::atomic<uint32_t> modelUp;
    std::atomic<uint32_t> helperFramesLo;
    std::atomic<uint32_t> helperFramesHi;
    std::atomic<uint32_t> helperEvalMsBits;
    std::atomic<uint32_t> helperUploadMsBits;
    std::atomic<uint32_t> helperReadbackMsBits;
    std::atomic<uint32_t> helperVramMB;
    std::atomic<uint32_t> helperFeatures;   // how many NGX features are actually built
    std::atomic<uint32_t> helperPassCeiling;  // what the VRAM budget currently allows

    // --- status, written by the layer ---------------------------------------------------------
    std::atomic<uint32_t> layerAttached;
    std::atomic<uint32_t> layerFramesLo;
    std::atomic<uint32_t> layerFramesHi;
    std::atomic<uint32_t> layerWidth;
    std::atomic<uint32_t> layerHeight;
    std::atomic<uint32_t> layerFormat;
    std::atomic<uint32_t> layerCompositionUp;
    std::atomic<uint32_t> layerMsBits;
    std::atomic<uint32_t> layerMeasuredWhiteBits;
    std::atomic<uint32_t> layerHeartbeat;

    // Free text, each guarded by its own sequence number: bumped after the bytes are written, so a
    // reader that sees an unchanged number is looking at a whole string.
    std::atomic<uint32_t> helperReasonSeq;
    char helperReason[kReasonBytes];
    std::atomic<uint32_t> layerReasonSeq;
    char layerReason[kReasonBytes];
    std::atomic<uint32_t> gameNameSeq;
    char gameName[kNameBytes];

    PassControl pass[kMaxPasses];

    // Appended after the pass array on purpose: everything before it has a pinned offset, and a new
    // field inserted higher up would move all of them. Motion vectors, from bmitch87's work.
    // Whether the layer waits for the model or lets it work alongside the game's next frame.
    //
    // Waiting is exact and it is also the whole cost of the pass: nothing overlaps, so the model's
    // time adds to the game's instead of hiding inside it. Running alongside costs one frame of
    // staleness -- in the *edit* only. The frame under it is always the one being presented, and the
    // shader takes an additive path for it so the game's own pixels survive; the layer keeps the
    // proxy that went with each answer so the difference it applies is the edit and nothing else.
    std::atomic<uint32_t> pipeline;

    // The layer asking the helper for the motion field, and the helper saying it put one there.
    //
    // Only the pipelined path needs it -- it is what lets a one-frame-old edit be moved to where its
    // content has got to -- and copying a frame-sized field every frame is not free, so the helper
    // writes it when asked and not otherwise. motionSeq carries the request the field belongs to, so
    // the layer can tell a field for the answer it is holding from one for a newer request.
    std::atomic<uint32_t> wantMotion;
    std::atomic<uint32_t> motionSeq;
    std::atomic<uint32_t> motionW;
    std::atomic<uint32_t> motionH;
    std::atomic<uint32_t> mvecEnabled;
    std::atomic<uint32_t> mvecScaleMode;
    std::atomic<uint32_t> mvecQuality;
    // How far the helper has answered *successfully*. seq_resp says a frame came back; this says it
    // was worth using, so the layer can present the game's own frame when it was not.
    std::atomic<uint32_t> seq_ok;

    // 0: the composition blends the model's edit onto the frame under the strength and guard limits.
    // 1: no composition at all -- the model's raw answer IS the presented frame, and the limits,
    // enlargement and compare overlays are moot. Default 1: the composition is off until the user
    // turns it on.
    std::atomic<uint32_t> compositionBypass;

    // Wall-clock milliseconds the helper waits after the last tuning change before it rebuilds a
    // feature, and between one rebuild and the next. NGX creation is expensive and back-to-back
    // creation was seen to exhaust the driver's latches on some setups, so the default spaces
    // rebuilds rather than firing them at once; 0 means no spacing -- build the moment the change
    // settles and chain the remaining builds back to back. It is time rather than frames because a
    // frame-counted wait crawls on a 30 fps game and races on a 144 fps one.
    std::atomic<uint32_t> rebuildSettleMs;

    // How fast the running pair walks toward a newly arrived answer, in hundredths.
    //
    // Pipelined, an answer lands every few frames and the edit it implies changes all at once when it
    // does. The frame under it is current, so nothing smears -- but the *edit* stepping between two
    // values on one frame and then holding for several is exactly what reads as a flicker, and the
    // faster the game runs the more often it steps. This walks the answer and the proxy it was
    // computed from toward the new pair by the same fraction each frame, which -- because the edit is
    // their difference, and a difference of two blends is the blend of the two differences -- walks
    // the edit itself. The step becomes a ramp.
    //
    // 100 means take the new answer whole the moment it lands, which is the old behaviour. 0 freezes
    // the edit at the first answer.
    //
    // It is not free: successive edits sit on different geometry, so where they disagree the blend
    // cancels them and the edit comes out weaker as well as smoother. Measured on a spinning vkcube
    // -- which turns far faster than a camera does, so this is the pessimistic end -- the step in the
    // edit between frames and the edit's own strength go: 100 -> 100% strength, 75 -> 84%, 60 -> 77%,
    // 40 -> 63%, 20 -> 51%, against a step of 0.0046, 0.0034, 0.0030, 0.0021, 0.0019. The default is
    // 60 because below it the step stops improving much and only the strength keeps falling.
    std::atomic<uint32_t> settlePercent;
};

static_assert(sizeof(ShmHeader) <= kHeaderBytes, "ShmHeader outgrew its region");

// The layout, pinned.
//
// Every process that maps this file agrees on where each field is only because they were compiled
// from the same header. A field inserted anywhere but the end silently moves everything after it, and
// a build that has not caught up then reads its neighbour's value -- which is not a crash, it is a
// status display quietly reporting 4861 for a flag that is 0 or 1, and it took a nonsensical number
// on screen to notice.
//
// The version check already existed to prevent exactly that; what was missing was anything to make
// someone remember to use it. If these fire, the layout changed: bump kShmVersion in the same commit,
// then update these numbers.
static_assert(sizeof(ShmHeader) == 1908, "the header layout changed -- bump kShmVersion");
static_assert(offsetof(ShmHeader, enabled) == 44, "layout changed -- bump kShmVersion");
static_assert(offsetof(ShmHeader, transferStrengthBits) == 88, "layout changed -- bump kShmVersion");
static_assert(offsetof(ShmHeader, helperState) == 176, "layout changed -- bump kShmVersion");
static_assert(offsetof(ShmHeader, pass) == 780, "layout changed -- bump kShmVersion");
static_assert(offsetof(ShmHeader, pipeline) == 1860, "layout changed -- bump kShmVersion");

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

inline void ShmStoreString(std::atomic<uint32_t>& seq, char* dst, size_t cap, const char* src) {
    std::memset(dst, 0, cap);
    if (src) std::strncpy(dst, src, cap - 1);
    seq.fetch_add(1);
}

inline std::string ShmLoadString(const std::atomic<uint32_t>& seq, const char* src, size_t cap) {
    for (int attempt = 0; attempt < 4; ++attempt) {
        const uint32_t before = seq.load();
        char buf[kReasonBytes > kNameBytes ? kReasonBytes : kNameBytes];
        std::memset(buf, 0, sizeof(buf));
        std::memcpy(buf, src, cap < sizeof(buf) ? cap : sizeof(buf));
        buf[(cap < sizeof(buf) ? cap : sizeof(buf)) - 1] = '\0';
        if (seq.load() == before) return std::string(buf);
    }
    return std::string();
}

// Every user-facing setting, and nothing else.
//
// Split out of ShmInitDefaults so the two cannot drift. Initialising a fresh mapping and resetting a
// live one now write the same values from the same place; written twice, one of them would quietly
// forget a field the other remembered, and the forgotten one is always the setting someone is
// looking at when they wonder why reset did not reset it.
//
// Nothing here touches identity, the transport or the status channel -- resetting a live mapping
// must not disturb the sequence numbers a running helper is answering, nor claim the model is up.
// captureRequest is likewise absent: it is a one-shot ask, not a setting.
inline void ShmDefaultSettings(ShmHeader* h) {
    h->enabled.store(1);
    h->passes.store(1);
    h->unlockPasses.store(0);
    h->preset.store(0);
    h->style.store(0);
    h->autoMask.store(1);
    h->intensityBits.store(FloatToBits(1.0f));
    h->localToneBits.store(FloatToBits(1.0f));
    h->localStructureBits.store(FloatToBits(1.0f));
    h->skinStructureBits.store(FloatToBits(-1.0f));
    h->sharpnessBits.store(FloatToBits(0.0f));

    h->transferStrengthBits.store(FloatToBits(1.0f));
    h->colourStrengthBits.store(FloatToBits(1.0f));
    h->maxRatioBits.store(FloatToBits(2.0f));
    h->transfer.store(1);
    h->debugView.store(0);
    h->debugScaleBits.store(FloatToBits(1.0f));
    h->whitePointBits.store(FloatToBits(1.0f));
    h->whitePointScaleBits.store(FloatToBits(1.0f));
    h->whitePointSource.store(kWhitePointManual);
    h->whitePointTrimBits.store(FloatToBits(1.0f));
    h->workingScaleBits.store(FloatToBits(1.0f));
    h->compareMode.store(0);
    h->compareSplitBits.store(FloatToBits(0.5f));
    h->compareZoomBits.store(FloatToBits(1.0f));
    h->compareSwap.store(0);
    h->colourMode.store(kColourAuto);
    h->toggleKey.store(0);
    h->reversibleMode.store(kReversibleKnee);
    h->applyModel.store(1);
    h->holdFrame.store(0);
    h->scalingDownscaler.store(kDownscaleLanczos3);

    h->pipeline.store(0);
    h->wantMotion.store(0);
    h->mvecEnabled.store(1);
    h->mvecScaleMode.store(kMVecPixels);
    h->mvecQuality.store(kMVecBalanced);
    h->compositionBypass.store(1);
    h->rebuildSettleMs.store(250);
    h->settlePercent.store(60);

    for (uint32_t i = 0; i < kMaxPasses; ++i) {
        h->pass[i].overrideMask.store(0);
        h->pass[i].intensityBits.store(FloatToBits(1.0f));
        h->pass[i].localToneBits.store(FloatToBits(1.0f));
        h->pass[i].localStructureBits.store(FloatToBits(1.0f));
        h->pass[i].skinStructureBits.store(FloatToBits(-1.0f));
        h->pass[i].sharpnessBits.store(FloatToBits(0.0f));
        // Inert until overrideMask names them, but initialised to the global defaults so a pass that
        // is switched on later starts from what the rest of the frame is already doing.
        h->pass[i].style.store(0);
        h->pass[i].preset.store(0);
        h->pass[i].autoMask.store(1);
    }
}

// Put a live mapping's settings back to those defaults, leaving the transport alone, and tell the
// other two processes to look again: controlSeq for what the layer reads every frame, tuningSeq for
// what the helper latches when it builds a feature.
inline void ShmResetSettings(ShmHeader* h) {
    if (!h) return;
    ShmDefaultSettings(h);
    h->controlSeq.fetch_add(1);
    h->tuningSeq.fetch_add(1);
}

inline void ShmInitDefaults(ShmHeader* h) {
    std::memset(static_cast<void*>(h), 0, sizeof(ShmHeader));
    h->magic.store(kShmMagic);
    h->version.store(kShmVersion);
    h->helperState.store(kHelperStopped);
    h->format.store(1);
    h->seq_ok.store(0);
    ShmDefaultSettings(h);
}

inline uint32_t ShmPassCeiling(const ShmHeader* h) {
    return h->unlockPasses.load() ? kMaxPasses : kDefaultMaxPasses;
}

inline uint32_t ShmPasses(const ShmHeader* h) {
    uint32_t p = h->passes.load();
    const uint32_t ceiling = ShmPassCeiling(h);
    if (p == 0) return 1;
    return p > ceiling ? ceiling : p;
}

inline bool ShmNeuralEnabled(const ShmHeader* h) { return h->enabled.load() != 0; }

// The global settings with one pass's overrides applied. A field the pass does not name follows the
// global value, which is what keeps a sparse override sparse.
inline PassTuning ShmResolvePass(const ShmHeader* h, uint32_t pass) {
    PassTuning t;
    t.intensity = BitsToFloat(h->intensityBits.load());
    t.localTone = BitsToFloat(h->localToneBits.load());
    t.localStructure = BitsToFloat(h->localStructureBits.load());
    t.skinStructure = BitsToFloat(h->skinStructureBits.load());
    t.sharpness = BitsToFloat(h->sharpnessBits.load());
    t.style = h->style.load();
    t.preset = h->preset.load();
    t.autoMask = h->autoMask.load();

    if (pass >= kMaxPasses) return t;
    const uint32_t mask = h->pass[pass].overrideMask.load();
    if (mask == 0) return t;
    if (mask & kOverrideIntensity) t.intensity = BitsToFloat(h->pass[pass].intensityBits.load());
    if (mask & kOverrideLocalTone) t.localTone = BitsToFloat(h->pass[pass].localToneBits.load());
    if (mask & kOverrideLocalStructure) t.localStructure = BitsToFloat(h->pass[pass].localStructureBits.load());
    if (mask & kOverrideSkinStructure) t.skinStructure = BitsToFloat(h->pass[pass].skinStructureBits.load());
    if (mask & kOverrideSharpness) t.sharpness = BitsToFloat(h->pass[pass].sharpnessBits.load());
    if (mask & kOverrideStyle) t.style = h->pass[pass].style.load();
    if (mask & kOverridePreset) t.preset = h->pass[pass].preset.load();
    if (mask & kOverrideAutoMask) t.autoMask = h->pass[pass].autoMask.load();
    return t;
}

inline uint64_t ShmLoad64(const std::atomic<uint32_t>& lo, const std::atomic<uint32_t>& hi) {
    return (uint64_t(hi.load()) << 32) | uint64_t(lo.load());
}

inline void ShmStore64(std::atomic<uint32_t>& lo, std::atomic<uint32_t>& hi, uint64_t v) {
    hi.store(uint32_t(v >> 32));
    lo.store(uint32_t(v & 0xFFFFFFFFu));
}

inline bool ShmPipelined(const ShmHeader* h) { return h && h->pipeline.load() != 0; }

inline bool ShmMVecEnabled(const ShmHeader* h) { return h->mvecEnabled.load() != 0; }

inline uint32_t ShmMVecScaleMode(const ShmHeader* h) {
    const uint32_t m = h->mvecScaleMode.load();
    return m <= kMVecUv01 ? m : kMVecNormalized;
}

inline uint32_t ShmMVecQuality(const ShmHeader* h) {
    const uint32_t q = h->mvecQuality.load();
    return q <= kMVecQuality ? q : kMVecBalanced;
}
