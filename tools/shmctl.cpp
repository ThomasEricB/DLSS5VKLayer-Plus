// dlssnr-shmctl — read and write the shared-memory header from shell.
//
// This exists because the helper CLI used to poke the header with
//
//     printf '\x01\x00\x00\x00' | dd of="$shm" bs=1 seek=20 count=4 conv=notrunc
//
// which is a hardcoded byte offset into a C++ struct. It was correct for exactly one layout: the
// field at offset 20 was `quit` in the v1 header and is `height` in v2, so the same line that used
// to stop the helper would instead have silently corrupted the frame size. Nothing about the shell
// could have caught that.
//
// Everything here derives its offsets from shm_protocol.h by including it, so the header and the
// tool cannot disagree. Adding a field is now a recompile rather than an arithmetic exercise.
#include "../common/shm_protocol.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// Every setting the header carries, by name, so the shell can drive the pass before there is an
// interface for it. Table-driven on purpose: a field added to the header and not to this table is a
// setting nobody can reach, and the table is short enough that the omission is obvious.
struct Setting {
    const char* name;
    std::atomic<uint32_t> ShmHeader::*field;
    bool isFloat;
    const char* help;
};

const Setting kSettings[] = {
    { "enabled", &ShmHeader::enabled, false, "0/1 run the model at all" },
    { "hdrmode", &ShmHeader::hdrMode, false, "0 auto, 1 off, 2 force float16 proxy" },
    { "passes", &ShmHeader::passes, false, "how many times the model runs over one frame" },
    { "unlockpasses", &ShmHeader::unlockPasses, false, "0/1 lift the pass ceiling" },
    { "preset", &ShmHeader::preset, false, "model preset" },
    { "style", &ShmHeader::style, false, "0 default, 1 natural, 2 cinematic" },
    { "automask", &ShmHeader::autoMask, false, "0/1 automatic skin mask" },
    { "intensity", &ShmHeader::intensityBits, true, "model intensity" },
    { "localtone", &ShmHeader::localToneBits, true, "local tone strength" },
    { "localstructure", &ShmHeader::localStructureBits, true, "local structure strength" },
    { "skinstructure", &ShmHeader::skinStructureBits, true, "-1 follows local structure" },
    { "sharpness", &ShmHeader::sharpnessBits, true, "sharpness" },
    { "detail", &ShmHeader::transferStrengthBits, true, "how much of the edit lands, 0-4" },
    { "colour", &ShmHeader::colourStrengthBits, true, "how much of its colour comes with it, 0-4" },
    { "guard", &ShmHeader::maxRatioBits, true, "highlight guard, the most a pixel may move" },
    { "transfer", &ShmHeader::transfer, false, "0 classic, 1 matched residual, 2 native + edit" },
    { "bypass", &ShmHeader::compositionBypass, false,
      "present the model's raw answer instead of composing its edit, 0 or 1" },
    { "rebuildms", &ShmHeader::rebuildSettleMs, false,
      "ms to settle before rebuilding a pass after a model setting changes" },
    { "pipeline", &ShmHeader::pipeline, false,
      "run the model alongside the frame instead of waiting for it, 0 or 1" },
    { "settle", &ShmHeader::settlePercent, false,
      "how fast the pipelined edit walks toward a new answer, 0-100 (100 = take it whole)" },
    { "ghostslack", &ShmHeader::ghostSlackPercent, false,
      "how far past its neighbours a pipelined pixel may land, in hundredths (0 = pinned, 50 default)" },
    { "ratiosmooth", &ShmHeader::ratioSmoothPercent, false,
      "how much of the relighting ratio comes from the neighbourhood, 0-100 (100 default, 0 = per-pixel)" },
    { "colourtrust", &ShmHeader::colourTrustPercent, false,
      "how far the model may move a pixel's colour from the frame's, in hundredths (200 default, 0 = frame's hue)" },
    { "smooth", &ShmHeader::motionSmoothPercent, false,
      "how hard the measured displacement is filtered over time, 0-100 (0 = raw, 100 default)" },
    { "gap", &ShmHeader::publishStride, false,
      "frames between pipelined answers, 0 = as soon as each arrives, 2-64 pins the cadence" },
    { "editblur", &ShmHeader::editBlurMilli, false,
      "radius splitting a stale edit's safe half from the half that ghosts, in thousandths of width" },
    { "mfg", &ShmHeader::mfgEnabled, false,
      "generate extra frames between the game's own, 0 or 1" },
    { "mfgfactor", &ShmHeader::mfgFactor, false,
      "generated frames per real frame, 1-3" },
    { "mfgmode", &ShmHeader::mfgMode, false,
      "0 only under a paced present mode (fifo), 1 under any" },
    { "mfgauto", &ShmHeader::mfgAuto, false,
      "find how many frames fit by measuring, using mfgfactor as the ceiling, 0 or 1" },
    { "mvec", &ShmHeader::mvecEnabled, false, "estimate motion vectors from the frames, 0 or 1" },
    { "mvecquality", &ShmHeader::mvecQuality, false, "0 fast, 1 balanced, 2 quality" },
    { "mvecunits", &ShmHeader::mvecScaleMode, false, "0 normalised, 1 pixels, 2 uv 0..1" },
    { "debugview", &ShmHeader::debugView, false, "0 off, 1 proxy, 2 model, 3 amplified edit" },
    { "debugscale", &ShmHeader::debugScaleBits, true, "what the debug views are multiplied by" },
    { "whitepoint", &ShmHeader::whitePointBits, true, "paper white" },
    { "whitepointscale", &ShmHeader::whitePointScaleBits, true, "multiplier on the white point" },
    { "whitepointsource", &ShmHeader::whitePointSource, false, "0 the slider, 1 measured off the frame" },
    { "whitepointtrim", &ShmHeader::whitePointTrimBits, true, "multiplier on a measured white point" },
    { "workingscale", &ShmHeader::workingScaleBits, true, "the fraction of the frame the model works at" },
    { "downscaler", &ShmHeader::scalingDownscaler, false, "1 bicubic, 2 catmull, 3 lanczos2, 4 lanczos3, 5 kaiser2, 6 kaiser3, 7 magic" },
    { "compare", &ShmHeader::compareMode, false, "0 off, 1 side by side, 2 wipe" },
    { "comparesplit", &ShmHeader::compareSplitBits, true, "where the split sits, 0-1" },
    { "comparezoom", &ShmHeader::compareZoomBits, true, "side by side only, 1-2" },
    { "compareswap", &ShmHeader::compareSwap, false, "0/1 which side the edited frame is on" },
    { "colourmode", &ShmHeader::colourMode, false, "0 auto, 1 display-referred, 2 linear HDR" },
    { "reversible", &ShmHeader::reversibleMode, false, "0 knee, 1 neutwo, 2 replace, 3 hybrid, 4 hybrid+replace" },
    { "applymodel", &ShmHeader::applyModel, false, "0 show the clean frame, 1 apply the edit" },
    { "hold", &ShmHeader::holdFrame, false, "0/1 freeze the frame the pass works on" },
    { "togglekey", &ShmHeader::toggleKey, false, "Linux KEY_ code the layer watches, 0 for none" },
};

void Usage() {
    std::fprintf(stderr,
                 "usage: dlssnr-shmctl <shm-path> <command> [args]\n"
                 "\n"
                 "commands:\n"
                 "  status          print the header, one 'key=value' per line\n"
                 "  quit            ask the helper and the layer to stand down\n"
                 "  resume          clear the quit flag and nudge the readers\n"
                 "  reset           put every setting back to its default, leaving a running\n"
                 "                  helper and its sequence numbers alone\n"
                 "  reinit          re-initialise the whole header, transport included. For a\n"
                 "                  stale or corrupt mapping; takes a live session down with it\n"
                 "  capture <n>     write n matched before/after frames\n"
                 "  toggle <key>    flip a setting between 0 and 1\n"
                 "  set <key> <v>   change one setting\n"
                 "  settings        list the settings and their current values\n");
    std::fprintf(stderr, "\nsettings:\n");
    for (const auto& s : kSettings) std::fprintf(stderr, "  %-16s %s\n", s.name, s.help);
}

// Maps the header only. The two pixel regions are megabytes and nothing here reads them.
ShmHeader* MapHeader(const char* path, bool create, void** base, int* fdOut) {
    const int flags = create ? (O_RDWR | O_CREAT) : O_RDWR;
    const int fd = open(path, flags, 0600);
    if (fd < 0) return nullptr;

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }

    if ((size_t) st.st_size < ShmTotalBytes()) {
        if (!create) {  // nothing has ever attached; there is nothing to talk to
            close(fd);
            return nullptr;
        }
        if (ftruncate(fd, (off_t) ShmTotalBytes()) != 0) {
            close(fd);
            return nullptr;
        }
    }

    void* m = mmap(nullptr, kHeaderBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        return nullptr;
    }

    *base = m;
    *fdOut = fd;
    return (ShmHeader*) m;
}

bool Initialised(const ShmHeader* h) {
    return h->magic.load() == kShmMagic && h->version.load() == kShmVersion;
}

void PrintSettings(ShmHeader* h) {
    for (const auto& s : kSettings) {
        const uint32_t raw = (h->*s.field).load();
        if (s.isFloat) std::printf("%s=%g\n", s.name, double(BitsToFloat(raw)));
        else std::printf("%s=%u\n", s.name, raw);
    }
}

// Returns false when the name is not a setting, so the caller can say so rather than silently
// succeeding at nothing.
bool ApplySetting(ShmHeader* h, const char* name, const char* value) {
    for (const auto& s : kSettings) {
        if (std::strcmp(s.name, name) != 0) continue;
        const double v = std::atof(value);
        (h->*s.field).store(s.isFloat ? FloatToBits(float(v)) : uint32_t(v < 0 ? 0 : v));
        h->controlSeq.fetch_add(1);
        // Anything the model latches when its feature is built also bumps the tuning sequence, which
        // is what tells the helper to rebuild rather than to keep using a feature built with the old
        // values. Sharpness is absent: it is read at evaluate, so a running feature follows it.
        static const char* kCreateTime[] = { "preset", "style", "automask", "intensity",
                                             "localtone", "localstructure", "skinstructure", "passes" };
        for (const char* k : kCreateTime) {
            if (std::strcmp(k, name) == 0) { h->tuningSeq.fetch_add(1); break; }
        }
        return true;
    }
    return false;
}

void PrintStatus(const ShmHeader* h) {
    std::printf("initialised=%d\n", Initialised(h) ? 1 : 0);
    std::printf("magic=%#x\nversion=%u\n", h->magic.load(), h->version.load());
    if (!Initialised(h)) return;
    std::printf("seq_req=%u\nseq_resp=%u\n", h->seq_req.load(), h->seq_resp.load());
    std::printf("width=%u\nheight=%u\n", h->width.load(), h->height.load());
    std::printf("quit=%u\nheartbeat=%u\ncontrol_seq=%u\n", h->quit.load(), h->heartbeat.load(),
                h->controlSeq.load());
    std::printf("helper_state=%u\nmodel_up=%u\nhelper_frames=%llu\n", h->helperState.load(),
                h->modelUp.load(),
                (unsigned long long) ShmLoad64(h->helperFramesLo, h->helperFramesHi));
    {
        const unsigned long long gen =
            ((unsigned long long)h->mfgGeneratedHi.load() << 32) | h->mfgGeneratedLo.load();
        const unsigned long long missed =
            ((unsigned long long)h->mfgMissedHi.load() << 32) | h->mfgMissedLo.load();
        static const char* kMfgState[] = { "off", "on, not generating", "generating",
                                           "unavailable on this swapchain",
                                           "waiting: needs pipeline=1" };
        const unsigned st = h->mfgState.load();
        const unsigned long long noimg =
            ((unsigned long long)h->mfgNoImageHi.load() << 32) | h->mfgNoImageLo.load();
        std::printf("mfg_state=%s\nmfg_per_frame=%u%s\nmfg_wait_us=%u (ceiling %u)\n"
                    "mfg_generated=%llu\nmfg_missed=%llu\nmfg_no_image=%llu\n"
                    "mfg_motion=%.2f,%.2f\n",
                    kMfgState[st < 5 ? st : 0],
                    h->mfgAuto.load() ? h->mfgActiveFactor.load() : h->mfgFactor.load(),
                    h->mfgAuto.load() ? " (measured)" : " (fixed)", h->mfgAcquireWaitUs.load(),
                    h->mfgWaitCeilingUs.load(), gen, missed, noimg,
                    BitsToFloat(h->mfgMotionXBits.load()), BitsToFloat(h->mfgMotionYBits.load()));
    }
    std::printf("layer_composition_up=%u\nlayer_frames=%llu\nlayer_ms=%.2f\n",
                h->layerCompositionUp.load(),
                (unsigned long long) ShmLoad64(h->layerFramesLo, h->layerFramesHi),
                double(BitsToFloat(h->layerMsBits.load())));
    std::printf("measured_white_point=%g\n", double(BitsToFloat(h->layerMeasuredWhiteBits.load())));
    std::printf("hdr_mode=%u\nhdr_detected=%u\nhdr_active=%u\nproxy_format=%u\nhdr_encode=%u\n",
                h->hdrMode.load(), h->hdrDetected.load(), h->hdrActive.load(),
                h->proxyFormat.load(), h->hdrEncode.load());
    const std::string reason = ShmLoadString(h->helperReasonSeq, h->helperReason, kReasonBytes);
    if (!reason.empty()) std::printf("helper_reason=%s\n", reason.c_str());
}

}  // namespace

// Frame generation is not a thing that can be switched on by itself, and the shape of the mistake is
// always the same: mfg goes to 1, nothing happens, and the only clue is a status field nobody looks
// at until they are told to. So say it at the moment the setting changes, in both directions.
static void WarnAboutPipeline(ShmHeader* h) {
    if (h->mfgEnabled.load() == 0 || h->pipeline.load() != 0) return;
    std::fprintf(stderr,
        "warning: frame generation will not run while pipeline=0.\n"
        "  It measures how far the picture moved by comparing the frame being encoded against the\n"
        "  frame an outstanding answer belongs to, and with the model waited on there is never an\n"
        "  answer in flight to measure against.\n"
        "  Fix:  dlssnr-shmctl <shm> set pipeline 1\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        Usage();
        return 2;
    }
    const char* path = argv[1];
    const char* cmd = argv[2];

    const bool create = std::strcmp(cmd, "resume") == 0 || std::strcmp(cmd, "reset") == 0 ||
                        std::strcmp(cmd, "reinit") == 0 ||
                        std::strcmp(cmd, "set") == 0 || std::strcmp(cmd, "capture") == 0 ||
                        std::strcmp(cmd, "toggle") == 0;

    void* base = nullptr;
    int fd = -1;
    ShmHeader* h = MapHeader(path, create, &base, &fd);
    if (!h) {
        // No mapping and none wanted: for 'quit' that means nothing is running, which is success.
        return std::strcmp(cmd, "quit") == 0 ? 0 : 1;
    }

    int rc = 0;
    if (std::strcmp(cmd, "status") == 0) {
        PrintStatus(h);
    } else if (std::strcmp(cmd, "quit") == 0) {
        if (Initialised(h)) {
            h->quit.store(1);
            h->controlSeq.fetch_add(1);
        }
    } else if (std::strcmp(cmd, "resume") == 0) {
        if (!Initialised(h)) ShmInitDefaults(h);
        h->quit.store(0);
        h->controlSeq.fetch_add(1);
    } else if (std::strcmp(cmd, "reset") == 0) {
        // Settings only. This used to re-initialise the whole header, which on a live mapping also
        // reset the sequence numbers a running helper was answering and declared the helper stopped.
        if (!Initialised(h)) ShmInitDefaults(h);
        else ShmResetSettings(h);
    } else if (std::strcmp(cmd, "reinit") == 0) {
        ShmInitDefaults(h);
    } else if (std::strcmp(cmd, "settings") == 0) {
        if (!Initialised(h)) ShmInitDefaults(h);
        PrintSettings(h);
    } else if (std::strcmp(cmd, "capture") == 0) {
        if (argc != 4) { Usage(); rc = 2; }
        else {
            if (!Initialised(h)) ShmInitDefaults(h);
            h->captureRequest.store(uint32_t(std::atoi(argv[3])));
            h->controlSeq.fetch_add(1);
        }
    } else if (std::strcmp(cmd, "toggle") == 0) {
        // The one command worth binding to a key.
        //
        // On Wayland a game is a client and its keys never reach this process, and /dev/input is not
        // readable without the 'input' group -- keyboards get no uaccess ACL, deliberately, because
        // that would let any program keylog. So on a Wayland game the layer cannot read a key at all,
        // and the way to get an in-game toggle is to bind this command to a shortcut in the desktop's
        // own settings, where the compositor already has the key and will deliver it over a fullscreen
        // window.
        if (argc != 4) { Usage(); rc = 2; }
        else {
            if (!Initialised(h)) ShmInitDefaults(h);
            bool found = false;
            for (const auto& st : kSettings) {
                if (std::strcmp(st.name, argv[3]) != 0) continue;
                found = true;
                const uint32_t now = (h->*st.field).load();
                const uint32_t next = now ? 0u : 1u;
                (h->*st.field).store(st.isFloat ? FloatToBits(float(next)) : next);
                h->controlSeq.fetch_add(1);
                std::printf("%s=%u\n", st.name, next);
                break;
            }
            if (!found) {
                std::fprintf(stderr, "unknown setting: %s\n", argv[3]);
                rc = 2;
            }
            WarnAboutPipeline(h);
        }
    } else if (std::strcmp(cmd, "set") == 0) {
        if (argc != 5) { Usage(); rc = 2; }
        else {
            if (!Initialised(h)) ShmInitDefaults(h);
            if (!ApplySetting(h, argv[3], argv[4])) {
                std::fprintf(stderr, "unknown setting: %s\n", argv[3]);
                rc = 2;
            } else {
                WarnAboutPipeline(h);
            }
        }
    } else {
        Usage();
        rc = 2;
    }

    msync(base, kHeaderBytes, MS_SYNC);
    munmap(base, kHeaderBytes);
    close(fd);
    return rc;
}
