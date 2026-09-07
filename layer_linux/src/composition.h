#pragma once
// The DLSS-NR pass, arranged for a present-time layer.
//
// OptiScaler runs this immediately after the game's upscaler, on surfaces it already holds. Here the
// only thing available is a finished swapchain image, and the model lives in another process behind a
// shared-memory round trip, so the same pass has to be split in two around that round trip:
//
//   leg 1   swapchain -> frame -> ENCODE -> proxy (+ the untouched keep)
//                              -> DOWNSAMPLE -> work, when the model runs below the frame
//                              -> host buffer
//     ...   the helper runs the model on those pixels and answers
//   leg 2   host buffer -> model image -> RESOLVE -> composed -> swapchain
//
// What crosses the shared memory is the proxy, not the frame: it is display-referred and 8-bit by
// construction, so the helper never has to know what format the game presents in, and the working
// scale reduces it quadratically.
//
// The composition itself -- what the resolve does with the model's answer -- is entirely the vendored
// shader's. Everything in this file is plumbing: which image is bound where, in what layout, and what
// goes in the constant block.
#include "capture.h"
#include "crossfade/crossfade_vk.h"
#include "motion/globalmotion_vk.h"
#include "dlssnr_pass.h"
#include "scaler_vk.h"
#include "vk_table.h"

#include "../../common/shm_protocol.h"

#include <memory>
#include <string>

namespace dlssnr {

// One frame's worth of settings, read from the shared header once so that a control changed
// mid-frame cannot make the encode and the resolve disagree about what they are doing.
struct FrameSettings {
    float transferStrength = 1.0f;
    float colourStrength = 1.0f;
    float maxRatio = 2.0f;
    float debugScale = 1.0f;
    // The white point is three numbers, not one: where it comes from, the multiplier that says what
    // the model should consider white, and the trim that belongs to a measured reading rather than to
    // the slider. Keeping them apart is upstream's fix for a real bug -- sharing one stored value
    // meant touching the slider in one mode silently destroyed the number found in the other.
    float whitePointManual = 1.0f;
    float whitePointScale = 1.0f;
    float whitePointTrim = 1.0f;
    uint32_t whitePointSource = kWhitePointManual;
    float compareSplit = 0.5f;
    float compareZoom = 1.0f;
    float workingScale = 1.0f;
    uint32_t transfer = 1;
    uint32_t debugView = 0;
    uint32_t compareMode = 0;
    uint32_t compareSwap = 0;
    uint32_t reversibleMode = kReversibleKnee;

    // Not a header setting: the layer decides it per frame, from whether it is running the round trip
    // in front of the frame or alongside it.
    bool pipelined = false;
    uint32_t applyModel = 1;
    uint32_t holdFrame = 0;
    uint32_t downscaler = kScalerLanczos3;
    // 1: present the model's raw answer as the frame -- no blend, no guard, no compare.
    uint32_t compositionBypass = 0;

    // How much of the way toward a newly arrived answer the running pair moves each frame, 0..1.
    // 1 is the old take-it-whole behaviour. Only consulted on the pipelined path.
    float settleRate = 0.4f;

    // How far past its neighbours' brightness a pipelined pixel may land. See ghostSlackPercent.
    float ghostSlack = 0.5f;

    // Radius splitting the stale edit's safe half from the half that can ghost, in uv.
    float editBlur = 0.04f;

    static FrameSettings Read(const ShmHeader* h);
};

// Whether a swapchain format can be composed at all, and what this pass works in when it can.
//
// Every internal surface uses the format's UNORM twin rather than the swapchain's own: sampling a
// _SRGB view would decode to linear on the way in and re-encode on the way out, and the composition
// wants exactly the display-referred numbers the game already wrote. Copies between the two are
// byte-for-byte, which is why the final result goes back with vkCmdCopyImage and not a blit -- a blit
// into an _SRGB image would apply the encode a second time.
VkFormat CompositionFormat(VkFormat swapchainFormat);

// Whether the frame the game presents holds light or a picture. 8-bit and 10-bit formats are already
// display-referred -- the game tone mapped before it got here -- and only a float swapchain is linear.
bool ColourIsLinearHdr(VkFormat swapchainFormat, uint32_t colourMode);

class Composition {
  public:
    Composition(const DeviceTable* vk, const InstanceTable* instance, VkDevice device,
                VkPhysicalDevice physicalDevice);
    ~Composition();

    Composition(const Composition&) = delete;
    Composition& operator=(const Composition&) = delete;

    bool Usable() const { return _usable; }

    // Why not, when not. Empty while it is working.
    const char* Reason() const { return _reason.c_str(); }

    // Build or rebuild everything sized to this frame and this model resolution. Cheap and a no-op
    // when nothing has changed, so it is safe to call every present.
    bool Prepare(uint32_t width, uint32_t height, VkFormat swapchainFormat, const FrameSettings& s,
                 bool linearHdr);

    uint32_t ModelWidth() const { return _modelW; }
    uint32_t ModelHeight() const { return _modelH; }
    size_t ModelBytes() const { return size_t(_modelW) * _modelH * 4; }

    // Leg 1. Leaves the proxy the model should see in the download buffer, and the swapchain image
    // back in PRESENT_SRC_KHR so a caller that gives up after this still presents something valid.
    bool RecordCapture(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s);

    // The two halves of it. A pipelined frame grabs before composing and encodes after, so that the
    // composition sees a proxy, a keep and a model that all belong to one frame.
    bool RecordGrab(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s);
    bool RecordEncode(VkCommandBuffer cb, const FrameSettings& s);

    // The encode's result on its way to the helper. Split from the encode because a pipelined frame
    // encodes every time and sends only when the helper is ready for another.
    bool RecordSend(VkCommandBuffer cb, const FrameSettings& s);

    // Keep a copy of what this frame is about to send, so the answer can be differenced against it
    // when it arrives. Recorded straight after the encode, in the same command buffer.
    bool RecordKeepSent(VkCommandBuffer cb);
    bool HasSentProxy() const { return _targetValid; }

    // How many round trips of motion the stale edit has to be moved by. The layer measures it; see
    // ShmMap::reprojScale.
    void SetReprojScale(float r) { _reprojScale = r; }

    // The answer for the proxy in flight has arrived, so that proxy becomes the matched one. A swap
    // of handles rather than a copy -- the surfaces are identical in every respect but their
    // contents. Must be called before the composition reads the pair.
    // A new answer has landed, so the proxy it was computed from stops being the one in flight and
    // becomes the one being walked toward. Not the one the composition reads: that is _proxySent, and
    // it moves there a fraction at a time so the edit ramps instead of stepping.
    void AdoptSentProxy() {
        if (!_flightValid) return;
        std::swap(_proxyTarget, _proxyFlight);
        std::swap(_workTarget, _workFlight);
        _flightValid = false;
        _targetValid = true;
    }

    // Whether this frame recorded a capture readback. The pipelined path does not wait for its own
    // submit, so it has to wait for this one before reading what the copy produced.
    bool CaptureRecorded() const { return _captureRecorded; }

    // Take a motion field for the answer currently held. Sized to the frame, two half floats a pixel,
    // in pixels of the frame. Without one the stale edit is laid down where it was computed, which is
    // where it ghosts.
    bool RecordMotion(VkCommandBuffer cb, VkBuffer from, uint32_t w, uint32_t h);
    bool HasMotion() const { return _motionValid; }
    void DropMotion() { _motionValid = false; }

    // The two ends of the round trip, when they have to be copied.
    //
    // Both return null once the shared regions have been imported as device memory: the capture leg
    // writes the proxy into the shared pages itself and the compose leg reads the answer from them,
    // so there is nothing for the caller to move and asking for a pointer would only invite a copy.
    const void* ProxyPixels() const { return _sharedIn ? nullptr : _download.mapped; }
    void* ModelPixels() { return _sharedOut ? nullptr : _upload.mapped; }

    // Use the caller's buffers -- the shared regions -- instead of this object's own staging. Passing
    // two null handles goes back to staging, which is what happens on a device that cannot import.
    void UseSharedBuffers(VkBuffer in, VkBuffer out) { _sharedIn = in; _sharedOut = out; }
    bool ZeroCopy() const { return _sharedIn && _sharedOut; }

    // Leg 2. Composes and leaves the swapchain image holding the result, in PRESENT_SRC_KHR.
    //
    // refreshModel says whether to take a new answer out of the shared region first. It is false on
    // a pipelined frame that is re-using the last answer, and that is not an optimisation: the helper
    // may be writing that region right now for the next request, and reading it would tear. The
    // model image already holds the last answer, so composing from it is both safe and correct.
    bool RecordCompose(VkCommandBuffer cb, VkImage swapchainImage, const FrameSettings& s,
                       bool refreshModel = true);

    bool HasModelFrame() const { return _haveModel; }
    void MarkModelFrame() { _haveModel = true; }

    // Write this many matched before/after pairs, starting with the next composed frame.
    void RequestCapture(uint32_t frames) { _capture.Begin(frames); }
    bool CaptureActive() const { return _capture.Active(); }

    // Called after leg 2's fence, when the readback the compose recorded has landed.
    void WriteCapturedFrame();

    // Called after leg 1's fence: turns the tile grid the meter wrote into a white point.
    void ConsumeMeter();

    // What the meter settled on, or 0 when it has not taken a usable reading. For the interface, so
    // the number in use is visible rather than inferred.
    float MeasuredWhitePoint() const { return _measuredWhitePoint; }

  private:
    struct Image {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t width = 0, height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct HostBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        size_t size = 0;
    };

    bool MakeImage(Image& img, uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage);
    void DropImage(Image& img);
    bool MakeHostBuffer(HostBuffer& buf, size_t bytes, VkBufferUsageFlags usage);
    void DropHostBuffer(HostBuffer& buf);
    void DropAll();

    void Transition(VkCommandBuffer cb, Image& img, VkImageLayout to);
    void TransitionSwapchain(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to);
    bool FormatSupportsBlit(VkFormat format) const;

    // Ordering against the process on the other side of the shared regions. See the definitions.
    void BarrierAfterExternalWrite(VkCommandBuffer cb) const;
    void BarrierBeforeExternalRead(VkCommandBuffer cb) const;
    void CopyWholeImage(VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, VkImage dst,
                        VkImageLayout dstLayout, uint32_t w, uint32_t h);

    bool FormatSupportsStorage(VkFormat format) const;
    DlssNrConstants BaseConstants(const FrameSettings& s) const;
    float ResolvedWhitePoint(const FrameSettings& s) const;

    const DeviceTable* _vk = nullptr;
    const InstanceTable* _instance = nullptr;
    VkDevice _device = VK_NULL_HANDLE;
    VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;

    std::unique_ptr<DlssNrPass> _pass;
    bool _usable = false;
    std::string _reason;

    uint32_t _width = 0, _height = 0;
    uint32_t _modelW = 0, _modelH = 0;
    VkFormat _swapchainFormat = VK_FORMAT_UNDEFINED;
    VkFormat _workFormat = VK_FORMAT_UNDEFINED;  // usually the swapchain's UNORM twin; see Prepare
    bool _blitSwapchain = false;                 // set when _workFormat is not the swapchain's twin
    VkFormat _keepFormat = VK_FORMAT_UNDEFINED;
    bool _linearHdr = false;
    bool _haveModel = false;

    Image _frame{}, _keep{}, _proxy{}, _work{}, _model{}, _composed{};

    // The proxy and working raster that went to the model with the answer now in _model.
    //
    // Only allocated for the pipelined path, where the answer is a frame behind what the encode has
    // since written. The resolve differences the model against the proxy it was actually computed
    // from; differencing it against the proxy of a newer frame is not an approximation, it is a
    // ratio between two unrelated pictures, and that is what put saturated pixels on moving edges.
    // Two of them, and the pair is the whole point.
    //
    // _proxySent is the picture the answer in _model was computed from. _proxyFlight is the one that
    // has been sent and is still being worked on. They cannot be the same surface: a send happens
    // several frames before its answer comes back, so overwriting the matched proxy at send time
    // leaves the composition differencing the old answer against the new picture -- which is not a
    // stale edit, it is the difference of two frames, and it puts a complete second exposure of the
    // scene on the screen.
    Image _proxySent{}, _workSent{};
    Image _proxyFlight{}, _workFlight{};

    // Where a newly arrived answer and its proxy are put, so the pair the composition reads can be
    // walked toward them over a few frames rather than replaced between one frame and the next.
    Image _proxyTarget{}, _workTarget{}, _modelTarget{};
    std::unique_ptr<CrossfadeVk> _crossfade;

    // Measures, in the layer, how far the picture moved between the frame the answer belongs to and
    // this one. That is the interval the composition actually needs, and the only place both ends of
    // it exist. See globalmotion.comp.
    std::unique_ptr<GlobalMotionVk> _globalMotion;

    // Set once the running pair holds a real answer. Until then there is nothing to walk away from
    // and the first arrival is taken whole -- blending toward it out of an uninitialised surface
    // would show whatever the allocation happened to contain.
    bool _settled = false;
    bool _targetValid = false;
    float _reprojScale = 1.0f;
    bool _flightValid = false;

    // The motion field for the answer being held, uploaded from the helper's shared region. Says
    // where each of this frame's pixels was in the frame that answer belongs to.
    Image _motion{};
    bool _motionValid = false;
    uint32_t _motionW = 0, _motionH = 0;

    // Supersampling: the model works above the frame, so the proxy is enlarged on the way in and the
    // answer averaged back on the way out. _modelNative holds that average; without it the resolve
    // would read the larger answer through a bilinear sampler and alias, which is the reason upstream
    // gave this its own filter rather than reusing the resolve's.
    Image _modelNative{};
    std::unique_ptr<ScalerVk> _superUp, _superDown;
    bool _superSample = false;
    uint32_t _scalerFilter = kScalerLanczos3;

    // The white point meter: a grid of tile peak luminances measured off the untouched copy, and the
    // percentile the host takes across it. See ConsumeMeter.
    Image _meter{};
    HostBuffer _meterBuf{};
    bool _meterRecorded = false;
    float _measuredWhitePoint = 0.0f;
    static constexpr uint32_t kMeterHistory = 16;
    float _meterHistory[kMeterHistory] = {};
    uint32_t _meterCount = 0;
    float _meterSteadiness = 0.0f;
    HostBuffer _download{}, _upload{}, _captureBuf{};

    // Not owned. The shared regions, imported by the transport and bound to buffers there.
    VkBuffer _sharedIn = VK_NULL_HANDLE, _sharedOut = VK_NULL_HANDLE;

    CaptureWriter _capture;
    bool _captureRecorded = false;

    // Frame hold. The freeze point is the raw colour the encode reads, not the proxy: both the proxy
    // and the untouched keep are derived from it, and freezing further down would stop a setting
    // change from re-encoding, which is the whole point of holding.
    bool _holding = false;
    bool _frameCaptured = false;
    float _heldWhitePoint = 1.0f;
};

}  // namespace dlssnr
