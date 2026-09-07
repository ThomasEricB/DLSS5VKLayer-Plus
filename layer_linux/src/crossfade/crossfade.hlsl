// Move one picture a fraction of the way toward another.
//
// This exists for one reason: the pipelined path's edit changes in steps. An answer arrives every few
// frames, and until this pass the pair the composition differences -- the model's answer and the proxy
// it was computed from -- was replaced whole on the frame the answer landed. The picture underneath is
// always current, so nothing smears; but the *edit* laid on it jumped on one frame and then held for
// several, and a game running fast enough steps it often enough to read as a flicker.
//
// Blending the two terms separately is the same as blending the edit, because the edit is their
// difference and lerp(m0,m1,a) - lerp(p0,p1,a) == lerp(m0-p0, m1-p1, a). So the composition shader
// needs to know nothing about any of this: it is handed a pair, as before, and the pair happens to be
// on its way from the last answer to the newest one.
//
// Applied every frame toward the newest answer rather than run as a fixed-length fade, which makes it
// a first-order approach -- the right shape when nobody controls how often answers arrive.
//
// What it costs, so nobody has to rediscover it: two answers taken a few frames apart were computed
// on slightly different geometry, and blending them cancels wherever they disagree. The edit comes
// out weaker as well as smoother, by roughly a quarter at the default rate on fast motion. Aligning
// the running pair with the motion field before blending it would recover that, and is the next thing
// to try here -- the field the composition already reprojects with runs the wrong way for it.

[[vk::binding(0, 0)]]
cbuffer Params : register(b0)
{
    uint  gWidth;
    uint  gHeight;
    float gAlpha;   // 0 stay put, 1 take the target whole
    uint  gPad;
};

[[vk::binding(1, 0)]]
Texture2D<float4> gTargetTex : register(t0);   // where the pair is heading

// Every surface this blends is R8G8B8A8_UNORM, so the format is declared rather than left unknown --
// an unknown-format *read* needs a device feature that an unknown-format write does not, and there is
// nothing to gain here by asking for it.
[[vk::binding(2, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> gCurrent : register(u0);   // read and written in place, one texel per invocation

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gWidth || id.y >= gHeight)
        return;

    const float4 target = gTargetTex.Load(int3(id.xy, 0));
    const float4 current = gCurrent[id.xy];
    gCurrent[id.xy] = lerp(current, target, saturate(gAlpha));
}
