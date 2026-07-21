/**
 * WGSL tone-mapping shader for the WebGPU tile viewer.
 *
 * Port of the GLSL shader at `../tone_mapping_shader.js`. Same Auto-STF
 * MTF math (PixInsight ScreenTransferFunction) applied per-pixel; only
 * the syntax + binding model differ.
 *
 * Binding layout:
 *   group 0:
 *     @binding(0) globals       — uniform Globals { e1, e5, max,
 *                                                   peakBrightness }
 *     @binding(1) viewport      — uniform Viewport { tx, ty, scale,
 *                                                    screenW, screenH }
 *     @binding(2) sampler       — linear, clamp-to-edge
 *   group 1:
 *     @binding(0) tileTexture   — texture_2d<f32>, r32float
 *     @binding(1) tileRect      — uniform TileRect { worldX, worldY,
 *                                                    tileW, tileH }
 *
 * Vertex shader emits a unit quad (4 verts, triangle-strip) whose
 * world-space corners come from `tileRect`. The viewport transform
 * places it into clip space. The same shader handles every tile;
 * tiles are differentiated by the per-tile bind group (group 1).
 *
 * SHADER_PREAMBLE (the structs + bindings 0–4 + group-1 bindings + vs_main
 * + normLin) is factored out and re-used by the data-driven finish shader
 * (generic_finish_shader.js) so the Globals struct layout can't drift
 * between the built-in GHS finish and a Python DisplayFinish — both bind
 * the SAME globals buffer.
 */
export const SHADER_PREAMBLE = /* wgsl */ `
struct Globals {
    e1: f32,
    e5: f32,
    maxVal: f32,
    peakBrightness: f32,
    // 0.0 = PixInsight STF (closed-form MTF using e1/e5 above).
    // 1.0 = LUT lookup (sample stretchLut at x normalized). Used by
    //       the constrained-max-entropy stretch (and any future
    //       LUT-based modes).
    stretchMode: f32,
    // >0.5 = baked-stretch RGB color image: sample .rgb and pass it
    // through (stretch is already applied), skipping all mono tone-
    // mapping. 0.0 = linear mono channel, tone-mapped below.
    isColor: f32,
    // >0.5 = LINEAR narrowband S/H/O tile: run the per-pixel colour
    // finish below (normalise → palette → black point → shaped stretch
    // → SCNR green → saturation → screen stars). Takes precedence over
    // isColor. The stretch reuses e1/e5 as the per-channel MTF.
    finishMode: f32,
    paletteMode: f32,   // 0 = Hubble (R=S,G=H,B=O), 1 = Foraxx dynamic
    blackPoint: f32,    // deep black point applied after the palette
    scnrAmount: f32,    // 0..1 green removal: G=min(G, mix(G,(R+B)/2, a))
    saturation: f32,    // luma-preserving chroma gain (1 = identity)
    addStars: f32,      // >0.5 = screen the stars texture on top
    // Per-channel normalise anchors (background median m, high pct p):
    // n = (c - m)/(p - m). Seeded from the tile metadata's per-channel
    // stats; trims are user gain on top (default 1).
    mS: f32, mH: f32, mO: f32,
    pS: f32, pH: f32, pO: f32,
    trimS: f32, trimH: f32, trimO: f32,
    // Per-channel background sigma + a shared shadow-clip strength: the low
    // normalise anchor is m + shadowClip·sigma PER CHANNEL, so a noisy channel
    // (SII/OIII) has its noise floor cut while clean Hα is untouched. Slider.
    sigmaS: f32, sigmaH: f32, sigmaO: f32,
    shadowClip: f32,
    // Per-channel Generalized Hyperbolic Stretch (GHS) — replaces the single MTF
    // in the finish. Each S/H/O is stretched INDEPENDENTLY: D_ = strength (shader
    // uses exp(D)-1, the ghsastro convFacD=1 convention), b_ = local intensity /
    // curve form (b<0 asinh-ish, 0 exp, >0 hyperbolic), SP_ = symmetry point
    // (seeded to each channel's dark-tail background so the sky lifts, never
    // crushes). LP/HP = shadow/highlight linear-protection; BP/WP = black/white
    // clip. Grid axes perturb per-channel D/b/SP; LP/HP/BP/WP are sliders.
    DS: f32, DH: f32, DO: f32,
    bS: f32, bH: f32, bO: f32,
    SPS: f32, SPH: f32, SPO: f32,
    LP: f32, HP: f32, BP: f32, WP: f32,
    _pad0: f32, _pad1: f32,
};
struct Viewport {
    tx: f32,
    ty: f32,
    scale: f32,
    screenW: f32,
    screenH: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
struct TileRect {
    worldX: f32,
    worldY: f32,
    tileW: f32,
    tileH: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> viewport: Viewport;
@group(0) @binding(2) var tileSampler: sampler;
// 1024×1 r32float texture carrying the stretch LUT. Indexed by
// normalized input (xRaw/maxVal). Updated whenever stats or the
// selected mode change.
@group(0) @binding(3) var stretchLut: texture_2d<f32>;
@group(0) @binding(4) var stretchLutSampler: sampler;
@group(1) @binding(0) var tileTexture: texture_2d<f32>;
@group(1) @binding(1) var<uniform> tileRect: TileRect;
// RGB stars for the narrowband finish (screened on top when addStars).
// A shared 1×1 black texture is bound for tiles/taps without stars, so
// the screen blend is an identity no-op there.
@group(1) @binding(2) var starsTexture: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // Unit quad as triangle-strip: (0,0), (1,0), (0,1), (1,1).
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let p = positions[idx];

    // World-space pixel coordinates of this vertex.
    let worldX = tileRect.worldX + p.x * tileRect.tileW;
    let worldY = tileRect.worldY + p.y * tileRect.tileH;

    // Apply viewport transform (translate then scale), produce screen
    // pixels, then convert to clip space [-1, 1].
    let screenX = (worldX + viewport.tx) * viewport.scale;
    let screenY = (worldY + viewport.ty) * viewport.scale;
    let clipX = (screenX / viewport.screenW) * 2.0 - 1.0;
    // Clip-space Y is up-positive; screen Y is down-positive — flip.
    let clipY = 1.0 - (screenY / viewport.screenH) * 2.0;

    var out: VsOut;
    out.pos = vec4<f32>(clipX, clipY, 0.0, 1.0);
    out.uv = p;
    return out;
}

// Per-channel affine normalise to a common scale: (c - m)/(p - m).
fn normLin(c: f32, m: f32, p: f32) -> f32 {
    return (c - m) / max(p - m, 1e-6);
}
`;

// The built-in finish (mono STF/LUT + baked color + per-channel GHS narrowband)
// = the shared preamble followed by its own helpers and fs_main.
export const TONE_MAPPING_WGSL = SHADER_PREAMBLE + /* wgsl */ `
// Closed-form PixInsight MTF stretch (reuses globals.e1/e5), per channel.
fn stfChannel(x: f32) -> f32 {
    let E2 = max(0.0, (x - globals.e1) / max(1.0 - globals.e1, 1e-12));
    let num = (globals.e5 - 1.0) * E2;
    let den = (2.0 * globals.e5 - 1.0) * E2 - globals.e5;
    if (abs(den) < 1e-12) {
        return 0.0;
    }
    return clamp(num / den, 0.0, 1.0);
}

// Foraxx dynamic narrowband palette (mirrors palettes.foraxx). Inputs
// are the normalised, brightness-matched channels.
fn foraxxPalette(s: f32, h: f32, o: f32) -> vec3<f32> {
    let sc = clamp(s, 0.0, 1.0);
    let hc = clamp(h, 0.0, 1.0);
    let oc = clamp(o, 0.0, 1.0);
    let wR = pow(oc, 1.0 - oc);
    let R = wR * sc + (1.0 - wR) * hc;
    let m = oc * hc;
    let wG = pow(m, 1.0 - m);
    let G = wG * hc + (1.0 - wG) * oc;
    let B = oc;
    return vec3<f32>(R, G, B);
}

// Generalized Hyperbolic Stretch (ghsastro / mikec1485 reference math), applied
// per channel. Dslider maps as D = exp(Dslider)-1 (convFacD=1). The curve is
// piecewise: linear below LP, generalized-hyperbolic LP→SP→HP (form set by b —
// exp for b=0, pow for b≷0), linear above HP. Coefficients computed inline (they
// depend only on the uniforms, constant across the tile — cheap on the GPU). SP
// is seeded to each channel's dark-tail background so the sky LIFTS, never crushes.
fn ghsChannel(x: f32, Dslider: f32, bIn: f32, SP: f32, LP: f32, HP: f32) -> f32 {
    let D = exp(Dslider) - 1.0;
    let z = clamp(x, 0.0, 1.0);
    if (D <= 0.0) { return z; }
    // Guard the log singularity at b=-1 (integral form divides by (|b|-1)).
    var b = bIn;
    if (abs(b + 1.0) < 0.02) { b = -0.98; }

    var b1 = 0.0;
    var a2 = 0.0; var b2 = 0.0; var c2 = 0.0; var d2 = 0.0; var e2 = 0.0;
    var a3 = 0.0; var b3 = 0.0; var c3 = 0.0; var d3 = 0.0; var e3 = 0.0;
    var a4 = 0.0; var b4 = 0.0;
    var isExp = false;

    if (b == 0.0) {                                   // exponential
        isExp = true;
        let qlp = exp(-D * (SP - LP));
        let q0 = qlp - D * LP * exp(-D * (SP - LP));
        let qwp = 2.0 - exp(-D * (HP - SP));
        let q1 = qwp + D * (1.0 - HP) * exp(-D * (HP - SP));
        let q = 1.0 / (q1 - q0);
        b1 = D * exp(-D * (SP - LP)) * q;
        a2 = -q0 * q; b2 = q; c2 = -D * SP; d2 = D;
        a3 = (2.0 - q0) * q; b3 = -q; c3 = D * SP; d3 = -D;
        a4 = (qwp - q0 - D * HP * exp(-D * (HP - SP))) * q;
        b4 = D * exp(-D * (HP - SP)) * q;
    } else if (b > 0.0) {                             // hyperbolic (pow)
        let B = b;
        let qlp = pow(1.0 + D * B * (SP - LP), -1.0 / B);
        let q0 = qlp - D * LP * pow(1.0 + D * B * (SP - LP), -(1.0 + B) / B);
        let qwp = 2.0 - pow(1.0 + D * B * (HP - SP), -1.0 / B);
        let q1 = qwp + D * (1.0 - HP) * pow(1.0 + D * B * (HP - SP), -(1.0 + B) / B);
        let q = 1.0 / (q1 - q0);
        b1 = D * pow(1.0 + D * B * (SP - LP), -(1.0 + B) / B) * q;
        a2 = -q0 * q; b2 = q; c2 = 1.0 + D * B * SP; d2 = -D * B; e2 = -1.0 / B;
        a3 = (2.0 - q0) * q; b3 = -q; c3 = 1.0 - D * B * SP; d3 = D * B; e3 = -1.0 / B;
        a4 = (qwp - q0 - D * HP * pow(1.0 + D * B * (HP - SP), -(B + 1.0) / B)) * q;
        b4 = D * pow(1.0 + D * B * (HP - SP), -(B + 1.0) / B) * q;
    } else {                                          // integral (b<0, pow)
        let B = -b;
        let qlp = (1.0 - pow(1.0 + D * B * (SP - LP), (B - 1.0) / B)) / (B - 1.0);
        let q0 = qlp - D * LP * pow(1.0 + D * B * (SP - LP), -1.0 / B);
        let qwp = (pow(1.0 + D * B * (HP - SP), (B - 1.0) / B) - 1.0) / (B - 1.0);
        let q1 = qwp + D * (1.0 - HP) * pow(1.0 + D * B * (HP - SP), -1.0 / B);
        let q = 1.0 / (q1 - q0);
        b1 = D * pow(1.0 + D * B * (SP - LP), -1.0 / B) * q;
        a2 = (1.0 / (B - 1.0) - q0) * q; b2 = -q / (B - 1.0); c2 = 1.0 + D * B * SP; d2 = -D * B; e2 = (B - 1.0) / B;
        a3 = (-1.0 / (B - 1.0) - q0) * q; b3 = q / (B - 1.0); c3 = 1.0 - D * B * SP; d3 = D * B; e3 = (B - 1.0) / B;
        a4 = (qwp - q0 - D * HP * pow(1.0 + D * B * (HP - SP), -1.0 / B)) * q;
        b4 = D * pow(1.0 + D * B * (HP - SP), -1.0 / B) * q;
    }

    var y: f32;
    if (z < LP) {
        y = b1 * z;                                   // a1 is always 0
    } else if (z < SP) {
        let u = c2 + d2 * z;
        if (isExp) { y = a2 + b2 * exp(u); } else { y = a2 + b2 * pow(max(u, 1e-12), e2); }
    } else if (z < HP) {
        let u = c3 + d3 * z;
        if (isExp) { y = a3 + b3 * exp(u); } else { y = a3 + b3 * pow(max(u, 1e-12), e3); }
    } else {
        y = a4 + b4 * z;
    }
    return clamp(y, 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sample both textures once, in uniform control flow (valid
    // derivatives), then branch. Color tiles carry the stretch baked in —
    // pass .rgb straight through (zeros → black is inherent, no sentinel
    // needed). starsTexture is always bound (a 1×1 black dummy when the
    // tap/tile has no stars) so sampling it here is always legal.
    let tex = textureSample(tileTexture, tileSampler, in.uv);
    let starTex = textureSample(starsTexture, tileSampler, in.uv);

    // Linear narrowband S/H/O: run the full per-pixel colour finish.
    if (globals.finishMode > 0.5) {
        let S = tex.r;
        let H = tex.g;
        let O = tex.b;
        // All-zero is the no-data footprint sentinel → opaque black.
        if (S == 0.0 && H == 0.0 && O == 0.0) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        // 1. Per-channel normalise to [0,1]. The low anchor sits BELOW the median
        //    (m - shadowClip·sigma) so the sky background SURVIVES as a small
        //    positive value — the GHS below lifts it to a grey rather than the old
        //    m + shadowClip·sigma anchor that clipped the sky to pure black.
        let loS = globals.mS - globals.shadowClip * globals.sigmaS;
        let loH = globals.mH - globals.shadowClip * globals.sigmaH;
        let loO = globals.mO - globals.shadowClip * globals.sigmaO;
        let nS = clamp((S - loS) / max(globals.pS - loS, 1e-6), 0.0, 1.0) * globals.trimS;
        let nH = clamp((H - loH) / max(globals.pH - loH, 1e-6), 0.0, 1.0) * globals.trimH;
        let nO = clamp((O - loO) / max(globals.pO - loO, 1e-6), 0.0, 1.0) * globals.trimO;
        // 2. Per-channel GHS view transform — each channel stretched INDEPENDENTLY
        //    with its own D/b/SP (SP seeded to its dark-tail background). Replaces
        //    the single MTF; this is the grid's per-channel stretch surface.
        let gS = ghsChannel(nS, globals.DS, globals.bS, globals.SPS, globals.LP, globals.HP);
        let gH = ghsChannel(nH, globals.DH, globals.bH, globals.SPH, globals.LP, globals.HP);
        let gO = ghsChannel(nO, globals.DO, globals.bO, globals.SPO, globals.LP, globals.HP);
        // 3. Palette combine (0 = Hubble R=S/G=H/B=O, 1 = Foraxx) on the stretched
        //    channels; SCNR green below turns the Hα green into the SHO gold.
        var rgb: vec3<f32>;
        if (globals.paletteMode > 0.5) {
            rgb = foraxxPalette(gS, gH, gO);
        } else {
            rgb = vec3<f32>(gS, gH, gO);
        }
        // 4. Black / white point clip.
        rgb = clamp((rgb - vec3<f32>(globals.BP)) / max(globals.WP - globals.BP, 1e-6),
                    vec3<f32>(0.0), vec3<f32>(1.0));
        // 5. SCNR green (average-neutral, amount-scaled).
        let cap = mix(rgb.g, (rgb.r + rgb.b) * 0.5, globals.scnrAmount);
        rgb.g = min(rgb.g, cap);
        // 6. Luma-preserving saturation.
        let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        rgb = clamp(mix(vec3<f32>(luma), rgb, globals.saturation),
                    vec3<f32>(0.0), vec3<f32>(1.0));
        // 7. Screen the (already display-space) RGB stars on top.
        if (globals.addStars > 0.5) {
            let st = clamp(starTex.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
            rgb = 1.0 - (1.0 - rgb) * (1.0 - st);
        }
        // 8. Output.
        return vec4<f32>(rgb * globals.peakBrightness, 1.0);
    }

    if (globals.isColor > 0.5) {
        return vec4<f32>(tex.rgb * globals.peakBrightness, 1.0);
    }

    let xRaw = tex.r;

    // Zero is the "no data" sentinel for the irregular mosaic
    // footprint inside the rectangular tile bbox. Opaque black.
    if (xRaw == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // Normalize-by-max — same as image/max(image) in the Python
    // original.
    let x = clamp(xRaw / globals.maxVal, 0.0, 1.0);

    var y: f32;
    if (globals.stretchMode > 0.5) {
        // LUT-based stretch (constrained max-entropy or any future
        // mode that needs an empirically-fitted curve). Texture is
        // 1024×1, so we sample at vec2(x, 0.5) with the linear
        // sampler — linear interpolation between adjacent LUT
        // entries gives sub-bin precision for free.
        //
        // textureSampleLevel (explicit LOD 0), NOT textureSample:
        // this call sits after the xRaw == 0.0 early-return, i.e.
        // in non-uniform control flow, where WGSL forbids the
        // derivative-computing textureSample (must only be called
        // from uniform control flow). A 1024x1 LUT has no mip chain,
        // so explicit-LOD sampling is identical and derivative-free.
        y = textureSampleLevel(stretchLut, stretchLutSampler, vec2<f32>(x, 0.5), 0.0).r;
    } else {
        // PixInsight Auto-STF: linear remap into [E1, 1] then MTF
        // (closed-form, no LUT — preserves the precision benefits
        // identified in the C4-era "kill the LUT" rewrite).
        let E2 = max(0.0, (x - globals.e1) / max(1.0 - globals.e1, 1e-12));
        let num = (globals.e5 - 1.0) * E2;
        let den = (2.0 * globals.e5 - 1.0) * E2 - globals.e5;
        if (abs(den) < 1e-12) {
            y = 0.0;
        } else {
            y = num / den;
        }
        y = clamp(y, 0.0, 1.0);
    }

    // HDR scaling. With the canvas configured for extended tone
    // mapping (rgba16float + toneMapping:{mode:'extended'}), values
    // >1.0 drive extended display luminance directly.
    return vec4<f32>(vec3<f32>(y) * globals.peakBrightness, 1.0);
}
`;
