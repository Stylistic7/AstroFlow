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
 */
export const TONE_MAPPING_WGSL = /* wgsl */ `
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
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let xRaw = textureSample(tileTexture, tileSampler, in.uv).r;

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
        y = textureSample(stretchLut, stretchLutSampler, vec2<f32>(x, 0.5)).r;
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
