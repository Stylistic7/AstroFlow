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

    // PixInsight Auto-STF: linear remap into [E1, 1] then MTF.
    let E2 = max(0.0, (x - globals.e1) / max(1.0 - globals.e1, 1e-12));
    let num = (globals.e5 - 1.0) * E2;
    let den = (2.0 * globals.e5 - 1.0) * E2 - globals.e5;
    var y: f32;
    if (abs(den) < 1e-12) {
        y = 0.0;
    } else {
        y = num / den;
    }
    y = clamp(y, 0.0, 1.0);

    // HDR scaling. With the canvas configured for extended tone
    // mapping (rgba16float + toneMapping:{mode:'extended'}), values
    // >1.0 drive extended display luminance directly.
    return vec4<f32>(vec3<f32>(y) * globals.peakBrightness, 1.0);
}
`;
