/**
 * Build the WGSL for a display finish (a Python DisplayFinish) — the ONE
 * rendering path of the tile viewer.
 *
 * The seam: the backend serves a finish's WGSL chunk — `struct DisplayParams`
 * (one f32 field per Python `Param`, in declared order) + the ops' helpers +
 * `fn displayFinish(src0[, src1..src3], P)`, one vec4 per declared input. We
 * splice it between the shared preamble (structs / bindings / vs_main) and a
 * generated fs_main that samples every input and hands them all to
 * displayFinish, whose colour is the pixel: the finish's own last pass
 * clamps it to [0, globals.displayWhite] — 1.0 on SDR, the viewer's HDR
 * display white above it (the finish's `white()`; see finishes/stdlib.py).
 *
 * Binding layout:
 *   group 0 (shared by every tile):
 *     @binding(0) globals   — uniform Globals { displayWhite }
 *     @binding(1) viewport  — uniform Viewport { tx, ty, scale, screenW, screenH }
 *     @binding(2) sampler   — linear, clamp-to-edge
 *     @binding(3) fparams   — uniform DisplayParams (declared AFTER the chunk,
 *                             which defines the struct; useTileRenderer packs
 *                             the values in Python's declared order, so field
 *                             i ↔ param i)
 *   group 1 (per tile):
 *     @binding(0) tileRect  — uniform TileRect { worldX, worldY, tileW, tileH }
 *     @binding(1..4)        — src0..src3, one texture per finish input
 *                             (a shared 1×1 black dummy where unbound)
 *
 * The vertex shader emits a unit quad (4 verts, triangle-strip) whose
 * world-space corners come from `tileRect`; the viewport transform places it
 * into clip space. The same shader handles every tile; tiles differ only by
 * their group-1 bind group.
 *
 * A 1-band input is BROADCAST to RGB before the call (`vec4(t.rrr, 1.0)`), so
 * a mono tap read as `src.g` equals `src.r` — the finish never needs to know
 * how many bands its input file had. See project_scene_display_seam.
 */
export const MAX_INPUTS = 4;

const SHADER_PREAMBLE = /* wgsl */ `
struct Globals {
    displayWhite: f32,
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
@group(1) @binding(0) var<uniform> tileRect: TileRect;

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
`;


/** The generated fs_main: one texture + one sample per declared input,
 * mono inputs broadcast to RGB, all handed to `displayFinish(src0.., P)`.
 * `monoMask[i]` is true when input i is a bound 1-band tap. */
function genericFsMain(monoMask) {
    const n = Math.max(1, monoMask.length);
    const decls = [];
    const samples = [];
    const args = [];
    for (let i = 0; i < n; i++) {
        decls.push(`@group(1) @binding(${1 + i}) var srcTexture${i}: texture_2d<f32>;`);
        samples.push(`    let t${i} = textureSample(srcTexture${i}, tileSampler, in.uv);`);
        samples.push(monoMask[i]
            ? `    let s${i} = vec4<f32>(t${i}.rrr, 1.0);`
            : `    let s${i} = t${i};`);
        args.push(`s${i}`);
    }
    return `
@group(0) @binding(3) var<uniform> fparams: DisplayParams;
${decls.join('\n')}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
${samples.join('\n')}
    // No-data footprint sentinel (all-zero primary) → opaque black.
    if (s0.r == 0.0 && s0.g == 0.0 && s0.b == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // RAW scene-linear sources straight into the finish — NO global normalise.
    // The finish owns ALL of its scaling (its black/white/mid, a normalise op,
    // or nothing): normalisation is part of the individual pipeline, not a
    // hidden global. See project_scene_display_seam. It owns its ceiling
    // too: its last pass clamps to [0, globals.displayWhite].
    return vec4<f32>(displayFinish(${args.join(', ')}, fparams), 1.0);
}
`;
}


/** Full WGSL for a finish, from its served chunk. `finishWgsl` = the backend's
 * `DisplayFinish.to_wgsl()` (DisplayParams struct + helpers + displayFinish);
 * `monoMask` = one boolean per declared input (true ⇒ bound 1-band tap). */
export function buildFinishShader(finishWgsl, monoMask) {
    return SHADER_PREAMBLE + '\n' + finishWgsl + '\n' + genericFsMain(monoMask);
}
