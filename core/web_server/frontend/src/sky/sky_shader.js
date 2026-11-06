/**
 * WGSL for the sky viewer — the SAME finish chunk the tile viewer splices
 * (DisplayParams + displayFinish, see generic_finish_shader.js) under a
 * full-screen pass that finds, for every pixel, the HEALPix cell under it
 * and the tile holding that cell.
 *
 * Per fragment: device pixel → Hammer–Aitoff plane (centred on the view
 * centre, east to the left) → a unit vector in the native frame → the map's
 * frame (one rotation, north kept up, the display frame folded in) →
 * HEALPix (face, x, y) in [0, 1)² → the tile at the order whose pixels are
 * at least as fine as this screen pixel, or the finest coarser one that has
 * loaded → a texel of that tile's atlas layer. The mip level inside the
 * tile comes from how fast the DIRECTION changes across a pixel (the
 * derivatives of the unit vector), which is continuous everywhere; the
 * raster coordinates jump at face edges and would pick a wrong level along
 * every seam.
 *
 * Twins that must stay line-for-line equivalent: the Hammer inverse and
 * the rotation ↔ sky_projection.js (the pointer side); healpixXyf, morton
 * and the index offset ↔ sky_tiles.js and allsky/healpix_faces.py (pinned
 * against astropy-healpix there).
 *
 * Binding layout:
 *   group 0: @0 globals { peakBrightness }, @1 skyView, @2 sampler, @3 fparams
 *   group 1: @0..3 src0..src3 — texture_2d_array, src0 = the tile atlas;
 *            @4 tileIndex — storage u32 per (order, npix): atlas layer + 1, 0 = not loaded
 */
import { MAX_INPUTS, finishTail } from '../tile_viewer/generic_finish_shader.js';

const PREAMBLE = /* wgsl */ `
struct Globals {
    peakBrightness: f32,
};
struct SkyView {
    rot: mat4x4<f32>,     // native → map frame (a 3×3 in the upper left)
    size: vec4<f32>,      // screenW, screenH (device px), device px per plane unit, cell side at nside 1 (rad)
    lod: vec4<f32>,       // finest tile order, log2(tile width), gauge (file units → pinned), 0
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> skyView: SkyView;
@group(0) @binding(2) var faceSampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    // The whole canvas as one triangle strip; everything happens per fragment.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(corners[idx], 0.0, 1.0);
    return out;
}

const PI: f32 = 3.141592653589793;

// (face, x, y) of a unit vector in the map frame: HEALPix ang2pix_nest in its
// real-valued form. Twin of allsky/healpix_faces.py::xyf_from_vec.
fn healpixXyf(v: vec3<f32>) -> vec3<f32> {
    let z = v.z;
    let za = abs(z);
    var phi = atan2(v.y, v.x);
    if (phi < 0.0) { phi = phi + 2.0 * PI; }
    let tt = phi * (2.0 / PI);                 // [0, 4): the azimuth in quadrants
    if (za <= 2.0 / 3.0) {
        // Equatorial belt: which pair of diagonal lines the point lies between.
        let t1 = 0.5 + tt;
        let t2 = 0.75 * z;
        let jp = t1 - t2;
        let jm = t1 + t2;
        let ifp = floor(jp);
        let ifm = floor(jm);
        var face = ifm + 8.0;
        if (ifp == ifm) {
            face = select(ifp + 4.0, 4.0, ifp == 4.0);
        } else if (ifp < ifm) {
            face = ifp;
        }
        return vec3<f32>(face, jm - ifm, 1.0 - (jp - ifp));
    }
    // Polar caps: one face per quadrant, the position from the equal-area radius.
    let ntt = min(3.0, floor(tt));
    let tp = tt - ntt;
    let tmp = sqrt(3.0 * (1.0 - za));
    let jpp = tp * tmp;
    let jmp = (1.0 - tp) * tmp;
    if (z >= 0.0) {
        return vec3<f32>(ntt, 1.0 - jmp, 1.0 - jpp);
    }
    return vec3<f32>(ntt + 8.0, jpp, jmp);
}

// Morton index of (x, y) within a face: x on the even bits, y on the odd —
// HEALPix NESTED order. Twin of sky_tiles.js::morton.
fn spreadBits(x: u32) -> u32 {
    var v = x & 0xFFFFu;
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333u;
    v = (v | (v << 1u)) & 0x55555555u;
    return v;
}
fn morton(x: u32, y: u32) -> u32 {
    return spreadBits(x) | (spreadBits(y) << 1u);
}
`;


function skyFsMain(monoMask) {
    const n = Math.max(1, monoMask.length);
    const decls = [];
    const samples = [];
    for (let i = 0; i < n; i++) {
        decls.push(`@group(1) @binding(${i}) var srcTexture${i}: texture_2d_array<f32>;`);
        samples.push(i === 0
            ? '    let t0 = textureSampleLevel(srcTexture0, faceSampler, uv, i32(slot - 1u), level) * skyView.lod.z;'
            : `    let t${i} = textureSampleLevel(srcTexture${i}, faceSampler, uv, 0, 0.0);`);
    }
    return `
@group(0) @binding(3) var<uniform> fparams: DisplayParams;
@group(1) @binding(${MAX_INPUTS}) var<storage, read> tileIndex: array<u32>;
${decls.join('\n')}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let W = skyView.size.x;
    let H = skyView.size.y;
    let ppu = skyView.size.z;
    // Device pixel → plane: centred, y up, east (+x) to the left.
    let px = -(in.pos.x - 0.5 * W) / ppu;
    let py = (0.5 * H - in.pos.y) / ppu;
    // Hammer–Aitoff inverse. z2 is its auxiliary; the ellipse rim is z2 = ½.
    let z2 = 1.0 - px * px / 16.0 - py * py / 4.0;
    let z = sqrt(max(z2, 0.0));
    let lam = 2.0 * atan2(z * px, 2.0 * (2.0 * z2 - 1.0));
    let bet = asin(clamp(z * py, -1.0, 1.0));
    let native = vec3<f32>(cos(bet) * cos(lam), cos(bet) * sin(lam), sin(bet));
    let v = (skyView.rot * vec4<f32>(native, 0.0)).xyz;
    // The HEALPix order whose cells are one device pixel wide (continuous),
    // from this pixel's angular size — before any branch (derivatives need
    // uniform control flow).
    let pixAngle = max(length(dpdx(v)), length(dpdy(v)));
    let pStar = log2(skyView.size.w / max(pixAngle, 1e-12));
    if (z2 < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let h = healpixXyf(normalize(v));
    let face = u32(h.x);
    let maxOrder = i32(skyView.lod.x);
    let tileShift = skyView.lod.y;
    // The tile whose pixels are at least as fine as this one, or the finest
    // coarser tile that has loaded.
    var k = clamp(i32(ceil(pStar - tileShift)), 0, maxOrder);
    var n: u32 = 1u;
    var tx: u32 = 0u;
    var ty: u32 = 0u;
    var slot: u32 = 0u;
    loop {
        n = 1u << u32(k);
        tx = min(u32(h.y * f32(n)), n - 1u);
        ty = min(u32(h.z * f32(n)), n - 1u);
        slot = tileIndex[4u * ((1u << (2u * u32(k))) - 1u) + face * n * n + morton(tx, ty)];
        if (slot != 0u || k == 0) { break; }
        k = k - 1;
    }
    if (slot == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let uv = vec2<f32>(h.y * f32(n) - f32(tx), h.z * f32(n) - f32(ty));
    let level = max(0.0, f32(k) + tileShift - pStar);
${samples.join('\n')}
${finishTail(monoMask)}
}
`;
}


/** Full WGSL for a finish over a HEALPix map. `finishWgsl` = the backend's
 * `DisplayFinish.to_wgsl()` chunk; `monoMask` = one boolean per declared
 * input (true ⇒ a 1-band map, which every map is). */
export function buildSkyShader(finishWgsl, monoMask) {
    return PREAMBLE + '\n' + finishWgsl + '\n' + skyFsMain(monoMask);
}
