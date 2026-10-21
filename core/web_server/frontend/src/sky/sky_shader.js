/**
 * WGSL for the sky viewer — the SAME finish chunk the tile viewer splices
 * (DisplayParams + displayFinish, see generic_finish_shader.js) under a
 * full-screen pass that finds, for every pixel, the HEALPix cell under it.
 *
 * Per fragment: device pixel → Hammer–Aitoff plane (centred on the view
 * centre, east to the left) → a unit vector in the native frame → the map's
 * frame (one rotation, north kept up) → HEALPix (face, x, y) in [0, 1)² → a
 * texel of that face's raster. The twelve faces are the layers of one
 * texture array with mips down to one cell. The mip level comes from how
 * fast the DIRECTION changes across a pixel (the derivatives of the unit
 * vector), which is continuous everywhere; the raster coordinates jump at
 * face edges and would pick a wrong level along every seam.
 *
 * Twins that must stay line-for-line equivalent: the Hammer inverse and the
 * rotation ↔ sky_projection.js (the pointer side); healpixXyf ↔
 * allsky/healpix_faces.py (pinned against astropy-healpix there).
 *
 * Binding layout:
 *   group 0: @0 globals { peakBrightness }, @1 skyView, @2 sampler, @3 fparams
 *   group 1: @0..3 src0..src3 — texture_2d_array, one layer per face
 */
import { finishTail } from '../tile_viewer/generic_finish_shader.js';

const PREAMBLE = /* wgsl */ `
struct Globals {
    peakBrightness: f32,
};
struct SkyView {
    rot: mat4x4<f32>,     // native → map frame (a 3×3 in the upper left)
    size: vec4<f32>,      // screenW, screenH (device px), device px per plane unit, cell side (rad)
    lod: vec4<f32>,       // coarsest mip level, 0, 0, 0
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
`;


function skyFsMain(monoMask) {
    const n = Math.max(1, monoMask.length);
    const decls = [];
    const samples = [];
    for (let i = 0; i < n; i++) {
        decls.push(`@group(1) @binding(${i}) var srcTexture${i}: texture_2d_array<f32>;`);
        samples.push(`    let t${i} = textureSampleLevel(srcTexture${i}, faceSampler, h.yz, i32(h.x), lod);`);
    }
    return `
@group(0) @binding(3) var<uniform> fparams: DisplayParams;
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
    // Mip level from this pixel's angular size, before any branch
    // (derivatives need uniform control flow).
    let pixAngle = max(length(dpdx(v)), length(dpdy(v)));
    let lod = clamp(log2(max(pixAngle, 1e-9) / skyView.size.w), 0.0, skyView.lod.x);
    if (z2 < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let h = healpixXyf(normalize(v));
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
