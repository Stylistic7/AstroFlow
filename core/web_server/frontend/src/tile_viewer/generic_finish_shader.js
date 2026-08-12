/**
 * Build the WGSL for a data-driven display finish (a Python DisplayFinish).
 *
 * The seam: the backend serves a finish's WGSL chunk — `struct DisplayParams`
 * (one f32 field per Python `Param`, in declared order) + the ops' helpers +
 * `fn displayFinish(cIn, P)`. We splice it onto the SHARED SHADER_PREAMBLE
 * (structs/bindings/vs_main) so the Globals buffer layout matches the built-in
 * shader — both bind the same globals buffer — then append a generic fs_main:
 *
 *   sample the S/H/O tile → per-channel normalise to [0,1] using the SAME
 *   anchors as the GHS finish (low = median − shadowClip·σ, so the sky survives
 *   as a small positive the finish can lift, never crushed) → displayFinish()
 *   with the params bound at binding(5) → × peakBrightness.
 *
 * binding(5) `fparams: DisplayParams` is declared AFTER the finish chunk (which
 * defines the struct). useTileRenderer packs the finish's param values into that
 * uniform in the same order Python declared them, so field i ↔ param i.
 *
 * This is the F2 (display-referred) half of the scene→display seam made live:
 * add a look = write a Python DisplayFinish; its shader arrives here as text.
 * See the project_scene_display_seam memory.
 */
import { SHADER_PREAMBLE } from './tone_mapping.js';


/** The generated fs_main: sample the primary tile plus `numInputs − 1` extra
 * source textures (group(1) bindings 3+; the tile layout always carries the
 * slots, dummy-black-bound when a source is absent) and hand ALL of them to
 * `displayFinish(src0.., P)` — the served chunk's signature has one vec4 per
 * declared input tap. A missing source samples the black dummy, so a finish
 * degrades gracefully (stars=0, Hα=0, L=0) instead of erroring. */
function genericFsMain(numInputs) {
    const n = Math.max(1, numInputs);
    const decls = [];
    const samples = ['    let s0 = textureSample(tileTexture, tileSampler, in.uv);'];
    const args = ['s0'];
    for (let i = 1; i < n; i++) {
        decls.push(`@group(1) @binding(${2 + i}) var srcTexture${i}: texture_2d<f32>;`);
        samples.push(`    let s${i} = textureSample(srcTexture${i}, tileSampler, in.uv);`);
        args.push(`s${i}`);
    }
    return `
@group(0) @binding(5) var<uniform> fparams: DisplayParams;
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
    // hidden global. See project_scene_display_seam.
    let rgb = clamp(displayFinish(${args.join(', ')}, fparams),
                    vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb * globals.peakBrightness, 1.0);
}
`;
}


/** Full WGSL for a finish, from its served chunk. `finishWgsl` = the backend's
 * `DisplayFinish.to_wgsl()` (DisplayParams struct + helpers + displayFinish);
 * `numInputs` = the finish's declared input-tap count (spec.inputs.length). */
export function buildFinishShader(finishWgsl, numInputs = 1) {
    return SHADER_PREAMBLE + '\n' + finishWgsl + '\n' + genericFsMain(numInputs);
}
