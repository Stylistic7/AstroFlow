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


const GENERIC_FS_MAIN = /* wgsl */ `
@group(0) @binding(5) var<uniform> fparams: DisplayParams;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex = textureSample(tileTexture, tileSampler, in.uv);
    let S = tex.r;
    let H = tex.g;
    let O = tex.b;
    // No-data footprint sentinel (all-zero) → opaque black.
    if (S == 0.0 && H == 0.0 && O == 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // RAW scene-linear S/H/O straight into the finish — NO global normalise.
    // The finish owns ALL of its scaling (its black/white/mid, a normalise op,
    // or nothing): normalisation is part of the individual pipeline, not a
    // hidden global. This is why "Sky floor" (a half-STF global anchor) is gone
    // from the generic path. See project_scene_display_seam.
    let rgb = clamp(displayFinish(vec3<f32>(S, H, O), fparams),
                    vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(rgb * globals.peakBrightness, 1.0);
}
`;


/** Full WGSL for a finish, from its served chunk. `finishWgsl` = the backend's
 * `DisplayFinish.to_wgsl()` (DisplayParams struct + helpers + displayFinish). */
export function buildFinishShader(finishWgsl) {
    return SHADER_PREAMBLE + '\n' + finishWgsl + '\n' + GENERIC_FS_MAIN;
}
