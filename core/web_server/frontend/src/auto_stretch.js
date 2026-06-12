/**
 * Auto-stretch parameter derivation — JS port of
 * `previews.operations.autostretch`.
 *
 * The shader applies the MTF curve directly (no LUT) using the two
 * scalars we compute here:
 *
 *   E1 = black point   = median − 2.8 · 1.4826 · MAD, clamped to [0,1]
 *   E5 = midtone `m`   = solved so mtf(median − E1, E5) = 0.20
 *
 * Hard-coded parameters (same as the Python original):
 *   stretch = 0.20   target background display brightness
 *   clip    = 2.8    shadow clip in σ-equivalent units
 *   curve   = 1.0    linear MTF (no arcsinh)
 *
 * The autostretch is intentionally not parameterized — the function is
 * "the automatic STF that PixInsight uses." Any slider would be
 * editorial; the tone-mapping operator chooses its parameters from
 * `(median, MAD)` and that's it.
 */


const STRETCH = 0.20;
const CLIP_SIGMA = 2.8;
const MAD_TO_SIGMA = 1.4826;


/**
 * Compute the two MTF scalars (`E1`, `E5`) the shader needs from
 * `ImageStatistics`.
 *
 * Returns the values normalized into [0, 1] over the histogram's
 * `max_value` — the shader's per-pixel `xRaw / max` is in the same
 * space.
 *
 * @param {object} stats - `statistics` field from `/api/metadata`.
 *     Uses `max_value`, `p50` (median), `mad`, and `sky_rms`.
 * @returns {{E1: number, E5: number, identity: boolean}}
 *     `identity` is true on degenerate inputs; the shader should pass
 *     the raw pixel through untouched (no stretch).
 */
export function computeAutoStretchParams(stats) {
    const max = stats.max_value;
    if (!(max > 0)) {
        return { E1: 0, E5: 0.5, identity: true };
    }
    const medianNorm = stats.p50 / max;
    // Pre-mad sidecars (raw subframes from the backfill) ship mad=0;
    // fall back to sky_rms/1.4826 — `σ = 1.4826 · MAD` for a Gaussian.
    // For stacked + pedestaled masters the sidecar regeneration step
    // populates a real per-pixel MAD, so this fallback only fires on
    // raw frames (where sky_rms IS the right magnitude).
    const madRaw = stats.mad > 0 ? stats.mad : (stats.sky_rms ?? 0) / MAD_TO_SIGMA;
    const madNorm = madRaw / max;

    const E1 = Math.max(0, Math.min(1, medianNorm - CLIP_SIGMA * MAD_TO_SIGMA * madNorm));
    const E3 = medianNorm - E1;

    // E5: solve mtf(E3, E5) = STRETCH for E5. The closed form is
    //   m = E3·(1 − stretch) / (stretch + E3 − 2·stretch·E3)
    // identical to what the Python algebra inlines.
    const E5_num = (STRETCH - 1) * E3;
    const E5_den = (2 * STRETCH - 1) * E3 - STRETCH;
    if (Math.abs(E5_den) < 1e-12 || !isFinite(E3) || E3 <= 0) {
        return { E1, E5: 0.5, identity: true };
    }
    const E5 = E5_num / E5_den;
    if (!isFinite(E5) || E5 <= 0 || E5 >= 1) {
        return { E1, E5: 0.5, identity: true };
    }
    return { E1, E5, identity: false };
}
