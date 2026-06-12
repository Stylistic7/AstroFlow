/**
 * Auto-stretch optimizer — JS port of `previews.operations.autostretch`.
 *
 * The Python original (`core/previews/operations.py:21`) implements
 * PixInsight's ScreenTransferFunction: shadow-clip the histogram at
 * `-2.8 · 1.4826 · MAD` below the median, then apply an MTF curve with
 * the midtone parameter chosen so the median lands at display value
 * `stretch` (the "target background brightness"). The port preserves
 * the exact algebra so the in-browser canvas reads the same as a
 * server-rendered thumbnail.
 *
 * Variables follow the original:
 *   E1 = black point (shadow clip)
 *   E2 = per-pixel value after black-shift + rescale
 *   E3 = median - E1                  (scalar)
 *   E4 = min(1, 1 / curve) = 1        (we hard-wire curve = 1 like the original)
 *   E5 = MTF midtone parameter solved so mtf(E3, E5) = stretch
 *
 * Input normalization: the original divides by `max(image)` before any
 * of this. We do the same here. The shader is configured with
 * `uHistRangeMin = 0` so its built-in normalize step computes
 * `x = xRaw / uHistRangeMax`, exactly matching `image / max`.
 *
 * The painterly slider varies `stretch` (the target background
 * brightness). The hard-coded value in the Python is `0.20`; we keep
 * that as the slider's default and span [0.10, 0.40] across the
 * naturalistic↔painterly range. Lower stretch → dimmer sky, higher
 * contrast; higher stretch → brighter sky, more lifted shadows.
 *
 * MAD travels in `stats.mad`. Computing it from the histogram alone
 * doesn't work for stacked-pedestaled masters — the actual sky-noise
 * spread is finer than the histogram bin width — so the value is
 * computed at pixel precision server-side and shipped.
 */


const LUT_SIZE = 256;

// Hard-coded from the original autostretch. `clip` is the shadow-cut
// in σ-equivalent units; `curve = 1` means a pure MTF (no arcsinh
// flavor). The painterly slider takes over the `stretch` parameter.
const CLIP_SIGMA = 2.8;
const CURVE = 1.0;
const MAD_TO_SIGMA = 1.4826;

// Slider endpoints for `stretch` (target background display brightness).
// 0.20 is the original autostretch default — slider centered there
// matches what the server-rendered thumbnails look like today.
const STRETCH_NATURALISTIC = 0.25;
const STRETCH_PAINTERLY = 0.10;


/**
 * Build the auto-stretch LUT.
 *
 * @param {object} stats - `statistics` field from /api/metadata.
 *     Uses `max_value`, `p50` (median), and `mad`.
 * @param {number} painterlyBias - 0.0 = naturalistic (stretch=0.25);
 *     1.0 = painterly (stretch=0.10).
 * @returns {Float32Array} - RGBA float data of length LUT_SIZE * 4.
 */
export function buildAutoStretchLut(stats, painterlyBias) {
    const max = stats.max_value;
    if (!(max > 0)) {
        return identityLut();
    }
    // The autostretch normalizes by max before anything else, so we
    // express median and MAD in [0, 1] units the same way.
    const medianNorm = stats.p50 / max;
    // Pre-mad sidecars (raw subframes from the early backfill, mostly)
    // ship `mad = 0`. For those, derive an estimate from `sky_rms` —
    // for a Gaussian sky noise, `σ = 1.4826 · MAD`, so `sky_rms / 1.4826`
    // is the right magnitude. Stacked + pedestaled masters have
    // `sky_rms ≈ 0` too and the right fix there is regenerating the
    // sidecar with the real per-pixel MAD; that one's already done.
    const madRaw = stats.mad > 0 ? stats.mad : (stats.sky_rms ?? 0) / 1.4826;
    const madNorm = madRaw / max;

    // E1: shadow-clip black point, clamped to [0, 1].
    const E1 = Math.max(0, Math.min(1, medianNorm - CLIP_SIGMA * MAD_TO_SIGMA * madNorm));
    const E3 = medianNorm - E1;
    const E4 = Math.min(1, 1 / CURVE);  // = 1 with curve = 1

    // stretch interpolates with the painterly slider.
    const stretch = STRETCH_NATURALISTIC +
        (STRETCH_PAINTERLY - STRETCH_NATURALISTIC) * painterlyBias;

    // E5: midtone parameter `m` so that `mtf(E3, m) = stretch`. This is
    // the algebraic solve embedded in the original Python expression.
    const E5_num = (stretch - 1) * E3;
    const E5_den = (2 * stretch - 1) * Math.pow(E3, E4) - stretch;
    const E5 = safeDiv(E5_num, E5_den);

    // Defensive: degenerate inputs (constant image, all pixels at max)
    // can produce E5 = 0 or NaN. Fall back to identity.
    if (!isFinite(E5) || E5 === 0) {
        return identityLut();
    }

    const lut = new Float32Array(LUT_SIZE * 4);
    for (let i = 0; i < LUT_SIZE; i++) {
        // The shader feeds xNorm = xRaw / max into the LUT (we set
        // uHistRangeMin = 0 so its (xRaw - min) / (max - min) reduces
        // to xRaw / max). LUT[i] is sampled at xNorm = i / 255.
        const xNorm = i / (LUT_SIZE - 1);
        const E2 = Math.max(0, (xNorm - E1) / (1 - E1));
        const num = (E5 - 1) * E2;
        const den = (2 * E5 - 1) * Math.pow(E2, E4) - E5;
        const y = safeDiv(num, den);
        const clamped = Math.max(0, Math.min(1, isFinite(y) ? y : 0));
        lut[i * 4 + 0] = clamped;
        lut[i * 4 + 1] = clamped;
        lut[i * 4 + 2] = clamped;
        lut[i * 4 + 3] = 1.0;
    }
    return lut;
}


function safeDiv(num, den) {
    return Math.abs(den) < 1e-12 ? 0 : num / den;
}


function identityLut() {
    const data = new Float32Array(LUT_SIZE * 4);
    for (let i = 0; i < LUT_SIZE; i++) {
        const v = i / (LUT_SIZE - 1);
        data[i * 4 + 0] = v;
        data[i * 4 + 1] = v;
        data[i * 4 + 2] = v;
        data[i * 4 + 3] = 1.0;
    }
    return data;
}
