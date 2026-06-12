/**
 * Auto-stretch optimizer: build a 1-D RGBA LUT from per-image
 * statistics and a user-controlled painterly bias.
 *
 * Modeled after PixInsight's ScreenTransferFunction. Three steps:
 *
 *   1. Find the sky pedestal — the input value where the cumulative
 *      distribution crosses 0.5 (or just `sky_median` from stats).
 *   2. Set the black/white points around it. Black sits just below the
 *      sky pedestal (effectively a 1-bin offset, because for stacked
 *      + pedestaled mosaics there is no per-pixel noise spread below
 *      sky — every "sky" pixel is at the pedestal value within
 *      single-precision rounding). White comes from a tail percentile
 *      of the input distribution.
 *   3. Solve for the MTF midtone parameter `m` such that the sky
 *      pedestal lands at a chosen DISPLAY brightness — the
 *      `targetSkyBrightness`. The painterly slider tunes this target:
 *
 *          painterlyBias = 0.0  →  sky at display 0.10 (dim, dramatic)
 *          painterlyBias = 1.0  →  sky at display 0.40 (lifted midtones)
 *
 *      Faint structure rides on top of the sky pedestal: as the slider
 *      lifts sky, the midtones lift with it, which is what makes the
 *      "painterly" half of the axis look painterly.
 *
 * Zero pixels are excluded from every calculation. They're the
 * "no data" sentinel for the irregular mosaic footprint inside the
 * rectangular bbox; the shader short-circuits exact zeros to opaque
 * black so they never reach the LUT.
 *
 * Implementation: 256-entry RGBA float LUT, recomputed on slider
 * change in ~0.3 ms. The shader looks it up with one texture sample
 * per pixel.
 */


const LUT_SIZE = 256;

// White point at the 99.99 % percentile of valid pixels — keeps the
// brightest 0.01 % of stars saturated, otherwise the LUT spreads its
// dynamic range across a long star tail and the canvas looks washed
// out in the midtones.
const WHITE_CDF = 0.9999;

// MTF curve: y = ((m − 1) t) / ((2m − 1) t − m). Solving mtf(t, m) = y
// for m gives the form below — the rest of the design is just
// picking t (where the sky pedestal sits in the bracket) and y
// (the display brightness we want the sky to show at).
const TARGET_SKY_NATURAL = 0.10;
const TARGET_SKY_PAINTERLY = 0.40;


/**
 * Build the auto-stretch LUT.
 *
 * @param {object} stats - the `statistics` field from /api/metadata.
 *     Uses `histogram_counts`, `histogram_range`, and (when present)
 *     `sky_median`.
 * @param {number} painterlyBias - 0.0 = naturalistic, dim sky;
 *     1.0 = painterly, lifted midtones.
 * @returns {Float32Array} - RGBA float data of length LUT_SIZE * 4.
 */
export function buildAutoStretchLut(stats, painterlyBias) {
    const counts = stats.histogram_counts;
    const [rangeMin, rangeMax] = stats.histogram_range;
    const nBins = counts.length;
    const span = rangeMax - rangeMin;
    if (span <= 0) {
        return identityLut();
    }

    // Defensive zero-bin drop for legacy sidecar stats. With the
    // current backend `compute_image_statistics` already filters
    // zeros, so bin 0 here is small and dropping it is a no-op; the
    // line stays as protection against pre-fix stats still on disk.
    const validCounts = counts.slice();
    validCounts[0] = 0;
    const totalPixels = validCounts.reduce((a, c) => a + c, 0);
    if (totalPixels === 0) {
        return identityLut();
    }

    const inputCDF = new Float32Array(nBins);
    let cumulative = 0;
    for (let i = 0; i < nBins; i++) {
        cumulative += validCounts[i];
        inputCDF[i] = cumulative / totalPixels;
    }

    const binForCDF = (target) => {
        for (let i = 0; i < nBins; i++) {
            if (inputCDF[i] >= target) return i;
        }
        return nBins - 1;
    };

    // Sky pedestal — prefer the sidecar's `sky_median` (computed off
    // the lower-25 %-cut on the full image data) when it's available
    // and lies inside the histogram range. Falls back to the
    // histogram median otherwise.
    const skyMedianBin = (
        typeof stats.sky_median === "number" &&
        stats.sky_median >= rangeMin &&
        stats.sky_median <= rangeMax
    )
        ? Math.floor((stats.sky_median - rangeMin) / span * nBins)
        : binForCDF(0.5);

    // Black point: one bin below sky_median in normalized space. For
    // pedestaled mosaics there is no histogram mass below sky — every
    // pixel is sky-or-brighter — so the black point is really a small
    // numerical offset that lets sky map to a non-zero `t` in the
    // bracket, which the MTF can then deliver as the target display
    // brightness.
    const skyNorm = (skyMedianBin + 0.5) / nBins;
    const blackNorm = Math.max(0, skyNorm - 1.0 / nBins);

    const whiteBin = Math.max(skyMedianBin + 2, binForCDF(WHITE_CDF));
    const whiteNorm = (whiteBin + 0.5) / nBins;

    if (whiteNorm - blackNorm < 1e-6) {
        return identityLut();
    }

    // t-position of the sky pedestal within the bracket.
    const skyT = Math.max(
        1e-6,
        Math.min(
            1 - 1e-6,
            (skyNorm - blackNorm) / (whiteNorm - blackNorm),
        ),
    );

    // Target display brightness for sky, lerped along the slider.
    const target = TARGET_SKY_NATURAL +
        (TARGET_SKY_PAINTERLY - TARGET_SKY_NATURAL) * painterlyBias;

    // Solve y = mtf(t, m) for m. Closed form:
    //   m = t·(1 − y) / (y + t − 2·y·t)
    // Clamp to a safe range — m = 0 and m = 1 are the degenerate
    // y = 0 / y = 1 limits where the curve flattens.
    const m = Math.max(
        0.05,
        Math.min(0.95, skyT * (1 - target) / (target + skyT - 2 * target * skyT)),
    );
    const mtf = (t) => ((m - 1) * t) / ((2 * m - 1) * t - m);

    const lut = new Float32Array(LUT_SIZE * 4);
    const bracketSpan = whiteNorm - blackNorm;
    for (let lutI = 0; lutI < LUT_SIZE; lutI++) {
        const xNorm = lutI / (LUT_SIZE - 1);
        let t = (xNorm - blackNorm) / bracketSpan;
        t = Math.max(0, Math.min(1, t));
        const y = mtf(t);
        lut[lutI * 4 + 0] = y;
        lut[lutI * 4 + 1] = y;
        lut[lutI * 4 + 2] = y;
        lut[lutI * 4 + 3] = 1.0;
    }
    return lut;
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
