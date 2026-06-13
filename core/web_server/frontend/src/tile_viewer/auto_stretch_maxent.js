/**
 * Constrained max-entropy stretch — LUT generator.
 *
 * Builds a 1-D lookup table that maps normalized input intensity in
 * [0, 1] to a display output value that:
 *
 *   1. Maps the sky background EXACTLY to a chosen dark target (the
 *      `SKY_TARGET` constant below — e.g. 0.05). Below the sky value
 *      the mapping is a plain linear ramp 0 → SKY_TARGET, so dark
 *      calibration artifacts don't drift into the visible range.
 *
 *   2. Above the sky background, the OUTPUT distribution is
 *      exponential with decay τ — equivalently, the log-histogram
 *      plotted against output brightness is a straight line. This
 *      is the max-entropy distribution on (SKY_TARGET, 1] given a
 *      fixed mean, so the displayed image carries the most
 *      information per displayable level under that constraint.
 *
 *      Achieved via histogram specification: for each input rank
 *      F(x) in the upper-tail input CDF, the output is G⁻¹(F(x))
 *      where G is the target exponential CDF on (SKY_TARGET, 1].
 *
 * The previous (pure max-entropy) attempt had the bulk of pixels at
 * sky background landing in the mid-tones (~0.55) because 80% of
 * the input rank maps to G⁻¹(0.80) under the exponential target.
 * Pinning the sky to a dark anchor fixes that without sacrificing
 * the linear-log-histogram property over the signal-bearing range.
 *
 * Output:
 *   Float32Array of length LUT_SIZE, indexed by `xNorm = i/(LUT-1)`
 *   where xNorm is the raw float pixel value divided by max_value
 *   (the same normalization the shader's xRaw / globals.maxVal
 *   produces). Values are in [0, 1].
 */

export const LUT_SIZE = 1024;

// Anchor for the sky background. Empirically dim enough that sky
// reads as "dark night" on an HDR-bright display but bright enough
// that there's still visible structure in the local noise.
const SKY_TARGET = 0.05;

// Exponential decay constant on the output. Smaller τ = more
// midtone-heavy (faster falloff from sky); larger τ = more highlight
// headroom. τ ≈ 0.3 puts the brightest 5% of input pixels above
// output 0.7, which gives the slider-driven HDR a useful pop.
const TAU = 0.30;


/**
 * @param {object} stats - `statistics` field from `/api/metadata`.
 *     Needs `max_value`, `sky_median` (preferred) or `p50`,
 *     `histogram_counts`, `histogram_range`.
 * @returns {Float32Array} length LUT_SIZE, in [0, 1].
 */
export function computeMaxEntropyLut(stats) {
    const lut = new Float32Array(LUT_SIZE);
    const maxVal = stats?.max_value;
    if (!(maxVal > 0) || !stats?.histogram_counts) {
        return identityLut();
    }

    const counts = stats.histogram_counts;
    const [rangeMin, rangeMax] = stats.histogram_range;
    const nBins = counts.length;
    let totalPixels = 0;
    for (let i = 0; i < nBins; i++) totalPixels += counts[i];
    if (totalPixels === 0) return identityLut();

    // Input CDF over the histogram bins. F[i] = fraction of pixels
    // with value ≤ the upper edge of bin i.
    const F = new Float32Array(nBins);
    let cum = 0;
    for (let i = 0; i < nBins; i++) {
        cum += counts[i];
        F[i] = cum / totalPixels;
    }

    // Sky value: prefer the lower-25%-cut median (background-only)
    // when the sidecar reports it; otherwise the global median.
    const skyValue = (stats.sky_median != null && stats.sky_median > 0)
        ? stats.sky_median
        : stats.p50;
    const skyNorm = Math.max(0, Math.min(1, skyValue / maxVal));

    // F(sky) — the rank percentile of the sky background. Below this
    // rank the input pixels are calibration noise / bottom-of-the-
    // pedestal; above are the signal we want to display.
    const binWidth = (rangeMax - rangeMin) / nBins;
    const skyBin = Math.min(
        nBins - 1, Math.max(0, Math.floor((skyValue - rangeMin) / binWidth)),
    );
    const F_sky = F[skyBin];
    const tailFraction = 1 - F_sky;

    // Precompute the inverse-CDF constant `k = 1 − exp(−L/τ)` where
    // L = 1 − SKY_TARGET is the output-side tail length. The inverse
    // is `y = SKY_TARGET − τ · ln(1 − p · k)` for p ∈ [0, 1].
    const L = 1 - SKY_TARGET;
    const k = 1 - Math.exp(-L / TAU);

    for (let lutI = 0; lutI < LUT_SIZE; lutI++) {
        const xNorm = lutI / (LUT_SIZE - 1);

        if (xNorm <= skyNorm) {
            // Linear ramp 0 → SKY_TARGET on the dark side. Anything
            // dimmer than sky is a calibration artifact or border;
            // display it dimly without injecting structure.
            lut[lutI] = skyNorm > 0 ? SKY_TARGET * (xNorm / skyNorm) : 0;
            continue;
        }

        // Above sky: look up F(x) and map through G⁻¹ on the tail.
        const xVal = xNorm * maxVal;
        const histBin = Math.min(
            nBins - 1, Math.max(0, Math.floor((xVal - rangeMin) / binWidth)),
        );
        const Fx = F[histBin];

        // p ∈ [0, 1] — rank within the signal-bearing tail.
        const p = tailFraction > 1e-12
            ? Math.max(0, Math.min(1, (Fx - F_sky) / tailFraction))
            : 0;

        // Inverse exponential CDF on (SKY_TARGET, 1].
        // y = SKY_TARGET − τ · ln(1 − p · k)
        const inner = Math.max(1e-12, 1 - p * k);
        const y = SKY_TARGET - TAU * Math.log(inner);
        lut[lutI] = Math.max(SKY_TARGET, Math.min(1, y));
    }
    return lut;
}


function identityLut() {
    const lut = new Float32Array(LUT_SIZE);
    for (let i = 0; i < LUT_SIZE; i++) lut[i] = i / (LUT_SIZE - 1);
    return lut;
}
