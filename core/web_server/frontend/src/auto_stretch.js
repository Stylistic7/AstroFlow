/**
 * Auto-stretch optimizer: build a 1-D RGBA LUT from per-image
 * statistics and a user-controlled painterly bias.
 *
 * Conceptual basis (see the long-form design discussion):
 *   * For brightness distributed perceptually evenly, the
 *     log-histogram should be linear in displayed brightness — i.e.
 *     the displayed histogram should be exponential, which is also
 *     the max-entropy distribution on [0, 1].
 *   * The cleanest single-pass mapping that takes ANY input
 *     distribution to a chosen target distribution is histogram
 *     specification: f(x) = G⁻¹(F(x)) where F is the input CDF and G
 *     is the target CDF.
 *   * The painterly slider biases G away from the pure max-entropy
 *     exponential toward a midtone-heavier curve. Pure naturalistic
 *     = exponential target; pure painterly = exponential weighted by
 *     a wide Gaussian centered at the midtones.
 *
 * Implementation: 256-entry RGBA float LUT, recomputed on slider
 * change in ~0.5 ms. The shader looks it up with one texture sample
 * per pixel.
 */


const LUT_SIZE = 256;


/**
 * Build the auto-stretch LUT from the given ImageStatistics and a
 * painterly slider value in [0, 1].
 *
 * @param {object} stats - the `statistics` field from
 *     /api/metadata. Must include `histogram_counts`,
 *     `histogram_range`, and at least `p99`, `p1` percentiles.
 * @param {number} painterlyBias - 0.0 = pure max-entropy (linear-
 *     log histogram); 1.0 = heavy midtone emphasis (painterly).
 * @returns {Float32Array} - RGBA float data of length LUT_SIZE * 4.
 */
export function buildAutoStretchLut(stats, painterlyBias) {
    const counts = stats.histogram_counts;
    const [rangeMin, rangeMax] = stats.histogram_range;
    const nBins = counts.length;

    // Step 1: input CDF F(x), normalized to [0, 1].
    const totalPixels = counts.reduce((a, c) => a + c, 0);
    if (totalPixels === 0) {
        // Degenerate: empty image. Fall back to identity.
        return identityLut();
    }
    const inputCDF = new Float32Array(nBins);
    let cumulative = 0;
    for (let i = 0; i < nBins; i++) {
        cumulative += counts[i];
        inputCDF[i] = cumulative / totalPixels;
    }

    // Step 2: define the target CDF G(y) on [0, 1]. Parameter τ
    // controls the shape — the LARGER τ, the more uniform the target
    // (pure max-entropy = exponential with reasonable τ). Painterly
    // tweaks the curve so more mass piles up in the midtones.
    //
    // At painterlyBias = 0: target_pdf(y) ∝ exp(-y/τ)  with τ ≈ 0.4
    //     (the max-entropy distribution on [0, 1] given a fixed mean
    //     of ~0.3). Histogram specification onto this gives the most
    //     informative linear-log display.
    //
    // At painterlyBias = 1: target_pdf(y) ∝ exp(-y/τ) · gauss(y; 0.5, 0.2)
    //     (the same exponential decay biased toward midtones).
    //     Histogram specification onto this gives painterly look.
    const TAU = 0.4;
    const MID = 0.5;
    const MID_SIGMA = 0.2;
    // Build target PDF on a fine grid, then integrate to a CDF.
    const targetCDF = new Float32Array(nBins);
    let pdfIntegral = 0;
    const pdfValues = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) {
        const y = i / (nBins - 1);
        const exponential = Math.exp(-y / TAU);
        const dy = y - MID;
        const midtoneBoost = Math.exp(-(dy * dy) / (2 * MID_SIGMA * MID_SIGMA));
        // Lerp between pure exponential (painterlyBias=0) and
        // exponential × midtone gaussian (painterlyBias=1).
        const pdf = exponential * (1 - painterlyBias + painterlyBias * midtoneBoost);
        pdfValues[i] = pdf;
        pdfIntegral += pdf;
    }
    let cumulativeTarget = 0;
    for (let i = 0; i < nBins; i++) {
        cumulativeTarget += pdfValues[i] / pdfIntegral;
        targetCDF[i] = cumulativeTarget;
    }

    // Step 3: build the LUT. For each output index i (representing
    // a display brightness y in [0, 1]), find the input brightness
    // x such that mapping x→y satisfies F(x) = G(y).
    //
    // Equivalently: for each input bin index, look up its CDF
    // value F, then find the output bin index where targetCDF
    // matches F. Invert the target CDF lookup table.
    //
    // The LUT is indexed by INPUT brightness (so the shader can
    // sample texture2D(uLUT, vec2(x, 0.5)) for input x in [0,1])
    // and stores the OUTPUT brightness.
    const lut = new Float32Array(LUT_SIZE * 4);
    let targetIdx = 0;
    for (let lutI = 0; lutI < LUT_SIZE; lutI++) {
        // Map LUT index to input range. The shader passes input
        // values already clamped to [0, 1] (the adapted value after
        // local-context mixing); we map those into our histogram
        // range to look up the right CDF bin.
        const xNorm = lutI / (LUT_SIZE - 1);
        const xData = rangeMin + xNorm * (rangeMax - rangeMin);
        // Find which histogram bin this input falls in.
        const histBin = Math.min(
            nBins - 1,
            Math.max(0, Math.floor((xData - rangeMin) / (rangeMax - rangeMin) * nBins)),
        );
        const F = inputCDF[histBin];
        // Find the output bin where targetCDF crosses F.
        while (targetIdx < nBins - 1 && targetCDF[targetIdx] < F) targetIdx++;
        const yNorm = targetIdx / (nBins - 1);
        lut[lutI * 4 + 0] = yNorm;
        lut[lutI * 4 + 1] = yNorm;
        lut[lutI * 4 + 2] = yNorm;
        lut[lutI * 4 + 3] = 1.0;
        // Reset target index search for the next input (which has
        // a smaller CDF value if we're scanning in order — but the
        // LUT input goes from small to large, so CDF goes from small
        // to large, so targetIdx only ever moves forward).
        // No reset needed; monotonic input → monotonic CDF.
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
