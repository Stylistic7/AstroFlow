/**
 * Generalized Hyperbolic Stretch (asinh form) — LUT generator.
 *
 * A third display stretch alongside STF (PixInsight MTF) and the
 * constrained max-entropy curve. Its defining behavior, vs STF, is a
 * GENTLY-LIFTED background: STF hard-clips shadows at median − 2.8σ,
 * crushing faint dust to zero; this maps the sky to a small positive
 * `background` anchor and applies an asinh transfer above it — the b→0
 * case of the Generalized Hyperbolic family — lifting faint signal hard
 * while rolling highlights off smoothly. That's the tonal *character*
 * reverse-engineered from a well-processed reference: lifted black,
 * strong shadow lift, soft highlights.
 *
 * Robust on star-dominated data (where the nebula occupies <1% of the
 * linear range): like max-entropy, it pins the sky to a fixed anchor
 * (ramp below, asinh above) rather than clipping at a knife-edge black
 * point, so the result doesn't collapse to gray when the black point is
 * a hair off.
 *
 * Auto-fit from per-image statistics → applies to ANY image untuned;
 * the two knobs (`bg`, `intensity`) adjust around the auto values.
 *
 * Output: Float32Array(LUT_SIZE) indexed by `xNorm = i/(LUT-1)` where
 * xNorm = rawPixel / max_value (same normalization the shader produces).
 * Values in [0,1]. Consumed via the shader's generic LUT path, like maxent.
 */

export const LUT_SIZE = 1024;

// Where the sky background lands after the stretch (display [0,1]).
const DEFAULT_BG = 0.10;
// Multiplier on the auto-fit aggressiveness; >1 brighter, <1 gentler.
const DEFAULT_INTENSITY = 1.0;
// Output level the brightest nebula (p99.9) maps to at intensity = 1.
const SIGNAL_TARGET = 0.75;

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));


/**
 * @param {object} stats - `statistics` from `/api/metadata`. Needs
 *     `max_value`, `sky_median` (or `p50`), and `p99_9` (or `p99`).
 * @param {object} [opts] - `{ bg, intensity }` slider overrides.
 * @returns {Float32Array} length LUT_SIZE, in [0, 1].
 */
export function computeGhsLut(stats, opts = {}) {
    const maxVal = stats?.max_value;
    if (!(maxVal > 0)) return identityLut();

    const bg = clamp(opts.bg ?? DEFAULT_BG, 0, 0.5);
    const intensity = opts.intensity ?? DEFAULT_INTENSITY;

    const skyVal = (stats.sky_median != null && stats.sky_median > 0)
        ? stats.sky_median : stats.p50;
    if (!(skyVal > 0)) return identityLut();
    const hiVal = stats.p99_9 ?? stats.p99 ?? (skyVal * 1.1);

    const skyN = clamp(skyVal / maxVal, 1e-6, 0.999);
    const hiN = clamp(hiVal / maxVal, skyN + 1e-5, 1);

    // asinh on the signal range (skyN, 1]. Auto-fit β so the brightest
    // nebula (p99.9) reaches SIGNAL_TARGET, then scale by `intensity`.
    const u99 = (hiN - skyN) / (1 - skyN);
    const yTail = clamp((SIGNAL_TARGET - bg) / (1 - bg), 1e-3, 0.999);
    const beta0 = solveBeta(u99, yTail);
    const beta = Math.max(1e-7, beta0 / Math.max(0.05, intensity));
    const denom = Math.asinh(1 / beta) || 1;

    const lut = new Float32Array(LUT_SIZE);
    for (let i = 0; i < LUT_SIZE; i++) {
        const x = i / (LUT_SIZE - 1);
        if (x <= skyN) {
            // Linear ramp 0 → bg below sky (lifted, never crushed).
            lut[i] = bg * (x / skyN);
        } else {
            const u = (x - skyN) / (1 - skyN);
            lut[i] = clamp(bg + (1 - bg) * Math.asinh(u / beta) / denom, 0, 1);
        }
    }
    return lut;
}


/**
 * Solve β such that asinh(u/β)/asinh(1/β) = target, for u,target in (0,1).
 * Monotonic decreasing in β (smaller β = harder lift), so log-space
 * bisection converges.
 */
function solveBeta(u, target) {
    let lo = 1e-6, hi = 10;
    for (let it = 0; it < 48; it++) {
        const mid = Math.sqrt(lo * hi);
        const y = Math.asinh(u / mid) / Math.asinh(1 / mid);
        if (y > target) lo = mid; else hi = mid;
    }
    return Math.sqrt(lo * hi);
}


function identityLut() {
    const lut = new Float32Array(LUT_SIZE);
    for (let i = 0; i < LUT_SIZE; i++) lut[i] = i / (LUT_SIZE - 1);
    return lut;
}
