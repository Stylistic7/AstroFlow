/**
 * Client-side seeding for display finishes — image-adaptive STARTING values from
 * the tap's generic per-channel stats (tile_metadata `channelStats`).
 *
 * Finishes run entirely in the frontend, so their startup logic lives here too;
 * the backend ships the scene + generic stats, never finish logic. `stfSeed` is
 * the PixInsight STF AutoStretch recipe for an MTF / black-white-mid stretch —
 * the reasonable default for essentially any stretch:
 *   black    = median − 2.8σ         (clip the noise floor)
 *   white    = p99                   (bracket the top so the midtones stays sane)
 *   midtones = solve(median → target)  (map the background to a target grey)
 * A finish only needs a custom seed if it wants something other than "auto-
 * stretch this image".
 */

/** The MTF midtones m with MTF(m, x0) == target (closed form). */
function solveMidtones(x0, target) {
    const denom = 2 * target * x0 - target - x0;
    if (Math.abs(denom) < 1e-12) return 0.5;
    return Math.min(0.999, Math.max(1e-4, (x0 * (target - 1)) / denom));
}

const mean = (a) => a.reduce((s, x) => s + x, 0) / Math.max(a.length, 1);

// STF for one channel's stats → { bp, wp, mid }.
//
// 2026-09-03 recalibration (validated on the γ Cyg starless mosaics):
//   * SKY anchoring — black + midtones derive from the sky-subset stats
//     (`skyMedian`/`skySigma`, shipped when the backend can compute them
//     on the COMPOSITE = starless + stars). Every channel's displayed
//     sky then lands on the same target grey, so the background renders
//     neutral instead of carrying a per-channel-slope colour cast.
//   * GAUGE-AWARE white floor — the old `bp + 1e-3` was an absolute
//     constant calibrated for display-normalized images (sky ~0.1). On
//     the photometrically pinned gauge (sky = 0.005, faint-channel
//     signal ~1e-4) it overrode the data's p99 and parked the white 10x
//     above any real signal. Relative to the sky level it preserves the
//     old behavior on normalized images and goes data-driven on ours.
function channelStf(c, target) {
    const med = c.skyMedian ?? c.median;
    const sig = c.skySigma ?? c.sigma;
    const bp = Math.max(0, med - 2.8 * sig);
    const wp = Math.max(c.p99, bp + Math.max(0.02 * med, 1e-6));
    const x0 = (med - bp) / Math.max(wp - bp, 1e-6);
    return { bp, wp, mid: solveMidtones(x0, target) };
}

const _CH = ['S', 'H', 'O'];   // matches the MTF finish's per-channel param suffixes

/** PixInsight UNLINKED STF/MTF starting values from per-channel stats.
 * `channelStats` = the tile_metadata array of `{ median, sigma, p99, ... }` per
 * channel. Seeds EACH channel independently (its own median → the target grey,
 * which is what neutralises the background). Returns per-channel keys
 * (`blackS/whiteS/midS`, …) PLUS combined keys (`blackpoint/whitepoint/midtones`
 * = the mean) so a linked finish still seeds; the frontend applies whichever keys
 * a finish's params actually use. `target` = the background grey. */
export function stfSeed(channelStats, target = 0.15) {
    if (!channelStats || !channelStats.length) return null;
    const seed = {};
    const per = channelStats.map((c) => channelStf(c, target));
    per.forEach((p, i) => {
        const s = _CH[i] ?? String(i);
        seed[`black${s}`] = p.bp; seed[`white${s}`] = p.wp; seed[`mid${s}`] = p.mid;
    });
    seed.blackpoint = mean(per.map((p) => p.bp));
    seed.whitepoint = mean(per.map((p) => p.wp));
    seed.midtones = mean(per.map((p) => p.mid));
    // Coverage floor for the framework's ragged-edge trim: partial-coverage
    // rolloff pixels (between no-data zero and the sky pedestal) render as
    // no-data. 60% of the faintest channel's sky level — comfortably above
    // any ramp, comfortably below real sky.
    const skyMin = Math.min(...channelStats.map((c) => c.skyMedian ?? c.median));
    if (skyMin > 0) seed.trim_floor = 0.6 * skyMin;
    return seed;
}
