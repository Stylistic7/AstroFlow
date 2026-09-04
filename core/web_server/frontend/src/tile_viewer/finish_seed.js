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
//   * BRIGHT white anchor — `high` (= max(p99.9, median + 60σ), shipped
//     in channelStats). NOT p99: an astro frame is ~99% background, so
//     p99 sits a few sigma above sky, the span collapses, and everything
//     brighter than sky+noise blows out (measured 2026-09-03: 1.3% of a
//     broadband target saturated, background lifted to 0.17 — the
//     "washed out" pop when the seed landed). The gauge-relative floor
//     stays as the degenerate guard for a channel with no bright pixels
//     at all; the old absolute `bp + 1e-3` was calibrated for
//     display-normalized images (sky ~0.1) and is wrong on the pinned
//     0.005 gauge.
function channelStf(c, target) {
    const med = c.skyMedian ?? c.median;
    const sig = c.skySigma ?? c.sigma;
    const bp = Math.max(0, med - 2.8 * sig);
    const bright = c.high ?? c.p999 ?? c.p99;
    const wp = Math.max(bright, bp + Math.max(0.02 * med, 1e-6));
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
// `target` = where the BACKGROUND lands in display units. 0.04, not the
// classic 0.15/0.25 STF convention, because AstroFlow's data is already
// photometrically pinned (sky = 0.005) and the finishes' own defaults are
// hand-tuned for that gauge — they render a nominal frame's background at
// 0.034. Seeding to 0.15 made the view JUMP ~5x brighter the moment stats
// arrived (measured 2026-09-03: first paint bg 0.034 -> seeded 0.166,
// "epic contrast to washed out"). Matching the defaults means the adaptive
// pass REFINES the stretch — correcting black/white for this image — with
// no visible pop.
export function stfSeed(channelStats, target = 0.04) {
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
