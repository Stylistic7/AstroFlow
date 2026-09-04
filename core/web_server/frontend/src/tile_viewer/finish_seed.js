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
function clampTo(v, b) {
    return b ? Math.min(b.hi, Math.max(b.lo, v)) : v;
}

// `bpB`/`wpB` are the finish's DECLARED ranges for this channel's black and
// white params. Clamping here — before solving midtones — is load-bearing:
// the viewer clamps whatever it applies, so a midtones solved for an
// out-of-range span renders the background at the wrong level. Measured
// 2026-09-03: the RGB finish declared whitepoint lo = 0.02 while the pinned
// gauge's natural white is 0.0088, so the applied white was 2.3x the solved
// one and the background rendered at 0.009 instead of 0.040 — near-black,
// and invisible to any test that didn't replay the clamp.
function channelStf(c, target, bpB, wpB) {
    const med = c.skyMedian ?? c.median;
    const sig = c.skySigma ?? c.sigma;
    const bright = c.high ?? c.p999 ?? c.p99;
    const bp = clampTo(Math.max(0, med - 2.8 * sig), bpB);
    let wp = clampTo(Math.max(bright, bp + Math.max(0.02 * med, 1e-6)), wpB);
    if (wp <= bp) wp = bp + 1e-6;
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
// `target` = where the BACKGROUND lands in display units. 0.18 sits
// between the classic PixInsight STF (0.25) and the darker end: measured
// on a real broadband target (2026-09-03), it puts the background at
// 0.197 and p99 at 0.629, so the frame actually uses the screen. Lower
// values (0.04 = matching the finishes' own defaults) render a frame
// that is ~99% background as almost entirely dark — correct by the
// recipe, unusable to look at.
export function stfSeed(channelStats, target = 0.10, boundsFor = null) {
    if (!channelStats || !channelStats.length) return null;
    const b = (name) => (boundsFor ? boundsFor(name) : null);
    const seed = {};
    const per = channelStats.map((c, i) => {
        const s = _CH[i] ?? String(i);
        return channelStf(c, target, b(`black${s}`), b(`white${s}`));
    });
    per.forEach((p, i) => {
        const s = _CH[i] ?? String(i);
        seed[`black${s}`] = p.bp; seed[`white${s}`] = p.wp; seed[`mid${s}`] = p.mid;
    });
    // LINKED keys are solved from POOLED stats under the LINKED params' own
    // ranges — not the mean of three separately-solved midtones, which is
    // only self-consistent when every channel shares a span.
    const pooled = {
        skyMedian: mean(channelStats.map((c) => c.skyMedian ?? c.median)),
        skySigma: mean(channelStats.map((c) => c.skySigma ?? c.sigma)),
        high: mean(channelStats.map((c) => c.high ?? c.p999 ?? c.p99)),
    };
    const linked = channelStf(pooled, target, b('blackpoint'), b('whitepoint'));
    seed.blackpoint = linked.bp;
    seed.whitepoint = linked.wp;
    seed.midtones = linked.mid;
    // Coverage floor for the framework's ragged-edge trim: partial-coverage
    // rolloff pixels (between no-data zero and the sky pedestal) render as
    // no-data. 60% of the faintest channel's sky level — comfortably above
    // any ramp, comfortably below real sky.
    const skyMin = Math.min(...channelStats.map((c) => c.skyMedian ?? c.median));
    if (skyMin > 0) seed.trim_floor = 0.6 * skyMin;
    // `star_white` normalises the STARS layer before its midtones curve.
    // Same bright anchor as the image white: the stats are composite-derived,
    // so `high` already sits at the bright-star level. Without a data-derived
    // value the curve saw raw linear star pixels (~0.005) and screened a 3%
    // brightening — the "stars invisible at star_s=2" report (2026-09-03).
    seed.star_white = pooled.high;
    return seed;
}
