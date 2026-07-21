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
function channelStf(c, target) {
    const bp = Math.max(0, c.median - 2.8 * c.sigma);
    const wp = Math.max(c.p99, bp + 1e-3);
    const x0 = (c.median - bp) / Math.max(wp - bp, 1e-6);
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
    return seed;
}
