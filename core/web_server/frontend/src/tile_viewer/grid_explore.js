/**
 * 3×3 taste-grid primitives — pure, no WebGPU/React.
 *
 * Cell 4 is the centre (the current finish). The 8 neighbours are SUGGESTIONS:
 * each is a "move" — a small signed vector of nudges across one or more of the
 * finish's param axes. Clicking a neighbour applies its move to the centre AND
 * keeps that same move under that cell, so clicking it again travels further
 * the same way ("please keep going along this axis"). The other 7 cells
 * reshuffle to fresh suggestions. Clicking the centre keeps the finish and
 * reshuffles all 8 ("give me new ideas").
 *
 * The goal is discovery: instead of hunting for a good subspace by dragging a
 * pile of unrelated sliders, the user is shown a handful of plausible, coherent
 * looks and picks by appearance. The axes come from the finish's own params —
 * see generic_grid.js; this module holds the finish-agnostic pieces: the
 * seeded RNG (so a given sequence of clicks always yields the same grids) and
 * the click → cell mapping.
 */

/** Deterministic PRNG (mulberry32). `seed` → a function returning [0, 1). */
export function rng(seed) {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/** Which 3×3 cell (0..8, row-major) a click at (x,y) in a w×h box falls in. */
export function cellFromClick(x, y, w, h) {
    const col = Math.min(2, Math.max(0, Math.floor(x / (w / 3))));
    const row = Math.min(2, Math.max(0, Math.floor(y / (h / 3))));
    return row * 3 + col;
}
