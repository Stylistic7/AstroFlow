/**
 * 3×3 taste-grid suggestion logic — pure, no WebGPU/React.
 *
 * Cell 4 is the centre (the current finish). The 8 neighbours are SUGGESTIONS:
 * each is a "move" — a small signed vector of nudges across one or more of the
 * finish axes (the stretch + colour knobs). Clicking a neighbour applies its
 * move to the centre AND keeps that same move under that cell, so clicking it
 * again travels further the same way ("please keep going along this axis"). The
 * other 7 cells reshuffle to fresh suggestions. Clicking the centre keeps the
 * finish and reshuffles all 8 ("give me new ideas").
 *
 * The goal is discovery: instead of hunting for a good subspace by dragging a
 * pile of unrelated sliders, the user is shown a handful of plausible, coherent
 * looks and picks by appearance. A move often nudges two or three axes at once,
 * because a combined move is frequently more obviously "better" than a single
 * knob — so the sampler is weighted toward multi-axis moves.
 *
 * Everything is data-agnostic and deterministic (seeded RNG) so the axis set /
 * step mix is a one-place edit below and the grid is reproducible.
 */

// The finish axes the grid perturbs. `range` clamps the knob, `step` is one
// nudge, `label` is the short human tag shown on the cell. Positive delta means
// "more of label" EXCEPT for the inverted axes below (see INVERT), where the
// knob runs backwards relative to how it reads.
export const AXES = {
    // Per-channel Generalized Hyperbolic Stretch — the heart of the grid. Each
    // S/H/O channel is stretched independently by strength (D), contrast (b), and
    // symmetry point (SP, the dark-tail pivot). D uses the ghsastro slider domain
    // (actual = exp(D)-1). See project_scene_display_seam: this is op #1 of the
    // display-referred finish.
    dS:  { range: [0.4, 4.0], step: 0.4,  label: 'S str' },
    dH:  { range: [0.4, 4.0], step: 0.4,  label: 'H str' },
    dO:  { range: [0.4, 4.0], step: 0.4,  label: 'O str' },
    bS:  { range: [-2.0, 4.0], step: 0.6, label: 'S con' },
    bH:  { range: [-2.0, 4.0], step: 0.6, label: 'H con' },
    bO:  { range: [-2.0, 4.0], step: 0.6, label: 'O con' },
    spS: { range: [0.02, 0.4], step: 0.03, label: 'S piv' },
    spH: { range: [0.02, 0.4], step: 0.03, label: 'H piv' },
    spO: { range: [0.02, 0.4], step: 0.03, label: 'O piv' },
    // Colour knobs.
    saturation: { range: [0, 3], step: 0.35, label: 'sat' },
    scnr:       { range: [0, 1], step: 0.20, label: 'green' },  // higher knob = less green
};

const KEYS = Object.keys(AXES);

// Axes whose knob runs opposite to its label: a positive knob delta reads as
// LESS of the labelled quantity (greener = lower scnr).
const INVERT = new Set(['scnr']);

// How many axes a single suggestion moves. Weighted toward two, since combined
// moves are often more obviously good than a single-knob nudge. Index = n − 1.
const AXIS_COUNT_WEIGHTS = [0.30, 0.45, 0.25];   // P(1 axis), P(2), P(3)

const NON_CENTRE_CELLS = [0, 1, 2, 3, 5, 6, 7, 8];

const clamp = (k, v) => Math.min(AXES[k].range[1], Math.max(AXES[k].range[0], v));

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

/** Draw an axis count (1..3) from AXIS_COUNT_WEIGHTS. */
function axisCount(rand) {
    let r = rand();
    for (let i = 0; i < AXIS_COUNT_WEIGHTS.length; i++) {
        if (r < AXIS_COUNT_WEIGHTS[i]) return i + 1;
        r -= AXIS_COUNT_WEIGHTS[i];
    }
    return AXIS_COUNT_WEIGHTS.length;
}

// Named focus groups so the grid explores a COHERENT subset at a time — 11 axes
// perturbed at once is unreadable. The user picks a focus; the sampler draws
// only from that group. These are the "soft steps" of the workflow.
export const AXIS_GROUPS = {
    Stretch: ['dS', 'dH', 'dO', 'bS', 'bH', 'bO'],  // per-channel strength + contrast
    Pivot:   ['spS', 'spH', 'spO'],                  // per-channel dark-tail pivot
    Color:   ['saturation', 'scnr'],
};

/** One suggestion = a move: n distinct axes (from `keys`), each a signed nudge. */
export function makeMove(rand, keys = KEYS) {
    const pool = [...keys];
    const n = Math.min(axisCount(rand), pool.length);
    const move = [];
    for (let i = 0; i < n; i++) {
        const key = pool.splice(Math.floor(rand() * pool.length), 1)[0];
        const mag = AXES[key].step * (rand() < 0.5 ? 1 : 2);
        move.push({ key, delta: (rand() < 0.5 ? -1 : 1) * mag });
    }
    return move;
}

/** Fresh suggestions for the 8 non-centre cells → { cellIndex: move }. */
export function sampleMoves(rand, keys = KEYS) {
    const moves = {};
    for (const c of NON_CENTRE_CELLS) moves[c] = makeMove(rand, keys);
    return moves;
}

/** Apply a move to the centre knobs → new clamped knobs. */
export function applyMove(center, move) {
    const knobs = { ...center };
    for (const { key, delta } of (move || []))
        knobs[key] = clamp(key, knobs[key] + delta);
    return knobs;
}

/** Merge knob values into the constant finish params → a setGlobals param
 * object. `base` carries the fixed finish fields (anchors, sigmas, paletteMode,
 * finishMode, isColor, peakBrightness, addStars, max, e1, stretchMode). */
function paramsFor(knobs, base) {
    return {
        ...base,
        ghsD: [knobs.dS, knobs.dH, knobs.dO],   // per-channel strength
        ghsB: [knobs.bS, knobs.bH, knobs.bO],   // per-channel contrast
        ghsSP: [knobs.spS, knobs.spH, knobs.spO], // per-channel dark-tail pivot
        saturation: knobs.saturation,
        scnrAmount: knobs.scnr,
    };
}

/** Build the 9 finish-param objects for the grid: cell 4 = centre, each other
 * cell = centre + its move. `moves` is a { cellIndex: move } map. */
export function buildCellParams(center, base, moves) {
    const cells = [];
    for (let i = 0; i < 9; i++)
        cells.push(paramsFor(i === 4 ? center : applyMove(center, moves[i]), base));
    return cells;
}

/** Short human label for a move, e.g. "bright+ · sat−". Empty for a no-op. */
export function describeMove(move) {
    if (!move || !move.length) return '';
    return move.map(({ key, delta }) => {
        const more = INVERT.has(key) ? delta < 0 : delta > 0;
        return AXES[key].label + (more ? '+' : '−');
    }).join(' · ');
}

/** Which 3×3 cell (0..8, row-major) a click at (x,y) in a w×h box falls in. */
export function cellFromClick(x, y, w, h) {
    const col = Math.min(2, Math.max(0, Math.floor(x / (w / 3))));
    const row = Math.min(2, Math.max(0, Math.floor(y / (h / 3))));
    return row * 3 + col;
}
