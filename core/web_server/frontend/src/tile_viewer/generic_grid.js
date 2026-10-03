/**
 * Data-driven 3×3 taste-grid for a Python DisplayFinish.
 *
 * Same suggestion mechanics as grid_explore.js (cell 4 = centre, the 8
 * neighbours are signed "moves" across the axes; click one to recenter + keep
 * that direction), but the AXES ARE BUILT FROM the finish's extracted params —
 * name, range, step, label — not a hardcoded set. Nothing here knows what a
 * param means; that's the whole point of the seam: the grid is generated from
 * the Python definition. Reuses grid_explore's mulberry32 rng for reproducible
 * grids. See project_scene_display_seam.
 */
import { rng } from './grid_explore.js';

export { rng };

const NON_CENTRE_CELLS = [0, 1, 2, 3, 5, 6, 7, 8];
// Each neighbour cell should read as a SMALL variation on the centre, not a big
// jump — so bias hard toward changing a single axis, occasionally two.
const AXIS_COUNT_WEIGHTS = [0.72, 0.28];   // P(1 axis), P(2)
// The Spread slider is a GLOBAL multiplier on each axis's declared step; 1 = use
// the finish's own steps as-is. HOW each axis nudges (add vs multiply) and its
// per-axis step come from the param's `scale`/`step` — see the Param docstring.
const GRID_SPREAD = 1.0;

/** Axis map { name: {lo, hi, step, label, scale} } from a finish spec's params. */
export function axesFromParams(params) {
    const axes = {};
    for (const p of params) {
        if (p.kind === 'data') continue;   // measured from the image, not taste
        axes[p.name] = { lo: p.lo, hi: p.hi, step: p.step, label: p.label, scale: p.scale || 'linear' };
    }
    return axes;
}

function axisCount(rand) {
    let r = rand();
    for (let i = 0; i < AXIS_COUNT_WEIGHTS.length; i++) {
        if (r < AXIS_COUNT_WEIGHTS[i]) return i + 1;
        r -= AXIS_COUNT_WEIGHTS[i];
    }
    return AXIS_COUNT_WEIGHTS.length;
}

const clampAxis = (axes, k, v) => Math.min(axes[k].hi, Math.max(axes[k].lo, v));

/** One suggestion = n distinct axes, each a small signed nudge sized + shaped by
 * the axis's own hints: a "log" axis moves × (1 ± step·spread) (a % of its value),
 * a "linear" axis moves ± step·spread (an absolute amount). A move records the
 * op, so "keep going" (clicking a tile) re-applies it and repeats compound the
 * right way. `spread` is the global Spread multiplier. */
export function makeMove(rand, axes, center, spread = GRID_SPREAD) {
    const pool = Object.keys(axes);
    const n = Math.min(axisCount(rand), pool.length);
    const move = [];
    for (let i = 0; i < n; i++) {
        const key = pool.splice(Math.floor(rand() * pool.length), 1)[0];
        const axis = axes[key];
        const dir = rand() < 0.5 ? -1 : 1;
        const mag = axis.step * spread;
        move.push(axis.scale === 'log'
            ? { key, op: 'mul', amount: 1 + dir * mag }   // × (1 ± mag)
            : { key, op: 'add', amount: dir * mag });     // ± mag
    }
    return move;
}

/** Fresh suggestions for the 8 non-centre cells → { cellIndex: move }.
 * `spread` = the global Spread multiplier the grid's slider passes. */
export function sampleMoves(rand, axes, center, spread = GRID_SPREAD) {
    const moves = {};
    for (const c of NON_CENTRE_CELLS) moves[c] = makeMove(rand, axes, center, spread);
    return moves;
}

/** Apply a move to the centre knobs → new clamped knobs (per-item add or mul). */
export function applyMove(axes, center, move) {
    const knobs = { ...center };
    for (const m of (move || [])) {
        const v = m.op === 'mul' ? knobs[m.key] * m.amount : knobs[m.key] + m.amount;
        knobs[m.key] = clampAxis(axes, m.key, v);
    }
    return knobs;
}

/** Short human label for a move, e.g. "black+ · mid−". */
export function describeMove(axes, move) {
    if (!move || !move.length) return '';
    return move.map((m) => {
        const up = m.op === 'mul' ? m.amount > 1 : m.amount > 0;
        return axes[m.key].label + (up ? '+' : '−');
    }).join(' · ');
}

/** The 9 cell param arrays for setGrid: this cell's knobs packed in `order`
 * (the finish's param order = the served DisplayParams struct field order).
 * Cell 4 = centre. */
export function buildCellFinishParams(axes, order, center, moves) {
    const cells = [];
    for (let i = 0; i < 9; i++) {
        const knobs = i === 4 ? center : applyMove(axes, center, moves[i]);
        cells.push(order.map((name) => knobs[name]));
    }
    return cells;
}
