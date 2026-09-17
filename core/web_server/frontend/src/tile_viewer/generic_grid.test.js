import {
    rng, axesFromParams, makeMove, sampleMoves, applyMove,
    buildCellFinishParams, describeMove,
} from './generic_grid.js';

const PARAMS = [
    { name: 'blackH', default: 0.004, lo: 0, hi: 0.05, step: 0.001, label: 'black', scale: 'linear' },
    { name: 'midH', default: 0.1, lo: 0.001, hi: 0.999, step: 0.2, label: 'mid', scale: 'log' },
    { name: 'saturation', default: 1.0, lo: 0, hi: 3, step: 0.35, label: 'sat', scale: 'linear' },
];
const AXES = axesFromParams(PARAMS);
const ORDER = PARAMS.map((p) => p.name);
const CENTER = { blackH: 0.004, midH: 0.1, saturation: 1.0 };

describe('makeMove', () => {
    test('moves 1–2 distinct declared axes, add on linear and mul on log', () => {
        const r = rng(3);
        for (let i = 0; i < 200; i++) {
            const move = makeMove(r, AXES, CENTER);
            expect(move.length).toBeGreaterThanOrEqual(1);
            expect(move.length).toBeLessThanOrEqual(2);
            const keys = move.map((m) => m.key);
            expect(new Set(keys).size).toBe(keys.length);      // distinct axes
            for (const { key, op, amount } of move) {
                expect(AXES[key]).toBeDefined();
                expect(op).toBe(AXES[key].scale === 'log' ? 'mul' : 'add');
                expect(amount).not.toBe(op === 'mul' ? 1 : 0);
            }
        }
    });
});

describe('sampleMoves', () => {
    test('fills the 8 non-centre cells and never cell 4', () => {
        const moves = sampleMoves(rng(9), AXES, CENTER);
        expect(Object.keys(moves).map(Number).sort((a, b) => a - b))
            .toEqual([0, 1, 2, 3, 5, 6, 7, 8]);
    });
});

describe('applyMove', () => {
    test('clamps each axis to its declared range', () => {
        const bottomed = applyMove(AXES, CENTER, [{ key: 'blackH', op: 'add', amount: -99 }]);
        expect(bottomed.blackH).toBe(AXES.blackH.lo);
        const topped = applyMove(AXES, CENTER, [{ key: 'saturation', op: 'add', amount: +99 }]);
        expect(topped.saturation).toBe(AXES.saturation.hi);
    });

    test('is pure — does not mutate the centre', () => {
        const snapshot = { ...CENTER };
        applyMove(AXES, CENTER, [{ key: 'saturation', op: 'add', amount: 0.5 }]);
        expect(CENTER).toEqual(snapshot);
    });

    test('"keep going" compounds: add twice doubles the delta, mul twice squares the ratio', () => {
        const add = [{ key: 'saturation', op: 'add', amount: 0.35 }];
        expect(applyMove(AXES, applyMove(AXES, CENTER, add), add).saturation).toBeCloseTo(1.70);
        const mul = [{ key: 'midH', op: 'mul', amount: 1.2 }];
        expect(applyMove(AXES, applyMove(AXES, CENTER, mul), mul).midH).toBeCloseTo(0.144);
    });
});

describe('buildCellFinishParams', () => {
    test('returns 9 param arrays in declared order with the centre (4) unperturbed', () => {
        const moves = sampleMoves(rng(11), AXES, CENTER);
        const cells = buildCellFinishParams(AXES, ORDER, CENTER, moves);
        expect(cells).toHaveLength(9);
        expect(cells[4]).toEqual([CENTER.blackH, CENTER.midH, CENTER.saturation]);
        for (const c of cells) expect(c).toHaveLength(ORDER.length);
    });

    test('a cell move reaches its slot in the packed array', () => {
        const moves = { 0: [{ key: 'saturation', op: 'add', amount: 0.5 }] };
        for (const c of [1, 2, 3, 5, 6, 7, 8]) moves[c] = [];
        const cells = buildCellFinishParams(AXES, ORDER, CENTER, moves);
        expect(cells[0]).toEqual([CENTER.blackH, CENTER.midH, 1.5]);
    });
});

describe('describeMove', () => {
    test('labels each axis with its direction and joins with ·', () => {
        expect(describeMove(AXES, [{ key: 'saturation', op: 'add', amount: 0.35 }])).toBe('sat+');
        expect(describeMove(AXES, [{ key: 'midH', op: 'mul', amount: 0.8 }])).toBe('mid−');
        expect(describeMove(AXES, [
            { key: 'saturation', op: 'add', amount: 0.35 },
            { key: 'blackH', op: 'add', amount: -0.001 },
        ])).toBe('sat+ · black−');
    });

    test('empty move → empty label', () => {
        expect(describeMove(AXES, [])).toBe('');
    });
});
