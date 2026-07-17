import {
    AXES, rng, makeMove, sampleMoves, applyMove,
    buildCellParams, describeMove, cellFromClick,
} from './grid_explore.js';

const CENTER = {
    midtone: 0.08, saturation: 1.35, scnr: 1.0, shadowClip: 1.0,
    blackPoint: 0.0, trimS: 1.0, trimH: 1.0, trimO: 1.0,
};

describe('rng', () => {
    test('is deterministic for a given seed', () => {
        const a = rng(7), b = rng(7);
        const seqA = [a(), a(), a()], seqB = [b(), b(), b()];
        expect(seqA).toEqual(seqB);
    });

    test('different seeds diverge', () => {
        expect(rng(1)()).not.toEqual(rng(2)());
    });

    test('stays in [0, 1)', () => {
        const r = rng(42);
        for (let i = 0; i < 1000; i++) {
            const v = r();
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThan(1);
        }
    });
});

describe('makeMove', () => {
    test('moves 1–3 distinct known axes with non-zero deltas', () => {
        const r = rng(3);
        for (let i = 0; i < 200; i++) {
            const move = makeMove(r);
            expect(move.length).toBeGreaterThanOrEqual(1);
            expect(move.length).toBeLessThanOrEqual(3);
            const keys = move.map((m) => m.key);
            expect(new Set(keys).size).toBe(keys.length);      // distinct axes
            for (const { key, delta } of move) {
                expect(AXES[key]).toBeDefined();
                expect(delta).not.toBe(0);
            }
        }
    });
});

describe('sampleMoves', () => {
    test('fills the 8 non-centre cells and never cell 4', () => {
        const moves = sampleMoves(rng(9));
        expect(Object.keys(moves).map(Number).sort((a, b) => a - b))
            .toEqual([0, 1, 2, 3, 5, 6, 7, 8]);
    });
});

describe('applyMove', () => {
    test('clamps each axis to its range', () => {
        const bottomed = applyMove(CENTER, [{ key: 'midtone', delta: -99 }]);
        expect(bottomed.midtone).toBe(AXES.midtone.range[0]);
        const topped = applyMove(CENTER, [{ key: 'saturation', delta: +99 }]);
        expect(topped.saturation).toBe(AXES.saturation.range[1]);
    });

    test('is pure — does not mutate the centre', () => {
        const snapshot = { ...CENTER };
        applyMove(CENTER, [{ key: 'saturation', delta: 0.5 }]);
        expect(CENTER).toEqual(snapshot);
    });

    test('"keep going" advances linearly: applying the same move twice doubles the delta', () => {
        const move = [{ key: 'saturation', delta: 0.35 }];
        const once = applyMove(CENTER, move);
        const twice = applyMove(once, move);
        expect(once.saturation).toBeCloseTo(1.70);
        expect(twice.saturation).toBeCloseTo(2.05);   // constant step, no acceleration
    });
});

describe('buildCellParams', () => {
    const base = { finishMode: true, isColor: false, paletteMode: 0 };

    test('returns 9 cells with the centre (4) unperturbed and trims propagated', () => {
        const moves = sampleMoves(rng(11));
        const cells = buildCellParams(CENTER, base, moves);
        expect(cells).toHaveLength(9);
        // Cell 4 is the identity finish.
        expect(cells[4].e5).toBe(CENTER.midtone);
        expect(cells[4].saturation).toBe(CENTER.saturation);
        expect(cells[4].trims).toEqual([CENTER.trimS, CENTER.trimH, CENTER.trimO]);
        // Base fields carry through to every cell.
        for (const c of cells) expect(c.finishMode).toBe(true);
    });

    test('per-cell trim moves reach the trims array', () => {
        const moves = { 0: [{ key: 'trimH', delta: 0.5 }] };
        for (const c of [1, 2, 3, 5, 6, 7, 8]) moves[c] = [];
        const cells = buildCellParams(CENTER, base, moves);
        expect(cells[0].trims).toEqual([1.0, 1.5, 1.0]);
    });
});

describe('describeMove', () => {
    test('polarity: brighter (lower midtone) and greener (lower scnr) read as "+"', () => {
        expect(describeMove([{ key: 'midtone', delta: -0.05 }])).toBe('bright+');
        expect(describeMove([{ key: 'midtone', delta: +0.05 }])).toBe('bright−');
        expect(describeMove([{ key: 'scnr', delta: -0.2 }])).toBe('green+');
        expect(describeMove([{ key: 'scnr', delta: +0.2 }])).toBe('green−');
    });

    test('non-inverted axes read naturally and multi-axis joins with ·', () => {
        expect(describeMove([{ key: 'saturation', delta: 0.35 }, { key: 'trimO', delta: -0.25 }]))
            .toBe('sat+ · O−');
    });

    test('empty move → empty label', () => {
        expect(describeMove([])).toBe('');
    });
});

describe('cellFromClick', () => {
    test('maps quadrants to row-major cell indices', () => {
        expect(cellFromClick(10, 10, 300, 300)).toBe(0);   // top-left
        expect(cellFromClick(150, 150, 300, 300)).toBe(4); // centre
        expect(cellFromClick(290, 290, 300, 300)).toBe(8); // bottom-right
        expect(cellFromClick(290, 10, 300, 300)).toBe(2);  // top-right
    });

    test('clamps out-of-box coordinates', () => {
        expect(cellFromClick(-50, -50, 300, 300)).toBe(0);
        expect(cellFromClick(9999, 9999, 300, 300)).toBe(8);
    });
});
