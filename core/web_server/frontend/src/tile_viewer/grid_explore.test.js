import { rng, cellFromClick } from './grid_explore.js';

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
