import { degradeFaces } from './sky_faces.js';

test('degradeFaces averages each 2×2 block of every face', () => {
    const n = 4;
    const faces = new Float32Array(12 * n * n).map((_, i) => i);
    const d = degradeFaces(faces, n);
    expect(d.length).toBe(12 * 2 * 2);
    // Face 0, block (0, 0): cells 0, 1, 4, 5.
    expect(d[0]).toBeCloseTo((0 + 1 + 4 + 5) / 4);
    // Face 11, block (1, 1): the last 2×2 of the last face.
    const base = 11 * n * n;
    expect(d[12 * 4 - 1]).toBeCloseTo((base + 10 + base + 11 + base + 14 + base + 15) / 4);
});
