import { ICRS_FROM_GALACTIC, galacticOf, mapFromDisplay, multiply, radecOf } from './frames.js';
import { vecOf } from './sky_projection.js';

const DEG = Math.PI / 180;

test('the Galactic axes land where astropy puts them', () => {
    // Galactic centre, l = 90° and the north Galactic pole (astropy, J2000).
    const cases = [
        [[0, 0], [266.404988, -28.936178]],
        [[90, 0], [318.004382, 48.329637]],
        [[0, 90], [192.859478, 27.128252]],
    ];
    for (const [[l, b], [ra, dec]] of cases) {
        const [r, d] = radecOf(vecOf(l * DEG, b * DEG));
        expect(r).toBeCloseTo(ra, 5);
        expect(d).toBeCloseTo(dec, 5);
        const [gl, gb] = galacticOf(ra, dec);
        if (Math.abs(b) < 90) expect(((gl - l + 180) % 360 + 360) % 360 - 180).toBeCloseTo(0, 5);
        expect(gb).toBeCloseTo(b, 5);
    }
});

test('the frame matrices are rotations and the identity is the identity', () => {
    const t = [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => ICRS_FROM_GALACTIC[i]);   // transpose
    const p = multiply(ICRS_FROM_GALACTIC, t);
    for (let i = 0; i < 9; i++) expect(p[i]).toBeCloseTo(i % 4 === 0 ? 1 : 0, 12);
    expect(mapFromDisplay('galactic')).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(mapFromDisplay('icrs')).toBe(ICRS_FROM_GALACTIC);
    expect(() => mapFromDisplay('ecliptic')).toThrow();
});
