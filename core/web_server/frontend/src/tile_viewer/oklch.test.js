import { colorAt, toHueSat, toHex, srgbToOklab, LINE_MARKERS } from './oklch.js';

const close = (a, b, tol) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

describe('colorAt', () => {
    test('always returns a full-value display colour inside [0, 1]', () => {
        for (let hue = 0; hue < 360; hue += 7) {
            for (const sat of [0, 0.3, 0.7, 1]) {
                const c = colorAt(hue, sat);
                expect(Math.max(...c)).toBeCloseTo(1, 6);
                expect(Math.min(...c)).toBeGreaterThanOrEqual(0);
            }
        }
    });

    test('sat 0 is white and sat 1 at red’s hue is sRGB red', () => {
        expect(close(colorAt(123, 0), [1, 1, 1], 1e-6)).toBe(true);
        const redHue = toHueSat([1, 0, 0]).hue;
        expect(close(colorAt(redHue, 1), [1, 0, 0], 0.01)).toBe(true);
    });
});

describe('toHueSat', () => {
    test('inverts colorAt across the plane', () => {
        for (let hue = 3; hue < 360; hue += 11) {
            for (const sat of [0.15, 0.5, 0.85, 1]) {
                const back = toHueSat(colorAt(hue, sat));
                const dh = Math.abs(((back.hue - hue + 540) % 360) - 180);
                expect(dh).toBeLessThan(0.5);
                expect(Math.abs(back.sat - sat)).toBeLessThan(0.01);
            }
        }
    });

    test('a grey keeps the fallback hue at zero saturation', () => {
        expect(toHueSat([1, 1, 1], 200)).toEqual({ hue: 200, sat: 0 });
    });
});

test('hex and line markers', () => {
    expect(toHex([1, 0.5, 0])).toBe('#ff8000');
    const hue = (rgb) => { const [, a, b] = srgbToOklab(rgb); return ((Math.atan2(b, a) * 180) / Math.PI + 360) % 360; };
    expect(LINE_MARKERS[0].hue).toBeCloseTo(hue([1, 0, 0]), 6);
    expect(LINE_MARKERS.map((m) => m.label)).toEqual(['Hα · NII · SII', 'OIII', 'Hβ']);
});
