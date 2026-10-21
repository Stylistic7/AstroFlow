import {
    apply, centerFor, fitPpu, hammerForward, hammerInverse, lonlatAt, nativeAt, planeOf,
    rotationFor, screenOf, skyAt, vecOf,
} from './sky_projection.js';

const close = (a, b, tol = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(tol);
const closeVec = (u, v, tol = 1e-9) => u.forEach((x, i) => close(x, v[i], tol));

// A deterministic PRNG so a failure reproduces.
function rng(seed) {
    let s = seed >>> 0;
    return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

test('the Hammer projection inverts across the sky', () => {
    for (let lam = -3.0; lam <= 3.0; lam += 0.5) {
        for (let bet = -1.5; bet <= 1.5; bet += 0.25) {
            const [x, y] = hammerForward(lam, bet);
            const [l2, b2] = hammerInverse(x, y);
            close(l2, lam, 1e-9); close(b2, bet, 1e-9);
        }
    }
});

test('the ellipse rim is the sky edge, beyond it is nothing', () => {
    expect(hammerInverse(2 * Math.SQRT2, 0)).not.toBeNull();
    close(hammerInverse(2 * Math.SQRT2 - 1e-9, 0)[0], Math.PI, 1e-4);
    expect(hammerInverse(2 * Math.SQRT2 + 1e-6, 0)).toBeNull();
    close(hammerInverse(0, Math.SQRT2)[1], Math.PI / 2, 1e-9);
    expect(hammerInverse(0, Math.SQRT2 + 1e-6)).toBeNull();
});

test('screen and plane invert, east to the left', () => {
    const s = { screenW: 800, screenH: 500, ppu: 100 };
    const [sx, sy] = screenOf(0.5, -0.25, s);
    closeVec(planeOf(sx, sy, s), [0.5, -0.25]);
    expect(planeOf(300, 250, s)[0]).toBeGreaterThan(0);   // left of centre = +x = east
});

test('the rotation puts the native centre on the centre with north up', () => {
    const R = rotationFor(123, -40);
    closeVec(apply(R, [1, 0, 0]), vecOf(123 * Math.PI / 180, -40 * Math.PI / 180));
    const north = apply(R, [0, 0, 1]);
    const centre = apply(R, [1, 0, 0]);
    close(north[0] * centre[0] + north[1] * centre[1] + north[2] * centre[2], 0);   // tangent
    expect(north[2]).toBeGreaterThan(0);                                             // toward the pole
    const east = apply(R, [0, 1, 0]);
    close(east[2], 0);                                                               // east stays level
});

test('centerFor recovers the centre that put a point under the pointer', () => {
    const r = rng(11);
    for (let i = 0; i < 500; i++) {
        const lon0 = r() * 360, lat0 = (r() - 0.5) * 178;
        const n = vecOf((r() - 0.5) * 2 * Math.PI * 0.999, (r() - 0.5) * Math.PI * 0.999);
        const P = apply(rotationFor(lon0, lat0), n);
        const [l, b] = centerFor(P, n, [lon0, lat0]);
        close(b, lat0, 1e-7);
        close(((l - lon0 + 540) % 360) - 180, 0, 1e-7);
    }
});

test('a drag keeps the grabbed sky under the pointer', () => {
    const s = { screenW: 1200, screenH: 800, ppu: 400, lon0: 80, lat0: 20 };
    const P = skyAt(700, 300, s);
    const n = nativeAt(500, 450, s);
    const [lon0, lat0] = centerFor(P, n, [s.lon0, s.lat0]);
    closeVec(skyAt(500, 450, { ...s, lon0, lat0 }), P, 1e-9);
});

test('a zoom about the pointer keeps the sky under it', () => {
    const s = { screenW: 1200, screenH: 800, ppu: 150, lon0: 10, lat0: -30 };
    const P = skyAt(700, 300, s);
    const zoomed = { ...s, ppu: 450 };
    const [lon0, lat0] = centerFor(P, nativeAt(700, 300, zoomed), [s.lon0, s.lat0]);
    closeVec(skyAt(700, 300, { ...zoomed, lon0, lat0 }), P, 1e-9);
});

test('lonlatAt reads the centre and wraps longitude', () => {
    const s = { screenW: 1000, screenH: 600, ppu: 100, lon0: 350, lat0: 5 };
    closeVec(lonlatAt(500, 300, s), [350, 5], 1e-9);
    const east = lonlatAt(500 - 20, 300, s)[0];
    expect(((east - 350 + 540) % 360) - 180).toBeGreaterThan(0);   // left = east = +lon (wrapped)
    expect(lonlatAt(5, 5, s)).toBeNull();                          // off the ellipse
});

test('fitPpu fits the ellipse inside the screen', () => {
    const ppu = fitPpu(1000, 400);
    expect(ppu * 2 * 2 * Math.SQRT2).toBeLessThanOrEqual(1000);
    expect(ppu * 2 * Math.SQRT2).toBeLessThanOrEqual(400);
});
