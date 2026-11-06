import {
    TileAtlas, degradeTile, healpixXyf, indexSize, morton, neededTiles, orderOffset,
    parseFitsTile, tileIndexOf, tileKey, tilePath, wantedOrder, CELL_ANGLE_NSIDE1,
} from './sky_tiles.js';

test('tile paths, keys and index offsets follow HiPS and NESTED order', () => {
    expect(tilePath(8, 400123)).toBe('Norder8/Dir400000/Npix400123.fits');
    expect(tilePath(0, 7)).toBe('Norder0/Dir0/Npix7.fits');
    expect(orderOffset(0)).toBe(0);
    expect(orderOffset(1)).toBe(12);
    expect(orderOffset(2)).toBe(12 + 48);
    expect(indexSize(8)).toBe(4 * (4 ** 9 - 1));
    expect(morton(5, 2)).toBe(0b011001);               // x=101 on the even bits, y=010 on the odd
    expect(morton(1, 0)).toBe(1);
    expect(morton(0, 1)).toBe(2);
    expect(morton(3, 3)).toBe(15);
    expect(tileIndexOf(3, 0.6, 0.1, 1)).toBe(3 * 4 + morton(1, 0));
    expect(tileKey(2, 5)).toBe(2 * 2 ** 32 + 5);
});

test('healpixXyf puts the twelve face centres at (½, ½)', () => {
    // Equatorial face 4 is centred on lon 0, lat 0; polar face 0 on lon 45°, lat 41.8°.
    expect(healpixXyf([1, 0, 0])).toEqual([4, 0.5, 0.5]);
    const b = Math.asin(2 / 3), l = Math.PI / 4;
    const [face, x, y] = healpixXyf([Math.cos(b) * Math.cos(l), Math.cos(b) * Math.sin(l), Math.sin(b)]);
    expect(face).toBe(0);
    expect(x).toBeCloseTo(0.5, 6);
    expect(y).toBeCloseTo(0.5, 6);
});

test('the wanted order gives tile pixels at least as fine as the screen pixel', () => {
    const tileShift = 9;                                   // 512-px tiles
    // A screen pixel exactly one nside-1024 cell wide asks for pixel order 10 → tile order 1.
    expect(wantedOrder(CELL_ANGLE_NSIDE1 / 1024, tileShift, 8)).toBe(1);
    // Slightly smaller than that cell → the next order up.
    expect(wantedOrder(CELL_ANGLE_NSIDE1 / 1100, tileShift, 8)).toBe(2);
    // Deep zoom clamps at the tree's finest order; the full sky at order 0.
    expect(wantedOrder(1e-9, tileShift, 8)).toBe(8);
    expect(wantedOrder(0.01, tileShift, 8)).toBe(0);
});

test('neededTiles lists each visible tile once, nearest the centre first', () => {
    // A fake sky: every screen point maps to face 4 with x, y from its position;
    // the pixel angle asks for order 1 (tiles of 2×2 per face).
    const tiles = neededTiles({
        screenW: 400, screenH: 400, step: 50,
        dirAt: (sx, sy) => (sx < 200 ? [1, 0, 0] : null),           // right half off the sky
        pixAngleAt: () => CELL_ANGLE_NSIDE1 / 1024,
        tileShift: 9, maxOrder: 8,
    });
    expect(tiles.length).toBe(1);
    expect(tiles[0]).toMatchObject({ order: 1, npix: 4 * 4 + morton(1, 1) });
});

test('parseFitsTile turns a HiPS tile a quarter turn into the raster', () => {
    const w = 4;
    const cards = [
        'SIMPLE  =                    T', 'BITPIX  =                  -32', 'NAXIS   =                    2',
        `NAXIS1  =                    ${w}`, `NAXIS2  =                    ${w}`, 'END',
    ];
    const header = cards.map((c) => c.padEnd(80)).join('').padEnd(2880);
    const buf = new ArrayBuffer(2880 * 2);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < header.length; i++) bytes[i] = header.charCodeAt(i);
    const dv = new DataView(buf, 2880);
    for (let r = 0; r < w; r++) for (let c = 0; c < w; c++) dv.setFloat32(4 * (r * w + c), r * w + c, false);
    dv.setFloat32(4 * (2 * w + 1), NaN, false);                     // a hole at row 2, col 1
    const raster = parseFitsTile(buf);
    for (let x = 0; x < w; x++) {
        for (let y = 0; y < w; y++) {
            const r = w - 1 - x, c = y;
            expect(raster[y * w + x]).toBe(r === 2 && c === 1 ? 0 : r * w + c);
        }
    }
});

test('degradeTile averages each 2×2 block', () => {
    const d = degradeTile(new Float32Array(16).map((_, i) => i), 4);
    expect(d.length).toBe(4);
    expect(d[0]).toBeCloseTo((0 + 1 + 4 + 5) / 4);
    expect(d[3]).toBeCloseTo((10 + 11 + 14 + 15) / 4);
});

test('the atlas evicts the least recently needed unpinned tile', () => {
    const a = new TileAtlas(2);
    expect(a.acquire(tileKey(0, 0), { pinned: true, stamp: 1 })).toEqual({ slot: 0, evicted: null });
    expect(a.acquire(tileKey(3, 7), { stamp: 2 })).toEqual({ slot: 1, evicted: null });
    // Full: the pinned tile stays, the other goes.
    expect(a.acquire(tileKey(3, 8), { stamp: 3 })).toEqual({ slot: 1, evicted: tileKey(3, 7) });
    // A tile the view needs is never the victim.
    expect(a.acquire(tileKey(3, 9), { stamp: 4, needed: new Set([tileKey(3, 8)]) }))
        .toEqual({ slot: -1, evicted: null });
    a.touch(tileKey(3, 8), 10);
    expect(a.has(tileKey(3, 8))).toBe(true);
    a.release(tileKey(3, 8));
    expect(a.size).toBe(1);
    expect(a.acquire(tileKey(3, 9), { stamp: 5 })).toEqual({ slot: 1, evicted: null });
});
