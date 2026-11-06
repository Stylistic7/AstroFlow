/**
 * Tiles of a HEALPix pyramid on the client — the HiPS shape every map is
 * served in (allsky/maps.py, allsky/hips.py): at order k the sky is 12·4^k
 * cells, each tile a `width`² raster of the cells log2(width) orders
 * deeper, NESTED within the tile. This is the pure part of streaming them:
 * which tiles a view shows, the slot atlas the renderer samples from, the
 * index the shader looks a cell's tile up in, FITS tile parsing and mip
 * averaging.
 *
 * Twins: healpixXyf ↔ sky_shader.js / allsky/healpix_faces.py;
 * parseFitsTile's quarter turn ↔ allsky/hips.py::raster_from_tile;
 * orderOffset / morton ↔ the shader's tileIndex lookup.
 */

// Side of a HEALPix cell at nside 1: sqrt(4π / 12) radians.
export const CELL_ANGLE_NSIDE1 = Math.sqrt(4 * Math.PI / 12);
const DIR_SPAN = 10000;

export function tileKey(order, npix) { return order * 2 ** 32 + npix; }
export function keyOrder(key) { return Math.floor(key / 2 ** 32); }
export function keyNpix(key) { return key % 2 ** 32; }

/** The HiPS path of a tile: `Norder<k>/Dir<10000·⌊n/10000⌋>/Npix<n>.fits`. */
export function tilePath(order, npix) {
    return `Norder${order}/Dir${Math.floor(npix / DIR_SPAN) * DIR_SPAN}/Npix${npix}.fits`;
}

/** Index entries of the orders below `order`: 12·(4^order − 1)/3. */
export function orderOffset(order) { return 4 * (4 ** order - 1); }
export function indexSize(maxOrder) { return orderOffset(maxOrder + 1); }

function spread(x) {
    let v = x & 0xFFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v >>> 0;
}
/** Morton index of (x, y): x on the even bits, y on the odd — HEALPix NESTED. */
export function morton(x, y) { return (spread(x) | (spread(y) << 1)) >>> 0; }

/** NESTED index of the order-`order` cell holding raster point (fx, fy) of `face`. */
export function tileIndexOf(face, fx, fy, order) {
    const n = 1 << order;
    const tx = Math.min(n - 1, Math.floor(fx * n));
    const ty = Math.min(n - 1, Math.floor(fy * n));
    return face * n * n + morton(tx, ty);
}

/** HEALPix [face, x, y], x and y in [0, 1), of a unit vector in the map
 * frame: ang2pix_nest in its real-valued form. */
export function healpixXyf(v) {
    const z = v[2];
    const za = Math.abs(z);
    let phi = Math.atan2(v[1], v[0]);
    if (phi < 0) phi += 2 * Math.PI;
    const tt = phi * (2 / Math.PI);
    if (za <= 2 / 3) {
        const t1 = 0.5 + tt, t2 = 0.75 * z;
        const jp = t1 - t2, jm = t1 + t2;
        const ifp = Math.floor(jp), ifm = Math.floor(jm);
        let face;
        if (ifp === ifm) face = ifp === 4 ? 4 : ifp + 4;
        else if (ifp < ifm) face = ifp;
        else face = ifm + 8;
        return [face, jm - ifm, 1 - (jp - ifp)];
    }
    const ntt = Math.min(3, Math.floor(tt));
    const tp = tt - ntt;
    const tmp = Math.sqrt(3 * (1 - za));
    const jpp = tp * tmp, jmp = (1 - tp) * tmp;
    return z >= 0 ? [ntt, 1 - jmp, 1 - jpp] : [ntt + 8, jpp, jmp];
}

/** The (continuous) HEALPix order whose cells are `pixAngle` radians wide. */
export function pixelOrderOf(pixAngle) {
    return Math.log2(CELL_ANGLE_NSIDE1 / Math.max(pixAngle, 1e-12));
}
/** The tile order whose pixels are at least as fine as a screen pixel of
 * `pixAngle` radians, within the tree. */
export function wantedOrder(pixAngle, tileShift, maxOrder) {
    return Math.max(0, Math.min(maxOrder, Math.ceil(pixelOrderOf(pixAngle) - tileShift)));
}

/** The tiles a view shows, nearest the screen centre first: the screen is
 * sampled every `step` px, which finds every tile because a tile at its
 * wanted order spans 256–512 device px. `dirAt(sx, sy)` is the map-frame
 * unit vector under a screen point (null off the sky), `pixAngleAt(sx, sy)`
 * the radians one device pixel spans there (null where unknowable). */
export function neededTiles({ screenW, screenH, step, dirAt, pixAngleAt, tileShift, maxOrder }) {
    const found = new Map();
    const cx = screenW / 2, cy = screenH / 2;
    for (let sy = step / 2; sy < screenH + step / 2; sy += step) {
        for (let sx = step / 2; sx < screenW + step / 2; sx += step) {
            const x = Math.min(sx, screenW - 0.5), y = Math.min(sy, screenH - 0.5);
            const v = dirAt(x, y);
            const pa = v && pixAngleAt(x, y);
            if (!pa) continue;
            const order = wantedOrder(pa, tileShift, maxOrder);
            const [face, fx, fy] = healpixXyf(v);
            const npix = tileIndexOf(face, fx, fy, order);
            const key = tileKey(order, npix);
            const dist = Math.hypot(x - cx, y - cy);
            const prev = found.get(key);
            if (!prev || dist < prev.dist) found.set(key, { order, npix, dist });
        }
    }
    return [...found.values()].sort((a, b) => a.dist - b.dist);
}

/** A FITS tile (one HDU, BITPIX −32) as the face raster. HiPS stores a tile
 * turned a quarter turn: FITS (column c, row r) is raster (x = W−1−r, y = c). */
export function parseFitsTile(buf) {
    const bytes = new Uint8Array(buf);
    let width = 0, height = 0, bitpix = 0, bzero = 0, bscale = 1, pos = 0;
    for (;;) {
        if (pos + 80 > bytes.length) throw new Error('FITS tile: header without END');
        const card = String.fromCharCode(...bytes.subarray(pos, pos + 80));
        pos += 80;
        const key = card.slice(0, 8).trim();
        const val = card.slice(10).split('/')[0].trim();
        if (key === 'END') break;
        if (key === 'NAXIS1') width = +val;
        else if (key === 'NAXIS2') height = +val;
        else if (key === 'BITPIX') bitpix = +val;
        else if (key === 'BZERO') bzero = +val;
        else if (key === 'BSCALE') bscale = +val;
    }
    if (bitpix !== -32 || width !== height || !width) {
        throw new Error(`FITS tile: unsupported BITPIX ${bitpix}, ${width}×${height}`);
    }
    pos = Math.ceil(pos / 2880) * 2880;
    const dv = new DataView(buf, pos, width * width * 4);
    const out = new Float32Array(width * width);
    for (let r = 0; r < width; r++) {
        for (let c = 0; c < width; c++) {
            const v = dv.getFloat32(4 * (r * width + c), false) * bscale + bzero;
            out[c * width + (width - 1 - r)] = Number.isFinite(v) ? v : 0;
        }
    }
    return out;
}

/** The next-coarser raster of an n×n tile: every 2 × 2 block averaged.
 * HEALPix cells are equal-area, so the plain mean IS the area-weighted mean. */
export function degradeTile(data, n) {
    const m = n >> 1;
    const out = new Float32Array(m * m);
    for (let y = 0; y < m; y++) {
        const r0 = 2 * y * n, r1 = r0 + n, o = y * m;
        for (let x = 0; x < m; x++) {
            const c = 2 * x;
            out[o + x] = 0.25 * (data[r0 + c] + data[r0 + c + 1] + data[r1 + c] + data[r1 + c + 1]);
        }
    }
    return out;
}

/** Which tile sits in which atlas slot. When full, the tile least recently
 * needed goes — never a pinned one, never one the current view needs. */
export class TileAtlas {
    constructor(slots) {
        this.free = Array.from({ length: slots }, (_, i) => slots - 1 - i);   // pop() hands out 0 first
        this.slotOf = new Map();
        this.lastUse = new Map();
        this.pinned = new Set();
    }
    get size() { return this.slotOf.size; }
    has(key) { return this.slotOf.has(key); }
    touch(key, stamp) { if (this.slotOf.has(key)) this.lastUse.set(key, stamp); }
    /** `{ slot, evicted }`: the slot now holding `key` (−1 when every resident
     * tile is pinned or needed) and the key that left for it, if any. */
    acquire(key, { pinned = false, needed = new Set(), stamp = 0 } = {}) {
        if (this.slotOf.has(key)) return { slot: this.slotOf.get(key), evicted: null };
        let evicted = null;
        if (!this.free.length) {
            let oldest = Infinity;
            for (const [k, t] of this.lastUse) {
                if (t < oldest && !this.pinned.has(k) && !needed.has(k)) { evicted = k; oldest = t; }
            }
            if (evicted === null) return { slot: -1, evicted };
            this.release(evicted);
        }
        const slot = this.free.pop();
        this.slotOf.set(key, slot);
        this.lastUse.set(key, stamp);
        if (pinned) this.pinned.add(key);
        return { slot, evicted };
    }
    release(key) {
        const slot = this.slotOf.get(key);
        if (slot === undefined) return;
        this.slotOf.delete(key);
        this.lastUse.delete(key);
        this.pinned.delete(key);
        this.free.push(slot);
    }
}
