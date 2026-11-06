/**
 * Streams a map's tiles into the renderer's atlas. On every view change the
 * tiles the screen shows (sky_tiles.neededTiles) are fetched nearest the
 * centre first, a bounded number at a time, and uploaded as they land. The
 * twelve order-0 tiles are pinned so every pixel has something to fall
 * back on, and the least recently needed tiles leave when the atlas is
 * full. A tile the tree lacks (404) is remembered as absent — the shader
 * keeps showing the coarser one — and a failed fetch is retried after a
 * pause. Remote trees (a HiPS URL in `meta.tiles`) are read straight from
 * their server as FITS; this app's maps come from its tile route as raw
 * float32.
 */
import { apply, nativeAt, skyAt } from './sky_projection.js';
import { TileAtlas, keyNpix, keyOrder, neededTiles, parseFitsTile, tileKey, tilePath } from './sky_tiles.js';
import { mapFromDisplay } from './frames.js';

const SAMPLE_STEP = 64;           // CSS px between screen samples
const RETRY_MS = 10000;
const ORDER0 = Array.from({ length: 12 }, (_, npix) => ({ order: 0, npix }));

export function makeTileLoader({ renderer, meta, slots }) {
    const atlas = new TileAtlas(slots);
    const M = mapFromDisplay(meta.frame);
    const tileShift = Math.log2(meta.tileWidth);
    const remote = Boolean(meta.tiles);
    // CDS answers sixteen at a time over HTTP/2; this app has four request threads.
    const maxInflight = remote ? 16 : 4;
    const inflight = new Map();
    const absent = new Set();
    const failedAt = new Map();
    let needed = new Map();
    let queue = [];
    let stamp = 0;
    let disposed = false;

    const url = (order, npix) => (remote
        ? `${meta.tiles}/${tilePath(order, npix)}`
        : `/api/healpix/${encodeURIComponent(meta.name)}/tile/${order}/${npix}.f32?v=${meta.mtime}`);

    function place(key, order, npix, data) {
        const { slot, evicted } = atlas.acquire(key, {
            pinned: order === 0, needed: new Set(needed.keys()), stamp,
        });
        if (slot < 0) return;                 // every resident tile is on screen; the next view retries
        if (evicted !== null) renderer.setIndex(keyOrder(evicted), keyNpix(evicted), -1);
        renderer.uploadTile(slot, data);
        renderer.setIndex(order, npix, slot);
        renderer.redraw();
    }

    async function load(order, npix) {
        const key = tileKey(order, npix);
        const ctl = new AbortController();
        inflight.set(key, ctl);
        try {
            const r = await fetch(url(order, npix), { signal: ctl.signal });
            if (r.status === 404) {
                absent.add(key);
                return;
            }
            if (!r.ok) throw new Error(`tile ${order}/${npix}: HTTP ${r.status}`);
            const buf = await r.arrayBuffer();
            if (disposed) return;
            place(key, order, npix, remote ? parseFitsTile(buf) : new Float32Array(buf));
        } catch (e) {
            if (e?.name !== 'AbortError') {
                console.warn(e);
                failedAt.set(key, performance.now());
            }
        } finally {
            inflight.delete(key);
            pump();
        }
    }

    function pump() {
        while (!disposed && inflight.size < maxInflight && queue.length) {
            const { order, npix } = queue.shift();
            const key = tileKey(order, npix);
            if (atlas.has(key) || inflight.has(key) || absent.has(key)) continue;
            const failed = failedAt.get(key);
            if (failed !== undefined && performance.now() - failed < RETRY_MS) continue;
            load(order, npix);
        }
    }

    /** The view changed: `s` is the viewport state (CSS px), `dpr` the
     * device pixel ratio the shader's pixel angles are in. */
    function update(s, dpr) {
        if (disposed || !s.screenW || !s.screenH) return;
        stamp += 1;
        const dirAt = (x, y) => {
            const v = skyAt(x, y, s);
            return v && apply(M, v);
        };
        const pixAngleAt = (x, y) => {
            const a = nativeAt(x, y, s);
            const b = nativeAt(x + 1, y, s) || nativeAt(x - 1, y, s);
            const c = nativeAt(x, y + 1, s) || nativeAt(x, y - 1, s);
            if (!a || !b || !c) return null;
            const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
            return Math.max(d(a, b), d(a, c)) / dpr;
        };
        const tiles = neededTiles({
            screenW: s.screenW, screenH: s.screenH, step: SAMPLE_STEP,
            dirAt, pixAngleAt, tileShift, maxOrder: meta.maxOrder,
        });
        needed = new Map(tiles.map((t) => [tileKey(t.order, t.npix), t]));
        for (const key of needed.keys()) atlas.touch(key, stamp);
        queue = [...ORDER0, ...tiles];
        pump();
    }

    function dispose() {
        disposed = true;
        for (const ctl of inflight.values()) ctl.abort();
    }

    return { update, dispose };
}
