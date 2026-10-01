/**
 * Tile-pyramid fetcher for the WebGPU tile viewer.
 *
 * Pixi-free re-implementation of `../linear_tile_loader.js`. The fetch
 * + brotli-decode path is identical (the browser handles
 * `Content-Encoding: br` transparently); only the upload helpers
 * differ — there's no `PIXI.BufferResource` / `BaseTexture`, just a
 * raw float16 Uint16Array that the WebGPU renderer uploads via
 * `device.queue.writeTexture` when it's ready.
 */

const TILE_SIZE = 256;


/**
 * Fetch one tile of EVERY finish input in one request. The body is the
 * inputs' float16 tiles concatenated in input order; `X-Tile-Bands` lists each
 * input's band count (3 colour, 1 mono, 0 = not served). Returns an array with
 * one { array: Uint16Array of IEEE half bits, width, height, bands } or null
 * per input, or null for a 404 (off the edge of the image).
 */
export async function fetchTileInputs(url, signal) {
    const resp = await fetch(url, { signal });
    if (!resp.ok) {
        if (resp.status === 404) return null;
        throw new Error(`tile fetch failed: ${resp.status} ${resp.statusText}`);
    }
    const buffer = await resp.arrayBuffer();
    const width = parseInt(resp.headers.get("X-Tile-Width") || TILE_SIZE, 10);
    const height = parseInt(resp.headers.get("X-Tile-Height") || TILE_SIZE, 10);
    const bandList = (resp.headers.get("X-Tile-Bands") || "1").split(",").map(Number);
    const out = [];
    let offset = 0;
    for (const bands of bandList) {
        if (!bands) { out.push(null); continue; }
        const n = width * height * bands;
        out.push({ array: new Uint16Array(buffer, offset, n), width, height, bands });
        offset += n * 2;
    }
    return out;
}
