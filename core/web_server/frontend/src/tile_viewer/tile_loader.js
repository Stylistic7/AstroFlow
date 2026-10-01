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
 * Fetch a tile as IEEE float16 bit patterns (a Uint16Array, uploaded as-is to
 * an r16float / rgba16float texture). Returns { array, width, height, bands }
 * or null for 404 (off the edge of the image). `bands` is 1 for a mono
 * channel or 3 for RGB; the array is band-interleaved (RGBRGB…).
 */
export async function fetchLinearTile(url, signal) {
    const resp = await fetch(url, { signal });
    if (!resp.ok) {
        if (resp.status === 404) return null;
        throw new Error(`tile fetch failed: ${resp.status} ${resp.statusText}`);
    }
    const buffer = await resp.arrayBuffer();
    const width = parseInt(resp.headers.get("X-Tile-Width") || TILE_SIZE, 10);
    const height = parseInt(resp.headers.get("X-Tile-Height") || TILE_SIZE, 10);
    const bands = parseInt(resp.headers.get("X-Tile-Bands") || "1", 10);
    return {
        array: new Uint16Array(buffer),
        width,
        height,
        bands,
    };
}
