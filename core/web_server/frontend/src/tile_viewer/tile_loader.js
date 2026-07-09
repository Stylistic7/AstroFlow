/**
 * Tile-pyramid fetcher for the WebGPU tile viewer.
 *
 * Pixi-free re-implementation of `../linear_tile_loader.js`. The fetch
 * + brotli-decode path is identical (the browser handles
 * `Content-Encoding: br` transparently); only the upload helpers
 * differ — there's no `PIXI.BufferResource` / `BaseTexture`, just a
 * raw Float32Array that the WebGPU renderer uploads via
 * `device.queue.writeTexture` when it's ready.
 */

const TILE_SIZE = 256;


/**
 * Fetch a tile as a Float32Array. Returns { array, width, height, bands }
 * or null for 404 (off the edge of the image). `bands` is 1 for a linear
 * mono channel (client-side tone-mapping) or 3 for a baked-stretch RGB
 * color image (pass-through); the array is band-interleaved (RGBRGB…).
 */
export async function fetchLinearTile(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
        if (resp.status === 404) return null;
        throw new Error(`tile fetch failed: ${resp.status} ${resp.statusText}`);
    }
    const buffer = await resp.arrayBuffer();
    const width = parseInt(resp.headers.get("X-Tile-Width") || TILE_SIZE, 10);
    const height = parseInt(resp.headers.get("X-Tile-Height") || TILE_SIZE, 10);
    const bands = parseInt(resp.headers.get("X-Tile-Bands") || "1", 10);
    return {
        array: new Float32Array(buffer),
        width,
        height,
        bands,
    };
}
