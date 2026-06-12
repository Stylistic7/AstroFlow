/**
 * Loaders for the linear-float tile pyramid + context map.
 *
 * Replaces the previous PNG-fetch / browser-image-decode path with a
 * fetch-as-ArrayBuffer + Float32Array path that hands raw linear
 * pixel data straight to the GPU. The brotli decompression happens
 * transparently on the network thread (browser handles the
 * `Content-Encoding: br` response header automatically).
 *
 * Two artifacts per image:
 *   * Tiles — `/tap/tile/.../{z}/{x}/{y}.f32` — raw float32 little-
 *     endian, brotli-compressed in transit. One fetch per visible
 *     tile per zoom level. Uploaded to a WebGL R32F texture.
 *   * Context map — `/tap/context_map` — a single sibling artifact,
 *     precompressed brotli + magic header + float16 pixels. Fetched
 *     once per image open. Uploaded to a WebGL R16F (HALF_FLOAT)
 *     texture; mipmaps generated on the GPU for multi-scale local
 *     tone-mapping context.
 *
 * The shader (see `tone_mapping_shader.js`) reads from both textures
 * to produce the final displayed pixel.
 */
import * as PIXI from 'pixi.js';


const _CONTEXT_MAGIC_BYTES = new TextEncoder().encode("CMv1");
const TILE_SIZE = 256;


/**
 * Fetch a tile as a Float32Array. Returns { array, width, height }.
 *
 * The browser's fetch+Content-Encoding pipeline handles the brotli
 * decompression for us; what we see is the raw float32 little-endian
 * bytes the server packed from `large_image`. We slice off the
 * server-supplied tile dimensions from the response headers so the
 * caller doesn't need to know the pyramid metadata.
 *
 * Returns null if the tile is missing (404). The caller should treat
 * that as "off the edge of the image" and skip rendering it.
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
    // The fetch already decompressed the brotli; what's in `buffer` is
    // raw little-endian float32. Float32Array reads in native byte
    // order (little-endian on every device the browser runs on; we
    // serve LE bytes regardless), so this is a pointer cast, no copy.
    return {
        array: new Float32Array(buffer),
        width,
        height,
    };
}


/**
 * Fetch the per-image context map and parse the magic + dims header.
 *
 * File layout after the browser's brotli decompression:
 *   offset  0: magic   = ASCII "CMv1" (4 bytes)
 *   offset  4: width   = little-endian uint32 (4 bytes)
 *   offset  8: height  = little-endian uint32 (4 bytes)
 *   offset 12: pixels  = width × height × float16 (2 bytes each), row-major
 *
 * Returns { array, width, height } where `array` is a Uint16Array view
 * of the float16 bytes (WebGL accepts these directly when paired with
 * the HALF_FLOAT type). Returns null on 404 — for very small images
 * the server doesn't generate a context map and the shader falls back
 * to sampling tile mipmaps.
 */
export async function fetchContextMap(url) {
    const resp = await fetch(url);
    if (resp.status === 404) return null;
    if (!resp.ok) {
        throw new Error(`context map fetch failed: ${resp.status} ${resp.statusText}`);
    }
    const buffer = await resp.arrayBuffer();
    const bytes = new Uint8Array(buffer);

    // Magic check — defensive against future format changes.
    for (let i = 0; i < _CONTEXT_MAGIC_BYTES.length; i++) {
        if (bytes[i] !== _CONTEXT_MAGIC_BYTES[i]) {
            throw new Error(
                `context map magic header mismatch at byte ${i}: ` +
                `expected ${_CONTEXT_MAGIC_BYTES[i]}, got ${bytes[i]}`
            );
        }
    }
    const dv = new DataView(buffer);
    const width = dv.getUint32(4, /*littleEndian=*/ true);
    const height = dv.getUint32(8, /*littleEndian=*/ true);
    // The float16 pixels start at byte 12. We expose them as Uint16Array
    // because WebGL's texImage2D wants the raw 16-bit container; the
    // GPU interprets them as HALF_FLOAT when the texture's pixel type
    // is set to gl.HALF_FLOAT.
    const pixels = new Uint16Array(buffer, 12, width * height);
    return { array: pixels, width, height };
}


/**
 * Upload a Float32Array to a WebGL R32F texture wrapped in a
 * PIXI.BaseTexture / PIXI.Texture pair.
 *
 * R32F is a single-channel (red) 32-bit float texture format,
 * available on every WebGL 2 context. The shader reads pixel values
 * via `texture2D(uTileTexture, vUV).r`.
 *
 * NB: PIXI 7's BufferResource handles the upload bookkeeping for us
 * but it defaults to FORMATS.RGBA / TYPES.UNSIGNED_BYTE. We override
 * via the BaseTexture options so the upload writes the float values
 * with the correct precision.
 */
export function uploadFloat32Tile(array, width, height) {
    const resource = new PIXI.BufferResource(array, { width, height });
    const baseTexture = new PIXI.BaseTexture(resource, {
        format: PIXI.FORMATS.RED,
        type: PIXI.TYPES.FLOAT,
        target: PIXI.TARGETS.TEXTURE_2D,
        // No mipmap on individual tiles — at the tile scale we're
        // rendering at native pixel resolution and don't sample at
        // lower LODs.
        mipmap: PIXI.MIPMAP_MODES.OFF,
        // Linear filtering inside the tile gives smooth zoom; nearest
        // would show pixel edges at non-1:1 viewport scales.
        scaleMode: PIXI.SCALE_MODES.LINEAR,
        // Float textures don't pre-multiply alpha (alpha is implicit
        // 1.0 for our luminance data).
        alphaMode: PIXI.ALPHA_MODES.NO_PREMULTIPLIED_ALPHA,
    });
    return new PIXI.Texture(baseTexture);
}


/**
 * Upload a Uint16Array (containing float16 bytes) to a WebGL R16F
 * texture for the context map. Generates mipmaps automatically so
 * the shader can do `textureLod(uContextMap, uv, level)` to get
 * multi-scale local context out of a single binding.
 */
export function uploadContextMap(array, width, height) {
    const resource = new PIXI.BufferResource(array, { width, height });
    const baseTexture = new PIXI.BaseTexture(resource, {
        format: PIXI.FORMATS.RED,
        // HALF_FLOAT tells WebGL to interpret each pair of bytes in
        // the Uint16Array as one float16. The shader sees normal
        // floats via texture2D — the precision conversion is free
        // on the GPU.
        type: PIXI.TYPES.HALF_FLOAT,
        target: PIXI.TARGETS.TEXTURE_2D,
        // Enable mipmaps so the shader has multi-scale context for
        // local tone mapping. POW2 is the strictest scaleMode-friendly
        // setting; ON is what we want for the auto-generated chain.
        mipmap: PIXI.MIPMAP_MODES.ON,
        // Trilinear filtering: bilinear in-level + smooth across LOD
        // boundaries. Critical for halo-free local tone mapping.
        scaleMode: PIXI.SCALE_MODES.LINEAR,
        alphaMode: PIXI.ALPHA_MODES.NO_PREMULTIPLIED_ALPHA,
    });
    return new PIXI.Texture(baseTexture);
}
