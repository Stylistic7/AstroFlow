/**
 * Custom WebGL shader for displaying linear-float tiles with
 * client-side tone mapping. One operator unifies the global stretch,
 * local-context-aware adaptation, and HDR brightness output — all
 * driven by a small set of uniforms.
 *
 *   y = LUT(mix(x, x - c + 0.5, uLocalMix)) * uPeakBrightness
 *
 * where:
 *   x          = the per-pixel linear value sampled from uTileTexture
 *   c          = the local context value sampled from uContextMap at
 *                the same world position (with auto-mipmap LOD chosen
 *                by the local-scale slider)
 *   uLocalMix  = naturalistic↔painterly axis. 0 = pure global LUT
 *                lookup; 1 = full local context adaptation.
 *   LUT        = the 1-D lookup table the auto-stretch optimizer
 *                produced from ImageStatistics. Indexed by the adapted
 *                value, returns the display RGB.
 *   uPeakBrightness = 1.0 in SDR mode; up to ~12.0 in HDR mode (the
 *                HLG-encoded canvas maps anything > 1.0 to extended
 *                display brightness).
 *
 * The whole operator is one fragment-shader execution — no separate
 * passes, no intermediate textures, no CPU work. Sliders update by
 * recomputing the LUT (or the uniforms) and re-rendering visible tiles.
 *
 * Shader source is plain GLSL ES 1.00 (WebGL 1 compatible) for the
 * widest browser support; conditional WebGL 2 features can be added
 * later via the `precision` / `version` declarations.
 */
import * as PIXI from 'pixi.js';


// ─── Vertex shader ──────────────────────────────────────────────────
//
// Standard pass-through over a unit-quad mesh. PIXI provides the
// projectionMatrix uniform; we just multiply through and pass the UV
// to the fragment stage.
//
const VERTEX_SHADER = `
precision highp float;

attribute vec2 aVertexPosition;
attribute vec2 aTextureCoord;

uniform mat3 projectionMatrix;
uniform mat3 translationMatrix;

varying vec2 vTextureCoord;

void main() {
    gl_Position = vec4(
        (projectionMatrix * translationMatrix * vec3(aVertexPosition, 1.0)).xy,
        0.0, 1.0
    );
    vTextureCoord = aTextureCoord;
}
`;


// ─── Fragment shader ────────────────────────────────────────────────
//
// The tone-mapping operator. Reads uTileTexture (linear float per
// pixel), optionally reads uContextMap at the world-projected
// position for local adaptation, applies the LUT, and outputs to the
// HDR/SDR canvas.
//
// uLutTexture is a 256-wide 1-row RGBA texture; we sample at
// (adaptedValue, 0.5) to get the display RGB. The auto-stretch
// optimizer fills the LUT from ImageStatistics; sliders adjust the
// LUT shape.
//
// uContextMap is the per-image context map (1:5 ratio, float16). When
// it's present, the local-context sample comes from it. When the
// shader is told there is no context map (uHasContext = 0), local
// adaptation falls back to a downsampled sample of the tile itself
// (single-tile small-image case).
//
const FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uTileTexture;      // R32F, this tile's linear values
uniform sampler2D uLutTexture;       // RGBA, 1D LUT (256×1)
uniform sampler2D uContextMap;       // R16F (HALF_FLOAT), whole-image context

uniform vec2  uTileWorldOrigin;      // tile's top-left in world coords
uniform vec2  uTileWorldSize;        // tile's bbox in world coords
uniform vec2  uWorldSize;            // total image size, world coords

uniform float uLocalMix;             // 0 = global, 1 = full-local
uniform float uPeakBrightness;       // 1.0 SDR; up to ~12.0 HDR
uniform float uHasContext;           // 1.0 if context map bound, else 0.0

varying vec2 vTextureCoord;

void main() {
    // Sample the linear pixel value from this tile's texture. R32F
    // gives us full float32 precision back from the GPU.
    float x = texture2D(uTileTexture, vTextureCoord).r;

    // Local context. When the context map is bound, sample at this
    // pixel's world coordinates (the context map IS the whole image
    // at 1:5). When it isn't, the local context degenerates to the
    // pixel itself, making the local-adaptation step a no-op and the
    // operator collapse to a pure global LUT lookup.
    float c = x;
    if (uHasContext > 0.5) {
        vec2 worldPos = uTileWorldOrigin + vTextureCoord * uTileWorldSize;
        vec2 contextUV = worldPos / uWorldSize;
        c = texture2D(uContextMap, contextUV).r;
    }

    // Adapted input for the LUT lookup. At uLocalMix = 0 this is
    // the raw pixel value (pure global stretch). At uLocalMix = 1
    // this subtracts the local context and recenters at 0.5,
    // mapping local CONTRAST onto the LUT instead of absolute
    // brightness — the painterly look. Smooth blend in between.
    float adapted = mix(x, x - c + 0.5, uLocalMix);
    adapted = clamp(adapted, 0.0, 1.0);

    // LUT lookup. The LUT is computed CPU-side from ImageStatistics
    // (auto-stretch) plus user slider input (painterly bias). One
    // texture sample per pixel.
    vec3 mapped = texture2D(uLutTexture, vec2(adapted, 0.5)).rgb;

    // HDR scaling. Output is allowed to exceed 1.0 on HDR canvases;
    // the browser composites those values at extended display
    // brightness on capable hardware. On SDR canvases the values
    // clamp at 1.0, which is also correct.
    gl_FragColor = vec4(mapped * uPeakBrightness, 1.0);
}
`;


/**
 * Build a PIXI.Shader instance with our vertex+fragment pair and a
 * default uniform set. The caller updates uniforms per tile (the
 * world-coordinate ones change between tiles) and on slider changes
 * (the LUT and the mix/brightness uniforms).
 */
export function createToneMappingShader() {
    return PIXI.Shader.from(VERTEX_SHADER, FRAGMENT_SHADER, {
        uTileTexture: PIXI.Texture.WHITE,
        uLutTexture: PIXI.Texture.WHITE,
        uContextMap: PIXI.Texture.WHITE,
        uTileWorldOrigin: new Float32Array([0, 0]),
        uTileWorldSize: new Float32Array([1, 1]),
        uWorldSize: new Float32Array([1, 1]),
        uLocalMix: 0.0,
        uPeakBrightness: 1.0,
        uHasContext: 0.0,
    });
}


/**
 * Build a 256×1 RGBA32F LUT texture. The default LUT is a straight
 * identity (linear input → linear output) so the very first frame
 * after image load looks "raw but correct" — the auto-stretch
 * optimizer replaces this with the real stretch as soon as the image
 * statistics arrive.
 *
 * Returns a PIXI.Texture that the shader binds as uLutTexture.
 */
export function createIdentityLut() {
    const SIZE = 256;
    const data = new Float32Array(SIZE * 4);
    for (let i = 0; i < SIZE; i++) {
        const v = i / (SIZE - 1);
        data[i * 4 + 0] = v;
        data[i * 4 + 1] = v;
        data[i * 4 + 2] = v;
        data[i * 4 + 3] = 1.0;
    }
    const resource = new PIXI.BufferResource(data, { width: SIZE, height: 1 });
    const baseTexture = new PIXI.BaseTexture(resource, {
        format: PIXI.FORMATS.RGBA,
        type: PIXI.TYPES.FLOAT,
        target: PIXI.TARGETS.TEXTURE_2D,
        mipmap: PIXI.MIPMAP_MODES.OFF,
        scaleMode: PIXI.SCALE_MODES.LINEAR,  // smooth LUT sampling
        alphaMode: PIXI.ALPHA_MODES.NO_PREMULTIPLIED_ALPHA,
    });
    return new PIXI.Texture(baseTexture);
}


/**
 * Update a LUT texture in place with new RGBA values. Cheap — copies
 * 1 KB per slider change, then a single texSubImage2D upload (<0.1
 * ms). Use this from the auto-stretch optimizer / slider handler.
 */
export function updateLutTexture(texture, rgbaData) {
    const resource = texture.baseTexture.resource;
    if (resource instanceof PIXI.BufferResource) {
        // Reuse the same buffer if the size matches; otherwise the
        // caller should rebuild the texture.
        if (resource.data.length === rgbaData.length) {
            resource.data.set(rgbaData);
            texture.baseTexture.update();
            return;
        }
    }
    // Fallback: rebuild. Caller's reference is still valid.
    const fresh = createLutFromRgba(rgbaData);
    texture.baseTexture.destroy();
    Object.assign(texture, fresh);
}


function createLutFromRgba(data) {
    const SIZE = data.length / 4;
    const resource = new PIXI.BufferResource(data, { width: SIZE, height: 1 });
    const baseTexture = new PIXI.BaseTexture(resource, {
        format: PIXI.FORMATS.RGBA,
        type: PIXI.TYPES.FLOAT,
        target: PIXI.TARGETS.TEXTURE_2D,
        mipmap: PIXI.MIPMAP_MODES.OFF,
        scaleMode: PIXI.SCALE_MODES.LINEAR,
        alphaMode: PIXI.ALPHA_MODES.NO_PREMULTIPLIED_ALPHA,
    });
    return new PIXI.Texture(baseTexture);
}
