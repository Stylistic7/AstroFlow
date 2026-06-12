/**
 * WebGL shader for the linear-float tile viewer.
 *
 * Applies the PixInsight Auto-STF (`previews.operations.autostretch`)
 * directly per-pixel via the Midtone Transfer Function:
 *
 *     x  = xRaw / max
 *     E2 = max(0, (x − E1) / (1 − E1))
 *     y  = ((E5 − 1) · E2) / ((2·E5 − 1) · E2 − E5)
 *
 * `E1` and `E5` are derived from `ImageStatistics` in JS (see
 * `auto_stretch.js`). No LUT — the curve is computed exactly per
 * pixel, which avoids the LUT-quantization artifact where a 256-bin
 * lookup misses the narrow transition near `E1` for stacked +
 * pedestaled masters and crushes the sky pedestal to black.
 *
 * The auto-stretch is intentionally not parameterized — `stretch`,
 * `clip`, and `curve` are baked into the JS-side `E1`/`E5` derivation
 * with the same hard-coded values the Python original uses (0.20, 2.8,
 * 1.0). HDR brightness is preserved as a uniform because the HDR
 * canvas toggle needs it as a separate output multiplier.
 */
import * as PIXI from 'pixi.js';


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


const FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uTileTexture;      // R32F, this tile's linear values

uniform float uMax;                  // max pixel value over the image
                                     // (per stats.max_value)
uniform float uE1;                   // STF black point in [0, 1]
uniform float uE5;                   // STF midtone parameter m in (0, 1)
uniform float uPeakBrightness;       // 1.0 SDR; up to ~12.0 HDR

varying vec2 vTextureCoord;

void main() {
    // Sample the linear pixel value from this tile's texture (R32F
    // gives us full float32 precision back from the GPU).
    float xRaw = texture2D(uTileTexture, vTextureCoord).r;

    // Zero is the "no data" sentinel for the irregular mosaic footprint
    // inside the rectangular tile bbox. Short-circuit to opaque black
    // so the canvas border is visually correct.
    if (xRaw == 0.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Normalize-by-max — same as image/max(image) in the Python
    // original. Clamp at 1.0 so over-bright pixels saturate at the
    // top of the bracket.
    float x = clamp(xRaw / uMax, 0.0, 1.0);

    // PixInsight Auto-STF: linear remap into [E1, 1] then MTF.
    float E2 = max(0.0, (x - uE1) / max(1.0 - uE1, 1e-12));
    float num = (uE5 - 1.0) * E2;
    float den = (2.0 * uE5 - 1.0) * E2 - uE5;
    float y = abs(den) < 1e-12 ? 0.0 : num / den;
    y = clamp(y, 0.0, 1.0);

    // HDR scaling. Output is allowed to exceed 1.0 on HDR canvases;
    // the browser composites those values at extended display
    // brightness on capable hardware. On SDR canvases the values
    // clamp at 1.0, which is also correct.
    gl_FragColor = vec4(vec3(y) * uPeakBrightness, 1.0);
}
`;


export function createToneMappingShader() {
    return PIXI.Shader.from(VERTEX_SHADER, FRAGMENT_SHADER, {
        uTileTexture: PIXI.Texture.WHITE,
        uMax: 1.0,
        uE1: 0.0,
        uE5: 0.5,
        uPeakBrightness: 1.0,
    });
}
