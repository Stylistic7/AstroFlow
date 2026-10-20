/**
 * Perceptual colour maths for the finish colour picker (2026-10-03).
 *
 * A finish's `Color` knob is a DISPLAY sRGB triple at FULL VALUE: its largest
 * component is 1, because the image data and the stretch supply brightness
 * (an emission line's colour multiplies its stretched intensity). The picker
 * therefore chooses only two things, both in OKLab (Björn Ottosson) so equal
 * steps look like equal changes:
 *
 *   hue — the OKLCh hue angle, in degrees;
 *   sat — 0..1, the OKLab mix from white (0) to the most saturated colour of
 *         that hue that sRGB can show (1).
 *
 * Every colour is then scaled so its largest linear component is 1. Scaling
 * linear RGB scales OKLab's L, a and b alike, so hue and the chroma/lightness
 * ratio survive the scaling — which is what makes `toHueSat` an exact
 * inverse of `colorAt`.
 */

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toDisplay = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** Display sRGB → OKLab [L, a, b]. */
export function srgbToOklab([r, g, b]) {
    const R = toLinear(r);
    const G = toLinear(g);
    const B = toLinear(b);
    const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
    const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
    const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ];
}

/** OKLab → LINEAR sRGB (may fall outside [0, 1]). */
function oklabToLinear([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
    return [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ];
}

const hueOf = ([, a, b]) => ((Math.atan2(b, a) * 180) / Math.PI + 360) % 360;

// The edge of the sRGB gamut at full value: every colour with one component
// 1 and one 0, i.e. HSV hue 0..360 at S = V = 1, tabulated with its OKLab.
// A colour of a given OKLCh hue at maximum saturation is the table entry of
// nearest hue (0.25° HSV steps keep the error far below a visible shift).
const EDGE = (() => {
    const out = [];
    for (let t = 0; t < 360; t += 0.25) {
        const h = t / 60;
        const rgb = [
            Math.min(1, Math.max(0, Math.abs(h - 3) - 1)),
            Math.min(1, Math.max(0, 2 - Math.abs(h - 2))),
            Math.min(1, Math.max(0, 2 - Math.abs(h - 4))),
        ];
        const lab = srgbToOklab(rgb);
        out.push({ hue: hueOf(lab), lab });
    }
    return out;
})();

function edgeLab(hue) {
    let best = EDGE[0];
    let bestD = Infinity;
    for (const e of EDGE) {
        const d = Math.abs(((e.hue - hue + 540) % 360) - 180);
        if (d < bestD) { bestD = d; best = e; }
    }
    return best.lab;
}

/** Display sRGB at full value for an OKLCh hue (degrees) and sat (0..1). */
export function colorAt(hue, sat) {
    const p = edgeLab(hue);
    const lab = [1 + sat * (p[0] - 1), sat * p[1], sat * p[2]];
    const lin = oklabToLinear(lab).map((c) => Math.max(0, c));
    const top = Math.max(...lin, 1e-9);
    return lin.map((c) => Math.min(1, toDisplay(c / top)));
}

/** The inverse of `colorAt`: { hue, sat } for a display sRGB colour. A grey
 *  has no hue, so `fallbackHue` (the picker's current one) is kept. */
export function toHueSat(rgb, fallbackHue = 0) {
    const lab = srgbToOklab(rgb);
    const chroma = Math.hypot(lab[1], lab[2]);
    if (chroma < 1e-5 || lab[0] <= 0) return { hue: fallbackHue, sat: 0 };
    const hue = hueOf(lab);
    const p = edgeLab(hue);
    const ratio = chroma / lab[0];
    const sat = ratio / (Math.hypot(p[1], p[2]) - ratio * (p[0] - 1));
    return { hue, sat: Math.min(1, Math.max(0, sat)) };
}

export const toHex = (rgb) => `#${rgb.map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255)
    .toString(16).padStart(2, '0')).join('')}`;

// Where each common emission line's own colour sits on the hue axis: its
// wavelength's colour clipped into sRGB (Bruton's spectral approximation),
// so the reds (Hα 656 nm, NII 658 nm, SII 672 nm) all land on sRGB red.
export const LINE_MARKERS = [
    { label: 'Hα · NII · SII', hue: hueOf(srgbToOklab([1, 0, 0])) },
    { label: 'OIII', hue: hueOf(srgbToOklab([0, 1, 0.465])) },
    { label: 'Hβ', hue: hueOf(srgbToOklab([0, 0.922, 1])) },
];
