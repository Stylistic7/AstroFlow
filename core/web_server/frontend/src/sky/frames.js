/**
 * Sky frames. The viewer always DISPLAYS in Galactic coordinates — the
 * Milky Way across the middle, l = 0 at the centre of the fit view —
 * whatever frame a map's cells are laid out in: a map in another frame gets
 * one constant rotation folded into the view's (the shader's `rot` is
 * mapFromDisplay · R). ICRS ↔ Galactic is the IAU 1958 definition carried
 * to J2000, the matrix astropy uses (pinned in frames.test.js).
 */
import { apply } from './sky_projection.js';

export const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// Row-major: ICRS x, y, z in terms of Galactic x, y, z.
export const ICRS_FROM_GALACTIC = [
    -0.0548756577125922, 0.4941094371927267, -0.8676661375596585,
    -0.8734370519556160, -0.4448297212232957, -0.1980763372730006,
    -0.4838350736167157, 0.7469821839866677, 0.4559838136873021,
];

/** Display (Galactic) → the map's frame. */
export function mapFromDisplay(frame) {
    if (frame === 'galactic') return IDENTITY;
    if (frame === 'icrs') return ICRS_FROM_GALACTIC;
    throw new Error(`unknown sky frame ${frame}`);
}

/** Row-major 3×3 product a · b. */
export function multiply(a, b) {
    const out = new Array(9);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            out[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
        }
    }
    return out;
}

const DEG = Math.PI / 180;

/** [RA, Dec] in degrees (RA in [0, 360)) of a Galactic unit vector. */
export function radecOf(vGal) {
    const v = apply(ICRS_FROM_GALACTIC, vGal);
    const ra = Math.atan2(v[1], v[0]) / DEG;
    return [(ra + 360) % 360, Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG];
}

const GALACTIC_FROM_ICRS = [0, 3, 6, 1, 4, 7, 2, 5, 8].map((i) => ICRS_FROM_GALACTIC[i]);

/** [l, b] in degrees (l in [0, 360)) of an ICRS position in degrees. */
export function galacticOf(raDeg, decDeg) {
    const c = Math.cos(decDeg * DEG);
    const v = apply(GALACTIC_FROM_ICRS, [c * Math.cos(raDeg * DEG), c * Math.sin(raDeg * DEG), Math.sin(decDeg * DEG)]);
    const l = Math.atan2(v[1], v[0]) / DEG;
    return [(l + 360) % 360, Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG];
}
