/**
 * The sky viewer's projection math — the pointer side. The fragment shader
 * carries a line-for-line twin (sky_shader.js); change both or neither.
 *
 * The view is a Hammer–Aitoff plane (FITS 'AIT': equal-area, closed-form
 * inverse, the whole sky inside one ellipse 4√2 wide and 2√2 tall) centred
 * on a sky point (lon0, lat0) with north up, drawn at `ppu` CSS pixels per
 * plane unit; a plane unit is one radian at the centre. East is to the
 * left, as on every sky chart.
 *
 * Two frames. NATIVE: x toward the view centre, z toward north, y toward
 * east — what the plane unprojects into. MAP: the map's own frame (Galactic
 * for the Hα template), the one the HEALPix cells are laid out in.
 * `rotationFor` takes native to map; `centerFor` solves the inverse problem
 * — which north-up centre puts map point P under native direction n — so a
 * drag or a zoom keeps the sky that was under the pointer under it exactly.
 */

export const HAMMER_HALF_WIDTH = 2 * Math.SQRT2;
export const HAMMER_HALF_HEIGHT = Math.SQRT2;

const DEG = Math.PI / 180;
const clamp1 = (v) => Math.max(-1, Math.min(1, v));

/** Plane (x, y) of native longitude/latitude in radians. */
export function hammerForward(lam, bet) {
    const d = Math.sqrt(1 + Math.cos(bet) * Math.cos(lam / 2));
    return [2 * Math.SQRT2 * Math.cos(bet) * Math.sin(lam / 2) / d,
            Math.SQRT2 * Math.sin(bet) / d];
}

/** Native [lam, bet] in radians of plane (x, y), or null outside the sky.
 * z² = 1 − x²/16 − y²/4 is the inverse's auxiliary; the ellipse rim is
 * z² = ½ (where lam = ±π), so anything with z² < ½ is off the sky. */
export function hammerInverse(x, y) {
    const z2 = 1 - x * x / 16 - y * y / 4;
    if (z2 < 0.5 - 1e-12) return null;      // the rim itself is sky
    const z = Math.sqrt(z2);
    return [2 * Math.atan2(z * x, 2 * (2 * z2 - 1)), Math.asin(clamp1(z * y))];
}

/** Screen (CSS px, y down) ↔ plane (y up, east to the left). */
export function planeOf(sx, sy, s) {
    return [-(sx - s.screenW / 2) / s.ppu, (s.screenH / 2 - sy) / s.ppu];
}
export function screenOf(x, y, s) {
    return [s.screenW / 2 - x * s.ppu, s.screenH / 2 - y * s.ppu];
}

export function vecOf(lam, bet) {
    const c = Math.cos(bet);
    return [c * Math.cos(lam), c * Math.sin(lam), Math.sin(bet)];
}
export function lonlatOf(v) {
    return [Math.atan2(v[1], v[0]), Math.asin(clamp1(v[2]))];
}

/** Native → map, row-major 3×3: Rz(lon0) · Ry(−lat0). The native x axis
 * lands on the centre, the native z axis on the centre's local north. */
export function rotationFor(lon0Deg, lat0Deg) {
    const ca = Math.cos(lon0Deg * DEG), sa = Math.sin(lon0Deg * DEG);
    const cb = Math.cos(lat0Deg * DEG), sb = Math.sin(lat0Deg * DEG);
    return [ca * cb, -sa, -ca * sb,
            sa * cb,  ca, -sa * sb,
            sb,        0,  cb];
}
export function apply(m, v) {
    return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}

/** The native direction under a screen point, or null off the sky. */
export function nativeAt(sx, sy, s) {
    const [x, y] = planeOf(sx, sy, s);
    const lb = hammerInverse(x, y);
    return lb && vecOf(lb[0], lb[1]);
}
/** The map-frame direction under a screen point, or null off the sky. */
export function skyAt(sx, sy, s) {
    const n = nativeAt(sx, sy, s);
    return n && apply(rotationFor(s.lon0, s.lat0), n);
}
/** Map longitude in [0, 360) and latitude, degrees, under a screen point. */
export function lonlatAt(sx, sy, s) {
    const v = skyAt(sx, sy, s);
    if (!v) return null;
    const [lon, lat] = lonlatOf(v);
    return [((lon / DEG) + 360) % 360, lat / DEG];
}

const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/** The north-up centre [lon0, lat0] (degrees) that puts map direction `P`
 * under native direction `n`. Solves R(lon0, lat0) · n = P: the latitude
 * from the z component (two solutions when the point sits far up the view;
 * the one nearer `current`'s latitude wins), then the longitude from the
 * azimuths. When no exact centre exists — P is nearer the pole than the
 * view can reach north-up — the nearest reachable one. */
export function centerFor(P, n, current) {
    const r = Math.hypot(n[0], n[2]);
    const delta = Math.atan2(n[2], n[0]);
    const a = Math.asin(clamp1(P[2] / Math.max(r, 1e-12)));
    const candidates = [wrapPi(a - delta), wrapPi(Math.PI - a - delta)]
        .map((lat) => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, lat)));
    const cur = (current?.[1] ?? 0) * DEG;
    const lat = candidates.sort((p, q) => Math.abs(p - cur) - Math.abs(q - cur))[0];
    const cb = Math.cos(lat), sb = Math.sin(lat);
    const m = [cb * n[0] - sb * n[2], n[1], sb * n[0] + cb * n[2]];   // Ry(−lat) · n
    const lon = Math.atan2(P[1], P[0]) - Math.atan2(m[1], m[0]);
    return [(((lon / DEG) % 360) + 360) % 360, lat / DEG];
}

/** Pixels per plane unit that fit the whole ellipse with a margin. */
export function fitPpu(screenW, screenH) {
    return 0.96 * Math.min(screenW / (2 * HAMMER_HALF_WIDTH), screenH / (2 * HAMMER_HALF_HEIGHT));
}
