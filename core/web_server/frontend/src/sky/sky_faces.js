/**
 * The twelve face rasters of a HEALPix map on the client: `(12, n, n)`
 * float32, face-major then row-major, as `/api/healpix/<name>/faces.f32`
 * serves them. Twin of allsky/healpix_faces.py::degrade.
 */

/** The next-coarser rasters: every 2 × 2 block averaged. HEALPix cells are
 * equal-area, so the plain mean IS the area-weighted mean. */
export function degradeFaces(faces, n) {
    const m = n >> 1;
    const out = new Float32Array(12 * m * m);
    for (let f = 0; f < 12; f++) {
        for (let y = 0; y < m; y++) {
            const r0 = f * n * n + 2 * y * n;
            const r1 = r0 + n;
            const o = f * m * m + y * m;
            for (let x = 0; x < m; x++) {
                const c = 2 * x;
                out[o + x] = 0.25 * (faces[r0 + c] + faces[r0 + c + 1] + faces[r1 + c] + faces[r1 + c + 1]);
            }
        }
    }
    return out;
}
