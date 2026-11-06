"""A HEALPix NESTED map as twelve square face rasters, and the sphere →
(face, x, y) projection the sky shader mirrors in WGSL (sky_shader.js).

NESTED numbering is a Morton order inside each of the twelve base faces: a
cell's index interleaves the bits of its (x, y) position in the face, x from
the even bits and y from the odd bits (HEALPix's `xyf` convention). Pulling
the bits apart turns a map into twelve nside × nside images that a texture
array holds as they are, and the continuous forward projection gives every
sky direction its (face, x, y) in [0, 1)², so a fragment shader samples those
images directly -- no tile mesh, no per-cell geometry on the CPU.

The forward projection is the real-valued form of HEALPix's `ang2pix_nest`
(Górski et al. 2005): the integer cell is floor(nside · x), floor(nside · y).
Pinned against astropy-healpix by test_healpix_faces.py; the WGSL copy must
stay line-for-line equivalent.
"""
from __future__ import annotations

import numpy as np


def compact_bits(v: np.ndarray) -> np.ndarray:
    """The even bits of each uint64, packed down: Morton index → x."""
    v = v & np.uint64(0x5555555555555555)
    v = (v | (v >> np.uint64(1))) & np.uint64(0x3333333333333333)
    v = (v | (v >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)
    return v


def spread_bits(v: np.ndarray) -> np.ndarray:
    """Each bit of a uint64 moved to the even positions: x → Morton index."""
    v = v & np.uint64(0x00000000FFFFFFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
    return v


def _order(nside: int) -> int:
    order = int(nside).bit_length() - 1
    if nside <= 0 or (1 << order) != nside:
        raise ValueError(f"nside must be a power of two, got {nside}")
    return order


def faces_from_nested(values: np.ndarray, nside: int) -> np.ndarray:
    """`(12, nside, nside)` float32 rasters, `faces[f, y, x]`, of a NESTED map."""
    order = _order(nside)
    n = 12 * nside * nside
    values = np.asarray(values).ravel()
    if values.size != n:
        raise ValueError(f"a nside={nside} map has {n} cells, got {values.size}")
    idx = np.arange(n, dtype=np.uint64)
    face = (idx >> np.uint64(2 * order)).astype(np.intp)
    within = idx & np.uint64(nside * nside - 1)
    ix = compact_bits(within).astype(np.intp)
    iy = compact_bits(within >> np.uint64(1)).astype(np.intp)
    faces = np.empty((12, nside, nside), np.float32)
    faces[face, iy, ix] = values
    return faces


def nested_index(face, ix, iy, nside: int) -> np.ndarray:
    """NESTED cell index of raster position (ix, iy) on `face`."""
    _order(nside)
    f = np.asarray(face, dtype=np.uint64)
    x = spread_bits(np.asarray(ix, dtype=np.uint64))
    y = spread_bits(np.asarray(iy, dtype=np.uint64))
    return (f * np.uint64(nside * nside) + (x | (y << np.uint64(1)))).astype(np.int64)


def xyf_from_vec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face, x, y) of unit vectors `v[..., 3]`: the face in 0..11 and the
    continuous position in [0, 1)² of each direction in that face's raster."""
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    za = np.abs(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    tt = phi * (2.0 / np.pi)                       # [0, 4): the azimuth in quadrants

    # Equatorial belt, |z| <= 2/3: the faces are diamonds cut by two families
    # of lines; which pair of lines a point falls between picks the face.
    t1 = 0.5 + tt
    t2 = 0.75 * z
    jp = t1 - t2
    jm = t1 + t2
    ifp = np.floor(jp)
    ifm = np.floor(jm)
    face_eq = np.where(ifp == ifm, np.where(ifp == 4.0, 4.0, ifp + 4.0),
                       np.where(ifp < ifm, ifp, ifm + 8.0))
    fx_eq = jm - ifm
    fy_eq = 1.0 - (jp - ifp)

    # Polar caps: one face per quadrant, the position from the cap's
    # equal-area radius.
    ntt = np.minimum(3.0, np.floor(tt))
    tp = tt - ntt
    tmp = np.sqrt(3.0 * (1.0 - za))
    jpp = tp * tmp
    jmp = (1.0 - tp) * tmp
    north = z >= 0.0
    face_po = np.where(north, ntt, ntt + 8.0)
    fx_po = np.where(north, 1.0 - jmp, jpp)
    fy_po = np.where(north, 1.0 - jpp, jmp)

    eq = za <= 2.0 / 3.0
    face = np.where(eq, face_eq, face_po).astype(np.int64)
    top = np.nextafter(1.0, 0.0)                   # the pole itself lands on 1.0
    fx = np.clip(np.where(eq, fx_eq, fx_po), 0.0, top)
    fy = np.clip(np.where(eq, fy_eq, fy_po), 0.0, top)
    return face, fx, fy


def vec_from_lonlat(lon_deg, lat_deg) -> np.ndarray:
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def degrade(faces: np.ndarray) -> np.ndarray:
    """The next-coarser rasters: every 2 × 2 block averaged. HEALPix cells are
    equal-area, so the plain mean IS the area-weighted mean."""
    n = faces.shape[-1]
    if n < 2:
        raise ValueError("cannot degrade a 1-cell face")
    return faces.reshape(faces.shape[0], n // 2, 2, n // 2, 2).mean(axis=(2, 4), dtype=np.float32)
