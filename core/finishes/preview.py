"""The canonical dashboard preview: ONE server-rendered look per image.

Narrowband present → autostretched Hubble (R=S, G=H, B=O) with RGB stars
screened in when the image has broadband; otherwise the linked-stretch RGB.
Rendered by the SAME traced-Python finishes the viewer runs live
(`Finish.bake` on numpy), over a LOW pyramid level of the linear scene
formats — pennies of compute, honest colours, and by construction it can
never drift from what the live sliders show.

Cached in the durable cache keyed on the finish's definition hash + the
source pyramids' mtimes, so edits to a finish or a rebuilt tap invalidate
naturally.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PREVIEW_MIN_PX = 512


def read_pyramid_level(path, min_px: int = _PREVIEW_MIN_PX) -> "np.ndarray":
    """The smallest stored pyramid level still >= `min_px` on its short side,
    as (H, W, bands) float32. Scene-format taps always carry `_tiles.tiff`
    pyramids, so no full-res decode ever happens here."""
    import pyvips
    p = os.fspath(path)
    base = pyvips.Image.tiffload(p, page=0, unlimited=True)
    try:
        n_pages = base.get("n-pages")
    except pyvips.Error:
        n_pages = 1
    level, dim = 0, min(base.width, base.height)
    while level + 1 < n_pages and dim // 2 >= min_px:
        level += 1
        dim //= 2
    img = pyvips.Image.tiffload(p, page=level, unlimited=True)
    arr = np.ndarray(buffer=img.write_to_memory(),
                     dtype={"float": np.float32, "double": np.float64,
                            "uchar": np.uint8}[img.format],
                     shape=(img.height, img.width, img.bands))
    arr = arr.astype(np.float32)
    if img.format == "uchar":
        arr /= 255.0
    return arr


def _stf_values(arr: "np.ndarray", suffixes: "tuple[str, ...] | None",
                target: float = 0.15, full_white: bool = False) -> dict:
    """Image-adaptive stretch params — the Python twin of the viewer's
    stfSeed recipe (finish_seed.js): per channel, black = median − 2.8σ
    (σ = 1.4826·MAD), white = p99, midtones solved so the median lands on
    the target grey. `suffixes=('S','H','O')` emits per-channel keys;
    None emits the linked keys from pooled stats. `full_white=True` uses
    white = 1.0 (the classic autostretch recipe — the narrowband Hubble
    preview matches the proven baked-palette look with it; p99 crushes the
    faint S/O channels relative to Hα)."""
    from mosaics.color.palettes import _solve_midtones
    chans = [arr[..., i] for i in range(arr.shape[-1])] if suffixes else [arr]
    out: dict = {}
    per = []
    for c in chans:
        finite = c[np.isfinite(c)]
        med = float(np.median(finite)) if finite.size else 0.0
        sigma = 1.4826 * float(np.median(np.abs(finite - med))) if finite.size else 1e-6
        p99 = 1.0 if full_white else (
            float(np.percentile(finite, 99)) if finite.size else 1.0)
        bp = max(0.0, med - 2.8 * max(sigma, 1e-6))
        wp = max(p99, bp + 1e-3)
        x0 = (med - bp) / max(wp - bp, 1e-6)
        per.append((bp, wp, _solve_midtones(max(x0, 1e-6), target)))
    if suffixes:
        for (bp, wp, mid), s in zip(per, suffixes):
            out[f"black{s}"], out[f"white{s}"], out[f"mid{s}"] = bp, wp, mid
    else:
        bp, wp, mid = per[0]
        out["blackpoint"], out["whitepoint"], out["midtones"] = bp, wp, mid
    return out


def _resolve(object_type: str, object_id: str, pipeline: str, display: str,
             ancestors=()) -> "Path | None":
    from catalog import Product, resolve_product
    filter_ = "L" if display == "Linear L" else "RGB"
    path, status = resolve_product(
        Product(object_type, object_id, pipeline, display, filter_), ancestors)
    return path if status in ("live", "out_of_date") else None


def render_canonical_preview(object_type: str, object_id: str,
                             ancestors=()) -> "bytes | None":
    """PNG bytes of the canonical look, or None when no scene format is
    servable yet. Hubble for anything with narrowband; RGB otherwise."""
    import cv2
    from catalog.queries import default_pipeline_for
    from context.cache import get_durable_cache
    from finishes.registry import get_finish

    pipeline = default_pipeline_for(object_type, object_id)

    sho_path = _resolve(object_type, object_id, pipeline, "Linear SHO",
                        ancestors)
    rgb_path = _resolve(object_type, object_id, pipeline, "Linear RGB",
                        ancestors)
    nb_partial_path = _resolve(object_type, object_id, pipeline,
                               "Linear Narrowband Starless", ancestors)
    stars_path = _resolve(object_type, object_id, pipeline,
                          "Linear RGB Stars", ancestors)

    # Ladder: full-SHO Hubble is THE canonical look; a partial narrowband
    # set (an Hα-only galaxy) would render Hubble as a monochrome tint
    # (zero-filled S/O bands), so broadband RGB wins there; the partial
    # Hubble is the last resort for narrowband-only partial images (an HOO
    # bicolor still reads fine).
    if sho_path is not None:
        finish, primary_path = get_finish("Hubble", primary="Linear SHO"), sho_path
    elif rgb_path is not None:
        finish, primary_path = get_finish("RGB", primary="Linear RGB"), rgb_path
    elif nb_partial_path is not None:
        finish = get_finish("Hubble", primary="Linear Narrowband Starless")
        primary_path = nb_partial_path
    else:
        return None
    if finish is None or primary_path is None:
        return None

    def _mtime(p):
        try:
            return int(os.path.getmtime(p))
        except OSError:
            return 0

    dc = get_durable_cache()
    # The literal 2 versions the SEEDING RECIPE (full-white narrowband
    # stretch) — bump it when the value derivation changes, since the
    # finish_hash only covers the finish definition.
    key = ("finish_preview", 2, object_type, object_id, pipeline,
           finish.name, finish.finish_hash,
           _mtime(primary_path), _mtime(stars_path) if stars_path else 0)
    cached = dc.get(key)
    if cached is not None:
        return cached

    primary = read_pyramid_level(primary_path)
    if primary.shape[-1] < 3:
        primary = np.repeat(primary[..., :1], 3, axis=-1)
    sources = [primary]
    if finish.name == "Hubble":
        if stars_path is not None:
            stars = read_pyramid_level(stars_path)
            if stars.shape[:2] != primary.shape[:2]:
                stars = cv2.resize(stars, (primary.shape[1], primary.shape[0]),
                                   interpolation=cv2.INTER_AREA)
            if stars.ndim == 2:
                stars = np.repeat(stars[..., None], 3, axis=-1)
        else:
            stars = np.zeros_like(primary)   # screen-with-black = identity
        sources.append(stars)
        values = _stf_values(primary, ("S", "H", "O"), full_white=True)
        star_finite = stars[np.isfinite(stars)]
        if star_finite.size and float(star_finite.max()) > 0:
            values["star_mid"] = _stf_values(stars, None)["midtones"]
    else:  # RGB
        stars = np.zeros_like(primary)
        sources.append(stars)
        values = _stf_values(primary, None)
        values["star_s"] = 1.0               # with-stars combine as-is

    out = finish.bake(sources, values)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(
        (out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    png = buf.tobytes()
    dc.set(key, png)
    return png
