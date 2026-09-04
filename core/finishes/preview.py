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
    stfSeed recipe (finish_seed.js, 2026-09-03 recalibration):
    per channel, black + midtones anchor on the SKY subset (pixels below
    the luminance p60 — so each channel's displayed sky lands on the same
    target grey and the background renders neutral), white on a BRIGHT
    anchor `max(p99.9, median + 60σ)` — the same `high` the backend ships
    and the GHS finish reads. NOT p99: an astro frame is ~99% background,
    so p99 sits a few σ above sky and the span collapses (measured
    2026-09-03: 1.3% saturated, background lifted to 0.17 — visibly
    washed out). The gauge-relative floor remains only as the degenerate
    guard, replacing the old absolute 1e-3 (calibrated for
    display-normalized images, wrong on the pinned 0.005 gauge).
    Callers pass the COMPOSITE (starless + stars) — the stretch is a
    property of the original image, not of the layer viewed.
    `suffixes=('S','H','O')` emits per-channel keys; None emits the
    linked keys from pooled stats. `full_white=True` uses white = 1.0
    (the classic autostretch recipe — the narrowband Hubble preview
    matches the proven baked-palette look with it; p99 crushes the faint
    S/O channels relative to Hα)."""
    from mosaics.color.palettes import _solve_midtones
    chans = [arr[..., i] for i in range(arr.shape[-1])] if suffixes else [arr]
    # Shared sky subset from composite luminance, as in the backend stats.
    lum = arr.mean(axis=-1) if arr.ndim == 3 else arr
    lum_f = lum[np.isfinite(lum) & (lum > 0)]
    sky_cut = float(np.percentile(lum_f, 60)) if lum_f.size > 100 else np.inf
    sky_mask = np.isfinite(lum) & (lum > 0) & (lum < sky_cut)
    def _split(c):
        """(finite values, sky values) for a mono (H,W) or pooled (H,W,3)
        channel — the pooled case masks per-PIXEL then flattens bands."""
        if c.ndim == 3:
            fin = np.isfinite(c).all(axis=-1)
            return c[fin].ravel(), c[sky_mask & fin].ravel()
        fin = np.isfinite(c)
        return c[fin], c[sky_mask & fin]

    out: dict = {}
    per = []
    for c in chans:
        finite, sky = _split(c)
        if sky.size < 100:
            sky = finite
        med = float(np.median(sky)) if sky.size else 0.0
        sigma = 1.4826 * float(np.median(np.abs(sky - med))) if sky.size else 1e-6
        if full_white:
            bright = 1.0
        elif finite.size:
            # `high`, computed exactly as the backend's channelStats does:
            # a genuinely BRIGHT level, floored well clear of the noise.
            all_med = float(np.median(finite))
            all_mad = float(np.median(np.abs(finite - all_med)))
            all_sig = 1.4826 * all_mad if all_mad > 0 else max(all_med * 1e-3, 1e-6)
            bright = max(float(np.percentile(finite, 99.9)),
                         all_med + 60.0 * all_sig)
        else:
            bright = 1.0
        bp = max(0.0, med - 2.8 * max(sigma, 1e-6))
        wp = max(bright, bp + max(0.02 * med, 1e-6))
        x0 = (med - bp) / max(wp - bp, 1e-6)
        per.append((bp, wp, _solve_midtones(max(x0, 1e-6), target)))
    if suffixes:
        for (bp, wp, mid), s in zip(per, suffixes):
            out[f"black{s}"], out[f"white{s}"], out[f"mid{s}"] = bp, wp, mid
    else:
        bp, wp, mid = per[0]
        out["blackpoint"], out["whitepoint"], out["midtones"] = bp, wp, mid
    def _sky_med(c):
        finite, sky = _split(c)
        v = sky if sky.size >= 100 else finite
        return float(np.median(v)) if v.size else 0.0

    sky_min = min(_sky_med(c) for c in chans)
    if sky_min > 0:
        out["trim_floor"] = 0.6 * sky_min
    return out


def _resolve(object_type: str, object_id: str, pipeline: str, display: str,
             ancestors=()) -> "tuple[Path, str] | None":
    """(path, status) for a servable scene format, or None. Status is
    "live" | "out_of_date" — callers must surface it (the 2026-08-31 lesson:
    this function used to DROP it, and every preview consumer silently showed
    last-good generations as if current)."""
    from catalog import Product, resolve_product
    filter_ = "L" if display.startswith("Linear L") else "RGB"
    path, status = resolve_product(
        Product(object_type, object_id, pipeline, display, filter_), ancestors)
    return (path, status) if status in ("live", "out_of_date") else None


def render_canonical_preview(object_type: str, object_id: str,
                             ancestors=(), pipeline: "str | None" = None,
                             ) -> "tuple[bytes, str] | None":
    """(PNG bytes, status) of the canonical look, or None when no scene
    format is servable yet. Status is the PRIMARY source's freshness
    ("live" | "out_of_date"); the stars layer never downgrades it.
    Hubble for anything with narrowband; RGB otherwise.

    `pipeline` PINS which pipeline's scene formats to render. Without it this
    picks the first that resolves, which is right for a single card but wrong
    for any A/B: the dashboard shows one preview PER pipeline, and auto-picking
    would render the same image under both headings (exactly the illusion the
    2026-08-15 colour-tail collision produced for real)."""
    import cv2
    from catalog.queries import default_pipeline_for
    from context.cache import get_durable_cache
    from finishes.registry import get_finish

    # WHICH PIPELINE: the scene formats live in whichever pipeline built
    # them, which is NOT necessarily the object's default. The Heart target
    # is processed by both "PixInsight Master" (no colour tail) and the
    # hybrid; preference picks PI Master, whose Linear* taps don't exist, so
    # the preview 404'd on an object whose colour was sitting right there.
    # Try the default first, then every other pipeline for this class, and
    # take the first that actually resolves a scene format.
    from catalog.queries import pipelines_for
    if pipeline is not None:
        candidates = [pipeline]
    else:
        default = default_pipeline_for(object_type, object_id)
        candidates = [default] + [p for p in pipelines_for(object_type, object_id)
                                  if p != default]
    pipeline = None
    for cand in candidates:
        if any(_resolve(object_type, object_id, cand, d, ancestors)
               for d in ("Linear SHO", "Linear RGB Starless",
                         "Linear Narrowband Starless")):
            pipeline = cand
            break
    if pipeline is None:
        return None

    def _split(r):
        return r if r is not None else (None, None)
    sho_path, sho_status = _split(_resolve(
        object_type, object_id, pipeline, "Linear SHO", ancestors))
    rgb_path, rgb_status = _split(_resolve(
        object_type, object_id, pipeline, "Linear RGB Starless", ancestors))
    nb_partial_path, nb_status = _split(_resolve(
        object_type, object_id, pipeline, "Linear Narrowband Starless",
        ancestors))
    stars_path, _stars_status = _split(_resolve(
        object_type, object_id, pipeline, "Linear RGB Stars", ancestors))

    # Ladder: full-SHO Hubble is THE canonical look; a partial narrowband
    # set (an Hα-only galaxy) would render Hubble as a monochrome tint
    # (zero-filled S/O bands), so broadband RGB wins there; the partial
    # Hubble is the last resort for narrowband-only partial images (an HOO
    # bicolor still reads fine).
    if sho_path is not None:
        finish, primary_path, primary_status = (
            get_finish("Hubble", primary="Linear SHO"), sho_path, sho_status)
    elif rgb_path is not None:
        finish, primary_path, primary_status = (
            get_finish("RGB", primary="Linear RGB Starless"),
            rgb_path, rgb_status)
    elif nb_partial_path is not None:
        finish = get_finish("Hubble", primary="Linear Narrowband Starless")
        primary_path, primary_status = nb_partial_path, nb_status
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
    # The literal 4 versions the SEEDING RECIPE (v4: composite-anchored,
    # sky-subset black/mid, BRIGHT `high` white anchor, 2026-09-03) — bump
    # it when the value derivation changes, since the finish_hash only
    # covers the finish definition.
    key = ("finish_preview", 4, object_type, object_id, pipeline,
           finish.name, finish.finish_hash,
           _mtime(primary_path), _mtime(stars_path) if stars_path else 0)
    cached = dc.get(key)
    if cached is not None:
        return cached, primary_status

    primary = read_pyramid_level(primary_path)
    if primary.shape[-1] < 3:
        primary = np.repeat(primary[..., :1], 3, axis=-1)
    if stars_path is not None:
        stars = read_pyramid_level(stars_path)
        if stars.shape[:2] != primary.shape[:2]:
            stars = cv2.resize(stars, (primary.shape[1], primary.shape[0]),
                               interpolation=cv2.INTER_AREA)
        if stars.ndim == 2:
            stars = np.repeat(stars[..., None], 3, axis=-1)
    else:
        stars = np.zeros_like(primary)   # absent stars layer binds black
    sources = [primary, stars]
    # Seed from the COMPOSITE in both branches — the stretch is a property
    # of the original image (starless + stars), never of the layer viewed.
    if finish.name == "Hubble":
        values = _stf_values(primary + stars, ("S", "H", "O"), full_white=True)
        star_finite = stars[np.isfinite(stars)]
        if star_finite.size and float(star_finite.max()) > 0:
            values["star_mid"] = _stf_values(stars, None)["midtones"]
    else:  # RGB
        values = _stf_values(primary + stars, None)
        values["star_s"] = 1.0

    out = finish.bake(sources, values)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(
        (out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    png = buf.tobytes()
    dc.set(key, png)
    return png, primary_status
