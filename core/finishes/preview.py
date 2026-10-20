"""Server-rendered finishes: a thumbnail of any bound finish, rendered by
the SAME traced-Python finish the viewer runs live (`Finish.bake` on
numpy) over a LOW pyramid level of its input taps, seeded exactly as the
viewer seeds its first frame — pennies of compute, honest colours, and by
construction it can never drift from what the live sliders show.

Cached in the durable cache keyed on the finish's definition hash + the
input pyramids' mtimes, so edits to a finish or a rebuilt tap invalidate
naturally.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_PREVIEW_MIN_PX = 512


def read_pyramid_level(path, min_px: int = _PREVIEW_MIN_PX) -> "np.ndarray":
    """The smallest zoom level still >= `min_px` on its short side, as
    (H, W, bands) float32 — a stored TIFF pyramid page or an XISF master
    shrunk from its memory map (`previews.pyramid_pages`); no full-res
    decode ever happens here."""
    from previews.pyramid_pages import _get_vips_page, _pyramid_levels
    p = os.fspath(path)
    base = _get_vips_page(p, 0)
    n_pages = _pyramid_levels(p)
    level, dim = 0, min(base.width, base.height)
    while level + 1 < n_pages and dim // 2 >= min_px:
        level += 1
        dim //= 2
    img = _get_vips_page(p, level)
    arr = np.ndarray(buffer=img.write_to_memory(),
                     dtype={"float": np.float32, "double": np.float64,
                            "uchar": np.uint8}[img.format],
                     shape=(img.height, img.width, img.bands))
    arr = arr.astype(np.float32)
    if img.format == "uchar":
        arr /= 255.0
    return arr


def _stf_values(arr: "np.ndarray", suffixes: "tuple[str, ...] | None",
                target: float = 0.10, display: bool = False,
                anchors_from: "np.ndarray | None" = None) -> dict:
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
    `target` is where the BACKGROUND lands in display units — 0.18, the
    twin of finish_seed.js: measured to put a real broadband frame's
    background at 0.197 with p99 at 0.629, i.e. actually using the
    display range. Darker targets render a ~99%-background frame as
    nearly black. `suffixes` are the band roles (`("S","H","O")`,
    `("R","G","B")`, `("H",)`): per-channel keys `black<role>…` are emitted
    for each, and the linked keys `blackpoint/whitepoint/midtones` from the
    pooled stats always — a finish reads the keys it declares.
    `display=True` (a display-referred payload) seeds the identity: the
    finish applies taste, not a second autostretch."""
    from mosaics.color.palettes import _solve_midtones
    if display:
        out = {"blackpoint": 0.0, "whitepoint": 1.0, "midtones": 0.5,
               "trim_floor": 0.0, "star_white": 1.0}
        for s_ in suffixes or ():
            out[f"black{s_}"], out[f"white{s_}"], out[f"mid{s_}"] = 0.0, 1.0, 0.5
        return out
    chans = [arr[..., i] for i in range(len(suffixes))] if suffixes else []
    chans.append(arr)   # the pooled channel, last
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
    anchors = []
    for c in chans:
        finite, sky = _split(c)
        if sky.size < 100:
            sky = finite
        med = float(np.median(sky)) if sky.size else 0.0
        sigma = 1.4826 * float(np.median(np.abs(sky - med))) if sky.size else 1e-6
        if finite.size:
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
        anchors.append((med, max(sigma, 1e-7), bright))
    for (bp, wp, mid), (med, sigma, bright), s_ in zip(per, anchors, suffixes or ()):
        out[f"black{s_}"], out[f"white{s_}"], out[f"mid{s_}"] = bp, wp, mid
        # Data anchors for the pinned-GHS finishes (Hubble and the channel finishes): the sky
        # median and noise the stretch pins, and the bright level it maps
        # to the nebula level. Twin of finish_seed.js.
        out[f"sky{s_}"], out[f"sig{s_}"], out[f"top{s_}"] = med, sigma, bright
    bp, wp, mid = per[-1]
    out["blackpoint"], out["whitepoint"], out["midtones"] = bp, wp, mid
    def _sky_med(c):
        finite, sky = _split(c)
        v = sky if sky.size >= 100 else finite
        return float(np.median(v)) if v.size else 0.0

    sky_min = min(_sky_med(c) for c in chans)
    if sky_min > 0:
        out["trim_floor"] = 0.6 * sky_min
    # Twin of finish_seed.js: normalise the stars layer by the same bright
    # anchor the image white uses, else the stars curve crushes raw linear
    # star pixels to invisibility.
    out["star_white"] = float(np.mean([wp for _, wp, _ in per[:-1]] or [per[-1][1]]))
    if anchors_from is not None and suffixes:
        # The pinned-GHS anchors come from the STARLESS primary: that
        # stretch works on the narrowband channels alone and never sees
        # the stars (Stylistic7, 2026-10-02). Twin of finish_seed.js.
        a = _stf_values(anchors_from, suffixes, target)
        for s_ in suffixes:
            for k in ("sky", "sig", "top"):
                out[f"{k}{s_}"] = a[f"{k}{s_}"]
    return out


def _fit(arr: "np.ndarray", shape) -> "np.ndarray":
    """`arr` resampled to the primary's (H, W) when a pyramid ladder differs."""
    import cv2
    if arr.shape[:2] != tuple(shape):
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        if arr.ndim == 2:
            arr = arr[..., None]
    return arr


def render_finish_preview(object_type: str, object_id: str, pipeline: str,
                          name: str, min_px: int = _PREVIEW_MIN_PX,
                          values: "dict | None" = None,
                          ) -> "tuple[bytes, str] | None":
    """(PNG bytes, status) of finish `name` bound to this object under
    `pipeline`, or None when its primary input is not servable yet. Status
    is the bound finish's (worst required input); callers must surface it
    (the 2026-08-31 lesson: dropping it showed last-good generations as
    current). `values` (a saved look's parameters) override the seeded
    ones; such renders are not cached."""
    import cv2
    from catalog import resolve_finish, stars_inputs, band_roles
    from context.cache import get_durable_cache
    from previews.create_tile_pyramid import payload_tags
    from previews.pyramid_pages import _get_vips_page

    b = resolve_finish(object_type, object_id, pipeline, name)
    if b is None or b.inputs[0].path is None:
        return None

    def _mtime(q):
        try:
            return int(os.path.getmtime(q))
        except OSError:
            return 0

    dc = get_durable_cache()
    # The literal 7 versions the SEEDING RECIPE (v7: pinned-GHS anchors from
    # the starless primary; v6: per-role keys from the
    # primary's band roles + pooled linked keys, identity seed for
    # display-referred payloads, viewer parity; 2026-09-16) — bump it when
    # the value derivation changes, since the finish_hash only covers the
    # finish definition.
    key = ("finish_preview", 7, object_type, object_id, pipeline, name, min_px,
           b.finish.finish_hash,
           *(_mtime(i.path) if i.path is not None else 0 for i in b.inputs))
    cached = dc.get(key) if values is None else None
    if cached is not None:
        return cached, b.status

    primary_path = os.fspath(b.inputs[0].path)
    primary = read_pyramid_level(primary_path, min_px)
    shape = primary.shape[:2]
    sources = [primary]
    for i in b.inputs[1:]:
        if i.path is None:
            sources.append(np.zeros((*shape, 3), np.float32))   # binds black
        else:
            sources.append(_fit(read_pyramid_level(i.path, min_px), shape))
    tags = payload_tags(_get_vips_page(primary_path, 0))
    roles = band_roles(primary.shape[-1], tags, b.inputs[0].filter)
    # Seed from the COMPOSITE — the stretch is a property of the picture
    # the finish screens the stars into, never of the layer viewed.
    composite = primary.astype(np.float32, copy=True)
    for i in stars_inputs(b, lambda q: payload_tags(_get_vips_page(os.fspath(q), 0))):
        st = sources[i.index]
        if st.shape[-1] >= 3 and composite.shape[-1] < 3:
            st = st[..., :3].mean(axis=-1, keepdims=True)
        composite += np.maximum(np.nan_to_num(st[..., :composite.shape[-1]]), 0.0)
    seeded = _stf_values(composite, tuple(roles), display=tags.get("COLORSPC") == "display",
                         anchors_from=primary.astype(np.float32, copy=False))

    out = b.finish.bake(sources, {**seeded, **(values or {})})
    # A saved look made for an HDR display carries its `display_white`, so
    # the bake runs past 1.0 there; the thumbnail is an SDR picture and clips
    # (uint8 would wrap a 1.35 to 88).
    ok, buf = cv2.imencode(".png", cv2.cvtColor(
        (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    png = buf.tobytes()
    if values is None:
        dc.set(key, png)
    return png, b.status
