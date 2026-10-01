"""HTTP endpoints for serving image tiles to the web frontend.

Tile pyramids are produced ahead of time as a side effect of the image
producers themselves (via @create_tile_pyramid). This module just looks
them up and serves them through pyvips. There is no on-demand
conversion: if a tile pyramid doesn't exist for a given object/level,
the request returns 404.

A small in-memory cache holds opened pyvips `Image` handles, one per
(path, pyramid-page) pair, with mtime-based invalidation. Handles are
lazy-decode (cheap to hold) so the cache is essentially free.

Two endpoints, both addressed by a finish (`catalog.finishes`): the
finish's inputs are the tap pyramids the viewer samples.

* `/tap/tile_metadata/.../<pipeline>/<finish>` — the finish spec plus, per
  input, image dimensions, tile size, level count, band roles and (for the
  primary) the per-channel statistics the finish seeds from.
* `/tap/tiles/.../<pipeline>/<finish>/{z}/{x}/{y}.f16` — one tile of every
  input: float16 little-endian pixels concatenated in input order, band counts
  in `X-Tile-Bands`; brotli-compressed for remote browsers only.
"""
import io
import os
import threading
import time

import brotli
import numpy as np
import pyvips
from flask import Blueprint, request, abort, jsonify, Response
from werkzeug.exceptions import HTTPException

from web_server.backend.router import BACKEND_PORTS, stateful_route
from web_server.backend._route import finish_route
from previews.create_tile_pyramid import payload_tags


_TILE_SIZE = 256

# Tile errors are RATE-LIMITED per (object, error-type). A stuck viewer can
# generate thousands per minute, and an unbounded print buries the pipeline's
# `### node_exec` telemetry — during the 2026-08-14 triage the log looked idle
# for 30k lines because a single tab's 404s had crowded everything out. One
# line per key per interval, with the suppressed count folded into the next.
_TILE_ERR_INTERVAL_S = 60.0
_tile_err_last: dict[tuple, float] = {}
_tile_err_suppressed: dict[tuple, int] = {}
_tile_err_lock = threading.Lock()


def _log_tile_error(object_type: str, object_id: str, exc: Exception) -> None:
    import time
    key = (object_type, object_id, type(exc).__name__)
    now = time.monotonic()
    with _tile_err_lock:
        last = _tile_err_last.get(key, 0.0)
        if now - last < _TILE_ERR_INTERVAL_S:
            _tile_err_suppressed[key] = _tile_err_suppressed.get(key, 0) + 1
            return
        _tile_err_last[key] = now
        dropped = _tile_err_suppressed.pop(key, 0)
    extra = f" ({dropped} more suppressed in the last {_TILE_ERR_INTERVAL_S:.0f}s)" if dropped else ""
    print(f"Error serving tile for {object_type, object_id}: {exc}{extra}")

tiles_bp = Blueprint("tiles", __name__)

# Brotli quality for per-tile compression on the fly. Quality 4 gives
# ~30 MB/s throughput on a single core and ~70 % the ratio of quality
# 11 — perfect for the request-time path where latency matters more
# than the last few bytes of size. Quality 11 is reserved for the
# build-time context map precompression.
_TILE_BROTLI_QUALITY = 4


from previews.pyramid_pages import _get_vips_page, _pyramid_levels  # noqa: E402


# Stats memo: computing them pulls a ~16 Mpx page (~190 MB) and runs per-band
# medians/percentiles — seconds of work that used to rerun on EVERY
# tile_metadata request (the viewer re-polls on each compute-status change,
# so an active pass hammered it into timeouts, 2026-08-12). Keyed on
# (path, mtime) so a rebuilt pyramid recomputes; the lock is single-flight —
# stacked gunicorn threads wait for the first computation instead of each
# pulling their own 190 MB.
_stats_cache: dict = {}
_stats_lock = threading.Lock()



_STATS_MAX_PX = 16_000_000
# Window sampling for a master with NO usable pyramid level. Reading a
# 3x3 grid of small windows off a TILED TIFF is a handful of tile reads;
# `subsample`/`resize` would DECODE THE WHOLE IMAGE first — measured at
# 123 s + ~100 s of medians for a 48207x46699 (27 GB) master, which is
# what made tile_metadata take 822 s and left the viewer black
# (no metadata -> no geometry -> no tiles). A scattered grid keeps the
# sample representative of sky AND signal, which a single centre crop
# would not be.
_STATS_WINDOW = 1024
_STATS_GRID = 3


def _stats_sample(img) -> "np.ndarray | None":
    """`(N, bands)` float32 sample of `img`, or None when no cheap sample
    exists — i.e. the resolved file is a huge master with NO pyramid page
    small enough to read. Those masters are strip-based, so even a 1024²
    crop decodes full-width strips: measured 70 s for a 3×3 grid, 250 s for
    the whole image, on a 48207×46699 master. Nothing in a web request may
    cost that. Callers omit `channelStats` instead; the finish then seeds
    from its param defaults until the pyramid is backfilled."""
    if img.width * img.height <= _STATS_MAX_PX:
        a = np.frombuffer(img.write_to_memory(), dtype=np.float32).reshape(
            img.height, img.width, img.bands)
        return a.reshape(-1, img.bands)
    return None


def _per_channel_linear_stats(path, star_paths=()):
    def _mt(q):
        try:
            return os.path.getmtime(os.fspath(q))
        except OSError:
            return 0
    p = os.fspath(path)
    key = (p, _mt(p), tuple((os.fspath(q), _mt(q)) for q in star_paths))
    with _stats_lock:
        cached = _stats_cache.get(key)
        if cached is None:
            cached = _per_channel_linear_stats_uncached(p, star_paths)
            _stats_cache[key] = cached
            if len(_stats_cache) > 64:            # a handful of taps in play
                _stats_cache.pop(next(iter(_stats_cache)))
    return cached


def _star_sample_for(page_img, star_path, band: int, primary_bands: int):
    """The stars-layer sample matching `page_img`'s pyramid page, as a
    flat (N,) array for primary band `band` — or None on any mismatch
    (dims must agree exactly; a stars mosaic shares its starless twin's
    canvas, so its pyramid has an identical page ladder). A 3-band stars
    layer over a mono primary contributes its band mean."""
    try:
        for p_idx in range(_pyramid_levels(star_path)):
            s = _get_vips_page(os.fspath(star_path), p_idx)
            if (s.width, s.height) == (page_img.width, page_img.height):
                a = _stats_sample(s)
                if a is None:
                    return None
                if a.shape[1] >= 3 and primary_bands < 3:
                    return a[:, :3].mean(axis=1)
                return a[:, band] if band < a.shape[1] else a[:, 0]
    except Exception:
        pass
    return None


def _per_channel_linear_stats_uncached(path, star_paths=()):
    """Per-band (background median, bright-signal anchor) for the shader's
    (c-m)/(p-m) normalise.

    The high anchor MUST be a genuinely bright level: if (p-m) is only a few σ,
    the per-pixel background NOISE fills most of [0,1] after normalise and the
    finish renders as pure speckle. Two things guard against that: (1) compute on
    a moderately-fine page (~16 Mpx), not the thumbnail — a coarse page averages
    the nebula down toward the sky pedestal and collapses the gap; (2) anchor on
    p99.9 (nebula cores/filaments) with a wide σ floor so the gap stays large
    relative to the noise even for a faint channel. Approximate but stable; the
    user fine-tunes via the sliders.

    `star_paths` (the finish's bound STARS-layer inputs) makes the stats
    COMPOSITE-derived: each band is sampled as starless + stars, so the
    seeds describe the picture the finish screens the stars into and never
    change character with the layer viewed (2026-09-03; a starless tap's
    razor-thin σ made the autostretch amplify ~1e-6 residuals into paint).

    Also ships per-band SKY-subset stats (`skyMedian`/`skySigma`): pixels
    below the composite-luminance p60 — empty sky, star-poor by
    construction. Seeding black/midtones from these pins each channel's
    DISPLAYED sky to the same target grey, which is what makes the
    background render neutral gray instead of carrying a channel-slope
    colour cast (the uniform-magenta finding, 2026-09-03)."""
    levels = _pyramid_levels(path)
    # Finest page still cheap to pull to numpy (~16 Mpx ≈ 190 MB at 3×float).
    page = levels - 1
    for p_idx in range(levels):
        img0 = _get_vips_page(path, p_idx)
        if img0.width * img0.height <= _STATS_MAX_PX:
            page = p_idx
            break
    img = _get_vips_page(path, page)
    arr = _stats_sample(img)
    if arr is None:
        print(f"### no pyramid page for stats: {path} "
              f"({img.width}x{img.height}) — serving without channelStats",
              flush=True)
        return []
    nb = min(3, img.bands)
    arr = arr[:, :nb].astype(np.float32, copy=True)
    for star_path in star_paths:
        for b in range(nb):
            s = _star_sample_for(img, star_path, b, img.bands)
            if s is not None and s.shape[0] == arr.shape[0]:
                arr[:, b] = arr[:, b] + np.maximum(np.nan_to_num(s, nan=0.0), 0.0)
    # COVERAGE COMES FROM THE COMPOSITE, not the primary. "Uncovered canvas is
    # exactly zero" holds for a starless layer, whose in-footprint sky sits on
    # the pedestal — but a STARS layer is zero everywhere except the stars, so
    # testing the primary would call almost the whole mosaic uncovered and
    # leave the stats to a handful of star pixels. The composite carries the
    # pedestal across the footprint either way round.
    coverage = np.all(np.isfinite(arr), axis=1) & np.all(arr > 0.0, axis=1)
    # Sky subset: covered pixels whose composite luminance sits below p60 —
    # background, largely star-free (stars are bright). Falls back to the
    # full covered set when coverage is tiny.
    sky_mask = coverage.copy()
    if coverage.sum() > 100:
        lum = arr[coverage].mean(axis=1)
        cut = np.percentile(lum, 60)
        idx = np.where(coverage)[0]
        sky_mask = np.zeros_like(coverage)
        sky_mask[idx[lum < cut]] = True
    # GENERIC per-channel statistics (median, robust σ/MAD, min/max, percentiles).
    # Finish-agnostic: the frontend's display finishes compute their own starting
    # values from these (e.g. the STF/MTF seed = median − 2.8σ, p99, midtones-to-
    # target). The backend ships scene + stats, never finish logic. Narrowband
    # channels differ wildly in SNR (Hα ~13σ, SII ~2σ, OIII <1σ), so everything is
    # per-channel.
    stats = []
    for b in range(nb):
        c = arr[:, b]
        good = np.isfinite(c) & (c > 0.0)
        c = c[good]
        if c.size == 0:
            stats.append({"median": 0.0, "mad": 0.0, "sigma": 1.0, "min": 0.0,
                          "max": 1.0, "p1": 0.0, "p95": 1.0, "p99": 1.0,
                          "p999": 1.0, "high": 1.0})
            continue
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med)))
        sigma = 1.4826 * mad if mad > 0 else max(med * 1e-3, 1e-6)
        p1, p95, p99, p999 = (float(np.percentile(c, q)) for q in (1, 95, 99, 99.9))
        sky = arr[:, b][sky_mask & good] if (sky_mask & good).any() else c
        sky_med = float(np.median(sky))
        sky_mad = float(np.median(np.abs(sky - sky_med)))
        stats.append({
            "median": med, "mad": mad, "sigma": sigma,
            "min": float(c.min()), "max": float(c.max()),
            "p1": p1, "p95": p95, "p99": p99, "p999": p999,
            "skyMedian": sky_med,
            "skySigma": 1.4826 * sky_mad if sky_mad > 0 else sigma,
            # Legacy HIGH anchor the built-in GHS finish reads: p99.9 floored at
            # median + 60σ so the gap stays wide relative to noise for a faint
            # channel.
            "high": max(p999, med + 60.0 * sigma),
        })
    return stats


def _declared_colorspace(base) -> "str | None":
    """The COLORSPC the PRODUCER stamped into this payload, or None — the
    authority on display-referredness (see `payload_tags`). Everything this
    app computes is LINEAR; only `mosaics/merging/stretched.py` stamps
    `COLORSPC: "display"` (decision 2026-08-21), so the viewer applies
    taste, not a second autostretch."""
    return payload_tags(base).get("COLORSPC")


def _is_display_referred(base) -> bool:
    """True only for a payload that says it is already display-referred.

    Linear is the DEFAULT — an untagged payload is linear scene data, which is
    what nearly everything is. An 8-bit payload is display-ready by its own
    encoding (the Subframe previews, which the tile route's uchar branch
    rescales /255) and must not be stretched a second time.
    """
    if str(getattr(base, "format", "")) in ("uchar", "char"):
        return True
    return _declared_colorspace(base) == "display"


def _colorspace(base) -> str:
    """`linear-sho` (a narrowband S,H,O join), `display` (already display-
    referred: an 8-bit payload or one tagged so), else `linear-scene` —
    LINEAR BY DEFAULT: everything the app computes is linear, only a payload
    that stamped itself display-referred opts out."""
    if _declared_colorspace(base) == "linear-sho":
        return "linear-sho"
    if _is_display_referred(base):
        return "display"
    return "linear-scene"


def _input_metadata(bound, path, backend_port, star_paths=()) -> dict:
    from catalog import band_roles
    base = _get_vips_page(path, 0)
    tags = payload_tags(base)
    meta = {
        "index": bound.index, "tap": bound.tap, "filter": bound.filter,
        "status": bound.status, "bound": True,
        "width": base.width, "height": base.height,
        "tileWidth": _TILE_SIZE, "tileHeight": _TILE_SIZE,
        "maxZoom": _pyramid_levels(path) - 1,
        "bands": base.bands,
        "bandRoles": band_roles(base.bands, tags, bound.filter),
        "port": backend_port,
        "mtime": os.path.getmtime(path),
        "colorspace": _colorspace(base),
    }
    if bound.index == 0 and meta["colorspace"] != "display":
        # Rich generic per-channel stats — the finish's seeds come from the
        # primary, composited with the stars layers it screens back in.
        meta["channelStats"] = _per_channel_linear_stats(path, star_paths=star_paths)
    return meta


def _unbound_input(bound) -> dict:
    return {"index": bound.index, "tap": bound.tap, "filter": bound.filter,
            "status": bound.status, "bound": False}


def _finish_metadata(b, backend_port) -> dict:
    """The viewer's contract for one bound finish: the finish spec (WGSL,
    params, inputs) plus one geometry/stats block per input. An extra input
    whose canvas differs from the primary's is served unbound — the shader
    samples black for it rather than misaligned pixels."""
    from catalog import stars_inputs
    primary = b.inputs[0]
    stars = [i.path for i in stars_inputs(
        b, lambda q: payload_tags(_get_vips_page(os.fspath(q), 0)))]
    inputs = [_input_metadata(primary, os.fspath(primary.path), backend_port, stars)]
    geometry = (inputs[0]["width"], inputs[0]["height"], inputs[0]["maxZoom"])
    for bound in b.inputs[1:]:
        if bound.path is None:
            inputs.append(_unbound_input(bound))
            continue
        m = _input_metadata(bound, os.fspath(bound.path), backend_port)
        if (m["width"], m["height"], m["maxZoom"]) != geometry:
            m = _unbound_input(bound)
        inputs.append(m)
    return {
        "name": b.name, "role": b.role, "status": b.status,
        # "live" | "out_of_date" — graceful versioning. When "out_of_date"
        # some input served is its last-good N-1 (the current version is
        # still building); the viewer badges it. See catalog.resolve_product.
        "finish": b.finish.spec(),
        "inputs": inputs,
        # Every tile server: the viewer spreads tiles across all of them by
        # tile coordinate (any backend serves any tile; the router's shard
        # only picks where a bare request is redirected).
        "tilePorts": list(BACKEND_PORTS),
    }


@stateful_route(
    finish_route(tiles_bp, "/tap/tile_metadata", hierarchy=("object_type", "object_id")),
    sharding_key="object_id",
)
def tile_metadata_finish(object_type: str, object_id: str, pipeline: str,
                         finish: str, backend_port: int = None):
    from catalog import resolve_finish
    if request.args.get("refresh"):
        # The finish author's edit loop: re-scan ~/AstroFlow/finishes.
        from finishes.registry import refresh_finishes
        refresh_finishes()
    b = resolve_finish(object_type, object_id, pipeline, finish)
    if b is None or b.inputs[0].path is None:
        return abort(404)
    return jsonify(_finish_metadata(b, backend_port))


# Every tile of a view resolves the same finish; resolving it per request
# (catalog + snapshot lookups) was a fixed cost on every one of the hundreds
# of tiles a zoom requests. Held briefly per process: a rebuilt product gets
# a new pyramid mtime and so new tile URLs anyway.
_FINISH_TTL_S = 5.0
_finish_cache: dict = {}


def _resolve_finish_cached(object_type, object_id, pipeline, finish):
    from catalog import resolve_finish
    key = (object_type, object_id, pipeline, finish)
    hit = _finish_cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < _FINISH_TTL_S:
        return hit[1]
    b = resolve_finish(object_type, object_id, pipeline, finish)
    _finish_cache[key] = (now, b)
    return b


def _geometry(path: str) -> tuple:
    base = _get_vips_page(path, 0)
    return base.width, base.height, _pyramid_levels(path)


def _tile_array(path: str, x: int, y: int, z: int) -> np.ndarray | None:
    """One _TILE_SIZE^2 tile of a pyramid as float16, (H, W) mono or (H, W, 3)
    colour, zero-padded past the image edge; None when (x, y) is off the image.

    float16 on the wire: float32 mantissa bits are noise to a compressor (a
    3-band tile compressed to 77 % of 768 KB), half floats are a quarter of
    that with 4e-4 worst-case relative error, and r16float / rgba16float
    textures are always filterable (2026-10-01). Edge padding is exact zero,
    which the shaders treat as "no data"."""
    max_zoom = _pyramid_levels(path) - 1
    img = _get_vips_page(path, max_zoom - z)
    x0, y0 = x * _TILE_SIZE, y * _TILE_SIZE
    w = min(_TILE_SIZE, img.width - x0)
    h = min(_TILE_SIZE, img.height - y0)
    if w <= 0 or h <= 0:
        return None
    tile = img.crop(x0, y0, w, h)
    raw = tile.write_to_memory()
    bands = tile.bands
    if tile.format == "uchar":
        # Legacy autostretched uint8 pyramids — rescale to [0, 1].
        arr = np.frombuffer(raw, np.uint8).reshape(h, w, bands).astype(np.float32) / 255.0
    else:
        arr = np.frombuffer(raw, np.float32).reshape(h, w, bands)
    arr = arr[..., :3] if bands >= 3 else arr[..., 0]
    out = np.zeros((_TILE_SIZE, _TILE_SIZE) + arr.shape[2:], "<f2")
    out[:h, :w] = np.nan_to_num(arr, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return out


def _is_local_request() -> bool:
    return request.remote_addr in ("127.0.0.1", "::1")


@stateful_route(
    finish_route(
        tiles_bp, "/tap/tiles",
        hierarchy=("object_type", "object_id"),
        suffix="/<int:z>/<int:x>/<int:y>.f16",
    ),
    sharding_key="object_id",
)
def serve_tile_finish(
    object_type: str, object_id: str, pipeline: str, finish: str,
    z: int, x: int, y: int,
):
    """One tile of EVERY input of the finish, in one response: the inputs'
    float16 tiles concatenated in input order. `X-Tile-Bands` lists each
    input's band count (3 colour, 1 mono, 0 = not served: unbound, absent,
    or a canvas that differs from the primary's -- the shader samples black
    for it, as tile_metadata already told the viewer). Input 0 off the image
    is a 404. One request per tile instead of one per input cut a three-input
    view's request count threefold (2026-10-01)."""
    b = _resolve_finish_cached(object_type, object_id, pipeline, finish)
    if b is None or b.inputs[0].path is None:
        return abort(404)
    try:
        primary = os.fspath(b.inputs[0].path)
        geometry = _geometry(primary)
        arrays = []
        for bound in b.inputs:
            if bound.path is None:
                arrays.append(None)
                continue
            path = os.fspath(bound.path)
            if path != primary and _geometry(path) != geometry:
                arrays.append(None)
                continue
            try:
                arrays.append(_tile_array(path, x, y, z))
            except Exception:
                if path == primary:
                    raise
                arrays.append(None)   # an extra input degrades to black
        if arrays[0] is None:
            return abort(404)
        body = b"".join(a.tobytes() for a in arrays if a is not None)
        bands = [0 if a is None else (3 if a.ndim == 3 else 1) for a in arrays]
        if _is_local_request():
            # Same machine: bandwidth is free and compressing costs CPU on
            # both ends, so send the half floats as they are.
            resp = Response(body, mimetype="application/octet-stream")
        else:
            resp = Response(brotli.compress(body, quality=_TILE_BROTLI_QUALITY),
                            mimetype="application/octet-stream")
            resp.headers["Content-Encoding"] = "br"
        resp.headers["X-Tile-Width"] = str(_TILE_SIZE)
        resp.headers["X-Tile-Height"] = str(_TILE_SIZE)
        resp.headers["X-Tile-Bands"] = ",".join(map(str, bands))
        if request.args.get("v"):
            # `v` carries the inputs' pyramid mtimes: a rebuilt pyramid gets a
            # new URL, so a versioned tile never changes and may be kept.
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp
    except HTTPException:
        # A real status (404) must reach the client as itself, never as a
        # 500 that reads as "retry": one viewer holding stale metadata across
        # a rebuild once looped at ~3,220 req/min (2026-08-14).
        raise
    except Exception as e:
        _log_tile_error(object_type, object_id, e)
        return abort(500)


