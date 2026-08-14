"""HTTP endpoints for serving image tiles to the web frontend.

Tile pyramids are produced ahead of time as a side effect of the image
producers themselves (via @create_tile_pyramid). This module just looks
them up and serves them through pyvips. There is no on-demand
conversion: if a tile pyramid doesn't exist for a given object/level,
the request returns 404.

A small in-memory cache holds opened pyvips `Image` handles, one per
(path, pyramid-page) pair, with mtime-based invalidation. Handles are
lazy-decode (cheap to hold) so the cache is essentially free.

Three endpoints:

* `/tap/tile_metadata` — image dimensions, tile size, level count.
* `/tap/tile/.../{z}/{x}/{y}.f32` — raw float32 little-endian pixels,
  brotli-compressed (`Content-Encoding: br`). Each tile is read by
  cropping the right pyvips pyramid page (the on-disk TIFF was also
  written by pyvips), packed as raw bytes, brotli-compressed on the
  fly, and shipped.
* `/tap/context_map` — the precompressed sibling context map, served
  verbatim with `Content-Encoding: br` (the file on disk is already
  in transit-ready form).
"""
import io
import os
import threading

import brotli
import numpy as np
import pyvips
from flask import Blueprint, request, send_file, abort, jsonify, Response
from werkzeug.exceptions import HTTPException

from previews.create_tile_pyramid import context_map_path_for
from web_server.backend.router import stateful_route
from web_server.backend._route import tap_route, parse_ancestors


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

# Display names whose 3-band tiles are LINEAR narrowband S/H/O (not baked RGB):
# the client must run the live colour finish in the shader rather than passing
# the tile through. Must match linear_sho's display= (mosaics/color/post_process.py
# LINEAR_SHO_DISPLAY). A 3-band tile under any other display is a baked deliverable.
LINEAR_FINISH_DISPLAYS = frozenset({"Linear SHO"})

# The linear SCENE FORMATS — the ≤3-band basis the multi-source live finishes
# sample (see display/finishes.py RGB_LIVE etc.). Marked "linear-scene" so the
# viewer offers data-driven finishes instead of baked pass-through; they get
# the same channelStats seeding as linear-sho.
LINEAR_SCENE_DISPLAYS = frozenset({
    "Linear RGB", "Linear RGB Stars", "Linear Narrowband",
    "Linear Narrowband Starless", "Linear L",
})


tiles_bp = Blueprint("tiles", __name__)

# Brotli quality for per-tile compression on the fly. Quality 4 gives
# ~30 MB/s throughput on a single core and ~70 % the ratio of quality
# 11 — perfect for the request-time path where latency matters more
# than the last few bytes of size. Quality 11 is reserved for the
# build-time context map precompression.
_TILE_BROTLI_QUALITY = 4


# (path, page) -> (pyvips.Image, mtime). Pyramid pages are lazy-decode
# handles, so caching them is cheap. Page 0 is always the full-resolution
# page; the metadata endpoint also uses it to compute width/height/levels.
_vips_page_cache: dict = {}
_vips_page_cache_lock = threading.Lock()


def _get_vips_page(path: str, page: int):
    current_mtime = os.path.getmtime(path)
    key = (path, page)
    with _vips_page_cache_lock:
        cached = _vips_page_cache.get(key)
        if cached is not None:
            img, cached_mtime = cached
            if cached_mtime == current_mtime:
                return img
        img = pyvips.Image.tiffload(path, page=page, unlimited=True)
        _vips_page_cache[key] = (img, current_mtime)
        return img


def _pyramid_levels(path: str) -> int:
    """Number of pyramid pages in the TIFF. Single-page TIFFs report 1."""
    base = _get_vips_page(path, 0)
    try:
        return int(base.get("n-pages"))
    except pyvips.Error:
        return 1


def _tile_pyramid_path_for_tap(
    object_type: str, object_id: str, pipeline: str, display: str,
    filter: str | None, ancestors: tuple[tuple[str, str], ...] = (),
):
    """The display tile pyramid (or a directly-servable master tiff) for a tap
    — the ONE "viewable" definition, now owned by the catalog. Gates on the
    pyramid's existence, so a tile request still resolves when the primary
    FITS/XISF/TIFF master was reclaimed by ephemeral cleanup; already-display-
    ready uint8 tiffs (e.g. Subframe previews) that have no separate pyramid
    fall back to the master.

    `ancestors` (a child leaf's cart-item chain, from the `ancestors` query
    param) lets a stack under a target resolve its tile path from the target's
    snapshot — the stack has none of its own."""
    from catalog import viewable_path, Product
    p = viewable_path(
        Product(object_type, object_id, pipeline, display, filter),
        ancestors=ancestors,
    )
    return str(p) if p is not None else None


def _request_ancestors() -> tuple[tuple[str, str], ...]:
    """The `ancestors` query param for the current tile request, parsed."""
    return parse_ancestors(request.args.get("ancestors"))


@stateful_route(
    tap_route(tiles_bp, "/tap/tile_metadata", hierarchy=("object_type", "object_id")),
    sharding_key="object_id",
)
def tile_metadata_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
    backend_port: int = None,
):
    from catalog import resolve_product, Product
    p, status = resolve_product(
        Product(object_type, object_id, pipeline, display, filter),
        ancestors=_request_ancestors())
    path = str(p) if p is not None else None
    return _tile_metadata_response(path, backend_port, display, status=status)


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


def _per_channel_linear_stats(path):
    p = os.fspath(path)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        mtime = 0
    key = (p, mtime)
    with _stats_lock:
        cached = _stats_cache.get(key)
        if cached is None:
            cached = _per_channel_linear_stats_uncached(p)
            _stats_cache[key] = cached
            if len(_stats_cache) > 64:            # a handful of taps in play
                _stats_cache.pop(next(iter(_stats_cache)))
    return cached


def _per_channel_linear_stats_uncached(path):
    """Per-band (background median, bright-signal anchor) for the shader's
    (c-m)/(p-m) normalise.

    The high anchor MUST be a genuinely bright level: if (p-m) is only a few σ,
    the per-pixel background NOISE fills most of [0,1] after normalise and the
    finish renders as pure speckle. Two things guard against that: (1) compute on
    a moderately-fine page (~16 Mpx), not the thumbnail — a coarse page averages
    the nebula down toward the sky pedestal and collapses the gap; (2) anchor on
    p99.9 (nebula cores/filaments) with a wide σ floor so the gap stays large
    relative to the noise even for a faint channel. Approximate but stable; the
    user fine-tunes via the sliders."""
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
    # GENERIC per-channel statistics (median, robust σ/MAD, min/max, percentiles).
    # Finish-agnostic: the frontend's display finishes compute their own starting
    # values from these (e.g. the STF/MTF seed = median − 2.8σ, p99, midtones-to-
    # target). The backend ships scene + stats, never finish logic. Narrowband
    # channels differ wildly in SNR (Hα ~13σ, SII ~2σ, OIII <1σ), so everything is
    # per-channel.
    stats = []
    for b in range(min(3, img.bands)):
        c = arr[:, b]
        c = c[np.isfinite(c) & (c > 0.0)]
        if c.size == 0:
            stats.append({"median": 0.0, "mad": 0.0, "sigma": 1.0, "min": 0.0,
                          "max": 1.0, "p1": 0.0, "p95": 1.0, "p99": 1.0,
                          "p999": 1.0, "high": 1.0})
            continue
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med)))
        sigma = 1.4826 * mad if mad > 0 else max(med * 1e-3, 1e-6)
        p1, p95, p99, p999 = (float(np.percentile(c, q)) for q in (1, 95, 99, 99.9))
        stats.append({
            "median": med, "mad": mad, "sigma": sigma,
            "min": float(c.min()), "max": float(c.max()),
            "p1": p1, "p95": p95, "p99": p99, "p999": p999,
            # Legacy HIGH anchor the built-in GHS finish reads: p99.9 floored at
            # median + 60σ so the gap stays wide relative to noise for a faint
            # channel.
            "high": max(p999, med + 60.0 * sigma),
        })
    return stats


def _tile_metadata_response(path, backend_port, display=None, status=None):
    if path is None:
        return abort(404)
    base = _get_vips_page(path, 0)
    levels = _pyramid_levels(path)
    meta = {
        "width": base.width,
        "height": base.height,
        "tileWidth": _TILE_SIZE,
        "tileHeight": _TILE_SIZE,
        "maxZoom": levels - 1,
        # Band count is the single source of truth for the client's
        # color-vs-mono branch: >=3 → 3-band RGB, 1 → linear mono channel.
        "bands": base.bands,
        "port": backend_port,
        "mtime": os.path.getmtime(path),
        # "live" | "out_of_date" — graceful versioning. When "out_of_date" the
        # `path` served is the last-good N-1 (the current version is still
        # building); the viewer badges it. See catalog.resolve_product.
        "status": status,
    }
    # `colorspace` disambiguates the cases the band count alone can't:
    #   "linear-sho"   → linear S/H/O, shader runs the live colour finish;
    #   "linear-scene" → a linear scene format (any band count) feeding the
    #                    multi-source live finishes;
    #   "display"      → baked RGB deliverable, shader passes it through.
    if display in LINEAR_FINISH_DISPLAYS and base.bands >= 3:
        meta["colorspace"] = "linear-sho"
    elif display in LINEAR_SCENE_DISPLAYS:
        meta["colorspace"] = "linear-scene"
    else:
        meta["colorspace"] = "display"
    if meta["colorspace"] != "display":
        stats = _per_channel_linear_stats(path)
        # Rich generic per-channel stats — frontend finishes seed from these.
        meta["channelStats"] = stats
        # Legacy arrays the built-in GHS finish still reads (median / σ / high).
        meta["channelMedians"] = [s["median"] for s in stats]
        meta["channelSigma"] = [s["sigma"] for s in stats]
        meta["channelP99"] = [s["high"] for s in stats]
    return jsonify(meta)


@stateful_route(
    tap_route(
        tiles_bp, "/tap/tile",
        hierarchy=("object_type", "object_id"),
        suffix="/<int:z>/<int:x>/<int:y>.f32",
    ),
    sharding_key="object_id",
)
def serve_tile_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
    z: int, x: int, y: int,
):
    path = _tile_pyramid_path_for_tap(
        object_type, object_id, pipeline, display, filter, _request_ancestors())
    return _serve_tile_response(path, object_type, object_id, x, y, z)


def _serve_tile_response(path, object_type, object_id, x, y, z):
    """Read one tile out of the pyramid as float32, brotli-compress,
    return with `Content-Encoding: br`. The browser's `fetch`
    decompresses brotli on the network thread; the bytes that arrive
    in JavaScript are Float32Array-compatible and upload to a WebGL
    R32F texture with no parse step.

    Pyvips is the natural read path because the tile pyramid TIFFs
    were also written by pyvips. The earlier large_image[vips] path
    routed `getTile()` through a JPEG-encoding step regardless of the
    requested format, which silently rounded our linear-float pixels
    to all-zero (the float values are << 1.0, JPEG quantizes to
    uint8 [0,255]). Going pyvips end-to-end keeps the pixels as-is.
    """
    if path is None:
        return abort(404)

    try:
        # Map URL zoom (0 = lowest res) to pyvips page (0 = full res).
        # tile_metadata returned `maxZoom = levels - 1`; pages are
        # symmetric to that.
        max_zoom = _pyramid_levels(path) - 1
        page = max_zoom - z
        img = _get_vips_page(path, page)

        x0 = x * _TILE_SIZE
        y0 = y * _TILE_SIZE
        w = min(_TILE_SIZE, img.width - x0)
        h = min(_TILE_SIZE, img.height - y0)
        if w <= 0 or h <= 0:
            return abort(404)
        tile = img.crop(x0, y0, w, h)
        raw = tile.write_to_memory()
        bands = tile.bands

        if tile.format == "float":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, bands)
        elif tile.format == "uchar":
            # Legacy autostretched uint8 pyramids — rescale to [0,1]
            # so the shader's per-pixel normalization still lines up.
            arr = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(h, w, bands)
                .astype(np.float32) / 255.0
            )
        else:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, bands).astype("<f4")

        # Color path: 3-band baked-stretch RGB served interleaved; the
        # shader samples .rgb and passes it through (stretch is baked in).
        # Mono path: a single linear channel the shader tone-maps client-
        # side. `X-Tile-Bands` tells the client which decode/texture to use.
        color = bands >= 3
        send_bands = 3 if color else 1
        arr = arr[..., :3] if color else arr[..., 0]

        # Edge tiles get padded to a full _TILE_SIZE × _TILE_SIZE with
        # zeros — the shader renders those pixels as opaque black, which
        # is the correct "no data" appearance past the image's actual
        # bounds (mono short-circuits on xRaw==0; color emits vec3(0)).
        # Padding eliminates the frontend's "texture smaller than the
        # mesh" case. Matches what `large_image.getTile()` did implicitly.
        if h < _TILE_SIZE or w < _TILE_SIZE:
            pad_shape = ((_TILE_SIZE, _TILE_SIZE, send_bands) if color
                         else (_TILE_SIZE, _TILE_SIZE))
            padded = np.zeros(pad_shape, dtype=np.float32)
            padded[:h, :w] = arr
            arr = padded
            h = w = _TILE_SIZE

        arr = np.ascontiguousarray(arr.astype("<f4"))
        compressed = brotli.compress(arr.tobytes(), quality=_TILE_BROTLI_QUALITY)
        resp = Response(compressed, mimetype="application/octet-stream")
        resp.headers["Content-Encoding"] = "br"
        resp.headers["X-Tile-Width"] = str(w)
        resp.headers["X-Tile-Height"] = str(h)
        resp.headers["X-Tile-Bands"] = str(send_bands)
        return resp
    except HTTPException:
        # An `abort(404)` raised INSIDE this try (the out-of-bounds tile check
        # below) is a NotFound, which `except Exception` used to swallow and
        # re-report as 500. A 500 reads as "server error, retry" — one viewer
        # holding stale tile_metadata across a rebuild then looped at ~3,220
        # req/min (36,150 errors on 2026-08-14), burying every `### node_exec`
        # line in the log. Let the real status through so the client can treat
        # it as "gone" and re-fetch metadata instead of hammering.
        raise
    except Exception as e:
        _log_tile_error(object_type, object_id, e)
        return abort(500)


@stateful_route(
    tap_route(
        tiles_bp, "/tap/context_map",
        hierarchy=("object_type", "object_id"),
    ),
    sharding_key="object_id",
)
def serve_context_map_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
):
    """Serve the per-master context map as precompressed brotli.

    The file on disk is already in transit-ready form: brotli-
    compressed binary blob whose decompressed payload is the magic
    header + dimensions + float16 pixels. We `sendfile()` it verbatim
    with `Content-Encoding: br` so the browser decompresses on the
    fetch path without us paying any per-request CPU.

    404 when the source was too small to produce a context map (the
    shader falls back to sampling tile mipmaps at no quality cost for
    that size).
    """
    pyramid_path = _tile_pyramid_path_for_tap(
        object_type, object_id, pipeline, display, filter, _request_ancestors(),
    )
    if pyramid_path is None:
        return abort(404)
    from pathlib import Path
    context_path = context_map_path_for(Path(pyramid_path))
    if not context_path.exists():
        return abort(404)
    resp = send_file(
        str(context_path), mimetype="application/octet-stream",
        conditional=True,  # ETag + Range support for caching
    )
    resp.headers["Content-Encoding"] = "br"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp
