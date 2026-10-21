"""The sky viewer's endpoints: a whole-sky HEALPix map (`allsky/maps.py`) in
the contract the tile viewer already speaks (tiles.py's tile_metadata), so
the same finish WGSL, params and seeding render the sphere.

* `/api/healpix` -- the maps in ~/AstroFlow/healpix and the single-input
  finishes that can show one.
* `/api/healpix/<name>/metadata[?finish=H]` -- the map's geometry and frame,
  the finish spec (WGSL + params) and the per-band stats it seeds from.
* `/api/healpix/<name>/faces.f32` -- the twelve face rasters, float32
  little-endian, face-major then row-major: `(12, nside, nside)`.
"""
from flask import Blueprint, Response, abort, jsonify, request

from allsky.maps import available_maps, load_map

healpix_bp = Blueprint("healpix", __name__)


def _single_input_finishes() -> list:
    """Finishes declaring exactly one input: a map is one band of one layer.
    A palette with an optional stars layer would run, reading the same band
    three times, but says nothing a channel finish doesn't."""
    from finishes.registry import all_finishes
    seen = {}
    for f in all_finishes():
        if len(f.inputs) == 1 and f.name not in seen:   # one entry per name, not per primary
            seen[f.name] = f
    return list(seen.values())


def _finish(name: str):
    for f in _single_input_finishes():
        if f.name == name:
            return f
    return abort(404, f"no single-input finish named {name!r}")


def _load(name: str):
    try:
        return load_map(name)
    except FileNotFoundError:
        return abort(404)


@healpix_bp.route("/api/healpix")
def healpix_index():
    return jsonify({"maps": available_maps(),
                    "finishes": [f.name for f in _single_input_finishes()]})


@healpix_bp.route("/api/healpix/<name>/metadata")
def healpix_metadata(name: str):
    m = _load(name)
    finish = _finish(request.args.get("finish", "H"))
    return jsonify({
        "name": m.name, "nside": m.nside, "frame": m.frame, "unit": m.unit,
        "gauge": m.gauge, "mtime": m.mtime,
        "finish": finish.spec(),
        "finishes": [f.name for f in _single_input_finishes()],
        # One input, one band, in the tile viewer's input shape: the map is
        # mono and linear, read through the finish's role band.
        "inputs": [{
            "index": 0, "tap": "HEALPix map", "filter": finish.role, "status": "live",
            "bound": True, "bands": 1, "bandRoles": [finish.role or "H"],
            "colorspace": "linear-scene", "mtime": m.mtime,
            "channelStats": [m.stats], "channelStatsStarless": [m.stats],
        }],
    })


@healpix_bp.route("/api/healpix/<name>/faces.f32")
def healpix_faces(name: str):
    m = _load(name)
    resp = Response(m.faces.tobytes(), mimetype="application/octet-stream")
    resp.headers["X-Healpix-Nside"] = str(m.nside)
    if request.args.get("v"):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp
