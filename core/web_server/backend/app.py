#!/usr/bin/env python
import sys

sys.path.append("../..")

# TODO: Don't connect to Ray until instructed to by the user.
# from ray_infra import ray_init

import io
import flask
from flask import request, jsonify, Response
from dataclasses import asdict
import operator
from dataclasses import replace
from flask.json.provider import DefaultJSONProvider
from pathlib import Path

from state.state_machine import enqueue_action
from context import local_context as context, get_cache
from cache.context_ref import use_local_context_for_refs

# Opt this gunicorn worker into reading ContextRef values directly from
# the local diskcache. Safe because the web server only runs on the Ray
# head node (see state_machine._init).
use_local_context_for_refs()
from stacks.composite_stack import Stack
from web_server.backend.files import files_bp
from web_server.backend.tiles import tiles_bp
from web_server.backend.metadata import metadata_bp
from web_server.backend.settings import settings_bp
from web_server.backend.dust import dust_bp
from web_server.backend.thumbnails import thumbnails_bp
from web_server.backend.compute_graphs import compute_graphs_bp
from web_server.backend.performance import performance_bp
from web_server.backend.decisions import decisions_bp
from web_server.backend.suggestions import suggestions_bp
from web_server.backend.asiair_panel import asiair_panel_bp
from web_server.backend.code_sync import code_sync_bp
from web_server.backend.finish_saves import finish_saves_bp
from web_server.backend.finish_export import finish_export_bp
from web_server.backend.router import stateful_route
from web_server.backend.planner import planner_bp

debug = False
STREAM_PORT = 8001

cache = get_cache()


def strtobool(val: str) -> bool:
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if val in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise ValueError(f"invalid truth value {val!r}")


from base_path import get_base_path

base_path = get_base_path()
print(f"### base_path {base_path}")

if getattr(sys, "frozen", False):
    print("### Starting web server: app is frozen")
    app = flask.Flask(
        __name__,
        template_folder=f"{base_path}/core/web_server/templates",
        static_folder=f"{base_path}/core/web_server/static",
    )
else:
    print("Starting web server: app is not frozen")
    app = flask.Flask(
        __name__,
        template_folder=f"{base_path}/core/web_server/frontend/build",
        static_folder=f"{base_path}/core/web_server/backend/static",
    )

from flask_cors import CORS
# Global CORS: tags every response (including `abort(404)` 404s and
# `redirect(...)` 307s from `stateful_route`) with the right headers,
# and handles OPTIONS preflights without each view needing its own
# decorator. The per-route `@cross_origin` inside `stateful_route` is
# idempotent with this.
# `expose_headers`: tiles are fetched cross-origin (frontend origin → the
# object's sharded backend port), and the WebGPU loader reads the tile's
# dimensions + band count from RESPONSE HEADERS. Without an explicit
# Access-Control-Expose-Headers, the browser hides all non-safelisted response
# headers from JS, so `X-Tile-Bands` reads as null → defaults to 1 → a colour
# (3-band) tile uploads as r32float → the shader's colour branch samples
# `(R,0,0)` and the image renders ALL RED. Mono is unaffected (bands defaults to
# 1 anyway). Expose the tile headers so colour tiles decode correctly.
CORS(app, resources={r"/*": {"origins": "*"}},
     expose_headers=["X-Tile-Width", "X-Tile-Height", "X-Tile-Bands"])

app.register_blueprint(files_bp)
app.register_blueprint(tiles_bp)
app.register_blueprint(metadata_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(dust_bp)
app.register_blueprint(thumbnails_bp)
app.register_blueprint(compute_graphs_bp)


import time
import sys
import faulthandler
import signal
from flask import g

# On-demand thread dump triggered by an external `kill -USR1 <pid>`.
# Faster + dependency-free vs py-spy when investigating a hung request:
#   pgrep -f 'gunicorn.*bind 0.0.0.0:8000' | tail -n +2 | xargs kill -USR1
# Stacks land on the worker's stderr, which gunicorn's `--capture-output`
# already collects. Trades a `SIGUSR1` signal slot for a always-available
# "what is this process doing right now" mechanism.
try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except (AttributeError, ValueError):
    # SIGUSR1 not available on this platform (Windows), or already taken.
    pass

# Auto-dump on requests that exceed a generous deadline. faulthandler's
# `dump_traceback_later` arms a watchdog that fires once after N seconds
# if the worker hasn't called `cancel_dump_traceback_later` first. Picks
# up reproductions automatically while we're not looking.
_SLOW_REQUEST_DEADLINE_S = 60

@app.before_request
def _start_timer():
    g.start_time = time.monotonic()
    faulthandler.dump_traceback_later(
        _SLOW_REQUEST_DEADLINE_S, repeat=False, file=sys.stderr,
    )

@app.after_request
def _log_slow_requests(response):
    faulthandler.cancel_dump_traceback_later()
    elapsed = time.monotonic() - g.start_time
    if elapsed > 0.2:
        print(f"SLOW [{elapsed:.2f}s] {request.method} {request.path}")
    return response
app.register_blueprint(performance_bp)
app.register_blueprint(decisions_bp)
app.register_blueprint(suggestions_bp)
app.register_blueprint(asiair_panel_bp)
app.register_blueprint(code_sync_bp)
app.register_blueprint(finish_saves_bp)
app.register_blueprint(finish_export_bp)
app.register_blueprint(planner_bp)


class CustomJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


app.json_provider_class = CustomJSONProvider
app.json = app.json_provider_class(app)


@app.route("/")
def index():
    """Displays the index page accessible at '/'"""
    return flask.render_template("index.html")


@app.route("/starmap")
def starmap():
    return flask.render_template("index.html")



@app.route("/api/shopping_cart")
def get_shopping_cart():
    return list(context.shopping_cart or [])


@app.route("/api/pipelines")
def get_pipelines():
    """Pipelines available for cart selection. Returns every distinct
    display name in the `@pipeline` registry — built-ins in
    `core/pipelines/*.py` (`PixInsight Master`, `Subframe`, the hybrids) and any user-registered pipelines in
    `~/AstroFlow/pipelines/`. A pipeline may register multiple entry
    points (per Stack / Target / Mosaic class it processes); they
    collapse to the single display name in this listing.
    """
    from pipelines.registry import registered_pipeline_names

    return {
        "pipelines": [
            {"name": name}
            for name in sorted(registered_pipeline_names())
        ],
    }


@app.route("/api/registry/pipelines")
def get_pipeline_listing():
    """Every pipeline present, built-in or installed, with its author,
    version, description, location and the entity classes it processes —
    the Pipelines page."""
    from pipelines.registry import pipeline_listing
    return {"pipelines": pipeline_listing()}


@app.route("/api/registry/finishes")
def get_finish_listing():
    """Every finish present, built-in or installed, with its author,
    version, description, location and the taps it reads — the Finishes
    page."""
    from finishes.registry import finish_listing
    return {"finishes": finish_listing()}


@app.route("/api/shopping_cart/<object_type>/<object_id>/<path:pipeline_name>", methods=["POST"])
def add_to_shopping_cart(object_type, object_id, pipeline_name):
    """Add a cart entry. Pipeline name is the sole variant axis — what
    quality / implementation / algorithm to run is encoded in the
    pipeline. The 3-tuple `(type, id, pipeline_name)` is what gets
    persisted in `context.shopping_cart`.

    Uses `<path:pipeline_name>` because pipeline display names contain
    spaces ('PixInsight Master').
    """
    from web_server.backend.announcer import announce
    from state.actor import state as state_handle
    cart = list(context.shopping_cart or [])
    entry = (object_type, object_id, pipeline_name)
    if entry not in cart:
        cart.append(entry)
        context.shopping_cart = cart
        state_handle.drain_and_restart()
    announce("reload_data")
    return Response(status=204)


@app.route("/api/shopping_cart/<object_type>/<object_id>/<path:pipeline_name>", methods=["DELETE"])
def remove_from_shopping_cart(object_type, object_id, pipeline_name):
    """Remove an item from the live cart. Accepts an optional
    `preserve` body field — `"all" | "finals" | "none"` (default
    `"none"`) — that controls whether the item's outputs are
    snapshotted into `context.archive`. See `state.archive` for
    the preservation semantics."""
    from web_server.backend.announcer import announce
    from state.actor import state as state_handle
    from state.archive import apply_remove_from_cart
    payload = request.get_json(silent=True) or {}
    preserve = payload.get("preserve", "none")
    entry = (object_type, object_id, pipeline_name)
    apply_remove_from_cart(entry, preserve=preserve)
    state_handle.drain_and_restart()
    announce("reload_data")
    return Response(status=204)


@app.route("/api/archive")
def get_archive():
    """Snapshot of `context.archive` for the UI's archive panel.
    Each entry is `[object_type, object_id, pipeline_name, [paths]]`.
    """
    from state.archive import get_archive as _get
    return list(_get())


@app.route(
    "/api/archive/<object_type>/<object_id>/<path:pipeline_name>",
    methods=["DELETE"],
)
def remove_from_archive(object_type, object_id, pipeline_name):
    """Drop an archive entry. Its preserved paths immediately
    become orphan-eligible on the next sweep."""
    from web_server.backend.announcer import announce
    from state.archive import apply_remove_from_archive
    apply_remove_from_archive((object_type, object_id, pipeline_name))
    announce("reload_data")
    return Response(status=204)


@app.route(
    "/api/archive/<object_type>/<object_id>/<path:pipeline_name>/to_cart",
    methods=["POST"],
)
def move_archive_to_cart(object_type, object_id, pipeline_name):
    """Re-activate an archived item: drop its archive entry and
    add it back to the live cart. The pipeline reruns under cart
    semantics (re-stacked / re-merged with current frames)."""
    from web_server.backend.announcer import announce
    from state.actor import state as state_handle
    from state.archive import apply_move_archive_to_cart
    apply_move_archive_to_cart((object_type, object_id, pipeline_name))
    state_handle.drain_and_restart()
    announce("reload_data")
    return Response(status=204)


def _cached_json(data, max_age=10):
    """Return a JSON response with Cache-Control headers."""
    response = jsonify(data)
    response.headers['Cache-Control'] = f'public, max-age={max_age}'
    return response


@app.route("/api/mosaics")
def mosaics():
    return _cached_json(context.ui_mosaics or {})


@app.route("/api/targets")
def targets():
    return _cached_json(context.ui_targets or {})


@app.route("/api/stacks")
def stacks():
    return _cached_json(context.ui_stacks or {})


@app.route("/api/catalog/entries")
def catalog_entries():
    """[designation, canonical name] for every catalog designation
    (NGC/IC/Sh2/LBN/LDN/B/vdB + Messier aliases). Lets the frontend
    resolve designation queries ("SH2 132", "M31") to the names that
    stacks/targets/mosaics are actually stored under."""
    from catalogs.ngc2000_catalog import designation_pairs

    return _cached_json(designation_pairs(), max_age=3600)


@app.route("/api/decision_overrides")
def decision_overrides():
    """Return all decision overrides for the frontend to resolve locally."""
    from web_server.backend.decisions import _serialize_key, _serialize_choice
    overrides = context.settings.decision_overrides or {}
    return {
        "overrides": [
            # Calibration choices are plain strings (a stack hash, a path,
            # or "none"); only the binary decisions carry an enum.
            {"key": _serialize_key(k), "choice": _serialize_choice(v.choice)}
            for k, v in overrides.items()
        ]
    }


@app.route("/api/stacks/<stack_id>/detail")
def stack_detail(stack_id):
    """Return full stack detail including contents (runs with dependencies).

    This is fetched on demand when the user opens a specific stack,
    not on every page load.
    """
    from flask import jsonify, abort
    from decisions.resolve import resolve_calibration
    from stacks.convert_to_dict import convert_to_dict
    import numpy as np
    from spherical_geometry.polygon import SingleSphericalPolygon

    stack_ref = (context.stacks or {}).get(stack_id)
    if stack_ref is None:
        return abort(404)

    stack = stack_ref.get()

    # Resolve calibration so the UI shows which bias/dark/flat is assigned.
    cal_stacks = context.calibration_stacks or {}
    decisions = context.settings.decision_overrides
    stack = resolve_calibration(stack, cal_stacks, decisions)

    def transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    result = convert_to_dict(
        stack, [["contents", "*", "metadata"]], transform=transform
    )

    # Per-run selected-vs-total so the RunsTab can show "10 frames
    # (6 selected)" — the runs here keep their FULL frame list (no
    # subframe selection applied), and this reflects the resolved
    # per-Stack quality setting (default keeps all).
    from decisions.subframe_selection import selected_counts_by_run

    counts = selected_counts_by_run(stack, decisions)
    for content_hash, run in result.get("contents", {}).items():
        if content_hash in counts:
            run["selected_frame_count"] = counts[content_hash][0]

    return jsonify(result)


@app.route("/api/calibration_stacks")
def calibration_stacks():
    return _cached_json(context.ui_calibration_stacks or {})


@app.route("/api/recent")
def recent():
    ui_targets = context.ui_targets or {}
    ui_mosaics = context.ui_mosaics or {}

    items = []
    for tid, t in ui_targets.items():
        if "newest_date" in t:
            items.append((("target", tid), t["newest_date"]))
    for mid, m in ui_mosaics.items():
        if "newest_date" in m:
            items.append((("mosaic", mid), m["newest_date"]))

    sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)
    return [i[0] for i in sorted_items]


# TODO: Loading stuff from the cache on every endpoint call is very slow.
@app.route("/api/flats/<width>/<height>/<filter>")
def flats_for_sensor_and_filter(width, height, filter):
    """Flats the user may pick from for a stack of this sensor and filter.

    Keyed on sensor GEOMETRY *and*, when the caller passes `?focal_length=`,
    the optical train — matching what `resolve_calibration` considers
    automatically (geometry bucket, then FOCALLEN refusal within 2%). This
    list is the manual-override path for the same decision, so offering a
    different candidate set than the automatic matcher would be its own bug:
    the two same-geometry rigs differ only by focal length, and a flat from
    the wrong train must not even appear as a tie-breaking option. Absent
    focal_length (or a candidate with none) is a wildcard, as in the matcher.
    It was keyed on the INSTRUME string, which silently returned nothing
    whenever a driver had renamed the camera.
    """
    from calibration.calibration_frames import focal_length_of, same_optical_train

    sensor = (int(width), int(height))
    raw = request.args.get("focal_length")
    target_focal = float(raw) if raw not in (None, "", "null") else None
    matching = [
        (k, s)
        for k, s in context.calibration_stacks.items()
        if s.image_type == "flat"
        and (s.width_pixels, s.height_pixels) == sensor
        and s.filter == filter  # calibration_stacks holds real Run objects
        and same_optical_train(target_focal, focal_length_of(s))
    ]
    # Night-sky flats first (spectrally matched to the lights), then star
    # flats, then the explicit masters — the order `_default_choice` picks in.
    return ([k for k, s in matching if getattr(s, "kind", None) == "star"]
            + [k for k, s in matching if getattr(s, "kind", None) != "star"])


@app.route("/api/ray/status")
def ray_status():
    """The cluster as the 1 Hz sampler last saw it (`ray_metrics`): node list
    and totals plus the utilization histories. Ray-free — a web worker
    never joins the cluster, so this reads the sampler's snapshot rather
    than calling `ray.nodes()`."""
    from web_server.backend.ray_metrics import (
        get_cluster_summary, get_cpu_history, get_mem_history,
        get_object_store_history,
    )

    summary = get_cluster_summary()
    if summary is None:
        return jsonify(
            {
                "healthyNodes": 0,
                "unhealthyNodes": 0,
                "totalNodes": 0,
                "totalCPUs": 0,
                "totalRAM": 0,
                "nodes": [],
                "error": "No cluster sample yet — the head has not started sampling.",
            }
        )
    return jsonify(
        {
            **summary,
            # Last 60 samples of each utilization stream, oldest first.
            # CPU / memory / object-store are sampled in lockstep so
            # indices line up.
            "cpuHistory": get_cpu_history(),
            "memHistory": get_mem_history(),
            "objectStoreHistory": get_object_store_history(),
        }
    )


@app.route("/api/ray/publish_status")
def publish_ray_status():
    from web_server.backend.announcer import announce

    announce("ray_status")
    return Response(status=200)


@app.route("/api/logs/last_run")
def get_last_run_logs():
    return cache["last_run_logs"] or {}


@app.route("/api/ray/head_node_enabled", methods=["GET", "POST"])
def ray_head_node_enabled():
    from state.network_settings import get_network_settings, update_network_settings

    if request.method == "POST":
        data = request.get_json() or request.form
        enabled_str = data.get("enabled")

        if enabled_str is None:
            return jsonify({"error": 'Missing "enabled" parameter'}), 400

        try:
            enabled = bool(strtobool(enabled_str))
        except ValueError:
            return jsonify({"error": "Invalid boolean value"}), 400

        update_network_settings(ray_head_node_enabled=enabled)
        enqueue_action(dict(type="restart_ray"))
        return Response(status=200)

    return jsonify({"enabled": get_network_settings().ray_head_node_enabled})


@app.route("/api/network_interfaces", methods=["GET"])
def network_interfaces():
    """List candidate network interfaces for the cluster bind dropdown.

    Returns up-interfaces that have at least one IPv4 address, with
    their name, current IPv4 address, link speed (Mbps; 0 = unknown),
    and MTU. Loopback is excluded — not useful for cluster bind.
    """
    import socket as _socket
    import psutil

    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    out = []
    for name, st in stats.items():
        if not st.isup:
            continue
        ipv4 = next(
            (a.address for a in addrs.get(name, []) if a.family == _socket.AF_INET),
            None,
        )
        if ipv4 is None or ipv4.startswith("127."):
            continue
        out.append({
            "name": name,
            "ip": ipv4,
            "speed_mbps": st.speed,
            "mtu": st.mtu,
        })
    out.sort(key=lambda i: (-i["speed_mbps"], i["name"]))
    return jsonify({"interfaces": out})


@app.route("/api/network_settings", methods=["GET", "PATCH"])
def network_settings_endpoint():
    """Read or partially-update the per-node network/Ray settings.

    PATCH body fields are optional; any combination of
    `ray_head_node_enabled` and `cluster_interface` is accepted.
    A successful PATCH dispatches `restart_ray` so the change takes
    effect immediately.
    """
    from state.network_settings import get_network_settings, update_network_settings

    if request.method == "PATCH":
        data = request.get_json() or {}
        updates = {}
        if "ray_head_node_enabled" in data:
            updates["ray_head_node_enabled"] = bool(data["ray_head_node_enabled"])
        if "cluster_interface" in data:
            iface = data["cluster_interface"]
            updates["cluster_interface"] = iface if iface else None
        if updates:
            update_network_settings(**updates)
            enqueue_action(dict(type="restart_ray"))

    ns = get_network_settings()
    return jsonify({
        "ray_head_node_enabled": ns.ray_head_node_enabled,
        "cluster_interface": ns.cluster_interface,
    })


