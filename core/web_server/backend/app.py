#!/usr/bin/env python
import sys

sys.path.append("../..")

# TODO: Don't connect to Ray until instructed to by the user.
# from ray_infra import ray_init

import ray
import io
import numpy as np
import flask
from flask import request, jsonify, Response
from dataclasses import asdict
import operator
from spherical_geometry.polygon import SingleSphericalPolygon
import threading
from dataclasses import replace
from flask.json.provider import DefaultJSONProvider
from pathlib import Path

from state.state_machine import enqueue_action
from context import context, get_cache
from stacks.composite_stack import Stack
from web_server.backend.files import files_bp
from web_server.backend.tiles import tiles_bp
from web_server.backend.metadata import metadata_bp
from web_server.backend.settings import settings_bp
from web_server.backend.thumbnails import thumbnails_bp
from web_server.backend.compute_graphs import compute_graphs_bp
from web_server.backend.videos import videos_bp
from web_server.backend.performance import performance_bp
from web_server.backend.decisions import decisions_bp
from web_server.backend.router import stateful_route

import plotly.express as px
import plotly.figure_factory as ff

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


def get_plot_html():
    buffer = io.StringIO()

    # df = px.data.iris()  # replace with your own data source
    # fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    hist_data = [x1, x2, x3]

    group_labels = ["Group 1", "Group 2", "Group 3"]
    colors = ["#A56CC1", "#A6ACEC", "#63F5EF"]

    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(
        hist_data,
        group_labels,
        colors=colors,
        curve_type="normal",
        bin_size=0.2,
        show_rug=False,
    )

    # Add title
    fig.update_layout(title_text="Hist and Curve Plot")
    # fig.show()

    fig.write_html(buffer, full_html=False)

    return buffer.getvalue()


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

app.register_blueprint(files_bp)
app.register_blueprint(tiles_bp)
app.register_blueprint(metadata_bp)
app.register_blueprint(settings_bp)
app.register_blueprint(thumbnails_bp)
app.register_blueprint(compute_graphs_bp)
app.register_blueprint(videos_bp)
app.register_blueprint(performance_bp)
app.register_blueprint(decisions_bp)


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



@app.route("/api/invalidated")
def get_invalidated():
    """Return the set of objects whose cached images are stale."""
    invalidated = context.get("invalidated", {})
    return jsonify([list(k) for k in invalidated.keys()])


# This stub redirects requests to a worker on a specific port. That worker
# will handle all SSE connections, and broadcast the announcer.announcements it receives
# over a diskcache queue.
@stateful_route(app.route("/stream"), fixed_port=STREAM_PORT)
def stream():
    pass


@app.route("/api/shopping_cart")
def get_shopping_cart():
    return list(context.get("shopping_cart", set()))


@app.route("/api/shopping_cart/<object_type>/<object_id>", methods=["POST"])
def add_to_shopping_cart(object_type, object_id):
    shopping_cart = context.shopping_cart or set()
    shopping_cart.add((object_type, object_id))
    context.shopping_cart = shopping_cart

    print("shopping_cart is now", context.shopping_cart)

    return Response(status=204)


@app.route("/api/shopping_cart/<object_type>/<object_id>", methods=["DELETE"])
def remove_from_shopping_cart(object_type, object_id):
    shopping_cart = context.shopping_cart or set()
    shopping_cart.remove((object_type, object_id))
    context.shopping_cart = shopping_cart

    print("shopping_cart is now", context.shopping_cart)

    return Response(status=204)


def _make_json_serializable(D):
    output = {}
    for k, v in asdict(D).items():
        if type(v) is np.ndarray:
            output[k] = v.tolist()
        elif type(v) is SingleSphericalPolygon:
            output[k] = v.points.tolist()
        elif type(v) is None:
            continue
        else:
            output[k] = v

    return output


@app.route("/api/mosaics")
def mosaics():
    return context.ui_mosaics or {}


@app.route("/api/targets")
def targets():
    return context.ui_targets or {}


@app.route("/api/stacks")
def stacks():
    return context.ui_stacks or {}


@app.route("/api/decision_overrides")
def decision_overrides():
    """Return all decision overrides for the frontend to resolve locally."""
    from customization.decisions import DecisionOverride
    overrides = context.settings.decision_overrides or {}
    # Serialize tuple keys to JSON-safe arrays
    return {
        "overrides": [
            {"key": list(k), "choice": str(v.choice)}
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
    from customization.resolve import resolve_calibration
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
    return jsonify(result)


@app.route("/api/calibration_stacks")
def calibration_stacks():
    return context.ui_calibration_stacks


@app.route("/get_plot")
def plot():
    mem = io.BytesIO()
    mem.write(get_plot_html().encode())
    mem.seek(0)
    return flask.send_file(mem, mimetype="text/html")


# @app.route('/get_num_stars_histogram_html/<stack_id>')
# def get_num_stars_histogram_html(stack_id):
#   print(f'Creating num stars histogram for stack id {stack_id}')
#   stack = context.stacks[stack_id]
#   plate_solutions = cache['plate_solutions']

#   light_frames = [f for run in stack.runs.values() for f in run.frames]
#   num_stars = [plate_solutions[f].num_stars for f in light_frames]
#   print('num_stars', num_stars)

#   colors = ['#A56CC1', '#A6ACEC']
#   bin_size = (np.max(num_stars) - np.min(num_stars)) / 10
#   fig = ff.create_distplot([num_stars], ['# Stars'],
#                            colors=colors,
#                            curve_type='normal',
#                            bin_size=bin_size,
#                            show_rug=False)

#   # fig.update_layout(title_text='Hist and Curve Plot')

#   buffer = io.StringIO()
#   fig.write_html(buffer, full_html=False)
#   mem = io.BytesIO()
#   mem.write(buffer.getvalue().encode())
#   mem.seek(0)
#   return flask.send_file(mem, mimetype='text/html')


@app.route("/api/recent")
def recent():
    stacks = context.stacks
    targets = context.targets
    mosaics = context.mosaics

    def last_stack_datetime(stack_id):
        return stacks[stack_id].get().newest_date

    def last_target_datetime(target_id):
        return max(
            [last_stack_datetime(s) for s in targets[target_id].get().stack_ids]
        )

    def last_mosaic_datetime(mosaic_id):
        return max(
            [last_target_datetime(t) for t in mosaics[mosaic_id].get().target_ids]
        )

    targets_and_dates = [
        (("target", s), last_target_datetime(s)) for s in targets.keys()
    ]
    mosaics_and_dates = [
        (("mosaic", s), last_mosaic_datetime(s)) for s in mosaics.keys()
    ]

    all_items = targets_and_dates + mosaics_and_dates
    sorted_items = sorted(all_items, key=operator.itemgetter(1), reverse=True)
    return [i[0] for i in sorted_items]


# TODO: Loading stuff from the cache on every endpoint call is very slow.
@app.route("/api/flats/<camera>/<filter>")
def flats_for_camera_and_filter(camera, filter):
    print(f"### getting flats for camera {camera} and filter {filter}")
    return [
        k
        for k, s in context.calibration_stacks.items()
        if s.image_type == "flat" and s.camera == camera and s.filter == filter  # calibration_stacks holds real Run objects
    ]


@app.route("/api/ray/status")
def ray_status():
    """Returns the current Ray cluster status."""
    try:
        nodes = ray.nodes()
        cluster_resources = ray.cluster_resources()
    except Exception:
        return jsonify(
            {
                "healthyNodes": 0,
                "unhealthyNodes": 0,
                "totalCPUs": 0,
                "totalRAM": 0,
                "nodes": [],
                "error": "Ray is not running. Start a head node to initialize the cluster.",
            }
        )

    healthy_nodes = sum(1 for node in nodes if node["Alive"])
    unhealthy_nodes = len(nodes) - healthy_nodes

    total_cpus = int(cluster_resources.get("CPU", 0))
    total_ram = int(
        cluster_resources.get("memory", 0) / 1e9
    )  # Convert bytes to GB

    node_list = []
    for node in nodes:
        ip = node["NodeManagerAddress"]
        health = "healthy" if node["Alive"] else "unhealthy"
        cpus = int(node["Resources"].get("CPU", 0))
        ram = int(
            node["Resources"].get("memory", 0) / 1e9
        )  # Convert bytes to GB
        message = None

        # Check for node failure reasons (if any)
        if not node["Alive"]:
            message = "Unknown failure"
            if "NodeID" in node:
                failure_reason = node.get("MetricsExportError", None)
                if failure_reason:
                    message = failure_reason

        node_list.append(
            {
                "ip": ip,
                "health": health,
                "cpus": cpus,
                "ram": ram,
                "message": message,
            }
        )

    return jsonify(
        {
            "healthyNodes": healthy_nodes,
            "unhealthyNodes": unhealthy_nodes,
            "totalCPUs": total_cpus,
            "totalRAM": total_ram,
            "nodes": node_list,
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
    if request.method == "POST":
        print(f"### ray_head_node_enabled method {request.method}")

        data = request.get_json() or request.form
        enabled_str = data.get("enabled")

        if enabled_str is None:
            return jsonify({"error": 'Missing "enabled" parameter'}), 400

        try:
            ray_head_node_enabled = bool(strtobool(enabled_str))
        except ValueError:
            return jsonify({"error": "Invalid boolean value"}), 400

        print(f"### setting ray_head_node_enabled {ray_head_node_enabled}")
        cache["ray_head_node_enabled"] = ray_head_node_enabled

        if ray_head_node_enabled:
            enqueue_action(dict(type="start_ray_head_node"))
        else:
            enqueue_action(dict(type="stop_ray_head_node"))

        print("### Response to /ray_head_node_enabled done")
        return Response(status=200)

    # GET request: return the current value
    return jsonify({"enabled": cache.get("ray_head_node_enabled", False)})


def flask_thread_body():
    print("### Starting AstroFlow web server...")
    app.run(host="0.0.0.0", port=80)


def start_flask():
    flask_thread = threading.Thread(target=flask_thread_body)
    flask_thread.start()

    return flask_thread
