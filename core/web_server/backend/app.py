#!/usr/bin/env python
import sys

sys.path.append('../..')

# TODO: Don't connect to Ray until instructed to by the user.
# from ray_infra import ray_init

import ray
import io
import numpy as np
import flask
from typing import Any
from flask import request, jsonify, Response
from dataclasses import asdict
import datetime
import operator
from spherical_geometry.polygon import SingleSphericalPolygon
import threading
from dataclasses import replace
import queue
from distutils.util import strtobool
import time
from flask.json.provider import DefaultJSONProvider
from pathlib import Path

from state import enqueue_action
from .message_announcer import MessageAnnouncer
from context import context
from catalogs import constellations
from context import get_cache
from compute_graph import build_compute_graph
from dependency_graph import run_graph, graph_status_to_progress
from dependency_graph.debugging import node_to_string
from metadata.classes import MetadataDict
from customization.types import CustomizedStack
from web_server.backend.files import files_bp
from web_server.backend.tiles import tiles_bp
from web_server.backend.metadata import metadata_bp
from web_server.backend.thumbnails import thumbnails_bp
from web_server.backend.videos import videos_bp

from stacks.composite_stack import CompositeStack

import plotly.express as px
import plotly.figure_factory as ff

debug = False

cache = get_cache()

announcer = MessageAnnouncer()


# TODO: Pull all of this transform code out into a separate module and test
# it properly.
def status_fn(status):
  global compute_graph_status
  print(f'### status_fn {len(status.ready_nodes)} ready '
        f'{len(status.waiting_nodes)} waiting '
        f'{len(status.done_nodes)} done')

  compute_graph_status = graph_status_to_progress(status)
  announcer.announce('compute_graph_status')

  if len(status.ready_nodes) == 0 and len(status.waiting_nodes) == 0:
    print('### Compute graph is done!')


def get_plot_html():
  buffer = io.StringIO()

  # df = px.data.iris()  # replace with your own data source
  # fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

  x1 = np.random.randn(200) - 2
  x2 = np.random.randn(200)
  x3 = np.random.randn(200) + 2

  hist_data = [x1, x2, x3]

  group_labels = ['Group 1', 'Group 2', 'Group 3']
  colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

  # Create distplot with curve_type set to 'normal'
  fig = ff.create_distplot(hist_data,
                           group_labels,
                           colors=colors,
                           curve_type='normal',
                           bin_size=.2,
                           show_rug=False)

  # Add title
  fig.update_layout(title_text='Hist and Curve Plot')
  # fig.show()

  fig.write_html(buffer, full_html=False)

  return buffer.getvalue()


from base_path import get_base_path

base_path = get_base_path()
print(f'### base_path {base_path}')

if getattr(sys, 'frozen', False):
  print('App is frozen')
  app = flask.Flask(__name__,
                    template_folder=f'{base_path}/core/web_server/templates',
                    static_folder=f'{base_path}/core/web_server/static')
else:
  print('App is not frozen')
  app = flask.Flask(
      __name__,
      template_folder=f'{base_path}/core/web_server/frontend/build',
      static_folder=f'{base_path}/core/web_server/frontend/build/static',
  )

app.register_blueprint(files_bp)
app.register_blueprint(tiles_bp)
app.register_blueprint(metadata_bp)
app.register_blueprint(thumbnails_bp)
app.register_blueprint(videos_bp)


class CustomJSONProvider(DefaultJSONProvider):

  def default(self, o):
    if isinstance(o, Path):
      return str(o)
    return super().default(o)


app.json_provider_class = CustomJSONProvider
app.json = app.json_provider_class(app)


# TODO: Move computations into new files.
def min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


@app.route('/')
def index():
  """ Displays the index page accessible at '/'"""
  return flask.render_template('index.html')


@app.route('/photo_sphere_viewer')
def photo_sphere_viewer():
  return flask.render_template('photo_sphere_viewer.html')


@app.route('/get_shopping_cart')
def get_shopping_cart():
  return list(context.get('shopping_cart', set()))


@app.route('/add_to_shopping_cart/<object_type>/<object_id>')
def add_to_shopping_cart(object_type, object_id):
  shopping_cart = context.shopping_cart or set()
  shopping_cart.add((object_type, object_id))
  context.shopping_cart = shopping_cart

  print('shopping_cart is now', context.shopping_cart)

  return Response(status=204)


@app.route('/remove_from_shopping_cart/<object_type>/<object_id>')
def remove_from_shopping_cart(object_type, object_id):
  shopping_cart = context.shopping_cart or set()
  shopping_cart.remove((object_type, object_id))
  context.shopping_cart = shopping_cart

  print('shopping_cart is now', context.shopping_cart)

  return Response(status=204)


@app.route('/starmap')
def starmap():
  return flask.render_template('index.html')


@app.route('/constellation_polylines')
def constellation_polylines():
  return constellations.polylines()


@app.route('/footprints')
def footprints():
  plate_solutions = metaflow_data_helper.find('plate_solve',
                                              'plate_solutions') or {}
  return [f for _, f, _ in list(plate_solutions.values())]


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


@app.route('/mosaics')
def mosaics():
  mosaics = context.get('mosaics', {})
  return {k: _make_json_serializable(v) for k, v in mosaics.items()}


@app.route('/targets')
def targets():
  targets = context.get('targets', {})
  return {k: _make_json_serializable(v) for k, v in targets.items()}


@app.route('/stacks')
def stacks():
  stacks = context.get('stacks', {})

  def prepare_stack(stack: CustomizedStack) -> dict[str, Any]:
    d = _make_json_serializable(stack)
    filenames = []
    for s in stack.contents.values():
      filenames += [m.filename for m in s.metadata.values()]

    d['filenames'] = filenames

    return d

  stacks = {h: prepare_stack(s) for h, s in stacks.items()}

  return stacks


@app.route('/calibration_stacks')
def calibration_stacks():
  calibration_stacks = context.get('calibration_stacks', {})
  calibration_stacks = {
      h: replace(s, metadata=None) for h, s in calibration_stacks.items()
  }

  return {k: _make_json_serializable(s) for k, s in calibration_stacks.items()}


@app.route('/send_datetime')
def send_datetime():
  return {'foo': datetime.datetime(2023, 5, 19, 4, 27, 17, 756893)}


@app.route('/targets_for_mosaic/<mosaic_id>')
def targets_for_mosaic(mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  return [t for t in context.targets if t.name in mosaic.target_ids]


@app.route('/stacks_for_target/<target>')
def stacks_for_target(target):
  return [s for s in context.stacks if s.target == target]


@app.route('/target/<target_id>')
def target(target_id):
  targets = test_app.targets if debug else context.targets
  print(f"Looking for target_id '{target_id}'")
  return asdict(targets[target_id])


@app.route('/target_polygon_points/<target_id>')
def target_polygon_points(target_id):
  targets = test_app.targets if debug else context.targets
  print(f"Looking for target_id '{target_id}'")
  # target = next(t for t in targets if t.name == target_id)
  target = targets[target_id]
  return ' '.join([','.join([str(x) for x in pt]) for pt in target.footprint])


@app.route('/stack/<stack_id>')
def stack(stack_id):
  stacks = test_app.stacks if debug else context.stacks
  print(f"Looking for stack_id '{stack_id}'")
  return asdict(stacks[stack_id])


@app.route('/flat_correction_thumbnails/<stack_id>/<flat_stack_id>')
def flat_correction_thumbnails(stack_id, flat_stack_id):
  key = ('flat_correction_thumbnail', stack_id, flat_stack_id)
  if key not in cache:
    return Response(status=404)

  buffer = io.BytesIO()
  buffer.write(cache[key])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='image/png')


@app.route('/get_plot')
def plot():
  mem = io.BytesIO()
  mem.write(get_plot_html().encode())
  mem.seek(0)
  return flask.send_file(mem, mimetype='text/html')


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


@app.route('/recent')
def recent():
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics

  def last_stack_datetime(stack_id):
    return stacks[stack_id].newest_date

  def last_target_datetime(target_id):
    return max([last_stack_datetime(s) for s in targets[target_id].stack_ids])

  def last_mosaic_datetime(mosaic_id):
    return max([last_target_datetime(t) for t in mosaics[mosaic_id].target_ids])

  targets_and_dates = [
      (('target', s), last_target_datetime(s)) for s in targets.keys()
  ]
  mosaics_and_dates = [
      (('mosaic', s), last_mosaic_datetime(s)) for s in mosaics.keys()
  ]

  all_items = targets_and_dates + mosaics_and_dates
  sorted_items = sorted(all_items, key=operator.itemgetter(1), reverse=True)
  return [i[0] for i in sorted_items]


# TODO: Loading stuff from the cache on every endpoint call is very slow.
@app.route('/flats_for_camera_and_filter/<camera>/<filter>')
def flats_for_camera_and_filter(camera, filter):
  print(f'### getting flats for camera {camera} and filter {filter}')

  stacks = cache['calibration_stacks']
  return [
      k for k, s in stacks.items()
      if s.image_type == 'flat' and s.camera == camera and s.filter == filter
  ]


@app.route('/get_stack_dependencies/<stack_id>')
def get_stack_dependencies(stack_id):
  stack = context.get('stacks', {}).get(stack_id, None)
  if not stack:
    return Response(status=404)

  key = ('stack_settings', stack_id)
  settings = context.get(key, {})

  return stack.dependencies | settings


def format_sse(event, data=None) -> str:
  msg = f'event: {event}\n'
  if data:
    msg += f'data: {data}\n'
  return msg + '\n'


@app.route('/stream')
def stream():
  print('### responding to request on /stream')

  def eventStream():
    messages = announcer.listen()
    while True:
      try:
        event, data = messages.get(timeout=5)  # Avoid indefinite blocking
      except queue.Empty:
        # Send a heartbeat every 5 seconds if no new messages arrive
        event, data = "heartbeat", f"keep-alive"
      if data == None:
        data = time.ctime(time.time())
      sse = format_sse(event, data)
      yield sse

  response = Response(eventStream(), mimetype="text/event-stream")
  response.headers['Access-Control-Allow-Origin'] = '*'
  response.headers['Connection'] = 'keep-alive'
  response.headers['Cache-Control'] = 'no-cache'

  return response


@app.route('/dispatch', methods=['POST'])
def dispatch():
  state_actor.dispatch(request.json,
                       callback=lambda e: announcer.announce('foo'))
  return Response(status=204)


compute_graph_status = {}


def populate_context():
  # TODO: Put these knobs in the UI.
  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160


# TODO: Make sure a second graph can't be started when one is already in
# progress.
#
# TODO: Something in here is causing `download_asiair()` to be called and
# preventing this from returning a response immediately.
@app.route('/run_graph')
def run_compute_graph():

  populate_context()

  # TODO: Put these in the UI.
  context.ignore_cached_results = True
  context.dangerously_accept_existing_files = True

  graph = build_compute_graph()

  def _run_graph(*args, **kwargs):
    # TODO: Perhaps the results and logs should be saved incrementally,
    # whenever a node finishes?
    results, logs = run_graph(*args, **kwargs)

    print('### Writing graph results and logs into context')
    context.last_run_results = {
        node_to_string(node): result
        for node, result in results.flatten().items()
    }

    context.last_run_logs = {
        node_to_string(node): log for node, log in logs.flatten().items()
    }

  print('### Running compute graph!')
  thread = threading.Thread(
      target=_run_graph,
      args=(graph,),
      kwargs={
          # 'status_fn': status_fn,
          'debug': False
      })
  thread.daemon = True
  thread.start()

  return Response(status=204)


@app.route('/compute_graph')
def get_compute_graph():
  return {
      'skeleton': [
          {
              "edgeLabel": "SUB-DAG 2",
              "id": "sub2",
              "type": "group",
              "definition": [{
                  "id": "Scan Texts",
                  "type": "execution",
                  "next": [{
                      "id": "Text Logs",
                      "type": "artifact"
                  }]
              }, {
                  "id": "Label Execution Logs",
                  "type": "execution",
                  "next": [{
                      "id": "Execution Logs",
                      "type": "artifact",
                  }]
              }],
              "next": [{
                  "id": "Send via APIs",
                  "type": "execution"
              }]
          },
          {
              'id': '3117593921912447883',
              'type': 'execution',
              "next": [{
                  "id": "sub2",
                  "type": "group"
              }]
          },
          {
              "id":
                  "Text2",
              "type":
                  "artifact",
              "next": [{
                  "edgeLabel": "obtuse-edge",
                  "id": "sub2",
                  "type": "group"
              }]
          },
      ],
      'state': {
          '3117593921912447883': {
              'displayName': 'scatter_compute_image_metadata',
              'description': 'scatter_compute_image_metadata',
              'icon': {
                  'text': 'SCIM',
                  'iconColors': 'inverted'
              }
          },
          "sub2": {
              "state":
                  "RUNNING",
              "hasControlNode":
                  True,
              "displayName":
                  "Log Ingestion (sub2)",
              "description":
                  "A simple subdag that shows edge-snapping logic above it",
              "groupMeta": {
                  "Text Logs": {
                      "state": "NO_STATE_RUNTIME",
                      "icon": {
                          "name": "logs",
                          "iconset": "shell",
                          "size": "small"
                      }
                  },
                  "Execution Logs": {
                      "state": "NO_STATE_RUNTIME",
                      "icon": {
                          "name": "logs",
                          "iconset": "shell",
                          "size": "small"
                      }
                  },
                  "Scan Texts": {
                      "state":
                          "SUCCEEDED",
                      "artifactRefs": [{
                          "id": "Text",
                          "path": []
                      }, {
                          "id": "Text2",
                          "path": []
                      }]
                  },
                  "Label Execution Logs": {
                      "state":
                          "RUNNING",
                      "artifactRefs": [{
                          "id": "Text",
                          "path": []
                      }, {
                          "id": "Entities",
                          "path": []
                      }]
                  }
              }
          }
      }
  }


@app.route('/compute_graph_status')
def get_compute_graph_status():
  return compute_graph_status


@app.route('/ray_status')
def ray_status():
  """Returns the current Ray cluster status."""
  try:
    nodes = ray.nodes()
    cluster_resources = ray.cluster_resources()
  except:
    return jsonify({
        "healthyNodes":
            0,
        "unhealthyNodes":
            0,
        "totalCPUs":
            0,
        "totalRAM":
            0,
        "nodes": [],
        "error":
            "Ray is not running. Start a head node to initialize the cluster."
    })

  healthy_nodes = sum(1 for node in nodes if node["Alive"])
  unhealthy_nodes = len(nodes) - healthy_nodes

  total_cpus = int(cluster_resources.get("CPU", 0))
  total_ram = int(cluster_resources.get("memory", 0) /
                  1e9)  # Convert bytes to GB

  node_list = []
  for node in nodes:
    ip = node["NodeManagerAddress"]
    health = "healthy" if node["Alive"] else "unhealthy"
    cpus = int(node["Resources"].get("CPU", 0))
    ram = int(node["Resources"].get("memory", 0) / 1e9)  # Convert bytes to GB
    message = None

    # Check for node failure reasons (if any)
    if not node["Alive"]:
      message = "Unknown failure"
      if "NodeID" in node:
        failure_reason = node.get("MetricsExportError", None)
        if failure_reason:
          message = failure_reason

    node_list.append({
        "ip": ip,
        "health": health,
        "cpus": cpus,
        "ram": ram,
        "message": message
    })

  return jsonify({
      "healthyNodes": healthy_nodes,
      "unhealthyNodes": unhealthy_nodes,
      "totalCPUs": total_cpus,
      "totalRAM": total_ram,
      "nodes": node_list
  })


@app.route('/publish_ray_status')
def publish_ray_status():
  announcer.announce('ray_status')
  return Response(status=200)


@app.route('/last_run_logs')
def get_last_run_logs():
  return cache['last_run_logs'] or {}


@app.route('/ray_head_node_enabled', methods=['GET', 'POST'])
def ray_head_node_enabled():
  if request.method == 'POST':
    print(f'### ray_head_node_enabled method {request.method}')

    data = request.get_json() or request.form
    enabled_str = data.get('enabled')

    if enabled_str is None:
      return jsonify({'error': 'Missing "enabled" parameter'}), 400

    try:
      ray_head_node_enabled = bool(strtobool(enabled_str))
    except ValueError:
      return jsonify({'error': 'Invalid boolean value'}), 400

    print(f'### setting ray_head_node_enabled {ray_head_node_enabled}')
    cache['ray_head_node_enabled'] = ray_head_node_enabled

    if ray_head_node_enabled:
      enqueue_action(dict(type='start_ray_head_node'))
    else:
      enqueue_action(dict(type='stop_ray_head_node'))

    print('### Response to /ray_head_node_enabled done')
    return Response(status=200)

  # GET request: return the current value
  return jsonify({'enabled': cache.get('ray_head_node_enabled', False)})


def flask_thread_body():
  print('### Starting AstroFlow web server...')
  app.run(host='0.0.0.0', port=80)


def start_flask():
  flask_thread = threading.Thread(target=flask_thread_body)
  flask_thread.start()
