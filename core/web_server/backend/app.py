#!/usr/bin/env python
import sys

sys.path.append('../..')

# from ray_infra import ray_init

import os
import io
import numpy as np
import flask
from flask import request, jsonify, Response
from dataclasses import asdict
from base64 import b64encode
import datetime
# import ray
from queue import Queue
import operator
from spherical_geometry.polygon import SingleSphericalPolygon
import threading
import json

from .message_announcer import MessageAnnouncer
from core.context import context
from state.actor import StateActor
from catalogs import constellations
from previews.operations import resize_array, autostretch, array_to_png
from context import get_cache
from artifacts import artifact_exists, read_artifact_image, CacheArtifact
from previews import array_to_png, autostretch
from astroflow import populate_context
from compute_graph import build_compute_graph
from dependency_graph import run_graph, graph_status_to_progress
from dependency_graph.debugging import node_to_string
from previews import QualityLevel
from quarantining.compute_graph import build_apply_stack_user_settings_subgraph
from compute_graph.compute_graph import load_metadata
from stacks.compute_graph import build_stack_subgraph

import plotly.express as px
import plotly.figure_factory as ff

debug = False

cache = get_cache()

announcer = MessageAnnouncer()


# TODO: Pull all of this transform code out into a separate module and test
# it properly.
def status_fn(status):
  global compute_graph_status

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


# Create the application.
app = flask.Flask(__name__, template_folder='./templates')


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
  return list(context.shopping_cart) or []


@app.route('/add_to_shopping_cart/<object_type>/<object_id>')
def add_to_shopping_cart(object_type, object_id):
  shopping_cart = set(context.shopping_cart) or set()
  shopping_cart.add((object_type, object_id))
  context.shopping_cart = shopping_cart

  print('shopping_cart is now', context.shopping_cart)

  return Response(status=204)


@app.route('/remove_from_shopping_cart/<object_type>/<object_id>')
def remove_from_shopping_cart(object_type, object_id):
  shopping_cart = set(context.shopping_cart) or set()
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
    else:
      output[k] = v

  return output


@app.route('/mosaics')
def mosaics():
  mosaics = test_app.mosaics if debug else cache.get('mosaics', {})
  return {k: _make_json_serializable(v) for k, v in mosaics.items()}


@app.route('/targets')
def targets():
  targets = test_app.targets if debug else cache.get('targets', {})
  return {k: _make_json_serializable(v) for k, v in targets.items()}


@app.route('/stacks')
def stacks():
  stacks = test_app.stacks if debug else cache.get('stacks', {})
  return {k: _make_json_serializable(v) for k, v in stacks.items()}


@app.route('/send_datetime')
def send_datetime():
  return {'foo': datetime.datetime(2023, 5, 19, 4, 27, 17, 756893)}


@app.route('/targets_for_mosaic/<mosaic_id>')
def targets_for_mosaic(mosaic_id):
  # mosaic = next(m for m in cache['mosaics'] if m.id == mosaic_id)
  mosaic = cache['mosaics'][mosaic_id]
  return [t for t in cache['targets'] if t.name in mosaic.target_ids]


@app.route('/stacks_for_target/<target>')
def stacks_for_target(target):
  return [s for s in cache['stacks'] if s.target == target]


@app.route('/target/<target_id>')
def target(target_id):
  targets = test_app.targets if debug else cache['targets']
  print(f"Looking for target_id '{target_id}'")
  # return asdict(next(t for t in targets if t.name == target_id))
  return asdict(targets[target_id])


@app.route('/target_polygon_points/<target_id>')
def target_polygon_points(target_id):
  targets = test_app.targets if debug else cache['targets']
  print(f"Looking for target_id '{target_id}'")
  # target = next(t for t in targets if t.name == target_id)
  target = targets[target_id]
  return ' '.join([','.join([str(x) for x in pt]) for pt in target.footprint])


@app.route('/stack/<stack_id>')
def stack(stack_id):
  stacks = test_app.stacks if debug else cache['stacks']
  print(f"Looking for stack_id '{stack_id}'")
  return asdict(stacks[stack_id])


# @app.route('/trial_flat_correction_image/<stack_id>/<index>')
# def trial_flat_correction_image(stack_id, index):
#   key = ('trial_flat_correction', stack_id)
#   if key not in cache:
#     return Response(status=404)

#   trial_flat_correction = cache[key][int(index)]

#   buffer = io.BytesIO()
#   buffer.write(trial_flat_correction['image'])
#   buffer.seek(0)
#   return flask.send_file(buffer, mimetype='image/png')


def tuple_starts_with(t, *values):
  if type(t) is not tuple:
    return False

  if len(values) > len(t):
    return False

  for i, v in enumerate(values):
    if t[i] != v:
      return False

  return True


@app.route('/stack_thumbnail_image/<stack_id>')
def stack_thumbnail_image(stack_id):
  for key in [('stack_master_thumbnail_image', stack_id),
              ('stack_subframe_thumbnail_image', stack_id)]:
    if key in cache:
      buffer = io.BytesIO()
      buffer.write(cache[key])
      buffer.seek(0)
      return flask.send_file(buffer, mimetype='image/png')

  return Response(status=404)


@app.route('/flat_correction_thumbnails/<stack_id>/<flat_stack_id>')
def flat_correction_thumbnails(stack_id, flat_stack_id):
  key = ('flat_correction_thumbnails', stack_id)
  hash = cache.get(key, {}).get(flat_stack_id, None)
  if not hash:
    return Response(status=404)

  key = ('flat_correction_thumbnail', hash)
  if key not in cache:
    return Response(status=404)

  buffer = io.BytesIO()
  buffer.write(cache[key])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='image/png')


@app.route('/stack_subframe_thumbnail_video/<stack_id>')
def stack_subframe_thumbnail_video(stack_id):
  key = ('stack_subframe_thumbnail_video', stack_id)
  if key not in cache:
    return Response(status=404)

  buffer = io.BytesIO()
  buffer.write(cache[key])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='video/webm')


# TODO: Thumbnails should always be stored as autostretched PNG, for speed.
@app.route('/mosaic_thumbnail_image_for_filter/<mosaic_id>/<filter>')
def mosaic_thumbnail_image_for_filter(mosaic_id, filter):
  # mosaic = context.mosaics[mosaic_id]
  # print('### mosaic', mosaic)

  # _, master_artifact = merge_graph_done_artifacts(QualityLevel.MASTER,
  #                                                 mosaic_id, filter)
  # subframe_artifact, = merge_graph_done_artifacts(QualityLevel.SUBFRAME,
  #                                                 mosaic_id, filter)

  # for artifact in [master_artifact, subframe_artifact]:
  #   if not artifact_exists(artifact):
  #     print('### artifact', artifact, 'did not exist')
  #     continue

  #   print('### sending artifact', artifact)
  #   image = read_artifact_image(artifact)
  #   image = np.nan_to_num(image)
  #   print('### image shape', image.shape, 'dtype', image.dtype, 'min',
  #         np.min(image), 'max', np.max(image))

  #   buffer = io.BytesIO()
  #   buffer.write(array_to_png(autostretch(image)))
  #   buffer.seek(0)
  #   return flask.send_file(buffer, mimetype='image/png')

  return Response(status=404)


@app.route('/mosaic_thumbnail_image/<mosaic_id>')
def mosaic_thumbnail_image(mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  print('### mosaic', mosaic)

  for mode in [QualityLevel.MASTER, QualityLevel.SUBFRAME]:
    for filter in ['H', 'L', 'S', 'O', 'G', 'R', 'B']:
      key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)
      if key in cache:
        return mosaic_thumbnail_image_for_filter(mosaic_id, filter)

  return Response(status=404)


# @app.route('/stack_master_image/<stack_id>')
# def stack_master_image(stack_id):
#   # stack = next(s for s in cache['stacks'] if s.id == stack_id)
#   stack = cache['stacks'][stack_id]

#   if not stack:
#     print(f'did not get stack for {stack_id}')
#   else:
#     print(f'got stack for {stack_id}')

#   if stack.master and os.path.exists(stack.master):
#     image = _read_image(stack.master)
#     image = resize_array(image, new_width=640)
#     image = autostretch(image)
#     image = array_to_png(image)
#     buffer = io.BytesIO()
#     buffer.write(image)
#     buffer.seek(0)
#     return flask.send_file(buffer, mimetype='image/png')


@app.route('/get_plot')
def plot():
  mem = io.BytesIO()
  mem.write(get_plot_html().encode())
  mem.seek(0)
  return flask.send_file(mem, mimetype='text/html')


# @app.route('/get_num_stars_histogram_html/<stack_id>')
# def get_num_stars_histogram_html(stack_id):
#   print(f'Creating num stars histogram for stack id {stack_id}')
#   stack = cache['stacks'][stack_id]
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

# TODO: All of the uses of the cache in this file have to be replaced by calls
# to the ContextActor. This also means the "front-door" to AstroFlow is via
# the website.

# @app.route('/quarantine_stack/<stack_id>')
# def quarantine_stack(stack_id):
#   context.or_equals('quarantined_stacks', {stack_id}, default_value=set())

#   graph = ray.get(build_apply_stack_user_settings_subgraph.remote())

#   thread = threading.Thread(target=run_graph,
#                             args=(graph,),
#                             kwargs={
#                                 'status_fn': status_fn,
#                                 'debug': False
#                             })
#   thread.daemon = True
#   thread.start()

#   return Response(status=204)

# @app.route('/quarantine_target/<target_id>')
# def quarantine_target(target_id):
#   context.or_equals('quarantined_targets', {target_id}, default_value=set())

#   graph = ray.get(build_apply_stack_user_settings_subgraph.remote())

#   thread = threading.Thread(target=run_graph,
#                             args=(graph,),
#                             kwargs={
#                                 'status_fn': status_fn,
#                                 'debug': False
#                             })
#   thread.daemon = True
#   thread.start()

#   return Response(status=204)

# @app.route('/quarantine_mosaic/<mosaic_id>')
# def quarantine_mosaic(mosaic_id):
#   context.or_equals('quarantined_mosaics', {mosaic_id}, default_value=set())

#   graph = ray.get(build_apply_stack_user_settings_subgraph.remote())

#   thread = threading.Thread(target=run_graph,
#                             args=(graph,),
#                             kwargs={
#                                 'status_fn': status_fn,
#                                 'debug': False
#                             })
#   thread.daemon = True
#   thread.start()

#   return Response(status=204)


@app.route('/recent')
def recent():
  stacks = cache['stacks']
  targets = cache['targets']
  mosaics = cache['mosaics']

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
  stacks = cache['stacks']
  return [
      k for k, s in stacks.items()
      if s.image_type == 'flat' and s.camera == camera and s.filter == filter
  ]


@app.route('/get_stack_dependencies/<stack_id>')
def get_stack_dependencies(stack_id):
  stack = cache.get('stacks', {}).get(stack_id, None)
  if not stack:
    return Response(status=404)

  key = ('stack_settings', stack_id)
  settings = cache.get(key, {})

  return stack.dependencies | settings


@app.route('/set_stack_settings/<stack_id>', methods=['POST'])
def set_stack_settings(stack_id):
  with cache.transact():
    key = ('stack_settings', stack_id)
    settings = cache.get(key, {}) | request.json
    cache[key] = settings

  return Response(status=204)


import time


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
      event, data = messages.get()
      print(f'### got event {event} from queue')
      if data == None:
        data = time.ctime(time.time())
      sse = format_sse(event, data)
      print('### yielding SSE', sse)
      yield sse

  response = Response(eventStream(), mimetype="text/event-stream")
  response.headers['Access-Control-Allow-Origin'] = '*'
  return response


@app.route('/dispatch', methods=['POST'])
def dispatch():
  state_actor.dispatch(request.json,
                       callback=lambda e: announcer.announce('foo'))
  return Response(status=204)


compute_graph_status = {}


# TODO: Make sure a second graph can't be started when one is already in
# progress.
#
# TODO: Something in here is causing `download_asiair()` to be called and
# preventing this from returning a response immediately.
@app.route('/run_graph')
def run_compute_graph():

  populate_context()

  # TODO: Put these in the UI.
  context.input_dirs = ['/Volumes/Astro/Raw', '/Volumes/Astro2/Raw']
  context.output_dir = '/Volumes/Astro/Processed'

  # TODO: Determine `regroup` automatically.
  graph = build_compute_graph(regroup=True,
                              make_previews=True,
                              run_main_processing_graph=True)

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
  thread = threading.Thread(target=_run_graph,
                            args=(graph,),
                            kwargs={
                                'status_fn': status_fn,
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


@app.route('/last_run_logs')
def get_last_run_logs():
  return context.last_run_logs or {}


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
