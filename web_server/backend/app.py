#!/usr/bin/env python
import sys

sys.path.append('../..')

import ray_init

import os
import io
import numpy as np
import flask
from flask import request, jsonify, Response
from dataclasses import asdict
import shutil
import threading
from base64 import b64encode
import datetime
import ray

import ray_diskcache
import constellations
import image_previews
from cache import cache
import subframe_selection
from context import Context

import plotly.express as px
import plotly.figure_factory as ff

debug = False


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


run_now = threading.Condition()


@app.route('/run')
def run():
  run_now.acquire()
  run_now.notify()
  run_now.release()

  return Response(status=204)


@app.route('/add_mosaic_to_shopping_cart/<mosaic_id>')
def add_mosaic_to_shopping_cart(mosaic_id):
  stacks = test_app.stacks if debug else cache.get('stacks', {})
  targets = test_app.targets if debug else cache.get('targets', {})
  mosaics = test_app.mosaics if debug else cache.get('mosaics', {})
  shopping_cart = test_app.shopping_cart if debug else cache.get(
      'shopping_cart', [])

  stack_ids = [
      s for t in mosaics[mosaic_id].target_ids for s in targets[t].stack_ids
  ]
  shopping_cart.extend([('stack_files', s) for s in stack_ids])

  shopping_cart.append(('merge_mosaic', mosaic_id))

  shopping_cart = list(set(shopping_cart))
  cache['shopping_cart'] = shopping_cart

  return Response(status=204)


@app.route('/add_target_to_shopping_cart/<target_id>')
def add_target_to_shopping_cart(target_id):
  targets = test_app.targets if debug else cache.get('targets', {})
  shopping_cart = test_app.shopping_cart if debug else cache.get(
      'shopping_cart', [])

  shopping_cart.extend([('stack_files', s) for s in targets[target_id].stack_ids
                       ])
  shopping_cart.append(('post_process_target', target_id))

  shopping_cart = list(set(shopping_cart))
  cache['shopping_cart'] = shopping_cart

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


@app.route('/mosaics')
def mosaics():
  return test_app.mosaics if debug else cache.get('mosaics', {})


@app.route('/targets')
def targets():
  return test_app.targets if debug else cache.get('targets', {})


@app.route('/stacks')
def stacks():
  return test_app.stacks if debug else cache.get('stacks', {})


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


@app.route('/trial_flat_corrections/<stack_id>')
def trial_flat_corrections(stack_id):
  key = ('trial_flat_correction', stack_id)
  if key not in cache:
    return Response(status=404)

  results = cache[key]
  for result in results:
    result['original_image'] = 'data:image/png;base64,' + b64encode(
        result['original_image']).decode()
    for t in result['trial_flat_corrections']:
      t['corrected_image'] = 'data:image/png;base64,' + b64encode(
          t['corrected_image']).decode()

  return results


@app.route('/flat_frame_associations/<stack_id>')
def flat_frame_associations(stack_id):
  stacks = test_app.stacks if debug else cache['stacks']
  return stacks[stack_id].flat_frame_associations


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
  for mode in ['master', 'subframe']:
    key = next(
        (k for k in cache.iterkeys()
         if tuple_starts_with(k, 'stack_thumbnail_image', mode, stack_id)),
        None)

    if key:
      buffer = io.BytesIO()
      buffer.write(cache[key])
      buffer.seek(0)
      return flask.send_file(buffer, mimetype='image/png')

  return Response(status=404)


@app.route('/stack_thumbnail_images/<stack_id>/<flat_stack_id>')
def stack_thumbnail_images(stack_id, flat_stack_id):
  key = ('stack_thumbnail_images', 'subframe', stack_id)
  image = cache.get(key, {}).get(flat_stack_id, None)
  if not image:
    return Response(status=404)

  buffer = io.BytesIO()
  buffer.write(image)
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='image/png')


# TODO: Maybe expose all of the levels of previews (master, subframe, etc.).
# @app.route('/stack_thumbnail_image/<mode>/<stack_id>')
# def stack_thumbnail_image(mode, stack_id):
#   key = next((k for k in cache.iterkeys()
#               if tuple_starts_with(k, 'stack_thumbnail_image', mode, stack_id)),
#              None)
#   print(f'found matching key {key}')

#   if not key:
#     return Response(status=404)

#   buffer = io.BytesIO()
#   buffer.write(cache[key])
#   buffer.seek(0)
#   return flask.send_file(buffer, mimetype='image/png')


@app.route('/stack_thumbnail_video/<stack_id>')
def stack_thumbnail_video(stack_id):
  key = next((k for k in cache.iterkeys()
              if tuple_starts_with(k, 'stack_thumbnail_video', stack_id)), None)
  print(f'found matching key {key}')

  if not key:
    return Response(status=404)

  buffer = io.BytesIO()
  buffer.write(cache[key])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='video/webm')


@app.route('/mosaic_image/<mosaic_id>/<filter>')
def mosaic_image(mosaic_id, filter):
  obj = prev_results.get_prev_result_object('post_process_mosaic', mosaic_id)
  if not obj:
    return Response(status=404)

  # TODO: Get the correct color channel.
  image = image_previews._read_image(obj.output_filenames[0])
  image = image_previews.resize_array(image, new_width=640)
  image = image_previews.autostretch(image)
  image = image_previews.array_to_png(image)
  buffer = io.BytesIO()
  buffer.write(image)
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='image/png')


@app.route('/mosaic_thumbnail_image/<mode>/<mosaic_id>/<filter>')
def mosaic_thumbnail_image_complete(mode, mosaic_id, filter):
  key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)
  if key in cache:
    result = cache[key]
    print('type(result)', type(result))
    buffer = io.BytesIO()
    buffer.write(result['merged']['image'])
    buffer.seek(0)
    return flask.send_file(buffer, mimetype='image/png')

  return Response(status=404)


@app.route('/mosaic_thumbnail_image_for_filter/<mosaic_id>/<filter>')
def mosaic_thumbnail_image_for_filter(mosaic_id, filter):
  for mode in ['master', 'subframe']:
    key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)
    if key in cache:
      result = cache[key]
      print('type(result)', type(result))
      buffer = io.BytesIO()
      buffer.write(result['merged']['image'])
      buffer.seek(0)
      return flask.send_file(buffer, mimetype='image/png')


@app.route('/mosaic_thumbnail_image/<mosaic_id>')
def mosaic_thumbnail_image(mosaic_id):
  for mode in ['master', 'subframe']:
    for filter in ['H', 'L', 'S', 'O', 'G', 'R', 'B']:
      key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)
      if key in cache:
        buffer = io.BytesIO()
        buffer.write(cache[key])
        buffer.seek(0)
        return flask.send_file(buffer, mimetype='image/png')


@app.route('/stack_master_image/<stack_id>')
def stack_master_image(stack_id):
  # stack = next(s for s in cache['stacks'] if s.id == stack_id)
  stack = cache['stacks'][stack_id]

  if not stack:
    print(f'did not get stack for {stack_id}')
  else:
    print(f'got stack for {stack_id}')

  if stack.master and os.path.exists(stack.master):
    image = image_previews._read_image(stack.master)
    image = image_previews.resize_array(image, new_width=640)
    image = image_previews.autostretch(image)
    image = image_previews.array_to_png(image)
    buffer = io.BytesIO()
    buffer.write(image)
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

# @app.route('/quarantine_target/<target_id>')
# def quarantine_target(target_id):
#   target = cache.get('targets', {})[target_id]
#   stacks = cache.get('stacks', {})
#   quarantined_files = cache.get('quarantined_files', [])

#   num_files_quarantined = 0
#   for stack_id in target.stack_ids:
#     stack = stacks[stack_id]
#     for run in stack.runs.values():
#       for h in run.frames:
#         quarantined_files.append(h)
#         num_files_quarantined += 1

#   print(f'### Quarantined {num_files_quarantined} files')
#   cache['quarantined_files'] = quarantined_files

#   return Response(status=204)

# @app.route('/quarantine_stack/<stack_id>')
# def quarantine_stack(stack_id):
#   stack = cache['stacks'][stack_id]
#   quarantined_files = cache['quarantined_files']

#   num_files_quarantined = 0
#   for run in stack.runs.values():
#     for h in run.frames:
#       quarantined_files.append(h)
#       num_files_quarantined += 1

#   print(f'### Quarantined {num_files_quarantined} files')
#   cache['quarantined_files'] = quarantined_files

#   return Response(status=204)


@app.route('/recent')
def recent():
  fits_headers = cache['fits_headers']
  stacks = cache['stacks']
  targets = cache['targets']

  def last_frame_datetime(target_id):
    target = targets[target_id]
    for stack_id in target.stack_ids:
      stack = stacks[stack_id]
      if stack.geometric_hash == None:
        continue

      return max(fits_headers[h]['DATE-OBS'] for h in stack.all_frames(stacks))

  return sorted(targets, key=last_frame_datetime, reverse=True)


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


def run_astroflow_repeatedly():

  def ok_to_run():
    with cache.transact():
      if not cache.get('astroflow_running', False):
        cache['astroflow_running'] = True
        return True

    return False

  while True:
    if ok_to_run():
      # TODO: An empty context won't work. The user will have to fill stuff in
      # in the UI before the first run here.
      print('Running AstroFlow...')
      # context = cache.get('context', {})
      # astro_flow_direct.main(context)
      print('AstroFlow done')
      cache['astroflow_running'] = False
    else:
      print('Cannot run AstroFlow because it is already running')

    print('Waiting 15 minutes to run AstroFlow again...')
    run_now.acquire()
    run_now.wait(15 * 60)
    run_now.release()


if __name__ == '__main__':
  cache['astroflow_running'] = False

  thread = threading.Thread(target=run_astroflow_repeatedly)
  thread.start()

  app.run(host='0.0.0.0', port=80)
