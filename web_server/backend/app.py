#!/usr/bin/env python
import sys

sys.path.append('../..')

import io
import numpy as np
import flask
from flask import jsonify, Response
from dataclasses import asdict
import argparse
import shutil
import threading
from base64 import b64encode
import time

import astro_flow_direct
import constellations
from astro_flow_direct import Context
import image_handling
from cache import cache
from . import test_app
import plate_solving
import prev_results

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
  return flask.render_template('index.html', stacks=cache['stacks'])


run_now = threading.Condition()


@app.route('/run')
def run():
  run_now.acquire()
  run_now.notify()
  run_now.release()

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
  return test_app.mosaics if debug else cache['mosaics']


@app.route('/targets')
def targets():
  return test_app.targets if debug else cache['targets']


@app.route('/stacks')
def stacks():
  return test_app.stacks if debug else cache['stacks']


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


@app.route('/stack_thumbnail_image/<stack_id>')
def stack_thumbnail_image(stack_id):
  buffer = io.BytesIO()
  buffer.write(cache[('stack_thumbnail_image', stack_id)])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='image/png')


@app.route('/stack_thumbnail_video/<stack_id>')
def stack_thumbnail_video(stack_id):
  buffer = io.BytesIO()
  buffer.write(cache[('stack_thumbnail_video', stack_id)])
  buffer.seek(0)
  return flask.send_file(buffer, mimetype='video/webm')


@app.route('/stack_master_image/<stack_id>')
def stack_master_image(stack_id):
  # stack = next(s for s in cache['stacks'] if s.id == stack_id)
  stack = cache['stacks'][stack_id]

  if not stack:
    print(f'### did not get stack for {stack_id}')
  else:
    print(f'### got stack for {stack_id}')

  if stack.master and os.path.exists(stack.master):
    image = image_handling._read_image(stack.master)
    image = image_handling.resize_array(image, new_width=640)
    image = image_handling.autostretch(image)
    image = image_handling.array_to_png(image)
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


@app.route('/get_num_stars_histogram_html/<stack_id>')
def get_num_stars_histogram_html(stack_id):
  print(f'Creating num stars histogram for stack id {stack_id}')
  stack = cache['stacks'][stack_id]
  num_stars = [
      p.num_stars for k, p in cache['plate_solutions'].items()
      if k in stack.light_frames
  ]
  print('num_stars', num_stars)

  colors = ['#A56CC1', '#A6ACEC']
  fig = ff.create_distplot([num_stars], ['# Stars'],
                           colors=colors,
                           curve_type='normal',
                           bin_size=20,
                           show_rug=False)

  # fig.update_layout(title_text='Hist and Curve Plot')

  buffer = io.StringIO()
  fig.write_html(buffer, full_html=False)
  mem = io.BytesIO()
  mem.write(buffer.getvalue().encode())
  mem.seek(0)
  return flask.send_file(mem, mimetype='text/html')


# TODO: Don't hardcode the quarantine directory.
quarantine_dir = "/Users/nobody/Projects/Astro/Raw/Bad/"


@app.route('/delete_target/<target_id>')
def delete_target(target_id):
  # target = next(t for t in cache['targets'] if t.id == target_id)
  target = cache['targets'][target_id]
  for stack_id in target.stack_ids:
    # stack = next(s for s in cache['stacks'] if s.id == stack_id)
    stack = cache['stacks'][stack_id]
    for filename in stack.light_frames:
      print(f'### Quarantining file {filename}')
      shutil.move(filename, quarantine_dir)

  resp = jsonify(success=True)
  return resp


def run_astroflow_repeatedly():
  def ok_to_run():
    with cache.transact():
      if not cache.get('astroflow_running', False):
        cache['astroflow_running'] = True
        return True

    return False

  while True:
    if ok_to_run():
      print('Running AstroFlow...')
      context = cache.get('context', {})
      # astro_flow_direct.main(context)
      time.sleep(10)
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
