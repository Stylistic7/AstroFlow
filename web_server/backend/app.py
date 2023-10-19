#!/usr/bin/env python
import sys

sys.path.append('../..')

import io
import numpy as np
import flask
from flask import jsonify
from dataclasses import asdict
import argparse
import shutil

import image_handling
from cache import cache
from . import test_app
import plate_solving

import plotly.express as px
from base64 import b64encode

debug = False


def get_plot_html():
  buffer = io.StringIO()

  df = px.data.iris()  # replace with your own data source
  fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
  fig.write_html(buffer)

  # html_bytes = buffer.getvalue().encode()
  # encoded = b64encode(html_bytes).decode()
  # return encoded

  return buffer.getvalue()


# def build_stack(stack):
#   print('build_stack', stack)

#   # FIXME where did the footprint go?
#   print('build_stack stack.footprint', stack.footprint)

#   return asdict(stack) | {
#       'name': stack.target,
#       'image': {
#           'url':
#           'https://img.freepik.com/free-photo/beagle-tricolor-puppy-is-posing_155003-31772.jpg?w=1480&t=st=1694757101~exp=1694757701~hmac=91cd6aaff87cb9502773c90d36e3bab81655a074a27dae5055a5dbf5a85d11df',
#       },
#       'video': {
#           'url': f'/thumbnail_video/{stack.target}/{stack.rig}/{stack.filter}'
#       },
#       'plot': {
#           # FIXME why do I need to include the hostname for iframes?
#           'url': f'http://localhost:80/get_plot'
#       },
#       'quantity': 1,
#       'status': 'Pending',
#       'updated_at': '2014-02-11',
#   }

# def build_target(index, target, stacks):
#   print('target', target)
#   print(f'target.stacks {target.stacks}')
#   stacks = [next(s for s in stacks if s.id == id) for id in target.stacks]
#   print(f'Found {len(stacks)} stacks')

#   return {
#       'id': index,
#       'target': f'Target {target.name}',
#       'status': 'In Progress',
#       'updated_at': '2014-02-11',
#       'user': {
#           'name': 'Stylistic7',
#           'email': 'Stylistic7@proton.me',
#           'phone': '408-910-3224',
#       },
#       'stacks': [build_stack(s) for s in stacks],
#       'total': 123.45,
#   }

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


# @app.route('/target_footprints')
# def target_footprints():
#   targets_and_light_frames = metaflow_data_helper.find(
#       'spatial_grouping', 'targets_and_light_frames') or []
#   plate_solutions = metaflow_data_helper.find('plate_solve',
#                                               'plate_solutions') or {}

#   output = {}
#   for target, light_frames in targets_and_light_frames:
#     print(f'target {target}')
#     ps = [plate_solutions[light_frame] for light_frame in light_frames]
#     footprints = [f for _, f, _ in ps]
#     # print('footprints', footprints)
#     output[target] = footprints

#   return output


@app.route('/')
def index():
  """ Displays the index page accessible at '/'"""
  return flask.render_template('index.html', stacks=cache['stacks'])


@app.route('/hello/<name>/')
def hello(name):
  """ Displays the page greats who ever comes to visit it."""
  return flask.render_template('hello.html', name=name)


@app.route('/starmap')
def starmap():
  return flask.render_template('index.html')


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


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)
