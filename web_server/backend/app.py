#!/usr/bin/env python
import sys

sys.path.append('../..')

import numpy as np
import flask

from cache import cache
import metaflow_data_helper
import plate_solving
import io

import plotly.express as px
from base64 import b64encode


def get_plot_html():
  buffer = io.StringIO()

  df = px.data.iris()  # replace with your own data source
  fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
  fig.write_html(buffer)

  # html_bytes = buffer.getvalue().encode()
  # encoded = b64encode(html_bytes).decode()
  # return encoded

  return buffer.getvalue()


def build_stack(stack):
  return {
      'id': '1',
      'name': 'foo',
      'image': {
          'url':
          'https://img.freepik.com/free-photo/beagle-tricolor-puppy-is-posing_155003-31772.jpg?w=1480&t=st=1694757101~exp=1694757701~hmac=91cd6aaff87cb9502773c90d36e3bab81655a074a27dae5055a5dbf5a85d11df',
      },
      'video': {
          'url': f'/thumbnail_video/{stack.target}/{stack.rig}/{stack.filter}'
      },
      'plot': {
          # FIXME why do I need to include the hostname for iframes?
          'url': f'http://localhost:80/get_plot'
      },
      'product': {
          'name': stack.target,
          'sku': f'{stack.rig} {stack.filter} {len(stack.footprint)} vertices',
      },
      'quantity': 1,
      'status': 'Pending',
      'updated_at': '2014-02-11',
  }


def build_mosaic(index, mosaic):
  return {
      'id': index,
      'order_number': f'Mosaic {len(mosaic)} stack',
      'status': 'In Progress',
      'updated_at': '2014-02-11',
      'user': {
          'name': 'Stylistic7',
          'email': 'Stylistic7@proton.me',
          'phone': '408-910-3224',
      },
      'order_items': [build_stack(s) for s in mosaic],
      'total': 123.45,
  }


# Create the application.
app = flask.Flask(__name__, template_folder='./templates')


@app.route('/targets')
def targets():
  targets = metaflow_data_helper.find('spatial_grouping',
                                      'targets_and_light_frames') or []
  return targets


# TODO: Move computations into new files.
def min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


@app.route('/target_footprints')
def target_footprints():
  targets_and_light_frames = metaflow_data_helper.find(
      'spatial_grouping', 'targets_and_light_frames') or []
  plate_solutions = metaflow_data_helper.find('plate_solve',
                                              'plate_solutions') or {}

  output = {}
  for target, light_frames in targets_and_light_frames:
    print(f'target {target}')
    ps = [plate_solutions[light_frame] for light_frame in light_frames]
    footprints = [f for _, f, _ in ps]
    # print('footprints', footprints)
    output[target] = footprints

  return output


@app.route('/')
def index():
  """ Displays the index page accessible at '/'"""
  return flask.render_template('index.html', stacks=cache['stacks'])


@app.route('/hello/<name>/')
def hello(name):
  """ Displays the page greats who ever comes to visit it."""
  return flask.render_template('hello.html', name=name)


@app.route('/footprints')
def footprints():
  plate_solutions = metaflow_data_helper.find('plate_solve',
                                              'plate_solutions') or {}
  return [f for _, f, _ in list(plate_solutions.values())]


@app.route('/stacks')
def stacks():
  return cache['stacks']


@app.route('/thumbnail_video/<target>/<rig>/<filter>')
def thumbnail_video(target, rig, filter):
  buffer = io.BytesIO()
  buffer.write(cache['thumbnail_videos'][(target, rig, filter)])
  buffer.seek(0)
  print(f'video buffer has size {buffer.getbuffer().nbytes}')
  return flask.send_file(buffer, mimetype='video/webm')


@app.route('/get_plot')
def plot():
  mem = io.BytesIO()
  mem.write(get_plot_html().encode())
  mem.seek(0)
  return flask.send_file(mem, mimetype='text/html')


@app.route('/state')
def state():
  return [build_mosaic(i, m) for i, m in enumerate(cache['mosaics'])]


if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0', port=80)
