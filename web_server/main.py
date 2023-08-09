#!/usr/bin/env python

import flask

import metaflow_data_helper
import plate_solving
import io

# Create the application.
app = flask.Flask(__name__, template_folder='./templates')


@app.route('/')
def index():
  """ Displays the index page accessible at '/'"""
  return flask.render_template('index.html')


@app.route('/hello/<name>/')
def hello(name):
  """ Displays the page greats who ever comes to visit it."""
  return flask.render_template('hello.html', name=name)


@app.route('/footprints')
def footprints():
  plate_solutions = metaflow_data_helper.find('plate_solve',
                                              'plate_solutions') or {}

  # TODO: This is quite slow.
  footprints = [
      plate_solving.calc_footprint(io.BytesIO(wcs)).tolist()
      for wcs, _ in list(plate_solutions.values())
  ]

  return footprints


if __name__ == '__main__':
  app.debug = True
  app.run(host='0.0.0.0', port=8000)
