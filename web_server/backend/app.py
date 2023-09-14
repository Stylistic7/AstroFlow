import time
from flask import Flask

app = Flask(__name__)


@app.route('/time')
def get_current_time():
  return {'time': time.time()}


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80)

# # Import flask and datetime module for showing date and time
# from flask import Flask
# import datetime

# x = datetime.datetime.now()

# # Initializing flask app
# app = Flask(__name__)

# # Route for seeing a data
# @app.route('/data')
# def get_time():

#   # Returning an api for showing in reactjs
#   return {'Name': "geek", "Age": "22", "Date": x, "programming": "python"}

# # Running app
# if __name__ == '__main__':
#   app.run(debug=True)

# #!/usr/bin/env python
# import numpy as np
# import flask

# from cache import cache
# import metaflow_data_helper
# import plate_solving
# import io

# # Create the application.
# app = flask.Flask(__name__, template_folder='./templates')

# @app.route('/targets')
# def targets():
#   targets = metaflow_data_helper.find('spatial_grouping',
#                                       'targets_and_light_frames') or []
#   return targets

# # TODO: Move computations into new files.
# def min_footprint(footprints):
#   min_ra = np.min([p[0] for f in footprints for p in f])
#   max_ra = np.max([p[0] for f in footprints for p in f])
#   min_dec = np.min([p[1] for f in footprints for p in f])
#   max_dec = np.max([p[1] for f in footprints for p in f])

#   return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
#           [min_ra, max_dec]]

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

# @app.route('/')
# def index():
#   """ Displays the index page accessible at '/'"""
#   return flask.render_template('index.html', stacks=cache['stacks'])

# @app.route('/hello/<name>/')
# def hello(name):
#   """ Displays the page greats who ever comes to visit it."""
#   return flask.render_template('hello.html', name=name)

# @app.route('/footprints')
# def footprints():
#   plate_solutions = metaflow_data_helper.find('plate_solve',
#                                               'plate_solutions') or {}
#   return [f for _, f, _ in list(plate_solutions.values())]

# @app.route('/stacks')
# def stacks():
#   return cache['stacks']

# @app.route('/thumbnail_video/<target>/<rig>/<filter>')
# def thumbnail_video(target, rig, filter):
#   buffer = io.BytesIO()
#   buffer.write(cache['thumbnail_videos'][(target, rig, filter)])
#   buffer.seek(0)
#   print(f'video buffer has size {buffer.getbuffer().nbytes}')
#   return flask.send_file(buffer, mimetype='video/webm')

# if __name__ == '__main__':
#   app.debug = True
#   app.run(host='0.0.0.0', port=8000)
