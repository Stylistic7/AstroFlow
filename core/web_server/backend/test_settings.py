import pytest

import datetime
import unittest
from web_server.backend import app

from compute_graph import build_compute_graph
from testing import make_test_context, copy_testdata_frames
from context import context
from dependency_graph import run_graph
from state.actor import state


class TestUpdateSettings(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

    app.config['TESTING'] = True
    self.client = app.test_client()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_stacks(self):
    copy_testdata_frames(prefix='light_run_0',
                         target_name='NGC7635',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0])

    run_graph(build_compute_graph())

    response = self.client.get('/stacks')
    self.assertEqual(response.status_code, 200)

    stacks = response.get_json()
    self.assertEqual(len(stacks), 1)
    stack_id = next(iter(stacks))
    stack = stacks[stack_id]
    print(f'### stack {stack}')

    runs = stack['contents']
    self.assertEqual(len(runs), 1)
    run_id = next(iter(runs))

    params = {
        'type': 'flat',
        'stack_id': stack_id,
        'run_id': run_id,
        'filename': 'foo.fits'
    }
    response = self.client.put('/api/select_file', json=params)
