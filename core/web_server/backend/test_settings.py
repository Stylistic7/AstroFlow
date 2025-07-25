import datetime
import httpx
import threading
import queue
import time
import pytest

from compute_graph import build_compute_graph
from testing import copy_testdata_frames
from context import context
from dependency_graph import run_graph
from state.gunicorn_control import GunicornManager
from state.gunicorn_configs import guincorn_configs


def wait_for_sse_event(q: queue.Queue, ignore_heartbeats=True):
  with httpx.stream('GET', 'http://localhost:8001/stream', timeout=None) as r:
    buffer = ''
    for line in r.iter_lines():
      if line == '':
        if buffer.startswith('event:') or buffer.startswith('data:'):
          if not buffer.startswith('event: heartbeat') or not ignore_heartbeats:
            q.put(buffer.strip())
            return
        buffer = ''
      else:
        buffer += line + '\n'


@pytest.mark.exclusive
def test_update_settings(test_context):
  copy_testdata_frames(prefix='light_run_0',
                       target_name='NGC7635',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0])

  run_graph(build_compute_graph())

  static_server = GunicornManager(guincorn_configs()['static_server']).start()
  stream_server = GunicornManager(guincorn_configs()['stream_server']).start()

  time.sleep(5)

  event_q = queue.Queue()
  listener_thread = threading.Thread(target=wait_for_sse_event, args=(event_q,))
  listener_thread.start()

  assert context.ui_stacks is not None

  response = httpx.get('http://localhost:8000/stacks')
  assert response.status_code == 200
  stacks = response.json()
  assert len(stacks) == 1
  stack_id = next(iter(stacks))
  stack = stacks[stack_id]

  runs = stack['contents']
  assert len(runs) == 1
  run_id = next(iter(runs))

  params = {
      'type': 'flat',
      'stack_id': stack_id,
      'run_id': run_id,
      'filename': 'foo.fits'
  }
  response = httpx.post('http://localhost:8000/api/select_file', json=params)

  try:
    sse = event_q.get(timeout=10)
    print("### Got SSE:", sse)
  except queue.Empty:
    raise TimeoutError("Timed out waiting for SSE event")

  response = httpx.get('http://localhost:8000/stacks')
  assert response.status_code == 200
  stacks = response.json()
  assert len(stacks) == 1
  stack_id = next(iter(stacks))
  stack = stacks[stack_id]

  assert len(stack['contents']) == 1
  run = next(iter(stack['contents'].values()))
  assert run['dependencies'].get('flat', None) == {'filename': 'foo.fits'}

  static_server.stop()
  stream_server.stop()
