import os
from typing import TypeVar
from pathlib import Path
import time
import signal
import sys
import datetime

from context import context, get_cache, queue
from ray_infra.ray_init import connect_to_ray_cluster
from state.ray_control import start_ray_head_node, start_ray_subordinate_node, stop_ray_node
from state.astroflow_zeroconf import AstroFlowZeroconf
from asiair.asiair_zeroconf import ASIAIRZeroconf
from state.types import State, Action
from dependency_graph import Graph, GraphStatus, TopologicalSorterWithStatus
from previews.types import QualityLevel
import pix

T = TypeVar('T')

# This state machine is intended to be low-level. It runs on a single machine,
# a single "node," and is never distributed. It is responsible for initializing
# Ray, using zeroconf for finding other Ray nodes, and other tasks.
#
# It does not do AstroFlow-specific work. That kind of work -- like running
# compute graphs, etc. -- should be done by a Ray actor, which is a singleton
# and accessible by any node in the Ray cluster.

# TODO: Actions to add:
# volume attached
# volume detached

_state = dict()

QUEUE_NAME = 'node_state_machine'


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def get_state(key: str):
  return _state.get(key, None)


def _init(state: State, action: Action) -> tuple[State, list[Action]]:
  new_actions = []

  new_actions.append(Action(type='prepare_pix'))
  new_actions.append(Action(type='start_gunicorn'))
  new_actions.append(Action(type='start_astroflow_zeroconf'))

  cache = get_cache()
  if cache.get('ray_head_node_enabled', False):
    new_actions.append(Action(type='start_ray_head_node'))

  return state, new_actions


def _populate_context(state: State,
                      action: Action) -> tuple[State, list[Action]]:
  # TODO: Put these in the UI.
  context.input_dirs = [Path('/Volumes/Astro/Raw'), Path('/Volumes/Astro2/Raw')]
  context.output_dir = Path('/Volumes/Astro/Processed')

  context.cosine_similarity_threshold = 0.9999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160

  context.asiair_devices = []
  context.compute_graphs = [TopologicalSorterWithStatus(Graph())]
  context.compute_graph_status = GraphStatus()

  return state, []


def _start_astroflow_zeroconf(state: State,
                              action: Action) -> tuple[State, list[Action]]:
  assert 'astroflow_zeroconf' not in state
  state['astroflow_zeroconf'] = AstroFlowZeroconf(enqueue_action)

  return state, []


def _start_asiair_zeroconf(state: State,
                           action: Action) -> tuple[State, list[Action]]:
  assert 'asiair_zeroconf' not in state
  state['asiair_zeroconf'] = ASIAIRZeroconf(enqueue_action)

  return state, []


def _advertise_zeroconf(state: State,
                        action: Action) -> tuple[State, list[Action]]:
  state['astroflow_zeroconf'].update_state(is_ray_head=True)
  return state, []


def _stop_zeroconf(state: State, action: Action) -> tuple[State, list[Action]]:
  if 'astroflow_zeroconf' in state:
    print("Stopping AstroFlow peer discovery...")
    state['astroflow_zeroconf'].close()
    del state['astroflow_zeroconf']

  if 'asiair_zeroconf' in state:
    print("Stopping ASIAir peer discovery...")
    state['asiair_zeroconf'].close()
    del state['asiair_zeroconf']

  return state, []


def _start_ray_head_node(state: State,
                         action: Action) -> tuple[State, list[Action]]:
  assert 'ray_process' not in state
  state['ray_process'] = start_ray_head_node()

  from web_server.backend.announcer import announce
  announce('ray_status')

  connect_to_ray_cluster()
  time.sleep(5)

  return state, [
      Action(type='populate_context'),
      Action(type='advertise_zeroconf'),
      Action(type='start_asiair_zeroconf'),
      Action(type='schedule_download_from_asiairs')
  ]


def _stop_ray_node(state: State, action: Action) -> tuple[State, list[Action]]:
  if 'ray_process' not in state:
    return state, []

  stop_ray_node()
  del state['ray_process']
  return state, []


def _update_astroflow_services(state: State,
                               action: Action) -> tuple[State, list[Action]]:
  cache = get_cache()
  if cache.get('ray_head_node_enabled', False):
    return state, []

  if 'ray_process' in state:
    return state, []

  # TODO: This should be a follow-on action.
  instance = next(iter(action['instances'].values()))
  print(f'### connecting to head node described by instance {instance}')
  state['ray_process'] = start_ray_subordinate_node(instance['ip'])

  from web_server.backend.announcer import announce
  announce('ray_status')

  connect_to_ray_cluster()

  return state, []


def _update_asiair_services(state: State,
                            action: Action) -> tuple[State, list[Action]]:
  print(f"### _update_asiair_services {action=}")
  context.asiair_devices = action['devices']

  return state, []


def _start_flask(state: State, action: Action) -> tuple[State, list[Action]]:
  from state.flask_control import FlaskThread
  server_thread = FlaskThread()
  server_thread.start()
  state['flask_thread'] = server_thread

  return state, []


def _stop_flask(state: State, action: Action) -> tuple[State, list[Action]]:
  if 'flask_thread' not in state:
    return state, []

  state['flask_thread'].stop()
  del state['flask_thread']

  return state, []


def _prepare_pix(state: State, action: Action) -> tuple[State, list[Action]]:
  pix.prepare_cache()
  return state, []


def _start_gunicorn(state: State, action: Action) -> tuple[State, list[Action]]:
  from state.gunicorn_control import GunicornManager
  from state.gunicorn_configs import guincorn_configs
  state['gunicorn_managers'] = [
      GunicornManager(c).start() for c in guincorn_configs().values()
  ]

  return state, []


def _stop_gunicorn(state: State, action: Action) -> tuple[State, list[Action]]:
  if 'gunicorn_managers' not in state:
    return state, []

  for manager in state['gunicorn_managers']:
    manager.stop()

  del state['gunicorn_managers']

  return state, []


def _schedule_download_from_asiairs(
    state: State, action: Action) -> tuple[State, list[Action]]:
  # TODO: Put the polling interval in the UI.
  import ray
  scheduler = ray.get_actor("scheduler", namespace=_get_namespace())
  scheduler.add.remote(interval_s=15 * 60,
                       action=dict(type='download_from_asiairs'),
                       start_delay_s=30,
                       ignore_if_busy=True)

  return state, []


def _quit(state: State, action: Action) -> tuple[State, list[Action]]:
  return state, [
      Action(type='stop_ray_node'),
      Action(type='stop_gunicorn'),
      Action(type='stop_zeroconf')
  ]


def _add_to_shopping_cart(state: State,
                          action: Action) -> tuple[State, list[Action]]:
  if not context.shopping_cart:
    context.shopping_cart = []

  if action['item'] not in context.shopping_cart:
    context.shopping_cart += [action['item']]
    new_action = dict(type='run_processing_graph', item=action['item'])
    return state, [new_action]

  return state, []


def _remove_from_shopping_cart(state: State,
                               action: Action) -> tuple[State, list[Action]]:
  if not context.shopping_cart:
    context.shopping_cart = []

  print(f'### _remove_from_shopping_cart item {action["item"]}')
  print(f'### current shopping cart is {context.shopping_cart}')
  if action['item'] in context.shopping_cart:
    # TODO: Is there a cleaner way to do this? Can I use the -= operator?
    new_shopping_cart = context.shopping_cart
    new_shopping_cart.remove(action['item'])
    context.shopping_cart = new_shopping_cart

  # TODO: Consider canceling running graphs when items are removed from
  # the shopping cart.
  return state, []


def _graph_done(state: State, action: Action) -> tuple[State, list[Action]]:
  state['processing'].remove(action['graph_id'])

  return state, []


def dispatch(action: Action) -> State:
  global _state

  action = dict(action)
  new_actions = []

  # TODO: Eliminate this match.
  match action['type']:
    case 'noop':
      _state, new_actions = _state, []

    case 'init':
      _state, new_actions = _init(_state, action)

    case 'start_astroflow_zeroconf':
      _state, new_actions = _start_astroflow_zeroconf(_state, action)

    case 'start_asiair_zeroconf':
      _state, new_actions = _start_asiair_zeroconf(_state, action)

    case 'advertise_zeroconf':
      _state, new_actions = _advertise_zeroconf(_state, action)

    case 'update_astroflow_services':
      _state, new_actions = _update_astroflow_services(_state, action)

    case 'update_asiair_services':
      _state, new_actions = _update_asiair_services(_state, action)

    case 'stop_zeroconf':
      _state, new_actions = _stop_zeroconf(_state, action)

    case 'prepare_pix':
      _state, new_actions = _prepare_pix(_state, action)

    case 'add_to_shopping_cart':
      _state, new_actions = _add_to_shopping_cart(_state, action)

    case 'remove_from_shopping_cart':
      _state, new_actions = _remove_from_shopping_cart(_state, action)

    case 'start_flask':
      _state, new_actions = _start_flask(_state, action)

    case 'stop_flask':
      _state, new_actions = _stop_flask(_state, action)

    case 'start_gunicorn':
      _state, new_actions = _start_gunicorn(_state, action)

    case 'stop_gunicorn':
      _state, new_actions = _stop_gunicorn(_state, action)

    case 'start_ray_head_node':
      _state, new_actions = _start_ray_head_node(_state, action)

    case 'stop_ray_node':
      _state, new_actions = _stop_ray_node(_state, action)

    case 'schedule_download_from_asiairs':
      _state, new_actions = _schedule_download_from_asiairs(_state, action)

    case 'populate_context':
      _state, new_actions = _populate_context(_state, action)

    case 'graph_done':
      _state, new_actions = _graph_done(_state, action)

    case 'quit':
      _state, new_actions = _quit(_state, action)

    case _:
      raise ValueError(f'Invalid action {action}')

  for action in new_actions:
    dispatch(action)

  return _state


def enqueue_action(action):
  queue.put(QUEUE_NAME, action)


ctrl_c_count = 0


def handle_sigint(sig, frame):
  global ctrl_c_count
  ctrl_c_count += 1
  if ctrl_c_count == 1:
    print(f'Got first ctrl-c, enqueuing graceful shutdown action')
    enqueue_action(dict(type='quit'))
  else:
    print(f'Got second ctrl-c, forcing exit')
    sys.exit(1)


signal.signal(signal.SIGINT, handle_sigint)


def main_event_loop():
  quit_received = False

  try:
    while True:
      action = queue.get(QUEUE_NAME)
      if action is None:
        time.sleep(0.1)
        continue

      try:
        if action.get("type") == "quit":
          quit_received = True

        dispatch(action)
      except Exception as e:
        import traceback
        print(f"main_event_loop action {action['type']} raised exception {e}")
        traceback.print_exc()

      if quit_received and queue.empty(QUEUE_NAME) and _state.get(
          'processing', []) == []:
        break

  except Exception as e:
    print(f"Unexpected error: {e}")
    raise

  print("Graceful shutdown complete.")
