import ray
import os
from typing import Any, Callable, TypeVar
import queue

from context import context

T = TypeVar('T')

# TODO: Actions to add:
# add_astroflow_instance (from zeroconf)
# remove_astroflow_instance (from zeroconf)
# start Ray head node (from web UI or startup code)
# start Ray follower node (from web UI or startup code)
# volume attached
# volume detached

_state = dict()
_action_queue = queue.Queue()


def get_state(key: str):
  return _state.get(key, None)


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


class Action(dict):
  pass


class State(dict):
  pass


def _start_ray_head_node(state: State,
                         action: Action) -> tuple[State, list[Action]]:
  pass


def _start_ray_subordinate_node(state: State,
                                action: Action) -> tuple[State, list[Action]]:
  pass


def _start_flask(state: State, action: Action) -> tuple[State, list[Action]]:
  from state.flask_control import FlaskThread
  server_thread = FlaskThread()
  server_thread.start()
  state['flask_server_thread'] = server_thread

  return state, []


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


def dispatch(action: Action) -> dict[str, Any]:
  global _state

  print(f'### handling action {action}')
  action = dict(action)
  new_actions = []

  match action['type']:
    case 'add_to_shopping_cart':
      _state, new_actions = _add_to_shopping_cart(_state, action)

    case 'remove_from_shopping_cart':
      _state, new_actions = _remove_from_shopping_cart(_state, action)

    # TODO: This will hang the main thread forever; these need to be
    # started asynchronously.
    case 'run_processing_graph':
      print('### Pretending to run processing graph')
      # run_graph(build_item_processing_graph(action['item']))

    case 'update_zeroconf_services':
      print('### Updating zeroconf services:')
      print(action['instances'])

    case 'start_flask':
      _state, new_actions = _start_flask(_state, action)

    case _:
      raise ValueError(f'Invalid action {action}')

  for action in new_actions:
    dispatch(action)

  return _state


def enqueue_action(action):
  _action_queue.put(action)


def main_event_loop():
  while True:
    action = _action_queue.get()
    dispatch(action)
    _action_queue.task_done()
