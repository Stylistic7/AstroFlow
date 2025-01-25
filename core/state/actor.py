import ray
import os
from typing import Any, Callable, TypeVar

from context import context

T = TypeVar('T')


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


class Action(dict):
  pass


class State(dict):
  pass


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


@ray.remote(resources={'diskcache': 0.001})
class StateActor:

  def __init__(self) -> None:
    self.state = dict()

  def dispatch(self, action: Action) -> dict[str, Any]:
    print(f'### handling action {action}')
    action = dict(action)

    match action['type']:
      case 'add_to_shopping_cart':
        self.state, new_actions = _add_to_shopping_cart(self.state, action)

      case 'remove_from_shopping_cart':
        self.state, new_actions = _remove_from_shopping_cart(self.state, action)

      # TODO: This will hang the main thread forever; these need to be
      # started asynchronously.
      case 'run_processing_graph':
        print('### Pretending to run processing graph')
        # run_graph(build_item_processing_graph(action['item']))
        new_actions = []

      case _:
        raise ValueError(f'Invalid action {action}')

    for action in new_actions:
      self.dispatch(action)

    return self.state


def dispatch(action: Action, callback: Callable[..., T] | None = None) -> T:
  actor = ray.get_actor('state', namespace=_get_namespace())
  result = ray.get(actor.dispatch.remote(action))
  if callback:
    print('### Calling callback')
    callback(result)
  return result
