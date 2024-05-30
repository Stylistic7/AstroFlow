import ray
from addict import Dict
import os

from context import context
from dependency_graph import Graph, run_graph
from compute_graph import build_item_processing_graph

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
  print(f'### Using mock namespace {namespace}')
else:
  namespace = 'AstroFlow'


class Action(Dict):
  pass


class State(Dict):
  pass


def _add_to_shopping_cart(state, action):
  if not context.shopping_cart:
    context.shopping_cart = []

  if action.item not in context.shopping_cart:
    context.shopping_cart += [action.item]
    new_action = Dict(type='run_processing_graph', item=action.item)
    return state, [new_action]

  return state, []


def _remove_from_shopping_cart(state, action):
  if not context.shopping_cart:
    context.shopping_cart = []

  print(f'### _remove_from_shopping_cart item {action.item}')
  print(f'### current shopping cart is {context.shopping_cart}')
  if action.item in context.shopping_cart:
    # TODO: Is there a cleaner way to do this? Can I use the -= operator?
    new_shopping_cart = context.shopping_cart
    new_shopping_cart.remove(action.item)
    context.shopping_cart = new_shopping_cart

  # TODO: Consider canceling running graphs when items are removed from
  # the shopping cart.
  return state, []


@ray.remote(resources={'diskcache': 0.001})
class StateActor:

  def __init__(self):
    self.state = Dict()

  def dispatch(self, action):
    print(f'### handling action {action}')
    action = Dict(action)

    match action.type:
      case 'add_to_shopping_cart':
        self.state, new_actions = _add_to_shopping_cart(self.state, action)

      case 'remove_from_shopping_cart':
        self.state, new_actions = _remove_from_shopping_cart(self.state, action)

      # TODO: This will hang the main thread forever; these need to be
      # started asynchronously.
      case 'run_processing_graph':
        print('### Pretending to run processing graph')
        # run_graph(build_item_processing_graph(action.item))
        new_actions = []

      case _:
        raise ValueError(f'Invalid action {action}')

    for action in new_actions:
      self.dispatch(action)

    return self.state.to_dict()


def dispatch(action, callback=None):
  actor = ray.get_actor('state', namespace=namespace)
  result = ray.get(actor.dispatch.remote(action))
  if callback:
    print('### Calling callback')
    callback(result)
  return result
