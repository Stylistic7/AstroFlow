import ray
from addict import Dict
from cache import cache
import os

from dependency_graph import Graph, run_graph

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
  print(f'### Using mock namespace {namespace}')
else:
  namespace = 'AstroFlow'


class Action(Dict):
  pass


class State(Dict):
  pass


def add_to_shopping_cart(state, action):
  if action.item not in state.shopping_cart:
    state.shopping_cart.append(action.item)

  return state


def remove_from_shopping_cart(state, action):
  if action.item in state.shopping_cart:
    state.shopping_cart.remove(action.item)

  return state


@ray.remote(resources={'diskcache': 0.001})
class StateActor:

  def __init__(self):
    self.state = Dict(shopping_cart=[])

  def dispatch(self, action):
    action = Dict(action)
    match action.type:
      case 'add_to_shopping_cart':
        self.state = add_to_shopping_cart(self.state, action)

      case 'remove_from_shopping_cart':
        self.state = remove_from_shopping_cart(self.state, action)

    return self.state.to_dict()


def dispatch(action):
  actor = ray.get_actor('ray_state', namespace=namespace)
  return ray.get(actor.dispatch.remote(action))
