import ray
import os
from enum import Enum
from state.types import Action, State

from context import context
from stacks.compute_graph import process_stacks_for_ui
from dependency_graph import Graph, run_graph
from customization.compute_graph import customize_stack, store_customized_stack
from mosaics.compute_graph import build_mosaic_subgraph, stacks_for_mosaic, targets_for_mosaic
from previews.types import QualityLevel


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def _announce(state: State, action: Action) -> tuple[State, list[Action]]:
  from web_server.backend.announcer import announce
  announce(action['message'])

  return state, []


def _update_settings(state: State,
                     action: Action) -> tuple[State, list[Action]]:
  stack = context.stacks[action['stack_id']]
  graph = Graph({
      store_customized_stack: {(customize_stack, stack, action['settings'])},
      process_stacks_for_ui: {store_customized_stack}
  })
  run_graph(graph)

  return state, [Action(type='announce', message='reload_data')]


@ray.remote
def command_done() -> None:
  pass


def _merge_mosaic(state: State, action: Action) -> tuple[State, list[Action]]:
  mosaic = context.mosaics[action['mosaic_id']]

  tasks = []
  for mode in ['subframe', 'master']:
    if action['mode'] in ['all', mode]:
      stacks = context.stacks
      targets = context.targets
      task = (build_mosaic_subgraph, QualityLevel(mode), mosaic,
              stacks_for_mosaic(mosaic, stacks,
                                targets), targets_for_mosaic(mosaic, targets))
      tasks.append(task)

  run_graph(Graph({command_done: tasks}))

  return state, [Action(type='announce', message='reload_data')]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class StateActor:

  def __init__(self):
    self.state = State()

  def dispatch(self, action: Action) -> State:
    new_actions = []

    print(f"### dispatch action type {action['type']}")

    match action['type']:
      case 'announce':
        self.state, new_actions = _announce(self.state, action)
      case 'merge_mosaic':
        self.state, new_actions = _merge_mosaic(self.state, action)
      case 'update_settings':
        self.state, new_actions = _update_settings(self.state, action)

    for action in new_actions:
      self.dispatch(action)

    return self.state


class State:

  def dispatch(self, action: Action) -> State:
    actor = ray.get_actor('state', namespace=_get_namespace())
    # TODO: Do I want this to synchronous?
    return ray.get(actor.dispatch.remote(action))


state = State()
