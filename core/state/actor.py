import ray
import os
from enum import Enum
from state.types import Action, State

from context import context
from stacks.compute_graph import process_stacks_for_ui
from dependency_graph import Graph, run_graph, chain
from customization.compute_graph import customize_stack, store_customized_stack
from mosaics.compute_graph import build_mosaic_subgraph, stacks_for_mosaic, targets_for_mosaic, process_mosaics_for_ui
from previews.types import QualityLevel
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from stacks.compute_graph import build_stacking_subgraph
from targets.post_process import post_process_target
from stacks.post_process import post_process_stack
from mosaics.post_process import post_process_mosaic


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


def _process_stack(state: State, action: Action) -> tuple[State, list[Action]]:
  stack = context.stacks[action['object_id']]

  run_graph(
      Graph({**chain(
          (build_stacking_subgraph, stack),
          process_stacks_for_ui,
      )}))

  return state, [Action(type='announce', message='reload_data')]


def _process_target(state: State, action: Action) -> tuple[State, list[Action]]:
  target = context.targets[action['object_id']]

  run_graph(
      Graph({**chain(
          (build_target_subgraph, target),
          process_targets_for_ui,
      )}))

  return state, [Action(type='announce', message='reload_data')]


def _process_mosaic(state: State, action: Action) -> tuple[State, list[Action]]:
  mosaic = context.mosaics[action['object_id']]

  tasks = []
  for mode in ['subframe', 'master']:
    if action['mode'] in ['all', mode]:
      stacks = context.stacks
      targets = context.targets
      task = (build_mosaic_subgraph, QualityLevel(mode), mosaic,
              stacks_for_mosaic(mosaic, stacks,
                                targets), targets_for_mosaic(mosaic, targets))
      tasks.append(task)

  tasks.append(process_mosaics_for_ui)

  run_graph(Graph({command_done: tasks}))

  return state, [Action(type='announce', message='reload_data')]


def _post_process(state: State, action: Action) -> tuple[State, list[Action]]:
  print(f'### _post_process {action=}')
  if action['object_type'] == 'stack':
    stack = context.stacks[action['object_id']]
    run_graph(Graph({(post_process_stack, stack): {}}))
  elif action['object_type'] == 'target':
    target = context.targets[action['object_id']]
    run_graph(Graph({(post_process_target, target): {}}))
  elif action['object_type'] == 'mosaic':
    mosaic = context.mosaics[action['object_id']]
    run_graph(Graph({(post_process_mosaic, mosaic): {}}))

  return state, [Action(type='announce', message='reload_data')]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class StateActor:

  def __init__(self):
    self.state = State()

  def dispatch(self, action: Action) -> State:
    new_actions = []

    print(f"### dispatch action type {action['type']}")

    match action['type']:
      case 'noop':
        new_actions = []
      case 'alert':
        print(f'### ALERT')
        new_actions = []
      case 'announce':
        self.state, new_actions = _announce(self.state, action)
      case 'process':
        if action['object_type'] == 'stack':
          self.state, new_actions = _process_stack(self.state, action)
        if action['object_type'] == 'target':
          self.state, new_actions = _process_target(self.state, action)
        elif action['object_type'] == 'mosaic':
          self.state, new_actions = _process_mosaic(self.state, action)
      case 'post_process':
        self.state, new_actions = _post_process(self.state, action)
      case 'update_settings':
        self.state, new_actions = _update_settings(self.state, action)
      case _:
        raise ValueError(f"Unexpected action type: {action['type']}")

    for action in new_actions:
      self.dispatch(action)

    return self.state


class State:

  def dispatch(self, action: Action) -> State:
    actor = ray.get_actor('state', namespace=_get_namespace())
    # TODO: Do I want this to synchronous?
    return ray.get(actor.dispatch.remote(action))


state = State()
