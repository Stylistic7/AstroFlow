import ray
import os
from enum import Enum
from state.types import Action, State
from threading import Thread, Condition
from collections import deque

from context import context
from compute_graph import build_compute_graph, build_regroup_graph, build_preview_graph, build_main_processing_graph
from stacks.compute_graph import process_stacks_for_ui
from dependency_graph import Graph, run_graph, chain, ResultDict, GraphStatus, TopologicalSorterWithStatus, graph_status_to_progress
from dependency_graph.debugging import print_result_dict, print_graph
from dependency_graph.workflow_graph_adapter import to_workflow_graph
from customization.compute_graph import customize_stack, store_customized_stack
from mosaics.compute_graph import build_mosaic_subgraph, stacks_for_mosaic, targets_for_mosaic, process_mosaics_for_ui
from previews.types import QualityLevel
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from stacks.compute_graph import build_stacking_subgraph
from targets.post_process import post_process_target
from stacks.post_process import post_process_stack
from mosaics.post_process import post_process_mosaic
from asiair.download_asiair import download_from_asiairs
from metadata.image_metadata import scatter_compute_image_metadata
from customization.compute_graph import build_customize_graph
from metadata.image_metadata import load_metadata


def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


def _announce(state: State, action: Action) -> tuple[State, list[Action]]:
  from web_server.backend.announcer import announce
  announce(action['message'])

  return state, []


def _download_from_asiairs() -> Graph:
  if len(context.asiair_devices) == 0:
    print(f'### _download_from_asiairs no ASIAir devices found')
    return Graph({})

  return Graph({
      **chain(
          download_from_asiairs,
          (scatter_compute_image_metadata, 'input'),
          build_regroup_graph,
          build_customize_graph,
          process_stacks_for_ui,
          process_targets_for_ui,
          process_mosaics_for_ui,
          build_preview_graph,
          build_main_processing_graph,
      )
  })


def _update_settings(action: Action) -> Graph:
  stack = context.stacks[action['stack_id']]
  return Graph({
      store_customized_stack: {(customize_stack, stack, action['settings'])},
      process_stacks_for_ui: {store_customized_stack}
  })


@ray.remote
def command_done() -> None:
  pass


def _process_stack(action: Action) -> Graph:
  stack = context.stacks[action['object_id']]

  return Graph(
      {**chain(
          (build_stacking_subgraph, stack),
          process_stacks_for_ui,
      )})


def _process_target(action: Action) -> Graph:
  target = context.targets[action['object_id']]

  return Graph(
      {**chain(
          (build_target_subgraph, target),
          process_targets_for_ui,
      )})


def _process_mosaic(action: Action) -> Graph:
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

  return Graph({command_done: tasks})


def _post_process(action: Action) -> Graph:
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


@ray.remote(resources={'diskcache': 0.001},
            max_restarts=-1,
            max_task_retries=-1)
class StateActor:

  def __init__(self):
    self.state = State()
    self._q: deque[Action] = deque()
    self._busy = False
    self._cv = Condition()
    self._runner = Thread(target=self._run_loop, daemon=True)
    self._runner.start()

  def enqueue(self, action: Action) -> int:
    with self._cv:
      self._q.append(action)
      self._cv.notify()
      return len(self._q)

  def is_busy(self) -> bool:
    with self._cv:
      return self._busy

  # TODO: The queue should explicitly contain Graphs to be run.
  def _run_loop(self) -> None:
    while True:
      with self._cv:
        while not self._q:
          self._cv.wait()
        action = self._q.popleft()
        self._busy = True
        print(f'### set _busy {self._busy}')
      try:
        self._handle(action)
      finally:
        with self._cv:
          self._busy = bool(self._q)
          print(f'### set _busy {self._busy}')

  def _run_graph(self, state: State,
                 graph: Graph) -> tuple[ResultDict, ResultDict]:

    def status_fn(status: GraphStatus):
      # TODO: Re-enable this, with some testing to make sure it's not too slow.
      # context.compute_graph_status = status
      pass

    def graph_update_fn(graphs: list[TopologicalSorterWithStatus]):
      # TODO: Re-enable this, with some testing to make sure it's not too slow.
      # for g in graphs:
      #   print(f'graph_update_fn type(g) {type(g)}')
      # context.compute_graphs = graphs
      pass

    # TODO: Should `state` be updated?
    results, logs = run_graph(graph,
                              status_fn=status_fn,
                              graph_update_fn=graph_update_fn,
                              status_interval=15)

    return state, [Action(type='announce', message='reload_data')]

  # TODO: Complete this. The goal is to iterate over the graph's dry run result,
  # deleting all of the cached results, and then running the Graph again. This
  # should be a generic operation.
  def _regroup(self, state: State) -> tuple[State, list[Action]]:
    graph = Graph({build_regroup_graph: {(scatter_compute_image_metadata, 'all')}})
    print_graph(graph)

    results, _ = run_graph(graph, dry_run=True)
    print_result_dict(results)

    for node, result in results.items():
      print(f'### {node=} {result=}')

    return state, [Action(type='announce', message='reload_data')]

  def _handle(self, action: Action) -> None:
    new_actions: list[Action] = []

    print(f"### handling action type {action['type']}")

    match action['type']:
      case 'noop':
        pass
      case 'download_from_asiairs':
        self.state, new_actions = self._run_graph(self.state,
                                                  _download_from_asiairs())
      case 'announce':
        self.state, new_actions = _announce(self.state, action)
      case 'process':
        if action['object_type'] == 'stack':
          self.state, new_actions = self._run_graph(self.state,
                                                    _process_stack(action))
        if action['object_type'] == 'target':
          self.state, new_actions = self._run_graph(self.state,
                                                    _process_target(action))
        elif action['object_type'] == 'mosaic':
          self.state, new_actions = self._run_graph(self.state,
                                                    _process_mosaic(action))
      case 'post_process':
        self.state, new_actions = self._run_graph(self.state,
                                                  _post_process(action))
      case 'update_settings':
        self.state, new_actions = self._run_graph(self.state,
                                                  _update_settings(action))
      case 'run_entire_graph':
        self.state, new_actions = self._run_graph(self.state,
                                                  build_compute_graph())
      case 'regroup':
        self.state, new_actions = self._regroup(self.state)

      case _:
        raise ValueError(f"Unexpected action type: {action['type']}")

    for action in new_actions:
      self._handle(action)

    return self.state


class State:

  def enqueue(self, action: Action) -> int:
    actor = ray.get_actor("state", namespace=_get_namespace())
    return ray.get(actor.enqueue.remote(action))

  def is_busy(self) -> bool:
    actor = ray.get_actor("state", namespace=_get_namespace())
    return ray.get(actor.is_busy.remote())


state = State()
