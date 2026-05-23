"""Decisions-aware calibration-stack selection.

`select_calibration_stacks(stack, config)` returns the topologically-
ordered tuple of `CalibrationStack`s that `stack` (transitively)
depends on — biases come before the darks/flats that need them.
The pipeline body iterates the result to build the cal-master
sub-graph explicitly, replacing the old `_build_calibration_graph`
machinery in `stacks/compute_graph.py`.

This is a plain Python function (NOT `@node`): pipelines call it
at trace time so the topology is visible to the tracer and the
pipeline can choose which cal stacks to wrap in `downscale_stack`.
"""
from stacks.composite_stack import BaseStack, CalibrationStack
from stacks.types import StackIdDependency
from decisions.resolve import resolve_calibration
from decisions.settings import ProcessingConfig


def select_calibration_stacks(
    stack: BaseStack, config: ProcessingConfig,
) -> tuple[CalibrationStack, ...]:
    """Return the cal stacks `stack` (transitively) depends on,
    bottom-up: biases first, then darks/flats that consume them.

    Each returned cal stack is enriched via `resolve_calibration`
    so its own `run.dependencies` are populated — the pipeline
    body can walk those to wire each cal `stack_files` call's
    cal-master data deps."""
    cal_stacks = config.get_calibration_stacks()
    decisions = config.decisions

    out: dict[str, CalibrationStack] = {}
    order: list[str] = []

    def visit(s: BaseStack) -> None:
        s_enriched = resolve_calibration(s, cal_stacks, decisions)
        for run in s_enriched.contents.values():
            for dep in run.dependencies.values():
                if not isinstance(dep, StackIdDependency):
                    continue
                cs = cal_stacks[dep.stack_id]
                visit(cs)
                if cs.stable_hash not in out:
                    out[cs.stable_hash] = resolve_calibration(
                        cs, cal_stacks, decisions,
                    )
                    order.append(cs.stable_hash)

    visit(stack)
    return tuple(out[h] for h in order)
