"""Calibration helpers exposed to pipeline code.

Two primitives, both plain Python (NOT `@node`) — they run at
trace time so the cal-master sub-graph is built explicitly:

* `calibration_stacks_for(stack, config)` — return the cal
  stacks `stack` (transitively) depends on, in topological order
  (biases first, then darks/flats that consume them). Use this
  when the pipeline wants to insert a per-stack transformation
  (downscale, denoise, etc.) before each stack is stacked.

* `calibrate(stack, config)` — convenience for the common case
  where no per-stack transform is needed. Returns the list of
  cal-master handles ready to spread into
  `stack_files(stack, config, *cal_masters)`.
"""
from file_io.types import WithSidecar
from stacks.composite_stack import BaseStack, CalibrationStack
from stacks.stacking.pixinsight import stack_files
from stacks.types import StackIdDependency
from decisions.resolve import resolve_calibration
from decisions.settings import ProcessingConfig


def calibration_stacks_for(
    stack: BaseStack, config: ProcessingConfig,
) -> tuple[CalibrationStack, ...]:
    """The cal stacks `stack` (transitively) depends on, in
    topological order (biases before the darks/flats that need
    them).

    Use when the pipeline wants to insert a per-stack transform
    before stacking each cal stack. Typical pattern:

        cal_masters = []
        for cs in calibration_stacks_for(stack, config):
            cs_transformed = my_transform(cs)
            cal_masters.append(stack_files(
                cs_transformed, config, *cal_masters,
            ))
        s_transformed = my_transform(stack)
        master = stack_files(s_transformed, config, *cal_masters)

    Each cal stack's `stack_files` call is passed every cal master
    built so far; `stack_files` matches them to its own dependencies
    by reading each master's `STBLHASH` header. (Bias masters need
    no cal masters, dark/flat masters need their own bias master,
    light masters need bias/dark/flat — the accumulator handles all
    cases.)
    """
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


def calibrate(
    stack: BaseStack, config: ProcessingConfig,
) -> list[WithSidecar]:
    """Common-case helper: build cal masters for `stack`'s
    calibration dependencies, with no per-stack transformation.

        cal_masters = calibrate(stack, config)
        master = stack_files(stack, config, *cal_masters)

    For pipelines that want to transform each cal stack (e.g.
    downscale) before stacking, use `calibration_stacks_for` and
    write the loop yourself."""
    cal_masters: list[WithSidecar] = []
    for cs in calibration_stacks_for(stack, config):
        cal_masters.append(stack_files(cs, config, *cal_masters))
    return cal_masters
