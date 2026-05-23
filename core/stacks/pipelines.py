"""Shared pipeline-body helpers for the PixInsight stacking flows.

The PI Master and PI Downscaled pipelines compose the same ops:
`select_calibration_stacks` (decisions-aware cal lookup),
optionally `downscale_stack` (preserves identity, differs content),
then `stack_files` for each cal stack and the light stack, then
`post_process_stack`. The ONLY difference between the two
pipelines is whether `downscale_stack` is applied — captured by
the `downscaler` argument here.
"""
from typing import Callable

from stacks.composite_stack import BaseStack
from stacks.types import StackIdDependency
from stacks.calibration_selection import select_calibration_stacks
from stacks.stacking.pixinsight import stack_files
from stacks.post_processing.pixinsight import post_process_stack


def _cal_deps_for(stack, masters_by_id):
    """Walk a stack's run.dependencies; return parallel
    `(cal_stack_ids, cal_masters)` tuples for the cal masters
    already built in `masters_by_id`.

    Dedup by stack_id — a run may reference the same bias as
    multiple cal types, and a stack with multiple runs may share
    deps; either way we want each master to flow into `stack_files`
    exactly once."""
    ids = []
    masters = []
    seen = set()
    for run in stack.contents.values():
        for dep in run.dependencies.values():
            if not isinstance(dep, StackIdDependency):
                continue
            if dep.stack_id in seen or dep.stack_id not in masters_by_id:
                continue
            seen.add(dep.stack_id)
            ids.append(dep.stack_id)
            masters.append(masters_by_id[dep.stack_id])
    return tuple(ids), tuple(masters)


def post_processed_master(s, config, mock, *, downscaler: Callable[[BaseStack], BaseStack]):
    """Build the per-stack `stack_files → post_process_stack`
    chain, including the cal-master sub-graph.

    `downscaler` is a function `BaseStack -> BaseStack`. PI Master
    passes the identity; PI Downscaled passes
    `lambda x: downscale_stack(x, 8)`. The cal-stack TOPOLOGY is
    identical between the two pipelines (decisions are sticky in
    the context actor); only the per-stack content (full-res vs
    downscaled frames) differs. That content difference flows
    through the `stack_files` call_hash and produces distinct
    on-disk outputs automatically."""
    cal_stacks = select_calibration_stacks(s, config)

    masters_by_id = {}
    for cs in cal_stacks:
        cs_d = downscaler(cs)
        ids, masters = _cal_deps_for(cs, masters_by_id)
        masters_by_id[cs.stable_hash] = stack_files(
            cs_d, config, ids, *masters,
        )

    s_d = downscaler(s)
    ids, masters = _cal_deps_for(s, masters_by_id)
    master = stack_files(s_d, config, ids, *masters)
    return post_process_stack(s_d, mock, master)
