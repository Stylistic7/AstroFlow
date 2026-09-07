"""Calibration helpers exposed to pipeline code.

Plain Python (NOT `@node`) — they run at trace time so the cal-master
sub-graph is built explicitly.

INVARIANT (the 2026-08-06 re-stack-loop fix): a calibration stack's
`stack_files` call receives EXACTLY the masters for its OWN resolved
dependencies — bias: none, dark: none, flat: its bias — in sorted
stable-hash order. Never "every master built so far": that accumulator
made a shared cal stack's arg list depend on WHICH CONSUMER's trace built
it (a dark visited late in the L-filter chain accumulated the L flat; the
same dark visited first accumulated nothing), minting up to one call_hash
per consumer for one physical stack. Each graph pass then re-stacked
whichever variant its trace demanded and superseded the others' files —
the perpetual calibration re-stack loop. The spurious masters were
pixel-inert (`_run_wbpp_stacking` matches masters to dependencies by
STBLHASH, ignoring extras), so the damage was cache identity and wasted
WBPP, never wrong pixels.

* `calibration_plan(stack, config)` — the build recipe: topologically
  ordered `(cal_stack, its_dep_ids)` steps plus `stack`'s own dep ids.
* `calibrate(stack, config)` — common case, no per-stack transform.
"""
from file_io.types import WithSidecar
from stacks.composite_stack import BaseStack, CalibrationStack
from stacks.stacking.pixinsight import stack_files
from stacks.types import StackIdDependency
from decisions.resolve import resolve_calibration
from decisions.settings import ProcessingConfig


def _own_dependency_ids(enriched: BaseStack) -> tuple[str, ...]:
    """The stable_hashes of the cal stacks `enriched`'s runs directly
    reference, deduplicated and SORTED — the canonical arg order, so the
    same stack yields the same `stack_files` arg list in every trace."""
    ids = {
        dep.stack_id
        for run in enriched.contents.values()
        for dep in run.dependencies.values()
        if isinstance(dep, StackIdDependency)
    }
    return tuple(sorted(str(i) for i in ids))


def calibration_plan(
    stack: BaseStack, config: ProcessingConfig,
) -> tuple[tuple[tuple[CalibrationStack, tuple[str, ...]], ...], tuple[str, ...]]:
    """The calibration build recipe for `stack`.

    Returns `(steps, stack_dep_ids)`:
      * `steps` — `(cal_stack_enriched, dep_ids)` in dependency order
        (a step's `dep_ids` only ever name earlier steps). `dep_ids` are
        the cal stack's OWN resolved dependencies, sorted.
      * `stack_dep_ids` — `stack`'s own direct dependency ids, sorted;
        index the built masters with these for the final `stack_files`
        call. Per CALIBRATION_TOPOLOGY a light's ids name its darks and
        flats — never a bias (Landmann §8.2: no master bias for lights).

    Typical per-stack-transform loop:

        steps, top_ids = calibration_plan(stack, config)
        masters = {}
        for cs, dep_ids in steps:
            masters[cs.stable_hash] = stack_files(
                my_transform(cs), config,
                *(masters[i] for i in dep_ids),
            )
        master = stack_files(
            my_transform(stack), config, *(masters[i] for i in top_ids),
        )

    (Transforms like `downscale_stack` preserve `stable_hash`, so the
    id-keyed indexing is transform-agnostic.)
    """
    cal_stacks = config.get_calibration_stacks()
    decisions = config.decisions

    steps: dict[str, tuple[CalibrationStack, tuple[str, ...]]] = {}
    order: list[str] = []

    def visit(s: BaseStack) -> None:
        enriched = resolve_calibration(s, cal_stacks, decisions)
        for dep_id in _own_dependency_ids(enriched):
            if dep_id in steps:
                continue
            cs = cal_stacks[dep_id]
            visit(cs)  # a dep's own deps must be planned first
            if dep_id not in steps:
                cs_enriched = resolve_calibration(cs, cal_stacks, decisions)
                steps[dep_id] = (cs_enriched, _own_dependency_ids(cs_enriched))
                order.append(dep_id)

    stack_enriched = resolve_calibration(stack, cal_stacks, decisions)
    visit(stack)
    return (
        tuple(steps[h] for h in order),
        _own_dependency_ids(stack_enriched),
    )


def calibrate(
    stack: BaseStack, config: ProcessingConfig,
) -> list[WithSidecar]:
    """Build `stack`'s cal masters with no per-stack transformation.

        cal_masters = calibrate(stack, config)
        master = stack_files(stack, config, *cal_masters)

    Returns the masters for `stack`'s OWN dependencies (sorted id order)
    — the exact set `_run_wbpp_stacking` will match by STBLHASH."""
    from stacks.star_flat import StarFlatStack, build_star_flat
    steps, top_ids = calibration_plan(stack, config)
    masters: dict[str, WithSidecar] = {}
    for cs, dep_ids in steps:
        if isinstance(cs, StarFlatStack):
            # Solved from photometry, not integrated: its own producer, keyed
            # on the group's membership (content_hash), no cal deps of its own.
            masters[str(cs.stable_hash)] = build_star_flat(cs)
        else:
            masters[str(cs.stable_hash)] = stack_files(
                cs, config, *(masters[i] for i in dep_ids),
            )
    return [masters[i] for i in top_ids]
