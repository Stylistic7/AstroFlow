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
* `calibration_master(cs, config, dep_masters, dep_ids)` — one step of that
  recipe: the producer a calibration stack is built by (solved from
  photometry for a star flat, combined from blank sky for a fringe frame,
  integrated from frames for everything else).
* `master_for(stack, cs, master, dark)` — the master `stack` is actually
  calibrated with: a star flat is re-scaled to the stack's own sky first
  (`scale_star_flat`, ephemeral, per light stack); anything else as built.
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

    (A `DownscaledStack` view keeps its source's `stable_hash`, so the
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


def has_own_producer(cs: CalibrationStack) -> bool:
    """Is this cal stack built by something other than a stacker?

    A star flat is solved from photometry and a fringe frame is combined
    from blank sky; neither is an integration of `contents`, so no stacking
    backend may be handed one. Every backend asks HERE rather than testing
    types itself — the OSS pipeline used to run `stack_files_oss` over every
    plan step, which for a star flat integrated an empty `contents`.
    """
    from stacks.star_flat import StarFlatStack
    from stacks.fringe import FringeStack
    return isinstance(cs, (StarFlatStack, FringeStack))


def calibration_master(
    cs: CalibrationStack, config: ProcessingConfig, dep_masters: list[WithSidecar],
    dep_ids: tuple[str, ...] = (),
) -> WithSidecar:
    """One calibration stack's master, by the producer its kind is built by.

      * a star flat is SOLVED from its members' star photometry — its own
        producer, keyed on the group's membership, no cal deps of its own;
      * a fringe frame is COMBINED from the blank-sky lights of the stack the
        user promoted, calibrated with its own dark and flat so that what
        survives is the residual the flat could not remove;
      * everything else is INTEGRATED from its frames with the masters of its
        own dependencies.

    `dep_ids` name `dep_masters` positionally (the sorted ids
    `calibration_plan` returns), which is how the fringe build tells its dark
    from its flat.
    """
    from stacks.star_flat import StarFlatStack, build_args, build_star_flat
    from stacks.fringe import FringeStack, build_fringe_frame
    if isinstance(cs, StarFlatStack):
        return build_star_flat(*build_args(cs))
    if isinstance(cs, FringeStack):
        by_type = _masters_by_type(config, dep_ids, dep_masters)
        # The RAW star flat, not the per-stack-scaled one: the amplitude the
        # map is applied at is fitted per Run anyway (`fringe_gain`), and the
        # gain's own residual is the same band-limited shape, so it lands in
        # A rather than in the map. Scaling here would key the fringe frame
        # to one consumer's sky, which is exactly what a shared calibration
        # product must not be.
        return build_fringe_frame(cs, by_type.get("dark"), by_type.get("flat"))
    return stack_files(cs, config, *dep_masters)


def _masters_by_type(
    config: ProcessingConfig, dep_ids: tuple[str, ...], dep_masters: list[WithSidecar],
) -> dict[str, WithSidecar]:
    """`{image_type: master}` for a step's own dependencies."""
    cal_stacks = config.get_calibration_stacks()
    out: dict[str, WithSidecar] = {}
    for dep_id, master in zip(dep_ids, dep_masters):
        cs = cal_stacks.get(dep_id)
        if cs is not None:
            out.setdefault(str(cs.image_type), master)
    return out


def master_for(
    stack: BaseStack, cs: CalibrationStack, master: WithSidecar, dark: WithSidecar | None,
) -> WithSidecar:
    """The master `stack` divides by. A star flat's pattern has the right shape
    for every stack in its group but its own amplitude for each one (the
    stack's sky sees the sensor's etalon at its own contrast), so the group's
    flat is re-scaled to this stack before WBPP sees it; an integrated master
    is used as built. `dark` lets the fit remove the pedestal first."""
    from stacks.star_flat import StarFlatStack, gain_args, scale_star_flat
    if isinstance(cs, StarFlatStack):
        return scale_star_flat(*gain_args(stack, master, dark))
    return master


def calibrate(
    stack: BaseStack, config: ProcessingConfig,
) -> list[WithSidecar]:
    """Build `stack`'s cal masters with no per-stack transformation.

        cal_masters = calibrate(stack, config)
        master = stack_files(stack, config, *cal_masters)

    Returns the masters for `stack`'s OWN dependencies (sorted id order)
    — the exact set `_run_wbpp_stacking` will match by STBLHASH."""
    steps, top_ids = calibration_plan(stack, config)
    cal_stacks: dict[str, CalibrationStack] = {}
    masters: dict[str, WithSidecar] = {}
    for cs, dep_ids in steps:
        cal_stacks[str(cs.stable_hash)] = cs
        masters[str(cs.stable_hash)] = calibration_master(
            cs, config, [masters[i] for i in dep_ids], dep_ids)
    dark = next((masters[i] for i in top_ids if getattr(cal_stacks[i], "image_type", None) == "dark"), None)
    return [master_for(stack, cal_stacks[i], masters[i], dark) for i in top_ids]
