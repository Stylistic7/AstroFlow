from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from beartype import beartype
from decorators import node
from stacks.composite_stack import Run, Stack, StacksDict, CalibrationStacksDict

from stacks.convert_to_dict import convert_to_dict
from decisions.decisions import decisions_for_stack
from decisions.subframe_selection import (
    effective_integration,
    get_light_frames_sorted_by_stars,
)
from dependency_graph import Graph
from decisions.settings import ProcessingConfig
from context import context
import structlog
from pix.tasks.filters import normalize_filter

log = structlog.get_logger()


# TODO: Calibrate the subframe returned here.
@node
def get_stack_subframe_image(stack: Stack) -> Path:
    assert stack is not None
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
    return Path(best_light_frame.filename)


def _my_transform(v):
    if type(v) is np.ndarray:
        return v.tolist()
    if type(v) is SingleSphericalPolygon:
        return v.points.tolist()
    return v



def _focal_length_of(stack) -> int | None:
    focal = next(
        (m.header.get("FOCALLEN") for m in stack.all_metadata().values()
         if m.header.get("FOCALLEN") is not None),
        None,
    )
    return round(float(focal)) if focal is not None else None


def _foreign_stack_for_ui(stack) -> dict:
    """A foreign stack's UI entry (`stacks.foreign`): no subframes, no
    calibration, no decisions — what its master says, plus whose it is and
    whether this instance's pipeline admits it."""
    return {
        "foreign": True,
        "same_pipeline": stack.same_pipeline,
        "stacked_by": "astroflow" if stack.pipeline_stamp else "unknown",
        "observer": stack.observer,
        "shared_dir": stack.shared_dir,
        "pipeline_stamp": stack.pipeline_stamp,
        "stable_hash": stack.stable_hash,
        "content_hash": stack.content_hash,
        "name": stack.name,
        "image_type": stack.image_type,
        "camera": stack.camera,
        "camera_id_hash": None,
        "camera_id_short": None,
        "filter": stack.filter,
        "role": normalize_filter(stack.filter) or "none",
        "exposure": stack.exposure,
        "integration": stack.integration,
        "raw_integration_s": stack.integration,
        "mean_temperature": stack.mean_temperature,
        "image_scale": stack.image_scale,
        "rotation_angle": stack.rotation_angle,
        "width_pixels": stack.width_pixels,
        "height_pixels": stack.height_pixels,
        "focal_length": None,
        "center": list(stack.center) if stack.center else None,
        "oldest_date": str(stack.oldest_date) if stack.oldest_date else None,
        "newest_date": str(stack.newest_date) if stack.newest_date else None,
        "state": dict(stack.state),
        "filenames": [stack.master_path],
    }


@node
def prepare_stack_for_ui(
    stack: Stack,
    cal_stacks: CalibrationStacksDict,
    decisions,
    min_frames: int,
) -> dict:
    from decisions.resolve import resolve_calibration, resolve_decisions
    from decisions.compute_graph import ALL_EXCLUSION_TYPES
    from stacks.foreign import is_foreign

    if is_foreign(stack):
        return _foreign_stack_for_ui(stack)
    stack = resolve_calibration(stack, cal_stacks, decisions)
    resolve_decisions(stack, ALL_EXCLUSION_TYPES, decisions, min_frames)
    return {
        "foreign": False,
        "stable_hash": stack.stable_hash,
        "content_hash": stack.content_hash,
        "name": stack.name,
        "image_type": stack.image_type,
        "camera": stack.camera,
        "camera_id_hash": stack.camera_id_hash,
        "camera_id_short": (
            stack.camera_id_hash[:8] if stack.camera_id_hash else None
        ),
        "filter": stack.filter,
        # The filter's ROLE (H, S, O, R, G, B, L, or "none") — the name of
        # the channel finish that shows this stack; sibling stacks of a
        # target are switched between by it.
        "role": normalize_filter(stack.filter) or "none",
        "exposure": stack.exposure,
        # `integration` is the *effective* (post-subframe-selection)
        # signal — what the viewer's exposure clock should reflect.
        # Run-level views still derive raw per-frame data themselves.
        "integration": effective_integration(stack),
        # `raw_integration_s` is the raw sum of EXPOSURE across every
        # subframe, with no quality filter applied. Used as the upper
        # bound (blue polygon) in the dual exposure widget on the
        # mosaic view, against which the WBPP-effective value
        # (`wbpp_exposure_s` carried in the /api/metadata output
        # polygon) is compared.
        "raw_integration_s": stack.integration,
        "mean_temperature": stack.mean_temperature,
        "image_scale": stack.image_scale,
        "rotation_angle": stack.rotation_angle,
        "width_pixels": stack.width_pixels,
        "height_pixels": stack.height_pixels,
        # Optical train, so the flat carousel can restrict candidates to
        # the same rig (same-geometry rigs differ only by focal length).
        "focal_length": _focal_length_of(stack),
        "center": list(stack.center) if stack.center else None,
        "oldest_date": str(stack.oldest_date) if stack.oldest_date else None,
        "newest_date": str(stack.newest_date) if stack.newest_date else None,
        "state": dict(stack.state),
        "filenames": [
            m.filename for s in stack.contents.values() for m in s.metadata.values()
        ],
    }


@ray.remote(num_cpus=0.1)
def gather_ui_stacks(prior_ref, *stack_dicts) -> dict:
    """Merge freshly-prepared per-stack UI dicts over the cache-hit base.

    `prior_ref` is a ContextRef holding the {stable_hash: dict} entries whose
    `prepare_stack_for_ui` results were already cached (collected by the
    builder's probe), or None when nothing hit."""
    base = dict(prior_ref.get()) if prior_ref is not None else {}
    base.update({d["stable_hash"]: d for d in stack_dicts})
    return base


@node
def process_stacks_for_ui(
    decisions,
    min_frames: int,
    entry_version: str,
    *ancestor_results,
) -> "Graph | dict":
    """Build (or simply return) the ui_stacks mapping — DELTA-ONLY.

    Probes each stack's `prepare_stack_for_ui` cache entry and emits graph
    nodes ONLY for misses; hits are collected as values. When every stack
    hits, this returns the assembled dict directly — no subgraph, no
    per-stack dispatches, no gather. And because this builder is itself
    @node-cached on (decisions, min_frames, entry_version, stacks,
    cal_stacks), the next identical startup short-circuits to ONE diskcache
    hit returning the finished dict: the ~1,020-task fan-out that previously
    re-dispatched on every startup (and helped wedge the web server's
    ContextActor reads) never enters the graph at all.

    `entry_version` is `prepare_stack_for_ui`'s source hash, key material
    only: the cached value EMBEDS the per-stack entries, so editing their
    producer must miss here as well.
    """
    from cache.context_ref import ContextRef

    # Ancestor results arrive in non-deterministic set order; unpack by type.
    stacks = next(r for r in ancestor_results if isinstance(r, StacksDict))
    cal_stacks = next(r for r in ancestor_results if isinstance(r, CalibrationStacksDict))

    hits: dict = {}
    per_stack_tasks = []
    for h, ref in stacks.items():
        stack = ref.get() if isinstance(ref, ContextRef) else ref
        stack_decisions = decisions_for_stack(decisions, stack.stable_hash)
        args = (stack, cal_stacks, stack_decisions, min_frames)
        hit, value = prepare_stack_for_ui.cached_value(*args)
        if hit:
            hits[value["stable_hash"]] = value
        else:
            per_stack_tasks.append((prepare_stack_for_ui, *args))

    log.info(
        "ui stacks delta", total=len(stacks), cached=len(hits),
        to_prepare=len(per_stack_tasks),
    )
    if not per_stack_tasks:
        return hits  # complete — no subgraph needed

    prior_ref = ContextRef.store_large(hits) if hits else None
    return Graph({(gather_ui_stacks, prior_ref): per_stack_tasks})


@node
def process_calibration_stacks_for_ui(cal_stacks: dict) -> dict:
    assert None not in cal_stacks.values()

    def prepare_stack(stack):
        result = convert_to_dict(stack, [["contents", "*", "metadata"]])
        metas = list(stack.all_metadata().values())
        result["filenames"] = [m.filename for m in metas]
        result["camera_id_hash"] = stack.camera_id_hash
        result["camera_id_short"] = (
            stack.camera_id_hash[:8] if stack.camera_id_hash else None
        )
        # Focal length distinguishes the two same-model rigs on the cards
        # (flats separate by it); None when the header lacks it.
        result["focal_length"] = _focal_length_of(stack)
        if getattr(stack, "kind", None) == "sky":
            # A night-sky flat: its "frames" are the blank-sky lights of the
            # stack the user promoted, so the card shows those, not the one
            # synthetic run that carries them.
            result["kind"] = "sky"
            result["focal_length"] = round(stack.focal_length) if stack.focal_length else None
            result["n_frames"] = stack.n_frames
            result["source_stack_id"] = stack.source_stack_id
            result["filenames"] = list(stack.member_files)
            result.pop("member_files", None)
        if getattr(stack, "kind", None) == "star":
            # A star flat: its "frames" are the group's light subframes.
            result["kind"] = "star"
            result["focal_length"] = round(stack.focal_length) if stack.focal_length else None
            result["n_subs"] = stack.n_subs
            result["filenames"] = list(stack.member_files)
            result.pop("member_files", None)

        return result

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_cal_stacks = {
        h: prepare_stack(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in cal_stacks.items()
    }
    return ui_cal_stacks
