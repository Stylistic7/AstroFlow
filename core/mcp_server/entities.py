"""Reading the domain model: mosaics, targets, stacks, runs, frames, and the
products the catalog knows for them. Snapshot reads only."""
from __future__ import annotations

import os
from typing import Literal

from decisions.decisions import BinaryChoice, DecisionKey, ObjectType as _OT
from mcp_server.server import NotAvailableYet, context, to_json, tool

ObjectType = Literal["mosaic", "target", "stack", "calibration_stack"]

_UI_KEY = {
    "mosaic": "ui_mosaics",
    "target": "ui_targets",
    "stack": "ui_stacks",
    "calibration_stack": "ui_calibration_stacks",
}
# Geometry blobs the UI viewer needs and a list does not.
_HEAVY = {"footprint", "healpix_cells", "projected_footprints", "filenames",
          "state", "contents"}
MAX_LIST = 500
MAX_FRAMES = 2000


def _ui(object_type: str) -> dict:
    table = context.get(_UI_KEY[object_type])
    if not table:
        raise NotAvailableYet(
            f"{_UI_KEY[object_type]} is not built yet this boot — it appears "
            "once identification (phase 1) finishes")
    return table


def _quarantined(object_type: _OT, object_id: str,
                 parent_stack_id: str | None = None) -> bool:
    key = DecisionKey.quarantine(object_type, object_id, parent_stack_id)
    override = context.settings.decision_overrides.get(key)
    return override is not None and override.choice == BinaryChoice.EXCLUDE


@tool()
def list_objects(object_type: ObjectType, name_contains: str | None = None,
                 filter: str | None = None, image_type: str | None = None,
                 limit: int = 100) -> dict:
    """List mosaics, targets, stacks or calibration stacks as light summaries
    keyed by stable_hash, newest first.

    `name_contains` matches case-insensitively; `filter` (H/S/O/R/G/B/L) and
    `image_type` (light/flat/dark/bias) apply to stacks. `limit` <= 500.
    Returns {count, truncated, items}. Use get_object for detail.
    """
    limit = max(1, min(limit, MAX_LIST))
    items = []
    for sid, d in _ui(object_type).items():
        if name_contains and name_contains.lower() not in str(d.get("name", "")).lower():
            continue
        if filter and d.get("filter") != filter:
            continue
        if image_type and d.get("image_type") != image_type:
            continue
        row = {k: v for k, v in d.items() if k not in _HEAVY}
        row.setdefault("stable_hash", sid)
        if "filenames" in d:
            row["n_frames"] = len(d["filenames"])
        items.append(row)
    items.sort(key=lambda r: str(r.get("newest_date") or ""), reverse=True)
    return {"count": len(items), "truncated": len(items) > limit,
            "items": to_json(items[:limit])}


@tool()
def get_object(object_type: ObjectType, object_id: str, depth: int = 1) -> dict:
    """One object by stable_hash as an identity-keyed hierarchy:
    mosaic -> panels (numbered) -> targets -> stacks -> runs -> frames.

    depth 0 = the object and its counts; 1 = its children (a stack's runs
    carry run_id, dates, selected_frame_count and quarantined); 2 =
    grandchildren (a stack's frames with filenames and star counts, capped
    at 2000 per stack); 3 = maximum. Reads the identification snapshot only.
    """
    from products.product import resolve_entity
    depth = max(0, min(depth, 3))
    entity = resolve_entity(object_type, object_id)
    if entity is None:
        raise ValueError(
            f"no {object_type} {object_id!r} in the current identification "
            "(ids are stable hashes — see list_objects)")
    return to_json(_describe(object_type, entity, depth))


def _describe(object_type: str, entity, depth: int) -> dict:
    if object_type == "mosaic":
        return _mosaic(entity, depth)
    if object_type == "target":
        return _target(entity, depth)
    return _stack(object_type, entity, depth)


def _mosaic(m, depth: int) -> dict:
    from catalog.queries import filters_for
    from products.product import resolve_entity
    numbers = m.panel_numbers()
    out = {
        "object_type": "mosaic", "stable_hash": m.stable_hash,
        "content_hash": m.content_hash, "name": m.name, "center": m.center,
        "image_scale": m.image_scale, "rotation_angle": m.rotation_angle,
        "oldest_date": m.oldest_date, "newest_date": m.newest_date,
        "filters": filters_for("mosaic", m.stable_hash),
        "n_panels": len(m.panel_target_ids),
        "strips": [list(s) for s in m.strips],
        "panels": [{"panel": numbers[tid], "target_id": tid}
                   for tid in m.panel_target_ids],
    }
    if depth >= 1:
        for p in out["panels"]:
            t = resolve_entity("target", p["target_id"])
            p["target"] = _target(t, depth - 1) if t is not None else None
    return out


def _target(t, depth: int) -> dict:
    from catalog.queries import filters_for
    from products.product import resolve_entity
    out = {
        "object_type": "target", "stable_hash": t.stable_hash,
        "content_hash": t.content_hash, "name": t.name, "center": t.center,
        "image_scale": t.image_scale, "rotation_angle": t.rotation_angle,
        "oldest_date": t.oldest_date, "newest_date": t.newest_date,
        # Not `t.filters`: that resolver goes through the Ray context proxy.
        "filters": filters_for("target", t.stable_hash),
        "stack_ids": list(t.stack_ids),
        "quarantined": _quarantined(_OT.TARGET, t.stable_hash),
    }
    if depth >= 1:
        out["stacks"] = [
            _stack("stack", s, depth - 1)
            for sid in t.stack_ids
            if (s := resolve_entity("stack", sid)) is not None
        ]
    return out


def _selected_counts(stack) -> dict:
    from stacks.composite_stack import Stack
    if not isinstance(stack, Stack):  # calibration stacks have no star gate
        return {}
    from decisions.subframe_selection import selected_counts_by_run
    return selected_counts_by_run(stack, context.settings.decision_overrides)


def _stack(object_type: str, s, depth: int) -> dict:
    ui = (context.get(_UI_KEY[object_type]) or {}).get(s.stable_hash) or {}
    out = {k: v for k, v in ui.items() if k not in _HEAVY}
    out.update({
        "object_type": object_type, "stable_hash": s.stable_hash,
        "content_hash": s.content_hash, "filter": s.filter,
        "image_type": s.image_type, "camera": s.camera,
        "oldest_date": s.oldest_date, "newest_date": s.newest_date,
        "n_runs": len(s.contents), "n_frames": len(s.all_frames()),
        "quarantined": _quarantined(_OT.STACK, s.stable_hash),
    })
    if depth < 1:
        return out
    selected = _selected_counts(s)
    budget = MAX_FRAMES
    runs = []
    for run_id, run in s.contents.items():
        r = {
            "run_id": run_id, "oldest_date": run.oldest_date,
            "newest_date": run.newest_date, "exposure": run.exposure,
            "n_frames": len(run.frames),
            "dependencies": {k: getattr(v, "stack_id", str(v))
                             for k, v in run.dependencies.items()},
            "quarantined": _quarantined(_OT.STACK, run_id, s.stable_hash),
        }
        if run_id in selected:
            r["selected_frame_count"], r["total_frame_count"] = selected[run_id]
        if depth >= 2:
            take = run.frames[:max(budget, 0)]
            budget -= len(take)
            r["frames"] = [
                {"hash": h, "filename": str(run.metadata[h].filename),
                 "num_stars": getattr(run.metadata[h].plate_solution,
                                      "num_stars", None)}
                for h in take
            ]
            r["frames_truncated"] = len(take) < len(run.frames)
        runs.append(r)
    out["runs"] = runs
    return out


@tool()
def finishes_for_object(object_type: ObjectType, object_id: str) -> dict:
    """Every view the catalog can show for an object: pipelines -> finishes
    (the user-facing renders: "Hubble", "H", ...), each with its status
    (live / out_of_date / absent), mtime and the taps it reads, plus the
    default view. The same matrix as `/api/finishes`. Snapshot-only: never
    traces.
    """
    from catalog import finishes_matrix
    matrix = finishes_matrix(object_type, object_id)
    if matrix is None:
        raise ValueError(f"no {object_type} {object_id!r} in the current identification")
    return matrix


def _plate(ps) -> dict | None:
    if ps is None:
        return None
    return to_json({k: getattr(ps, k, None)
                    for k in ("center", "image_scale", "rotation_angle", "num_stars")})


def _meta_summary(meta) -> dict:
    st = meta.statistics
    return to_json({
        "content_hash": meta.content_hash, "stable_hash": meta.stable_hash,
        "width": meta.width_pixels, "height": meta.height_pixels,
        "channels": meta.channels, "dtype": str(meta.dtype),
        "noise_estimate": meta.noise_estimate,
        "master_integration_s": meta.master_integration_s,
        "sky_median": getattr(st, "sky_median", None),
        "sky_rms": getattr(st, "sky_rms", None),
        "p50": getattr(st, "p50", None),
        "plate_solution": _plate(meta.plate_solution),
    })


@tool()
def resolve_finish(object_type: ObjectType, object_id: str,
                   pipeline: str | None = None, finish: str | None = None) -> dict:
    """Resolve one finish to files. Omit pipeline and finish for the object's
    default view. Returns the finish's status and, per declared input, the
    tap, its filter, the master (the real deliverable — read pixels, headers
    and the sidecar from it), the servable path (pyramid or master) with its
    status, and a sidecar summary when the master exists. `absent` inputs
    are a normal answer, not an error.
    """
    from catalog import Product, default_finish_for, master_path, predict_path
    from catalog import resolve_finish as bind
    from metadata.sidecar_io import read_image_metadata
    from pathlib import Path
    if pipeline is None and finish is None:
        b = default_finish_for(object_type, object_id)
        if b is None:
            raise NotAvailableYet(f"nothing of {object_type} {object_id} is on disk yet")
    else:
        if pipeline is None or finish is None:
            raise ValueError("give both pipeline and finish, or neither")
        b = bind(object_type, object_id, pipeline, finish)
        if b is None:
            raise ValueError(f"{finish!r} is not a finish {pipeline!r} can show for {object_type} {object_id}")
    inputs = []
    for i in b.inputs:
        product = Product(object_type, object_id, b.pipeline, i.tap, i.filter)
        master = master_path(product) if i.filter is not None else None
        row = {"tap": i.tap, "filter": i.filter, "optional": i.optional, "status": i.status,
               "predicted_path": str(predict_path(product)) if i.filter is not None else None,
               "master_path": str(master) if master else None,
               "path": str(i.path) if i.path else None}
        if master is not None:
            st = os.stat(master)
            row["mtime"], row["size_bytes"] = st.st_mtime, st.st_size
            meta = read_image_metadata(Path(master))
            if meta is not None:
                row["sidecar"] = _meta_summary(meta)
        inputs.append(row)
    return {"finish": {"object_type": object_type, "object_id": object_id, "pipeline": b.pipeline,
                       "name": b.name, "role": b.role, "status": b.status, "mtime": b.mtime},
            "inputs": inputs}


@tool()
def image_metadata(path: str, histogram: bool = False) -> dict:
    """The sidecar record of an image file (no pixel read): identity hashes,
    geometry, the full header, statistics (percentiles, sky median/rms, MAD,
    clipped fractions; histogram counts only when histogram=true), the MRS
    noise estimate, integration time, plate solution and the SPFC
    calibration certificate when present.
    """
    from pathlib import Path
    from file_io.sidecar import get_section
    from metadata.sidecar_io import (_serialize_header, _statistics_to_obj,
                                     read_image_metadata)
    from mcp_server.images import checked_path
    p = checked_path(path)
    meta = read_image_metadata(Path(p))
    if meta is None:
        raise ValueError(f"no sidecar for {p} (expected {p}.json with an "
                         "image_metadata section)")
    stats = _statistics_to_obj(meta.statistics) if meta.statistics else None
    if stats and not histogram:
        stats = {k: v for k, v in stats.items() if not k.startswith("histogram")}
    return to_json({
        "path": str(p), "content_hash": meta.content_hash,
        "stable_hash": meta.stable_hash, "width": meta.width_pixels,
        "height": meta.height_pixels, "channels": meta.channels,
        "dtype": str(meta.dtype), "header": _serialize_header(meta.header),
        "statistics": stats, "noise_estimate": meta.noise_estimate,
        "master_integration_s": meta.master_integration_s,
        "plate_solution": _plate(meta.plate_solution),
        "spfc_calibration": get_section(Path(p), "spfc_calibration"),
    })
