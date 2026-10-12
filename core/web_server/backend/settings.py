import os
from dataclasses import replace
from pathlib import Path
from flask import request, jsonify, Blueprint
from frozendict import frozendict

from context import local_context as context
from decisions.settings import Settings
from decisions.decisions import ObjectType, BinaryChoice, DecisionType
from decisions.decisions import DecisionKey, DecisionOverride
from decisions.helpers import exclude
from web_server.backend.announcer import announce

settings_bp = Blueprint("settings", __name__)


def _update_settings(updater):
    """Read-modify-write the settings in the context."""
    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings
    return new_settings


@settings_bp.route("/api/project_kind/<object_type>/<object_id>", methods=["PUT", "DELETE"])
def put_project_kind(object_type, object_id):
    """Override what a target or mosaic is for: `{"kind": "lrgb" |
    "narrowband"}` (narrowband: R/G/B are star colour, capped at their goal).
    DELETE returns it to the planner's own classification from the hours
    shot. Planner-only: no graph restart."""
    try:
        key = DecisionKey.project_kind(ObjectType(object_type), object_id)
    except ValueError:
        return jsonify({"error": "object_type must be target or mosaic"}), 400
    current = dict(context.settings.decision_overrides)
    if request.method == "DELETE":
        current.pop(key, None)
    else:
        kind = (request.get_json(silent=True) or {}).get("kind")
        if kind not in ("lrgb", "narrowband"):
            return jsonify({"error": "kind must be lrgb or narrowband"}), 400
        current[key] = DecisionOverride(choice=kind)
    _update_settings(lambda s: replace(s, decision_overrides=frozendict(current)))
    return jsonify({"object_type": object_type, "object_id": object_id,
                    "kind": None if request.method == "DELETE" else kind})


@settings_bp.route("/api/mosaic_membership/target/<target_id>", methods=["PUT", "DELETE"])
def put_mosaic_membership(target_id):
    """PUT keeps a target out of every mosaic ("standalone"), however its
    footprint overlaps; DELETE lets identification group it again. The
    mosaic re-identifies — and if the target held its oldest frames, the
    mosaic comes back under a new id (its stable hash is its oldest
    panel's)."""
    from mosaics.identification import STANDALONE
    key = DecisionKey.mosaic_membership(target_id)
    current = dict(context.settings.decision_overrides)
    if request.method == "DELETE":
        current.pop(key, None)
    else:
        current[key] = DecisionOverride(choice=STANDALONE)
    _update_settings(lambda s: replace(s, decision_overrides=frozendict(current)))
    announce("reload_data")
    from state.actor import state
    state.drain_and_restart()
    return jsonify({"target_id": target_id, "standalone": request.method == "PUT"})


@settings_bp.route("/api/quarantine/<object_type>/<object_id>", methods=["POST"])
def toggle_quarantine(object_type, object_id):
    """Toggle quarantine for a stack, target, or mosaic.

    Quarantine is a decision override: EXCLUDE means quarantined,
    absent means not quarantined.

    For run-level quarantine (quarantining a run within a stack), pass
    ?parent=<stack_hash> so the decision key carries the stack hash as
    its parent_stack_id (required for per-stack decision filtering).
    """
    type_map = {
        "stack": ObjectType.STACK, "target": ObjectType.TARGET,
        "mosaic": ObjectType.MOSAIC, "run": ObjectType.RUN,
    }
    obj_type = type_map.get(object_type)
    if obj_type is None:
        return jsonify({"error": f"Unknown object type: {object_type}"}), 400

    parent_stack_id = request.args.get("parent")
    if obj_type is ObjectType.RUN and not parent_stack_id:
        return jsonify({"error": "a run quarantine needs ?parent=<stack hash>"}), 400
    key = DecisionKey.quarantine(obj_type, object_id, parent_stack_id)

    current_overrides = context.settings.decision_overrides
    is_currently_quarantined = (
        key in current_overrides
        and current_overrides[key].choice == BinaryChoice.EXCLUDE
    )

    if is_currently_quarantined:
        new_overrides = frozendict({k: v for k, v in current_overrides.items() if k != key})
    else:
        new_overrides = frozendict(dict(current_overrides) | exclude(key))

    _update_settings(lambda s: replace(s, decision_overrides=new_overrides))

    announce("reload_data")

    from state.actor import state
    state.drain_and_restart()

    return jsonify({
        "object_type": object_type,
        "object_id": object_id,
        "is_quarantined": not is_currently_quarantined,
    })


@settings_bp.route("/api/sky_flat_source/<stack_id>", methods=["POST"])
def toggle_sky_flat_source(stack_id):
    """Promote a stack to a night-sky flat source, or demote it.

    Blank-sky frames arrive stamped `Light` like any other, so nothing in a
    header can identify them — this decision is the only place that knowledge
    exists (`stacks/sky_flat.py`). Promoting mints a `SkyFlatStack` whose
    frames are this stack's; it is then an ordinary flat, and resolves onto
    lights of the same optical train exactly the way any flat does — pickable
    in the flat selector alongside panel, twilight and star flats.
    """
    key = DecisionKey.sky_flat_source(stack_id)
    current = context.settings.decision_overrides
    promoted = key in current and current[key].choice == BinaryChoice.INCLUDE

    if promoted:
        new_overrides = frozendict({k: v for k, v in current.items() if k != key})
    else:
        from decisions.helpers import include
        new_overrides = frozendict(dict(current) | include(key))

    _update_settings(lambda s: replace(s, decision_overrides=new_overrides))
    announce("reload_data")

    from state.actor import state
    state.drain_and_restart()

    from stacks.sky_flat import sky_flat_id_for
    return jsonify({
        "stack_id": stack_id,
        "is_sky_flat_source": not promoted,
        "sky_flat_stack_id": None if promoted else str(sky_flat_id_for(stack_id)),
    })


def _quarantined_frame_files() -> list[Path]:
    """Collect all raw frame files belonging to quarantined stacks/runs."""
    overrides = context.settings.decision_overrides
    stacks = context.stacks or {}
    calibration_stacks = context.calibration_stacks or {}

    quarantined_stack_ids = set()
    quarantined_run_ids = {}  # parent_stack_id → set of run_ids

    for key, override in overrides.items():
        if key.decision_type != DecisionType.QUARANTINE:
            continue
        if override.choice != BinaryChoice.EXCLUDE:
            continue
        if key.object_type == ObjectType.RUN:
            quarantined_run_ids.setdefault(key.parent_stack_id, set()).add(key.subject_id)
        elif key.object_type == ObjectType.STACK:
            quarantined_stack_ids.add(key.subject_id)

    files = []
    for sid in quarantined_stack_ids:
        ref = stacks.get(sid) or calibration_stacks.get(sid)
        if ref is None:
            continue
        stack = ref.get() if hasattr(ref, 'get') else ref
        for run in stack.contents.values():
            for fm in run.metadata.values():
                if fm.filename and os.path.exists(fm.filename):
                    files.append(Path(fm.filename))

    for parent_sid, run_ids in quarantined_run_ids.items():
        ref = stacks.get(parent_sid)
        if ref is None:
            continue
        stack = ref.get() if hasattr(ref, 'get') else ref
        for run_hash, run in stack.contents.items():
            if run_hash not in run_ids:
                continue
            for fm in run.metadata.values():
                if fm.filename and os.path.exists(fm.filename):
                    files.append(Path(fm.filename))

    return files


@settings_bp.route("/api/quarantined_files", methods=["GET"])
def quarantined_files_summary():
    """Return a summary of files that would be trashed."""
    files = _quarantined_frame_files()
    total_bytes = sum(f.stat().st_size for f in files if f.exists())
    return jsonify({
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": [str(f) for f in files[:50]],
    })


@settings_bp.route("/api/trash_quarantined", methods=["POST"])
def trash_quarantined():
    """Move all quarantined frame files to the macOS Trash.

    After moving, clears the quarantine overrides (the objects they
    reference no longer exist) and drains/restarts the compute graph
    so identification re-runs without the trashed frames.
    """
    from send2trash import send2trash

    files = _quarantined_frame_files()
    if not files:
        return jsonify({"trashed": 0, "error": None})

    trashed = 0
    for f in files:
        if f.exists():
            send2trash(str(f))
            sidecar = Path(str(f) + ".json")
            if sidecar.exists():
                send2trash(str(sidecar))
            trashed += 1

    # Clear all quarantine overrides — the files are gone, so the
    # quarantine state is moot. The objects will vanish from context
    # on the next graph rebuild.
    overrides = context.settings.decision_overrides
    new_overrides = frozendict({
        k: v for k, v in overrides.items()
        if k.decision_type != DecisionType.QUARANTINE
        or v.choice != BinaryChoice.EXCLUDE
    })
    _update_settings(lambda s: replace(s, decision_overrides=new_overrides))

    announce("reload_data")
    from state.actor import state
    state.drain_and_restart()

    return jsonify({"trashed": trashed, "error": None})
