import os
from dataclasses import replace
from pathlib import Path
from flask import request, jsonify, Blueprint
from frozendict import frozendict

from context import local_context as context
from decisions.settings import Settings
from decisions.decisions import ObjectType, BinaryChoice, DecisionType
from decisions.decisions import DecisionKey
from decisions.helpers import exclude
from web_server.backend.announcer import announce

settings_bp = Blueprint("settings", __name__)


def _update_settings(updater):
    """Read-modify-write the settings in the context."""
    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings
    return new_settings


@settings_bp.route("/api/quarantine/<object_type>/<object_id>", methods=["POST"])
def toggle_quarantine(object_type, object_id):
    """Toggle quarantine for a stack, target, or mosaic.

    Quarantine is a decision override: EXCLUDE means quarantined,
    absent means not quarantined.

    For run-level quarantine (quarantining a run within a stack), pass
    ?parent=<stack_hash> so the decision key carries the stack hash as
    its parent_stack_id (required for per-stack decision filtering).
    """
    type_map = {"stack": ObjectType.STACK, "target": ObjectType.TARGET, "mosaic": ObjectType.MOSAIC}
    obj_type = type_map.get(object_type)
    if obj_type is None:
        return jsonify({"error": f"Unknown object type: {object_type}"}), 400

    parent_stack_id = request.args.get("parent")
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
        if key.object_type != ObjectType.STACK:
            continue
        if key.parent_stack_id != key.subject_id:
            quarantined_run_ids.setdefault(key.parent_stack_id, set()).add(key.subject_id)
        else:
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
