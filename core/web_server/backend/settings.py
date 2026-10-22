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


@settings_bp.route("/api/sharing/<object_type>/<object_id>", methods=["PUT", "DELETE"])
def put_sharing(object_type, object_id):
    """`{"folder": <a configured shared master folder>}`: PUT publishes this
    stack's, target's or mosaic's masters into that folder from the end of
    the next pass on (a target or mosaic sends all its stacks); DELETE stops
    — the published copies are retired then (`stacks.publish_shared`).
    Opt-in only. The graph restarts: publishing happens at the end of a
    pass."""
    from stacks.publish_shared import SHARED
    try:
        kind = ObjectType(object_type)
    except ValueError:
        return jsonify({"error": "object_type must be stack, target or mosaic"}), 400
    folder = str((request.get_json(silent=True) or {}).get("folder") or "")
    if folder not in {str(d) for d in (context.shared_dirs or [])}:
        return jsonify({"error": "folder must be one of the configured shared master folders"}), 400
    key = DecisionKey.sharing(kind, object_id, folder)
    current = dict(context.settings.decision_overrides)
    if request.method == "DELETE":
        current.pop(key, None)
    else:
        if kind == ObjectType.STACK:
            verdict = _share_readiness(object_id)
            if not verdict["ready"]:
                return jsonify({"error": "this stack's master is not ready to share", **verdict}), 409
        current[key] = DecisionOverride(choice=SHARED)
    _update_settings(lambda s: replace(s, decision_overrides=frozendict(current)))
    announce("reload_data")
    from state.actor import state
    state.drain_and_restart()
    return jsonify({"object_type": object_type, "object_id": object_id, "folder": folder,
                    "shared": request.method == "PUT"})


_readiness_cache: dict = {"at": 0.0, "value": {}}
_READINESS_TTL_S = 60.0

NO_MASTER = {"ready": False, "warnings": [], "blocking": [{
    "code": "master", "title": "No master yet",
    "detail": "This stack has no post-processed master to share; one is made at the end of a pass."}]}


def _share_readiness(stack_id: str) -> dict:
    """`metadata.share_gate` over every current master, cached briefly: the
    verdict needs the train's typical PSF, which is a median over its peers."""
    import time
    from context import get_cache
    from metadata.share_gate import concerns_from_cache, stacks_readiness
    from stacks.publish_shared import current_masters
    now = time.monotonic()
    if now - _readiness_cache["at"] > _READINESS_TTL_S:
        ui_stacks = context.ui_stacks or {}
        cache = get_cache()
        _readiness_cache["value"] = stacks_readiness(
            ui_stacks, current_masters(set(ui_stacks)), lambda sid: concerns_from_cache(cache, sid))
        _readiness_cache["at"] = now
    return _readiness_cache["value"].get(stack_id, NO_MASTER)


@settings_bp.route("/api/share_readiness/stack/<stack_id>", methods=["GET"])
def share_readiness(stack_id):
    """Whether this stack's master may go into a shared master folder, and
    every reason it may not or should be thought about first
    (`metadata.share_gate`): `{"ready", "blocking": [...], "warnings": [...]}`."""
    if stack_id not in (context.ui_stacks or {}):
        return jsonify({"error": "stack not found"}), 404
    return jsonify(_share_readiness(stack_id))


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

    A stack shot as SkyFlat panes (`stacks/sky_flat_plans.py`) is a source by
    default; any other is not. Toggling flips the stack's EFFECTIVE state and
    stores a decision only when the result differs from that default — so a
    promoted ordinary stack holds an INCLUDE, a demoted SkyFlat stack an
    EXCLUDE, and a stack back at its default holds nothing. Promoting mints a
    `SkyFlatStack` whose frames are this stack's; it is then an ordinary flat,
    pickable in the flat selector alongside panel, twilight and star flats.
    """
    from decisions.helpers import exclude, include
    from stacks.sky_flat import is_sky_flat_source, sky_flat_id_for
    from stacks.sky_flat_plans import is_named_sky_flat

    key = DecisionKey.sky_flat_source(stack_id)
    current = context.settings.decision_overrides
    filenames = ((context.ui_stacks or {}).get(stack_id) or {}).get("filenames") or []
    now = is_sky_flat_source(stack_id, filenames, current)
    default = is_named_sky_flat(filenames)
    rest = {k: v for k, v in current.items() if k != key}
    if (not now) == default:
        new_overrides = frozendict(rest)
    else:
        new_overrides = frozendict(rest | (exclude(key) if now else include(key)))

    _update_settings(lambda s: replace(s, decision_overrides=new_overrides))
    announce("reload_data")

    from state.actor import state
    state.drain_and_restart()

    return jsonify({
        "stack_id": stack_id,
        "is_sky_flat_source": not now,
        "sky_flat_stack_id": None if now else str(sky_flat_id_for(stack_id)),
    })


@settings_bp.route("/api/sky_flat_sources", methods=["GET"])
def sky_flat_sources():
    """Every stack that is a night-sky flat source NOW: the explicit
    promotions, plus the SkyFlat-named stacks nobody demoted. Read from the
    UI stack summaries — no stack is loaded."""
    from stacks.sky_flat import is_sky_flat_source
    overrides = context.settings.decision_overrides or {}
    ui = context.ui_stacks or {}
    return jsonify(sorted(sid for sid, s in ui.items()
                          if is_sky_flat_source(sid, s.get("filenames") or [], overrides)))


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
