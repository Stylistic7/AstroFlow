"""Decisions — the only writes. Each mutating tool previews its effect
unless confirm=true; a confirmed write goes through the same path as the
UI (`decisions.helpers.with_override`) and re-plans the whole graph."""
from __future__ import annotations

from typing import Literal

from decisions.decisions import DecisionKey, DecisionType, ObjectType, _decisions_key
from mcp_server.server import context, to_json, tool
from web_server.backend.decisions import (_deserialize_choice, _serialize_choice,
                                          _serialize_decision, _serialize_key)

DecisionTypeName = Literal["quarantine", "rotation_outlier", "subframe_quality",
                           "calibration_bias", "calibration_dark", "calibration_flat"]
ObjectTypeName = Literal["stack", "target", "mosaic", "run"]

_RESTART_EFFECT = ("the head stops admitting tasks, lets in-flight ones finish, "
                   "then re-plans and re-runs the whole graph (finished work "
                   "cache-hits; whatever this decision re-keys is recomputed)")


def _key(decision_type: str, object_type: str, parent_stack_id: str,
         subject_id: str) -> DecisionKey:
    return DecisionKey(DecisionType(decision_type), ObjectType(object_type),
                       parent_stack_id, subject_id)


def _current(key: DecisionKey) -> dict:
    overrides = context.settings.decision_overrides
    override = overrides.get(key)
    points = context.get(_decisions_key(key.object_type, key.parent_stack_id), {})
    dp = points.get(key)
    return {
        "override": _serialize_choice(override.choice) if override else None,
        "default": _serialize_choice(dp.default_choice) if dp else None,
    }


def _apply(key: DecisionKey, choice) -> dict:
    from decisions.helpers import with_override
    from state.actor import state
    from web_server.backend.announcer import announce
    context.settings = with_override(context.settings, key, choice)
    announce("reload_data")
    state.drain_and_restart()
    return {"applied": True, "key": _serialize_key(key),
            "choice": _serialize_choice(choice), "restarted": True}


@tool()
def list_decisions(object_type: ObjectTypeName, object_id: str) -> dict:
    """Decision points recorded for an object (default and effective
    choice, whether overridden, title, description) plus every override
    whose parent is this object — quarantines included, which never enter
    the decision cache. Keys are (decision_type, object_type,
    parent_stack_id, subject_id); a run-level key has the run's content
    hash as subject_id.
    """
    overrides = context.settings.decision_overrides
    points = context.get(_decisions_key(ObjectType(object_type), object_id), {})
    return to_json({
        "decisions": [_serialize_decision(k, dp, overrides) for k, dp in points.items()],
        "overrides": [{"key": _serialize_key(k), "choice": _serialize_choice(o.choice)}
                      for k, o in overrides.items() if k.parent_stack_id == object_id],
    })


@tool()
def set_decision(decision_type: DecisionTypeName, object_type: ObjectTypeName,
                 parent_stack_id: str, subject_id: str, choice: str,
                 confirm: bool = False) -> dict:
    """Override a decision. Binary types (quarantine, rotation_outlier)
    take "include"/"exclude"; calibration types take a calibration-stack
    hash, a file path, or "none" (pinned no-flat). Keys are exactly as
    list_decisions reports them. To quarantine a run: decision_type=
    "quarantine", object_type="run", parent_stack_id=<stack>, subject_id=
    <run id>, choice="exclude"; a whole stack: object_type="stack",
    subject_id = parent_stack_id = <stack>. Without confirm=true only the
    preview is returned; a confirmed write restarts the compute graph.
    """
    key = _key(decision_type, object_type, parent_stack_id, subject_id)
    parsed = _deserialize_choice(choice, key.decision_type)
    preview = {"action": "set", "key": _serialize_key(key), **_current(key),
               "new_choice": _serialize_choice(parsed), "will_restart": True,
               "effect": _RESTART_EFFECT, "confirm": confirm}
    if not confirm:
        return preview
    return {**preview, **_apply(key, parsed)}


@tool()
def clear_decision(decision_type: DecisionTypeName, object_type: ObjectTypeName,
                   parent_stack_id: str, subject_id: str,
                   confirm: bool = False) -> dict:
    """Remove an override so the decision reverts to its default. Without
    confirm=true only the preview is returned; when no override exists
    nothing happens either way.
    """
    key = _key(decision_type, object_type, parent_stack_id, subject_id)
    current = _current(key)
    preview = {"action": "clear", "key": _serialize_key(key), **current,
               "will_restart": current["override"] is not None,
               "effect": _RESTART_EFFECT, "confirm": confirm}
    if not confirm or current["override"] is None:
        return preview
    return {**preview, **_apply(key, None)}


@tool()
def drain_and_restart(confirm: bool = False) -> dict:
    """Stop admitting tasks, let in-flight ones finish, then re-plan and
    re-run the compute graph. Use after files change on disk outside the
    downloader's notice. Without confirm=true only the preview is returned.
    """
    running = context.get("node_running") or {}
    progress = context.get("compute_graph_status") or {}
    preview = {"action": "drain_and_restart", "in_flight": len(running),
               "remaining": (progress.get("_totals") or {}).get("remaining_number"),
               "effect": _RESTART_EFFECT, "confirm": confirm}
    if not confirm:
        return preview
    from state.actor import state
    state.drain_and_restart()
    return {**preview, "applied": True}
