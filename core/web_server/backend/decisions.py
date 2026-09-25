from dataclasses import replace
from flask import request, jsonify, Blueprint
from frozendict import frozendict

from context import local_context as context
from decisions.decisions import (
    DecisionKey,
    DecisionOverride,
    BinaryChoice,
    DecisionType,
    ObjectType,
    get_decisions_for_object,
)
from state.actor import state

decisions_bp = Blueprint("decisions", __name__)


def _serialize_key(key: DecisionKey) -> dict:
    """Serialize a DecisionKey to a JSON-safe dict."""
    return {
        "decision_type": key.decision_type.value,
        "object_type": key.object_type.value,
        "parent_stack_id": key.parent_stack_id,
        "subject_id": key.subject_id,
    }


def _deserialize_key(raw: dict) -> DecisionKey:
    """Deserialize a JSON dict back to a DecisionKey."""
    return DecisionKey(
        decision_type=DecisionType(raw["decision_type"]),
        object_type=ObjectType(raw["object_type"]),
        parent_stack_id=raw["parent_stack_id"],
        subject_id=raw["subject_id"],
    )


def _serialize_choice(choice):
    """Serialize a choice value (enum or plain string/int) to JSON."""
    return choice.value if hasattr(choice, "value") else choice


def _serialize_decision(key, dp, overrides):
    override = overrides.get(key)
    return {
        "key": _serialize_key(key),
        "decision_type": dp.decision_type.value,
        "default_choice": _serialize_choice(dp.default_choice),
        "effective_choice": (
            _serialize_choice(override.choice) if override
            else _serialize_choice(dp.default_choice)
        ),
        "is_overridden": override is not None,
        "title": dp.title,
        "description": dp.description,
        "metadata": dict(dp.metadata),
        "object_type": dp.object_type.value,
        "object_id": dp.object_id,
        "subject_id": dp.subject_id,
    }


@decisions_bp.route(
    "/api/decisions/for/<object_type>/<object_id>", methods=["GET"]
)
def list_decisions(object_type, object_id):
    """Return decisions relevant to a specific object."""
    decisions = get_decisions_for_object(ObjectType(object_type), object_id)
    overrides = context.settings.decision_overrides

    result = [
        _serialize_decision(key, dp, overrides)
        for key, dp in decisions.items()
    ]

    return jsonify(result)


def _deserialize_choice(choice, decision_type: DecisionType):
    """Parse a choice value from JSON based on the decision type."""
    if choice is None:
        return None
    binary_types = {
        DecisionType.ROTATION_OUTLIER,
        DecisionType.SUBFRAME_QUALITY,
        DecisionType.QUARANTINE,
        DecisionType.SKY_FLAT_SOURCE,
    }
    if decision_type in binary_types:
        return BinaryChoice(choice)
    # Calibration and other decisions: raw string (stack hash, path, or the
    # pinned "none" — `decisions.resolve.NO_FLAT`).
    return choice


@decisions_bp.route("/api/decisions/override", methods=["POST"])
def override_decision():
    """Set or clear an override for a decision point.

    Body: {"key": [...], "choice": <value>|null}
    Pass choice=null to clear the override and revert to default.
    For binary decisions, choice is "include" or "exclude".
    For calibration decisions, choice is a stack hash, a filesystem path,
    or "none" (a pinned no-flat, distinct from clearing the override).
    """
    data = request.get_json()
    key = _deserialize_key(data["key"])
    raw_choice = data.get("choice")

    def updater(settings):
        current = dict(settings.decision_overrides)
        if raw_choice is None:
            current.pop(key, None)
        else:
            choice = _deserialize_choice(raw_choice, key.decision_type)
            current[key] = DecisionOverride(choice=choice)
        return replace(settings, decision_overrides=frozendict(current))

    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings

    state.drain_and_restart()

    return jsonify({
        "key": _serialize_key(key),
        "choice": raw_choice,
    })
