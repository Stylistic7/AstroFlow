from dataclasses import replace
from flask import request, jsonify, Blueprint
from frozendict import frozendict

from context import context
from customization.decisions import (
    DecisionKey,
    DecisionOverride,
    BinaryChoice,
    DecisionType,
    ObjectType,
)
from state.actor import state

decisions_bp = Blueprint("decisions", __name__)


def _serialize_key(key: DecisionKey) -> list[str]:
    """Serialize a DecisionKey tuple to JSON-safe list of strings."""
    return [k.value if hasattr(k, "value") else str(k) for k in key]


def _deserialize_key(raw: list[str]) -> DecisionKey:
    """Deserialize a JSON list back to a DecisionKey tuple."""
    return (
        DecisionType(raw[0]),
        ObjectType(raw[1]),
        raw[2],
        raw[3],
    )


def _serialize_decision(key, dp, overrides):
    override = overrides.get(key)
    return {
        "key": _serialize_key(key),
        "decision_type": dp.decision_type.value,
        "default_choice": dp.default_choice.value,
        "effective_choice": (
            override.choice.value if override else dp.default_choice.value
        ),
        "is_overridden": override is not None,
        "title": dp.title,
        "description": dp.description,
        "metadata": dict(dp.metadata),
        "object_type": dp.object_type.value,
        "object_id": dp.object_id,
        "subject_id": dp.subject_id,
    }


@decisions_bp.route("/api/decisions", methods=["GET"])
def get_decisions():
    """Return all detected decision points with their resolved state."""
    decisions = context.get("decisions", {})
    overrides = context.settings.decision_overrides

    result = [
        _serialize_decision(key, dp, overrides)
        for key, dp in decisions.items()
    ]

    return jsonify(result)


@decisions_bp.route(
    "/api/decisions/for/<object_type>/<object_id>", methods=["GET"]
)
def get_decisions_for_object(object_type, object_id):
    """Return decisions relevant to a specific object."""
    decisions = context.get("decisions", {})
    overrides = context.settings.decision_overrides

    relevant = [
        _serialize_decision(key, dp, overrides)
        for key, dp in decisions.items()
        if dp.object_type.value == object_type and dp.object_id == object_id
    ]

    return jsonify(relevant)


@decisions_bp.route("/api/decisions/override", methods=["POST"])
def override_decision():
    """Set or clear an override for a decision point.

    Body: {"key": [...], "choice": "include"|"exclude"|null}
    Pass choice=null to clear the override and revert to default.
    """
    data = request.get_json()
    key = _deserialize_key(data["key"])
    choice = data.get("choice")

    def updater(settings):
        current = dict(settings.decision_overrides)
        if choice is None:
            current.pop(key, None)
        else:
            current[key] = DecisionOverride(choice=BinaryChoice(choice))
        return replace(settings, decision_overrides=frozendict(current))

    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings

    state.enqueue(
        dict(type="update_settings", settings=new_settings)
    )

    return jsonify({
        "key": _serialize_key(key),
        "choice": choice,
    })
