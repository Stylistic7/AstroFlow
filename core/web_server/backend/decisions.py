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
    get_decisions_for_object,
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
        DecisionType.SMALL_RUN,
        DecisionType.SUBFRAME_QUALITY,
        DecisionType.QUARANTINE,
    }
    if decision_type in binary_types:
        return BinaryChoice(choice)
    # Calibration and other decisions: raw string (stack hash or path).
    return choice


@decisions_bp.route("/api/decisions/override", methods=["POST"])
def override_decision():
    """Set or clear an override for a decision point.

    Body: {"key": [...], "choice": <value>|null}
    Pass choice=null to clear the override and revert to default.
    For binary decisions, choice is "include" or "exclude".
    For calibration decisions, choice is a stack hash or filesystem path.
    """
    data = request.get_json()
    key = _deserialize_key(data["key"])
    raw_choice = data.get("choice")

    def updater(settings):
        current = dict(settings.decision_overrides)
        if raw_choice is None:
            current.pop(key, None)
        else:
            choice = _deserialize_choice(raw_choice, key[0])
            current[key] = DecisionOverride(choice=choice)
        return replace(settings, decision_overrides=frozendict(current))

    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings

    state.enqueue(
        dict(type="update_settings", settings=new_settings)
    )

    # Mark the affected object as stale so the UI shows an indicator.
    object_type = key[1].value  # ObjectType enum → string
    object_id = key[2]
    context.or_equals("invalidated", {(object_type, object_id): True})

    return jsonify({
        "key": _serialize_key(key),
        "choice": raw_choice,
    })
