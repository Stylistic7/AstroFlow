from dataclasses import replace
from flask import request, jsonify, Blueprint
from frozendict import frozendict

from context import context
from customization.settings import Settings
from customization.decisions import (
    DecisionType, ObjectType, DecisionOverride, BinaryChoice,
)
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
    """
    type_map = {"stack": ObjectType.STACK, "target": ObjectType.TARGET, "mosaic": ObjectType.MOSAIC}
    obj_type = type_map.get(object_type)
    if obj_type is None:
        return jsonify({"error": f"Unknown object type: {object_type}"}), 400

    key = (DecisionType.QUARANTINE, obj_type, object_id, object_id)

    current_overrides = context.settings.decision_overrides
    is_currently_quarantined = (
        key in current_overrides
        and current_overrides[key].choice == BinaryChoice.EXCLUDE
    )

    if is_currently_quarantined:
        # Un-quarantine: delete the override
        new_overrides = frozendict({k: v for k, v in current_overrides.items() if k != key})
    else:
        # Quarantine: add EXCLUDE override
        new_overrides = frozendict(dict(current_overrides) | {key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})

    _update_settings(lambda s: replace(s, decision_overrides=new_overrides))
    announce("reload_data")

    return jsonify({
        "object_type": object_type,
        "object_id": object_id,
        "is_quarantined": not is_currently_quarantined,
    })
