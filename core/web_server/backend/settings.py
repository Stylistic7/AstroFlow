from dataclasses import replace
from flask import request, jsonify, Blueprint, Response
from frozendict import frozendict

from context import context
from customization.settings import (
    Settings,
    StackSettings,
    TargetSettings,
    MosaicSettings,
)
from state.actor import state

settings_bp = Blueprint("settings", __name__)


def _update_settings(updater):
    """Read-modify-write the settings in the context."""
    settings = context.settings
    new_settings = updater(settings)
    context.settings = new_settings
    return new_settings


@settings_bp.route("/api/settings/stack", methods=["POST"])
def update_stack_settings():
    data = request.get_json()
    stack_id = data["stack_id"]
    run_id = data["run_id"]
    dependencies = data["dependencies"]

    def updater(settings):
        stack_settings = settings.stack_settings.get(run_id, StackSettings())
        new_deps = dict(stack_settings.dependencies) | dependencies
        new_stack_settings = replace(stack_settings, dependencies=frozendict(new_deps))
        return replace(
            settings,
            stack_settings=frozendict(dict(settings.stack_settings) | {run_id: new_stack_settings}),
        )

    settings = _update_settings(updater)

    state.enqueue(
        dict(type="update_settings", stack_id=stack_id, settings=settings)
    )

    return jsonify({"status": "ok"})


@settings_bp.route("/api/quarantine/stack/<stack_id>", methods=["POST"])
def toggle_quarantine_stack(stack_id):
    """Toggle quarantine state for a stack (or substack/run)."""
    def updater(settings):
        current = settings.stack_settings.get(stack_id, StackSettings())
        toggled = replace(current, is_quarantined=not current.is_quarantined)
        return replace(
            settings,
            stack_settings=frozendict(dict(settings.stack_settings) | {stack_id: toggled}),
        )

    _update_settings(updater)
    is_quarantined = context.settings.stack_settings.get(stack_id, StackSettings()).is_quarantined
    return jsonify({"stack_id": stack_id, "is_quarantined": is_quarantined})


@settings_bp.route("/api/quarantine/target/<target_id>", methods=["POST"])
def toggle_quarantine_target(target_id):
    """Toggle quarantine state for a target."""
    def updater(settings):
        current = settings.target_settings.get(target_id, TargetSettings())
        toggled = replace(current, is_quarantined=not current.is_quarantined)
        return replace(
            settings,
            target_settings=frozendict(dict(settings.target_settings) | {target_id: toggled}),
        )

    _update_settings(updater)
    is_quarantined = context.settings.target_settings.get(target_id, TargetSettings()).is_quarantined
    return jsonify({"target_id": target_id, "is_quarantined": is_quarantined})


@settings_bp.route("/api/quarantine/mosaic/<mosaic_id>", methods=["POST"])
def toggle_quarantine_mosaic(mosaic_id):
    """Toggle quarantine state for a mosaic."""
    def updater(settings):
        current = settings.mosaic_settings.get(mosaic_id, MosaicSettings())
        toggled = replace(current, is_quarantined=not current.is_quarantined)
        return replace(
            settings,
            mosaic_settings=frozendict(dict(settings.mosaic_settings) | {mosaic_id: toggled}),
        )

    _update_settings(updater)
    is_quarantined = context.settings.mosaic_settings.get(mosaic_id, MosaicSettings()).is_quarantined
    return jsonify({"mosaic_id": mosaic_id, "is_quarantined": is_quarantined})


@settings_bp.route("/api/quarantine", methods=["GET"])
def get_quarantine_state():
    """Return all quarantine states for the UI."""
    settings = context.settings
    return jsonify({
        "stacks": {
            k: v.is_quarantined
            for k, v in settings.stack_settings.items()
            if v.is_quarantined
        },
        "targets": {
            k: v.is_quarantined
            for k, v in settings.target_settings.items()
            if v.is_quarantined
        },
        "mosaics": {
            k: v.is_quarantined
            for k, v in settings.mosaic_settings.items()
            if v.is_quarantined
        },
    })
