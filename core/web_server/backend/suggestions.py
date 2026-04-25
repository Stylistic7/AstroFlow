"""Flask blueprint for the Suggestions feature.

Suggestions surface as small badges on objects (targets, mosaics). The user
clicks a suggestion to "accept" it — which appends it to a persistent
observing plan in `context.observing_plan`. The plan resets when the user
clears it (typically once per clear night, like starting a new fishing trip).
"""
import datetime
import flask
from flask import Blueprint, jsonify, request, Response, abort

from astropy.coordinates import EarthLocation  # type: ignore
import astropy.units as u  # type: ignore

from context import local_context as context
from suggestions.generator import generate_suggestions
from suggestions.moon import compute_conditions
from suggestions.export import telescopius_csv, rich_json, plain_text
from suggestions.types import ObservingPlan, PlanEntry

suggestions_bp = Blueprint("suggestions", __name__)

_ALL_FILTERS = frozenset({"H", "S", "O", "L", "R", "G", "B"})


def _get_plan_entries() -> list[dict]:
    """Plan is stored as a list of plain-dict entries keyed by suggestion_key."""
    return list(context.observing_plan or [])


def _set_plan_entries(entries: list[dict]) -> None:
    context.observing_plan = entries


@suggestions_bp.route("/api/suggestions")
def all_suggestions():
    """All suggestions across the Shopping Cart. Used for counts and summaries."""
    sugs = generate_suggestions(_ALL_FILTERS)
    return jsonify({"suggestions": [s.to_dict() for s in sugs]})


@suggestions_bp.route("/api/suggestions/by_object/<object_type>/<object_id>")
def suggestions_by_object(object_type: str, object_id: str):
    """Suggestions scoped to a single target or mosaic."""
    sugs = generate_suggestions(_ALL_FILTERS)
    if object_type == "target":
        scoped = [s for s in sugs if s.target_id == object_id]
    elif object_type == "mosaic":
        scoped = [s for s in sugs if s.mosaic_id == object_id]
    else:
        scoped = []
    return jsonify({"suggestions": [s.to_dict() for s in scoped]})


@suggestions_bp.route("/api/moon")
def moon():
    settings = context.settings
    if settings is None or settings.observer_location is None:
        return jsonify({"error": "observer_location not set"}), 400
    date_str = request.args.get("date")
    for_date = (
        datetime.date.fromisoformat(date_str) if date_str else datetime.date.today()
    )
    lat, lon, elev = settings.observer_location
    observer = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
    conditions = compute_conditions(for_date, observer)
    return jsonify({
        "for_date": for_date.isoformat(),
        "illumination": conditions.illumination,
        "altitude_at_midnight_deg": conditions.altitude_at_midnight_deg,
        "recommended_filters": sorted(conditions.recommended_filters),
    })


@suggestions_bp.route("/api/observing_plan", methods=["GET"])
def get_observing_plan():
    return jsonify({"entries": _get_plan_entries()})


@suggestions_bp.route("/api/observing_plan/accept", methods=["POST"])
def accept_suggestion():
    """Add (or update) a plan entry. Idempotent on suggestion_key."""
    body = request.get_json(force=True)
    key = body.get("suggestion_key")
    if not key:
        abort(400, description="missing suggestion_key")
    exposure = float(body.get("exposure_per_frame_s"))
    count = int(body.get("frame_count"))
    filter_name = body.get("filter")
    panel_label = body.get("panel_label")
    entries = _get_plan_entries()
    entries = [e for e in entries if e.get("suggestion_key") != key]
    entries.append({
        "suggestion_key": key,
        "exposure_per_frame_s": exposure,
        "frame_count": count,
        "filter": filter_name,
        "panel_label": panel_label,
    })
    _set_plan_entries(entries)
    return jsonify({"entries": entries})


@suggestions_bp.route("/api/observing_plan/entry/<key>", methods=["DELETE"])
def remove_plan_entry(key: str):
    entries = [e for e in _get_plan_entries() if e.get("suggestion_key") != key]
    _set_plan_entries(entries)
    return jsonify({"entries": entries})


@suggestions_bp.route("/api/observing_plan/clear", methods=["POST"])
def clear_observing_plan():
    _set_plan_entries([])
    return jsonify({"entries": []})


@suggestions_bp.route("/api/observing_plan/export", methods=["POST"])
def export_plan():
    body = request.get_json(force=True)
    fmt = body.get("format", "text")
    for_date_str = body.get("for_date")
    for_date = (
        datetime.date.fromisoformat(for_date_str)
        if for_date_str
        else datetime.date.today()
    )
    entries_in = body.get("entries") or _get_plan_entries()
    entries = tuple(
        PlanEntry(
            suggestion_key=e["suggestion_key"],
            exposure_per_frame_s=float(e["exposure_per_frame_s"]),
            frame_count=int(e["frame_count"]),
        )
        for e in entries_in
    )
    plan = ObservingPlan(for_date=for_date, entries=entries)
    sugs = generate_suggestions(_ALL_FILTERS)

    if fmt == "telescopius":
        return Response(
            telescopius_csv(plan, sugs),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=plan_{for_date}.csv"},
        )
    if fmt == "json":
        return Response(rich_json(plan, sugs), mimetype="application/json")
    if fmt == "text":
        return Response(plain_text(plan, sugs), mimetype="text/plain")
    abort(400, description=f"unknown format {fmt!r}")
