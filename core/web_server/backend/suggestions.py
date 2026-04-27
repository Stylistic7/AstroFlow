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


def _annotate_with_panel_numbers(sugs: list) -> list[dict]:
    """Attach the per-mosaic panel_number to each suggestion dict so the
    frontend can group + sort suggestions by panel — the same numbering
    used by the polygon labels and the Telescopius export."""
    mosaics = context.mosaics or {}
    cache: dict[str, dict] = {}
    out = []
    for s in sugs:
        d = s.to_dict()
        if s.mosaic_id not in cache:
            ref = mosaics.get(s.mosaic_id)
            cache[s.mosaic_id] = ref.get().panel_numbers() if ref is not None else {}
        d["panel_number"] = cache[s.mosaic_id].get(s.target_id)
        out.append(d)
    return out


@suggestions_bp.route("/api/suggestions")
def all_suggestions():
    """All suggestions across the Shopping Cart. Used for counts and summaries."""
    sugs = generate_suggestions(_ALL_FILTERS)
    return jsonify({"suggestions": _annotate_with_panel_numbers(sugs)})


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
    return jsonify({"suggestions": _annotate_with_panel_numbers(scoped)})


@suggestions_bp.route("/api/moon")
def moon():
    date_str = request.args.get("date")
    for_date = (
        datetime.date.fromisoformat(date_str) if date_str else datetime.date.today()
    )
    lat, lon, elev = context.settings.observer_location
    observer = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
    conditions = compute_conditions(for_date, observer)
    return jsonify({
        "for_date": for_date.isoformat(),
        "illumination": conditions.illumination,
        "altitude_at_midnight_deg": conditions.altitude_at_midnight_deg,
        "recommended_filters": sorted(conditions.recommended_filters),
    })


def _entries_with_panel_numbers(entries: list[dict]) -> list[dict]:
    """Annotate plan entries with panel_number (looked up from the
    suggestion the entry came from), and sort by panel_number so the
    Observing Plan dropdown matches the order of the polygon labels
    and the Suggestions dialog. Entries whose suggestion can't be
    resolved keep panel_number=None and sort to the end."""
    sugs = generate_suggestions(_ALL_FILTERS)
    by_key = {s.stable_key(): s for s in sugs}
    mosaics = context.mosaics or {}
    panel_cache: dict[str, dict] = {}
    out = []
    for e in entries:
        s = by_key.get(e.get("suggestion_key"))
        pn = None
        if s is not None:
            if s.mosaic_id not in panel_cache:
                ref = mosaics.get(s.mosaic_id)
                panel_cache[s.mosaic_id] = ref.get().panel_numbers() if ref else {}
            pn = panel_cache[s.mosaic_id].get(s.target_id)
        out.append({**e, "panel_number": pn})
    out.sort(key=lambda x: (x["panel_number"] is None, x["panel_number"] or 0, x.get("filter") or ""))
    return out


@suggestions_bp.route("/api/observing_plan", methods=["GET"])
def get_observing_plan():
    return jsonify({"entries": _entries_with_panel_numbers(_get_plan_entries())})


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

    # Panel numbers come from Mosaic — single source of truth so the
    # polygon labels in the viewer can't disagree with the export.
    by_key = {s.stable_key(): s for s in sugs}
    plan_mosaic_ids = {
        by_key[e.suggestion_key].mosaic_id
        for e in entries
        if e.suggestion_key in by_key
    }
    mosaics = context.mosaics or {}
    panel_numbers: dict[str, int] = {}
    for mid in plan_mosaic_ids:
        ref = mosaics.get(mid)
        if ref is not None:
            panel_numbers.update(ref.get().panel_numbers())

    if fmt == "telescopius":
        return Response(
            telescopius_csv(plan, sugs, panel_numbers),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=plan_{for_date}.csv"},
        )
    if fmt == "json":
        return Response(rich_json(plan, sugs, panel_numbers), mimetype="application/json")
    if fmt == "text":
        return Response(plain_text(plan, sugs, panel_numbers), mimetype="text/plain")
    abort(400, description=f"unknown format {fmt!r}")
