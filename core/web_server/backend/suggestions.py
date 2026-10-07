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


def _resolve_plan(body: dict) -> tuple[ObservingPlan, list, dict[str, int]]:
    """The request's plan (its entries or the stored ones), the suggestions
    they point at, and the panel numbers of every mosaic involved. Panel
    numbers come from Mosaic — single source of truth so the polygon
    labels in the viewer can't disagree with an export."""
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
    return plan, sugs, panel_numbers


@suggestions_bp.route("/api/observing_plan/export", methods=["POST"])
def export_plan():
    body = request.get_json(force=True)
    fmt = body.get("format", "text")
    plan, sugs, panel_numbers = _resolve_plan(body)
    for_date = plan.for_date

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


@suggestions_bp.route("/api/observing_plan/send_asiair", methods=["POST"])
def send_plan_to_asiair():
    """Write the observing plan into an ASIAir's Plan mode — coordinates
    AND exposure rows — as a NEW, disabled plan beside whatever is armed.

    Body: {"ip": "10.0.0.28", "dry_run": true, "plan_name": "...",
    "replace": true, "entries": [...]?}. `dry_run` (the default) returns the
    plan exactly as it would be sent plus what the device holds now; a real
    send backs up the device's plan list first, then writes and verifies by
    reading back. `replace` (the default — Stylistic7's workflow) deletes the
    device's plans and writes ours fresh, which is the only way to drop a
    target; `replace: false` adds ours beside the existing ones. Nothing
    here starts a plan or moves the mount; Start is a tap on the tablet.
    """
    import json
    import time
    from asiair.plan_push import (
        ExposureRow, PlanTarget, build_plan, next_plan_id, push_plan, replace_plan, wheel_slots,
    )
    from asiair.rpc_session import AsiairSession
    from asiair.rsa_sign import default_key_path
    from base_path import app_support_dir
    from suggestions.export import _match

    body = request.get_json(force=True)
    ip = body.get("ip")
    if not ip:
        abort(400, description="missing ip")
    dry_run = bool(body.get("dry_run", True))
    replace = bool(body.get("replace", True))
    key = default_key_path()
    if not key.exists():
        abort(409, description=f"no ASIAir signing key at {key}")

    plan, sugs, panel_numbers = _resolve_plan(body)
    grouped: dict[str, dict] = {}
    for entry, s in _match(plan, sugs):
        n = panel_numbers.get(s.target_id)
        # Target names are identity, not prose (they end up in file names):
        # "<mosaic hash prefix> Panel N", or "<target hash prefix> <label>"
        # for a panel outside any mosaic — Stylistic7's shorthand ("75a1 Panel 10").
        t = grouped.setdefault(s.target_id, {
            "order": (n is None, n or 0, s.panel_label),
            "name": f"{s.mosaic_id[:4]} Panel {n}" if n is not None else f"{s.target_id[:4]} {s.panel_label}",
            "ra": s.ra_deg, "dec": s.dec_deg, "rows": [],
        })
        t["rows"].append(ExposureRow(s.filter, entry.exposure_per_frame_s, entry.frame_count))
    targets = [
        PlanTarget(t["name"], t["ra"], t["dec"], tuple(t["rows"]))
        for t in sorted(grouped.values(), key=lambda t: t["order"])
    ]
    if not targets:
        abort(400, description="the observing plan is empty")

    with AsiairSession(ip) as session:
        if not session.verify(key):
            abort(502, description=f"ASIAir {ip} refused the handshake")
        existing = session.call("get_plan") or []
        like = next((p for p in existing if p.get("enable")), existing[0] if existing else None)
        device_plan = build_plan(
            1 if replace else next_plan_id(session), body.get("plan_name") or f"AstroFlow {plan.for_date}",
            targets, wheel_slots(session), like=like,
        )
        summary = [{"id": p.get("id"), "plan_name": p.get("plan_name"), "enable": p.get("enable"),
                    "targets": len(p.get("targets", []))} for p in existing]
        if dry_run:
            return jsonify({"dry_run": True, "plan": device_plan, "existing": summary})
        backups = app_support_dir() / "asiair_backups"
        backups.mkdir(parents=True, exist_ok=True)
        (backups / f"{ip}_get_plan_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(existing, indent=2))
        try:
            result = (replace_plan if replace else push_plan)(session, device_plan)
        except RuntimeError as e:
            abort(502, description=str(e))
    return jsonify({"dry_run": False, "plan_name": device_plan["plan_name"], "existing": summary, **result})
