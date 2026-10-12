"""`GET /api/planner/next` — the next night's plan with its explanation.
`GET /api/planner/next_v2` — the v2 shadow plan (month-aware, weather-aware).
`PUT|DELETE /api/planner/override/<day>` — the user's word for one night.
`GET /api/planner/stars`, `PUT|DELETE /api/planner/star/<type>/<id>` — the starred (shoot-first) cart entries.

Served from the plan `planner.refresh` keeps on disk. A plan for another
night is never shown: the page waits (202) while tonight's is made. A plan
for tonight from older inputs is shown at once, marked `refreshing`, while
the new one is computed. The SchedulerActor's hourly job keeps it current,
so the wait is rare.
"""
from __future__ import annotations

from datetime import date

from flask import Blueprint, jsonify, request

from context import local_context as context
from planner import refresh

planner_bp = Blueprint("planner", __name__)


@planner_bp.route("/api/planner/next")
def next_plan():
    state = refresh.refresh_if_stale(context)
    if state == "stale":
        return jsonify({"status": "computing"}), 202
    return jsonify({**refresh.stored(), "refreshing": state == "refreshing"})


@planner_bp.route("/api/planner/next_v2")
def next_plan_v2():
    """The shadow plan (planner.v2), written beside every v1 plan."""
    state = refresh.refresh_if_stale(context)
    plan = refresh.stored(refresh.plan_path_v2())
    if plan is None:
        return jsonify({"status": "computing"}), 202
    return jsonify({**plan, "refreshing": state != "current"})


OVERRIDE_WORDS = ("clear", "cloudy", "away")


@planner_bp.route("/api/planner/stars")
def get_stars():
    """The starred cart entries, `[[type, id], ...]`."""
    return jsonify(context.get("planner_stars") or [])


@planner_bp.route("/api/planner/star/<object_type>/<object_id>", methods=["PUT", "DELETE"])
def put_star(object_type, object_id):
    """Star (PUT) or unstar (DELETE) a cart entry. Two tiers: the planner
    gives the starred entries every hour they can still use and plans the
    unstarred, by improvement, from the hours the stars cannot use.
    Improvement per hour alone never finishes an image — a fresh panel
    always beats a deep one — so this is said, not inferred. Planner-only:
    no graph restart."""
    if object_type not in ("target", "mosaic"):
        return jsonify({"error": "object_type must be target or mosaic"}), 400
    from web_server.backend.announcer import announce
    stars = {tuple(s) for s in context.get("planner_stars") or []}
    (stars.add if request.method == "PUT" else stars.discard)((object_type, object_id))
    context["planner_stars"] = sorted(stars)
    announce("reload_data")
    return jsonify({"stars": context["planner_stars"]})




@planner_bp.route("/api/planner/override/<day>", methods=["PUT", "DELETE"])
def put_override(day):
    """The user's word for one night (the evening's local date, YYYY-MM-DD):
    `{"state": "clear" | "cloudy" | "away"}`; DELETE forgets it. Wins over
    every forecast in the v2 plan (weather.probability)."""
    try:
        date.fromisoformat(day)
    except ValueError:
        return jsonify({"error": "day must be YYYY-MM-DD"}), 400
    current = dict(context.get("planner_overrides") or {})
    if request.method == "DELETE":
        current.pop(day, None)
    else:
        state = (request.get_json(silent=True) or {}).get("state")
        if state not in OVERRIDE_WORDS:
            return jsonify({"error": f"state must be one of {OVERRIDE_WORDS}"}), 400
        current[day] = state
    context["planner_overrides"] = current
    return jsonify(current)
