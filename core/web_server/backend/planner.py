"""`GET /api/planner/next` — the next night's plan with its explanation.

Served from the plan `planner.refresh` keeps on disk. A plan for another
night is never shown: the page waits (202) while tonight's is made. A plan
for tonight from older inputs is shown at once, marked `refreshing`, while
the new one is computed. The SchedulerActor's hourly job keeps it current,
so the wait is rare.
"""
from __future__ import annotations

from flask import Blueprint, jsonify

from context import local_context as context
from planner import refresh

planner_bp = Blueprint("planner", __name__)


@planner_bp.route("/api/planner/next")
def next_plan():
    state = refresh.refresh_if_stale(context)
    if state == "stale":
        return jsonify({"status": "computing"}), 202
    return jsonify({**refresh.stored(), "refreshing": state == "refreshing"})
