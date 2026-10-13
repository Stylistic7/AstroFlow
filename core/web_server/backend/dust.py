"""`GET /api/dust/epochs` — the dust-mote epochs of every optical system
(`stacks/dust_runs.py`): per epoch its first run's date, run count and
model (motes, state, split-half error). Read from the context the graph
stored it in; `[]` until the first fold has run."""
from __future__ import annotations

from flask import Blueprint, jsonify

from context import context

dust_bp = Blueprint("dust", __name__)


@dust_bp.route("/api/dust/epochs")
def dust_epochs():
    table = context.get("dust_epochs")
    return jsonify(list(table.report) if table is not None else [])
