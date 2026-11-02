"""`GET /api/sites` — every observing site the library's frames state
(`light_pollution`): its runs and frames, its sky-glow model when the data
support one (with the model's map: percent brighter than the zenith by
altitude and azimuth) or the reason it has none, and where its frames were
on the dome. Read from the context the graph stored it in; `[]` until the
first fold has run."""
from __future__ import annotations

from flask import Blueprint, jsonify

from context import context
from light_pollution.model import brightness_grid

sites_bp = Blueprint("sites", __name__)


@sites_bp.route("/api/sites")
def sites():
    table = context.get("site_models")
    if table is None:
        return jsonify([])
    out = []
    for report in table.report:
        model = table.models.get(report["key"])
        out.append({**report, "grid": brightness_grid(model) if model is not None else None})
    return jsonify(sorted(out, key=lambda r: -r["frames"]))
