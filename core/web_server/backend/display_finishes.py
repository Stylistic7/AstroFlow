"""Serve the finish registry to the frontend.

The finish picker, grid axes, sliders, and real-time shader are ALL generated
from these specs, not hand-coded on the frontend. Adding a look = writing a
short traced-Python function (`finishes/stock.py`, or a user file in
`~/AstroFlow/finishes/`); it appears here automatically with its extracted
params and traced WGSL. Finishes RUN entirely in the frontend (WGSL shader)
and seed their starting values there from the generic per-channel image stats
in tile-metadata; the backend ships scene + stats, never finish logic.
"""
from flask import Blueprint, jsonify, request

from finishes.registry import finish_specs, refresh_finishes

display_finishes_bp = Blueprint("display_finishes", __name__)


@display_finishes_bp.route("/api/display/finishes")
def list_finishes():
    """Every registered finish's spec (name, input tap contract, extracted
    params, traced WGSL). `?refresh=1` re-scans `~/AstroFlow/finishes/` —
    the author edit loop."""
    if request.args.get("refresh"):
        refresh_finishes()
    return jsonify(finish_specs())
