"""Serve the display-referred finish registry to the frontend.

This is the F1↔F2 seam made concrete for the UI: the finish picker, the grid
axes, the sliders, and the real-time shader are ALL generated from these specs,
not hand-coded on the frontend. Adding a look = writing a Python DisplayFinish in
display/finishes.py; it appears here automatically with its extracted params and
assembled WGSL. See the project_scene_display_seam memory.
"""
from flask import Blueprint, jsonify

from display.finishes import FINISHES

display_finishes_bp = Blueprint("display_finishes", __name__)


@display_finishes_bp.route("/api/display/finishes")
def list_finishes():
    """Every registered finish's spec (name, input tap contract, extracted
    params, assembled WGSL) — the single source the frontend reads to build the
    picker + grid + sliders + shader. Finishes RUN entirely in the frontend
    (WGSL shader) and compute their own starting values there from the generic
    per-channel image stats in tile-metadata; the backend ships scene + stats,
    never finish logic."""
    return jsonify([f.spec() for f in FINISHES.values()])
