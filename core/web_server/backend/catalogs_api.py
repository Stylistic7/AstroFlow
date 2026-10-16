"""The Catalogs page's API: the reference data AstroFlow reads.

  * astrometry.net index files: a folder the user chooses, with a download
    (`catalogs/astrometry_index.py`). Plate solving can't run without them,
    so nothing is built until the folder is set (`state/actor.py`).
  * the MARS database, for the PixInsight pipelines' gradient correction: a
    file that comes with PixInsight's own downloads.
  * the Gaia XP catalogue: ships inside AstroFlow.app; shown, not chosen.

The choices live in the durable store and reach every Mac through
`node_settings`.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from flask import Blueprint, abort, jsonify, request

import node_settings
from catalogs import astrometry_index
from context import local_context as context
from photometry.gaia_catalog import CATALOG_PATH

catalogs_bp = Blueprint("catalogs", __name__)

CORE = Path(__file__).resolve().parents[2]


def _gaia() -> dict:
    try:
        sources = int(np.load(CATALOG_PATH / "dec.npy", mmap_mode="r").shape[0])
    except OSError:
        sources = None
    return {"path": str(CATALOG_PATH), "sources": sources}


def _current() -> dict:
    index, mars = context.astrometry_index_dir, context.mars_database
    return {
        "astrometry_index_dir": str(index) if index else None,
        "mars_database": str(mars) if mars else None,
        "gaia": _gaia(),
    }


@catalogs_bp.route("/api/catalogs", methods=["GET"])
def get_catalogs():
    return jsonify(_current())


@catalogs_bp.route("/api/catalogs", methods=["PUT"])
def put_catalogs():
    from state.actor import state

    body = request.get_json(force=True)
    index = Path(str(body.get("astrometry_index_dir") or "").strip()).expanduser()
    if not index.is_absolute():
        abort(400, description="The index folder must be a full path, like /Volumes/Astro/astrometry-net")
    if not index.exists() and index.parent.is_dir():
        index.mkdir()
    if not index.is_dir():
        abort(400, description=f"Index folder {index} is not a folder this Mac can see")
    mars = str(body.get("mars_database") or "").strip()
    if mars and not Path(mars).is_file():
        abort(400, description=f"MARS database {mars} is not a file this Mac can see")

    # Every Mac first: if one can't take the settings, nothing is saved.
    node_settings.push_to_cluster({**node_settings.current(), "astrometry_index_dir": str(index),
                                   "mars_database": mars or None})
    first = not context.astrometry_index_dir
    context.astrometry_index_dir = index
    context.mars_database = Path(mars) if mars else None
    if first:
        state.drain_and_restart()
    return jsonify(_current())


@catalogs_bp.route("/api/catalogs/index_status", methods=["GET"])
def index_status():
    index = context.astrometry_index_dir
    return jsonify(astrometry_index.status(Path(index)) if index else None)


@catalogs_bp.route("/api/catalogs/download_index", methods=["POST"])
def download_index():
    """Downloads the missing index files into the saved index folder, in the
    background; `index_status` reports progress."""
    index = context.astrometry_index_dir
    if not index:
        abort(400, description="Save an index folder first")
    if not astrometry_index.status(Path(index))["downloading"]:
        subprocess.Popen([sys.executable, "-m", "catalogs.astrometry_index", str(index)], cwd=CORE,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return jsonify({**astrometry_index.status(Path(index)), "downloading": True})


@catalogs_bp.route("/api/node_settings", methods=["GET"])
def get_node_settings():
    """What every Mac keeps a copy of (`node_settings`); a joining worker
    reads it here."""
    return jsonify(node_settings.current())
