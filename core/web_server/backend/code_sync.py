"""Head-only endpoints for GitOps-style pull-deploy.

The head freezes the cluster's code into a checkout dir (committed HEAD or, in
uncommitted mode, the working tree) and runs its tasks from it. These endpoints
expose that exact frozen snapshot: a worker reads `/api/cluster-version` (also
broadcast via zeroconf) and, when it differs from what the worker is running,
GETs `/api/code-snapshot` and runs it. Serving reads only the frozen checkout —
no git at request time — so head and workers are guaranteed byte-identical.
Runs only on the head (workers don't start gunicorn).
"""
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

from base_path import checkout_dir
from state.code_version import read_synced_sha, tar_directory

code_sync_bp = Blueprint("code_sync", __name__)


def _frozen_version() -> str:
    return read_synced_sha(checkout_dir()) or ""


@code_sync_bp.route("/api/cluster-version")
def cluster_version():
    return jsonify({"cluster_version": _frozen_version()})


@code_sync_bp.route("/api/code-snapshot")
def code_snapshot():
    frozen = _frozen_version()
    if not frozen:
        return jsonify({"error": "head has not frozen a snapshot yet"}), 503
    # The worker passes the version it saw advertised; serve only that exact
    # snapshot so a head that has since redeployed can't hand back a mismatch.
    requested = request.args.get("sha")
    if requested is not None and requested != frozen:
        return jsonify({"error": f"head is at {frozen}, not {requested}"}), 404
    tar = tar_directory(Path(checkout_dir()))
    return Response(
        tar,
        mimetype="application/gzip",
        headers={
            "X-AstroFlow-SHA": frozen,
            "Content-Disposition": f'attachment; filename="astroflow-{frozen}.tar.gz"',
        },
    )
