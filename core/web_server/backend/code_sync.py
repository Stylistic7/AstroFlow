"""Head-only: the cluster's code version over HTTP.

Workers read `/api/cluster-version` and FENCE on it — they refuse to join
unless their own git checkout computes the same version
(`state_machine._join_head`). Code itself moves via git (`git pull` on the
worker), never through the app; the old `/api/code-snapshot` pull-deploy
endpoint is gone with the rest of the snapshot machinery (see
`state.code_version`).

Computed once per process: gunicorn workers boot with the app, so this is
the version the head booted with — deploying is restarting, and a restart
recomputes it.
"""
import os
from pathlib import Path

from flask import Blueprint, jsonify

from base_path import get_base_path
from state.code_version import head_cluster_version

code_sync_bp = Blueprint("code_sync", __name__)

_version_cache: "str | None" = None


def _uncommitted_mode() -> bool:
    """Same env-only rule as `state_machine.is_uncommitted_mode` (per-launch,
    never persisted — a sticky dev mode is what defeated the start gate)."""
    return os.environ.get("ASTROFLOW_UNCOMMITTED", "").lower() in (
        "1", "true", "yes")


def _version() -> str:
    global _version_cache
    if _version_cache is None:
        try:
            _version_cache = head_cluster_version(
                Path(get_base_path()), _uncommitted_mode())
        except Exception:
            _version_cache = ""
    return _version_cache


@code_sync_bp.route("/api/cluster-version")
def cluster_version():
    return jsonify({"cluster_version": _version()})
