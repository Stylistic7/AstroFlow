#!/usr/bin/env bash
# Bootstrap a Mac for running an AstroFlow node (head or worker).
#
# Every node gets the BYTE-IDENTICAL Python environment, built from the
# committed uv.lock — no node re-resolves dependencies on its own, so
# the cluster can't drift (a node on a different numpy / spherical-geometry
# pickles objects the other nodes can't unpickle).
#
# Usage:
#   scripts/setup-node.sh            # set up ./.venv from uv.lock
#   scripts/setup-node.sh --refresh  # delete ./.venv and rebuild
#
# After the first run, `cd core && make install` keeps the venv in sync
# with the lock (same `uv sync --frozen` under the hood).
#
# Requires: macOS with Homebrew.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
VENV_DIR="$REPO_ROOT/.venv"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

if [[ "${1:-}" == "--refresh" ]]; then
    log "Removing existing venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found — install from https://brew.sh first." >&2
    exit 1
fi

log "Installing system libraries via Homebrew"
# vips: native lib behind pyvips (tile pyramids). pkg-config helps native
# extensions find their libs. hdf5: needed by h5py.
for pkg in vips pkg-config hdf5; do
    if ! brew list --formula "$pkg" >/dev/null 2>&1; then
        brew install "$pkg"
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv (Astral's Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin — make it available for the rest of this script.
    export PATH="$HOME/.local/bin:$PATH"
fi

log "Building $VENV_DIR from uv.lock (frozen — no re-resolution)"
# `uv sync --frozen` creates .venv using the pinned interpreter
# (.python-version = 3.13), then installs EXACTLY what uv.lock specifies.
# It errors if pyproject.toml and uv.lock disagree, which is the guard
# that keeps every node identical. HDF5_DIR helps h5py if it ever builds
# from source on Apple Silicon.
export HDF5_DIR="$(brew --prefix hdf5)"
uv sync --frozen

log "Smoke test: import the packages most likely to fail on a fresh Mac"
"$VENV_DIR/bin/python" -c "
import pyvips, h5py, astropy, astroalign, astroscrappy, photutils
import cv2, ray, diskcache, spherical_geometry, numpy
print('✓ all critical imports OK')
print('  numpy', numpy.__version__, '| spherical-geometry', spherical_geometry.__version__)
"

cat <<EOF

────────────────────────────────────────────────────────
Done. Every node built this way shares one environment.

Activate it (fish):
    source $VENV_DIR/bin/activate.fish
(bash/zsh: source $VENV_DIR/bin/activate)

Or skip activation and prefix commands with: uv run

Then join the cluster (replace HEAD_IP with the head node's LAN IP):
    cd $REPO_ROOT/core
    python -c "from state.ray_control import start_ray_subordinate_node; \\
               start_ray_subordinate_node('HEAD_IP')"
────────────────────────────────────────────────────────
EOF
