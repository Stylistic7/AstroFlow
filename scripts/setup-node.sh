#!/usr/bin/env bash
# Bootstrap a Mac for running an AstroFlow node (head or worker).
#
# Every node gets the same Python environment so behaviour is uniform
# across the cluster — no "worker-only" or "head-only" subsets.
#
# Usage:
#   scripts/setup-node.sh            # set up ./.venv from pyproject.toml
#   scripts/setup-node.sh --refresh  # delete ./.venv and rebuild
#
# Requires: macOS with Homebrew. Installs uv (astral's fast pip/venv
# replacement), libvips (needed by large_image), and every Python
# dependency declared in pyproject.toml.

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
# libvips: needed by large_image. pkg-config helps native extensions find it.
# hdf5: needed by h5py.
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

log "Creating Python environment at $VENV_DIR"
# uv venv creates a venv; uv pip install -e . reads pyproject.toml and
# installs everything in that one dependencies list.
uv venv "$VENV_DIR" --python 3.13

log "Installing AstroFlow and all its dependencies (editable)"
# HDF5_DIR helps h5py's wheel build on Apple Silicon find the homebrew libs.
export HDF5_DIR="$(brew --prefix hdf5)"
uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_ROOT"

log "Smoke test: import the packages most likely to fail on a fresh Mac"
"$VENV_DIR/bin/python" -c "
import large_image, h5py, astropy, astroalign, astroscrappy, photutils
import cv2, ray, diskcache
print('✓ all critical imports OK')
"

cat <<EOF

────────────────────────────────────────────────────────
Done. Activate with:
    source $VENV_DIR/bin/activate

Then join the cluster (replace HEAD_IP with the head node's LAN IP):
    cd $REPO_ROOT/core
    python -c "from state.ray_control import start_ray_subordinate_node; \\
               start_ray_subordinate_node('HEAD_IP')"
────────────────────────────────────────────────────────
EOF
