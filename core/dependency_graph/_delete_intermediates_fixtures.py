"""Real `@node` ops + a subgraph builder for `test_delete_intermediates`.

These live in a normal (non-test) module — exactly like production pipeline
ops — so cloudpickle ships them the way it ships real ops and the op body's
`_di_*.filename_fn(...)` self-reference (the same idiom `crop_panel` uses)
resolves to the decorated `@node` wrapper on the worker.

NOTE: deliberately NO `from __future__ import annotations`. `@node`'s
artifact-producer detection (`_looks_like_file_return`) reads the RUNTIME
return annotation and matches the real `FitsFilePath` TYPE; under PEP 563 the
annotation would be the bare string "FitsFilePath", detection would fail, and
no `filename_fn` / CONTHASH stamping would be wired up.

Shape — three ops so the test proves reclaim isn't merely "the immediate
predecessor of the terminal":

    _di_source ─▶ _di_middle ─▶ _di_terminal
    └────────── intermediates ──────────┘ └── escapes ──┘
"""
import os
from pathlib import Path

import numpy as np
import ray
from astropy.io import fits as pyfits

from context import context
from decorators import node
from dependency_graph.classes import Graph
from dependency_graph.tracing import trace
from file_io.types import FitsFilePath


# Each op body appends to a log under the shared output dir: a module-level
# counter would live in the driver and never see a Ray worker's increments.
# The path derives from `context.output_dir`, NOT a call arg, so it can never
# perturb an op's call_hash.
EXEC_LOG_NAME = "_di_execs.log"


def _note_execution(op_name: str) -> None:
    with open(Path(context.output_dir) / EXEC_LOG_NAME, "a") as f:
        f.write(op_name + "\n")


def executions(out_dir: Path, op_name: str) -> int:
    """How many times `op_name`'s BODY actually ran (across processes)."""
    log = Path(out_dir) / EXEC_LOG_NAME
    if not log.exists():
        return 0
    return log.read_text().split().count(op_name)


def _write_small_fits(path, seed) -> None:
    """A tiny, non-degenerate float32 FITS. float (not uint16) keeps
    `compute_image_metadata` off the defect-extraction path."""
    data = (np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
            + float(seed)) / 1000.0
    hdu = pyfits.PrimaryHDU(data=data)
    hdu.header["EXPOSURE"] = 1.0
    hdu.writeto(str(path), overwrite=True)


# Also interior to the scope, and also disposable.
@node(plate_solve=False, ephemeral=True)
def _di_source(seed: int) -> FitsFilePath:
    _note_execution("_di_source")
    out = _di_source.filename_fn(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, seed)
    return FitsFilePath(out)


# The interior of the scope: declares itself disposable, so the runtime
# reclaims it when the subgraph concludes.
@node(plate_solve=False, ephemeral=True)
def _di_middle(config: str, source) -> FitsFilePath:
    _note_execution("_di_middle")
    out = _di_middle.filename_fn(config, source)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, len(source.content_hash))
    return FitsFilePath(out)


@node(plate_solve=False)
def _di_terminal(config: str, middle) -> FitsFilePath:
    _note_execution("_di_terminal")
    out = _di_terminal.filename_fn(config, middle)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, len(middle.content_hash))
    return FitsFilePath(out)


@ray.remote(num_cpus=0.1)
def build_scoped_subgraph(seed, config, dry_run=False):
    """A subgraph whose author declares its intermediates disposable.

    Traces source → middle → terminal, then returns that graph wrapped with
    Its interior op declares `ephemeral=True`: the terminal escapes and persists, and
    `_di_source` / `_di_middle` are deleted once this subgraph concludes.
    """
    del dry_run

    def _body():
        source = _di_source(seed)
        middle = _di_middle(config, source)
        return _di_terminal(config, middle)

    return Graph(trace(_body))


def build_top_level_graph(seed, config) -> Graph:
    """The one-node parent graph whose node expands into the subgraph."""
    return Graph({(build_scoped_subgraph, seed, config): []})
