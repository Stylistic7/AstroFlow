"""Real `@node` ops + `@graph_builder` used by `test_subgraph_preflight_skip`.

These live in a normal (non-test) module — exactly like production pipeline
ops — so that Ray/cloudpickle ships them the way it ships real ops and the
op body's `_ss_source.filename_fn(...)` self-reference (the same idiom
`bxt_correct_only` uses) resolves to the decorated `@node` wrapper on the
worker.

NOTE: deliberately NO `from __future__ import annotations`. `@node`'s
artifact-producer detection (`_looks_like_file_return`) reads the runtime
return annotation and matches the real `FitsFilePath` TYPE; under PEP 563 the
annotation would be the bare string "FitsFilePath", detection would fail, and
no `filename_fn` / CONTHASH stamping would be wired up. Production producer
modules avoid the future-import for the same reason.

Shape (mirrors a real pipeline):

    source ─▶ intermediate (ephemeral) ─▶ final (final=True, on disk)
                                             └─▶ pipeline-done sentinel
"""
import os
from pathlib import Path

import numpy as np
import ray
from astropy.io import fits as pyfits

from cache.ephemeral_scope import ephemeral_intermediates
from context import context
from decorators import node
from dependency_graph.classes import Graph
from dependency_graph.ephemeral_registry import EphemeralFiles
from dependency_graph.tracing import graph_builder, record, trace
from file_io.types import FitsFilePath


# The ephemeral op runs in a Ray WORKER process, so a module-level int counter
# in the driver would never see its increments. Executions of the ephemeral
# body are counted via an append-only log under the shared output dir instead
# — cross-process-visible, and its path is derived from `context.output_dir`
# (NOT a call arg) so it never perturbs any op's call_hash.
EXEC_LOG_NAME = "_ss_intermediate_execs.log"


def _write_small_fits(path, seed):
    """Write a tiny, non-degenerate float32 FITS. float (not uint16) keeps
    `compute_image_metadata` off the defect-extraction path."""
    data = (np.arange(32 * 32, dtype=np.float32).reshape(32, 32) + float(seed)) / 1000.0
    hdu = pyfits.PrimaryHDU(data=data)
    hdu.header["EXPOSURE"] = 1.0
    hdu.writeto(str(path), overwrite=True)


def _note_ephemeral_execution():
    log = Path(context.output_dir) / EXEC_LOG_NAME
    with open(log, "a") as f:
        f.write("run\n")


# ──────────────────────────────────────────────────────────────────────────
# Real @node artifact producers. Each writes a real FITS at its own
# `filename_fn` path, gets a CONTHASH header stamped by `makes_image_wrap`,
# and returns `ImageMetadata` (via the composed `compute_image_metadata`).
# plate_solve=False keeps them off the solver.
# ──────────────────────────────────────────────────────────────────────────

@node(plate_solve=False)
def _ss_source(seed: int) -> FitsFilePath:
    out = _ss_source.filename_fn(seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, seed)
    return FitsFilePath(out)


@node(plate_solve=False)
def _ss_intermediate(config: str, source) -> FitsFilePath:
    # The EPHEMERAL intermediate. Every real execution of THIS body is what
    # the test counts — it must run exactly once across both runs.
    _note_ephemeral_execution()
    out = _ss_intermediate.filename_fn(config, source)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, len(source.content_hash))
    return FitsFilePath(out)


@node(final=True, display="Deliverable", plate_solve=False)
def _ss_final(config: str, intermediate) -> FitsFilePath:
    out = _ss_final.filename_fn(config, intermediate)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_small_fits(out, len(intermediate.content_hash))
    return FitsFilePath(out)


@ray.remote(num_cpus=0.1)
def _ss_pipeline_done() -> None:
    """No-output terminal sentinel — the `run_pipeline_done` analogue. Funnels
    the real deliverable(s) into a single subgraph terminal so the
    `GraphRuntime` single-terminal invariant holds. Produces no file."""
    pass


@graph_builder
def build_subgraph(seed, config):
    """Traced subgraph builder: source → ephemeral intermediate → final
    deliverable, funneled into the no-output sentinel."""
    source = _ss_source(seed)
    with ephemeral_intermediates():
        intermediate = _ss_intermediate(config, source)
    final = _ss_final(config, intermediate)
    # Funnel the deliverable into the sentinel (ordering-only dep), exactly
    # like `graph[run_pipeline_done] = terminals` in production.
    record(_ss_pipeline_done, _after=(final,))
    return final


def build_top_level_graph(seed, config):
    """Top-level graph wired the way production's `build_main_processing_graph`
    is: a subgraph-returning builder node PLUS a pre-built `ephemeral_files`
    registry from an EAGER trace of that same subgraph.

    Mirrors `_eager_trace_pipeline` (`trace(...)` the pipeline on the head) +
    `_build_ephemeral_registry` (`register_from_graph` each traced graph). The
    root `GraphRuntime` merges this registry into the shared one so every
    ephemeral producer's consumer set is known BEFORE any child runtime
    dispatches — the invariant that lets in-run GC reclaim an intermediate only
    after its last consumer, without a cross-pipeline FileNotFound race."""
    traced = trace(build_subgraph._function, seed, config)
    registry = EphemeralFiles()
    if traced is not None:
        registry.register_from_graph(traced)
    return Graph({(build_subgraph, seed, config): []}, ephemeral_files=registry)
