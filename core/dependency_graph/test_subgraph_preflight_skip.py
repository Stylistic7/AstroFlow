"""Integration test: preflight must prune a fully-satisfied dynamic subgraph.

This reproduces the real production churn where an ephemeral intermediate
inside a `@graph_builder`-produced subgraph got re-dispatched (and
recomputed) on every later run — because `run_graph` preflighted only the
TOP-LEVEL graph, never the child `GraphRuntime` a subgraph-returning node
spawns.

The subgraph shape (mirrors a real pipeline) is built by real `@node`
artifact producers + a real `@graph_builder` in
`dependency_graph/_preflight_skip_fixtures.py`:

    source ─▶ intermediate (ephemeral) ─▶ final (final=True, on disk)
                                             └─▶ pipeline-done sentinel

This runs the TRUE production path end-to-end:

  * in-run GC is ON (`run_graph` wires `cleanup_ephemeral_artifact`), and
  * the top-level graph carries a pre-built `ephemeral_files` registry from an
    eager trace of the subgraph (exactly like `build_main_processing_graph`),

so RUN 1 AUTO-reclaims the ephemeral intermediate once its consumer (`final`)
completes — no hand-deletion anywhere. RUN 2 runs the identical graph: the
`final=True` deliverable is already on disk with a matching CONTHASH, so
preflight of the child subgraph must prune the WHOLE subtree and the ephemeral
op's body must NOT execute again.

On unfixed code the child subgraph is never preflighted, the auto-reclaimed
intermediate cache-MISSES (its output file is gone AND its cache entry was
invalidated by GC), and its body re-runs.
"""
from pathlib import Path

from context import context
from dependency_graph import run_graph
from dependency_graph._preflight_skip_fixtures import (
    EXEC_LOG_NAME,
    build_top_level_graph,
)


def _count_ephemeral_executions(out_dir: Path) -> int:
    log = out_dir / EXEC_LOG_NAME
    if not log.exists():
        return 0
    return log.read_text().count("run")


def test_subgraph_preflight_skips_satisfied_ephemeral(test_context) -> None:
    # Production output dirs are real paths. macOS's temp dir is a symlink
    # (/var -> /private/var), and `compute_image_metadata` stores the producer
    # result's `.resolve()`d filename while `expected_output_path` (the
    # registry's key) joins the UNRESOLVED output dir. Resolving here makes the
    # two agree — the same invariant a real output dir already satisfies — so
    # the ephemeral registry's producer/consumer keys match and GC can fire.
    context.output_dir = Path(context.output_dir).resolve()
    out_dir = Path(context.output_dir)
    seed, config = 7, "cfg-v1"

    # ---- RUN 1: build everything; ephemeral auto-reclaimed by GC ------
    run_graph(build_top_level_graph(seed, config))

    runs_after_1 = _count_ephemeral_executions(out_dir)
    assert runs_after_1 == 1, (
        f"expected the ephemeral op to run exactly once on the first pass, "
        f"ran {runs_after_1}x"
    )

    final_files = list((out_dir / "_ss_final").glob("*.fits"))
    assert len(final_files) == 1, (
        f"expected exactly one final deliverable on disk, found {final_files}"
    )

    # The ephemeral intermediate's output was AUTO-reclaimed once `final`
    # (its only consumer) completed — no manual unlink. This exercises the real
    # `ephemeral_intermediates()` cleanup path, not a hand-deletion of it.
    inter_fits = list((out_dir / "_ss_intermediate").glob("*.fits"))
    assert inter_fits == [], (
        f"expected the ephemeral intermediate to be auto-reclaimed by GC after "
        f"run 1, but it is still on disk: {inter_fits}"
    )

    # ---- RUN 2: identical graph, deliverable already satisfied --------
    run_graph(build_top_level_graph(seed, config))

    runs_after_2 = _count_ephemeral_executions(out_dir)
    assert runs_after_2 == runs_after_1, (
        f"the ephemeral op re-ran on the second pass even though the final "
        f"deliverable was already on disk: executions went "
        f"{runs_after_1} -> {runs_after_2}. Preflight must prune the whole "
        f"satisfied subgraph."
    )

    # The deliverable is still correct/on disk.
    final_files_2 = list((out_dir / "_ss_final").glob("*.fits"))
    assert len(final_files_2) == 1, (
        f"the final deliverable went missing on the second pass: {final_files_2}"
    )


# ── single-terminal restoration (the "Subgraph has N terminals" crash) ──────
#
# When a `final=True` master's output is reclaimed (unsatisfied) but its
# downstream consumers + the funnel sentinel are still on disk (satisfied),
# preflight's reverse-walk keeps the master(s) and DROPS the funnel — leaving a
# subgraph with multiple bare terminals that tripped GraphRuntime.__init__'s
# single-terminal assertion. `_restore_single_terminal` re-attaches the funnel.

from dependency_graph.classes import Graph
from dependency_graph.dependency_graph import _restore_single_terminal, _terminals


def test_restore_single_terminal_refunnels_dropped_sentinel():
    # norm: sentinel S ─▶ merge M ─▶ {A1,A2,A3}. One terminal (S).
    norm = Graph({"S": ["M"], "M": ["A1", "A2", "A3"],
                  "A1": [], "A2": [], "A3": []})
    # pruned: preflight dropped S and M (satisfied), kept the 3 unsatisfied
    # masters — three bare terminals, which would trip the assertion.
    pruned = Graph({"A1": [], "A2": [], "A3": []})
    assert len(_terminals(pruned)) == 3

    fixed = _restore_single_terminal(norm, pruned)

    assert _terminals(fixed) == ["S"], "funnel must be restored as the sole terminal"
    assert set(fixed["S"]) == {"A1", "A2", "A3"}, "funnel must join the surviving terminals"


def test_restore_single_terminal_noop_when_already_single():
    norm = Graph({"S": ["A1"], "A1": []})
    pruned = Graph({"S": ["A1"], "A1": []})  # already one terminal
    assert _restore_single_terminal(norm, pruned) is pruned


def test_restore_single_terminal_noop_when_original_multiterminal():
    # If the ORIGINAL subgraph itself had >1 terminal there's no single funnel
    # to restore — leave pruned untouched rather than guess.
    norm = Graph({"A1": [], "A2": []})   # two terminals, no funnel
    pruned = Graph({"A1": [], "A2": []})
    assert _restore_single_terminal(norm, pruned) is pruned


def test_node_completed_preflights_subgraph_with_is_subgraph_true():
    """The subgraph preflight in node_completed MUST pass is_subgraph=True, or
    the single-terminal restoration never runs and the crash returns."""
    import inspect
    from dependency_graph import graph_runtime
    src = inspect.getsource(graph_runtime.GraphRuntime.node_completed)
    assert "_apply_preflight(result, is_subgraph=True)" in src
