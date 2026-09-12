"""Drain-time completeness guard — does the disk hold what the cart promises?

`run_graph` returning cleanly means "every node the PLAN contained concluded".
It does NOT mean "every product the TRACE promises exists": if the plan was
built missing a subtree (2026-08-25 boot 7 dropped the whole WBPP-Cygnus-SHO
plan and still drained green), the run completes and nobody notices until a
viewer 404s days later. The trace and the plan are supposed to be two
projections of the same intent — this guard checks that they agree where it
matters, at the moment the graph claims it is done.

Scope: TAP entries (`display` set) and finals only. Working values
(`@node(ephemeral=True)` crops/feathers) are deliberately reclaimed in-run
and are never taps, so checking taps is exactly the "what the user can see"
contract without re-deriving ephemerality. Over-reporting is the failure
mode to avoid — a noisy guard gets ignored — so the report is one compact
`### completeness` block, and the guard NEVER raises into the run loop.
"""

from __future__ import annotations

import os


def missing_products(ctx) -> dict[tuple, list[dict]]:
    """`{cart_item: [trace entries whose path is absent from disk]}`.

    Cart items come from the same declaration the sweep uses
    (`orphan_sweep._cart_items`: shopping cart + implicit per-mosaic
    pipelines). An unresolvable trace is itself reported as missing — the
    trace failing AT DRAIN means the just-finished run cannot have built
    what the item needs.
    """
    from cache.orphan_sweep import _cart_items
    from dependency_graph.trace_paths import TraceResolutionError, traced_paths_for

    missing: dict[tuple, list[dict]] = {}
    for item in _cart_items(ctx):
        try:
            entries = traced_paths_for(item, recompute=True, strict=True)
        except TraceResolutionError as e:
            missing[item] = [{"path": f"<unresolvable: {e}>", "op_name": "<trace>"}]
            continue
        gone = [
            e for e in entries
            if (e.get("display") is not None or e.get("is_final"))
            and e.get("path")
            and not os.path.exists(e["path"])
        ]
        if gone:
            missing[item] = gone
    return missing


def log_completeness(ctx) -> None:
    """One loud `### completeness` block after every clean drain."""
    from cache.orphan_sweep import _cart_items

    try:
        n_items = len(_cart_items(ctx))
        missing = missing_products(ctx)
    except Exception as e:  # the guard must never kill the run loop
        print(f"### completeness: guard failed to run: {e!r}")
        return
    if not missing:
        print(f"### completeness: all {n_items} cart items fully on disk")
        return
    total = sum(len(v) for v in missing.values())
    print(
        f"### completeness: {len(missing)}/{n_items} cart items INCOMPLETE "
        f"({total} promised products absent) — the plan and the trace disagree"
    )
    for item, gone in sorted(missing.items()):
        print(f"### completeness:   {item}: {len(gone)} missing")
        for e in gone[:6]:
            print(f"### completeness:     {e.get('op_name', '?')}  {e.get('path')}")
        if len(gone) > 6:
            print(f"### completeness:     ... and {len(gone) - 6} more")
