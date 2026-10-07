"""The compute graph as the head publishes it: per-op progress, one node's
recorded outcome, and what is running where."""
from __future__ import annotations

import time

from mcp_server.server import NotAvailableYet, context, to_json, tool


@tool()
def graph_status() -> dict:
    """The compute graph right now: per op, done / running / ready / waiting /
    total / remaining (the head's 5-second status tick), sorted by what is
    running, plus totals and when the last status event was seen.
    """
    from mcp_server import events
    progress = context.get("compute_graph_status")
    if not progress:
        raise NotAvailableYet(
            "compute_graph_status has not been published yet this boot (the "
            "StateActor writes it every 5 s once a graph is running)")
    ops = [{"op": k, **v} for k, v in progress.items() if not k.startswith("_")]
    ops.sort(key=lambda o: (-o.get("running_number", 0), -o.get("remaining_number", 0)))
    return {"totals": progress.get("_totals"), "ops": ops,
            "last_status_event_at": events.last_seen("compute_graph_status")}


@tool()
def node_status(node_id: str, max_logs: int = 50) -> dict:
    """What one graph node recorded when it finished: result type, output
    path, exception and stack trace if it failed, its last log entries — and
    its placement if it is running now. Node ids are `<op>_<hash>`, as in
    running_tasks and the workflow-graph UI.
    """
    nid = node_id.partition("@")[0]
    status = context.get(("node_status", nid))
    running = (context.get("node_running") or {}).get(nid)
    if status is None and running is None:
        raise ValueError(f"no status recorded for node {nid!r}: it has not "
                         "run this boot, or its record was evicted")
    out = dict(status or {})
    logs = out.pop("logs", None) or []
    out["logs"] = logs[-max(0, max_logs):]
    out["n_logs"] = len(logs)
    out["outcome"] = context.get(("node_outcome", nid))
    out["running"] = running
    return to_json({"node_id": nid, **out})


@tool()
def running_tasks() -> dict:
    """Every task in flight: node id, op, the host it was placed on (or null
    for an unpinned light task), when it started and how long it has run.
    From the head's status tick, so at most ~5 s stale.
    """
    running = context.get("node_running")
    if running is None:
        raise NotAvailableYet("node_running has not been published yet this boot")
    now = time.time()
    rows = [{"node_id": nid, **r, "elapsed_s": round(now - r["started_at"], 1)}
            for nid, r in running.items()]
    rows.sort(key=lambda r: r["started_at"])
    return {"count": len(rows), "tasks": rows}
