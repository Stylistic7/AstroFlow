"""Keep the stored next-night plan current.

A plan is CURRENT when it was made for the upcoming night and from the
inputs that exist now (the cart, and the newest master on disk). Callers:
the hourly scheduler job (so the page almost never waits) and the page's
endpoint. Planning runs as a subprocess (`scripts/plan_night.py --json`):
~a minute of numpy that must not block a web worker or the scheduler's
event loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1]
_proc: subprocess.Popen | None = None


def plan_path() -> Path:
    from base_path import app_support_dir
    return app_support_dir() / "planner_next.json"


def plan_path_v2() -> Path:
    return plan_path().with_name("planner_next_v2.json")


def upcoming_night_start(context, now: datetime | None = None) -> str:
    """ISO start of the dark window a plan made now would be for."""
    from planner.explain import _upcoming
    from planner.inventory import site
    lon, lat = site(context)
    night, _ = _upcoming(lon, lat, now or datetime.now(timezone.utc))
    return night.times[0].to_datetime(timezone=timezone.utc).replace(microsecond=0).isoformat()


def inputs_key(context, now: datetime | None = None) -> list:
    """What a plan was made from. The 6-hour bucket is for v2's weather: a
    forecast a day old is stale, so the shadow plan is remade four times a
    day even when nothing else moved."""
    masters = Path(context.output_dir, "stack_files")
    newest = max((p.stat().st_mtime for p in masters.glob("*.xisf.json")), default=0.0)
    now = now or datetime.now(timezone.utc)
    return [repr(sorted(map(tuple, context.shopping_cart or []))), round(newest),
            repr(sorted((context.get("planner_overrides") or {}).items())),
            repr(sorted((str(k), str(v.choice)) for k, v in (context.settings.decision_overrides or {}).items()
                        if k.decision_type.value == "project_kind")),
            f"{now:%Y-%m-%d}/{now.hour // 6}"]


def stored(path: Path | None = None) -> dict | None:
    p = path or plan_path()
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def computing() -> bool:
    return _proc is not None and _proc.poll() is None


def refresh_if_stale(context) -> str:
    """'current', 'refreshing' (for tonight, older inputs) or 'stale' (for
    another night, or none). Starts a recompute unless current or running."""
    global _proc
    plan, night, key = stored(), upcoming_night_start(context), inputs_key(context)
    if plan and plan.get("night_start") == night and plan.get("inputs") == key:
        return "current"
    if not computing():
        _proc = subprocess.Popen(
            [sys.executable, "scripts/plan_night.py", "--json", str(plan_path()), "--inputs", json.dumps(key)],
            cwd=_CORE, env={**os.environ, "PYTHONPATH": str(_CORE)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "refreshing" if plan and plan.get("night_start") == night else "stale"
