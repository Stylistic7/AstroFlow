"""Plan tonight for every rig from the shopping cart. Read-only: prints the
plans, pushes nothing.

    PYTHONPATH=. python scripts/plan_night.py [--date 2026-10-08] [--utc-offset -7]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="the evening's local date")
    ap.add_argument("--utc-offset", type=float, default=-7.0)
    a = ap.parse_args()

    from cache.context_ref import use_local_context_for_refs
    use_local_context_for_refs()
    from context import local_context as context
    from planner import inventory, report, schedule, sky

    lon, lat = inventory.site(context)
    noon_utc = datetime.fromisoformat(a.date).replace(hour=12, tzinfo=timezone.utc) - timedelta(hours=a.utc_offset)
    night = sky.tonight(lon, lat, noon_utc)
    rigs = inventory.rigs(context)
    cands, unassignable = inventory.candidates(context, rigs)
    from planner import efficiency
    rarity = efficiency.scarcity(efficiency.availability(lon, lat, noon_utc))
    print("filter scarcity this lunation:", {f: round(v, 2) for f, v in sorted(rarity.items())}, end="\n\n")
    for rig in rigs:
        inventory.set_urgency(cands.get(rig.key, []), lon, lat, noon_utc)
        plan = schedule.plan_rig(night, rig, cands.get(rig.key, []), rarity)
        print(report.render(night, plan, a.utc_offset), end="\n\n")
    if unassignable:
        print("in the cart but no rig shoots it:", ", ".join(unassignable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
