"""Plan the next night for every rig from the shopping cart. Read-only:
prints the plans, or writes the planner page's JSON; pushes nothing.

    PYTHONPATH=. python scripts/plan_night.py              # text, the upcoming night
    PYTHONPATH=. python scripts/plan_night.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="write the explanation JSON here instead of printing text")
    ap.add_argument("--inputs", default=None, help="the inputs key to stamp on the plan (planner.refresh)")
    ap.add_argument("--v2", action="store_true", help="also print the v2 (month-aware, weather-aware) plan")
    a = ap.parse_args()

    import ray

    def _refuse(*_a, **_k):
        raise RuntimeError("the planner reads the head's cache directly and never joins Ray")
    ray.init = ray.get_actor = _refuse
    from cache.context_ref import use_local_context_for_refs
    use_local_context_for_refs()
    from context import local_context as context
    from planner.explain import next_night

    plan = next_night(context)
    if a.json:
        if a.inputs:
            plan["inputs"] = json.loads(a.inputs)
        tmp = a.json.with_suffix(".partial")
        tmp.write_text(json.dumps(plan))
        tmp.replace(a.json)
        from planner.v2 import write_shadow
        write_shadow(context, a.json.with_name("planner_next_v2.json"), inputs=plan.get("inputs"))
        return 0
    from datetime import datetime
    t = lambda i: datetime.fromisoformat(plan["times"][i]).astimezone().strftime("%H:%M")
    for rig in plan["rigs"]:
        flip = t(rig["flip"]) if rig["flip"] is not None else "none"
        print(f"{rig['name']} · {rig['focal']:.0f} mm   flip: {flip}")
        for v in rig["visits"]:
            print(f"  {t(v['start'])}–{t(v['end'])}  {v['label']:<24} {v['filter']}  {v['subs']:3d} × {v['exposure']:.0f} s   [{v['why']}]")
    if plan["unassignable"]:
        print("in the cart but no rig shoots it:", ", ".join(plan["unassignable"]))
    if a.v2:
        from planner.v2 import plan as plan_v2, text as text_v2
        print()
        print(text_v2(plan_v2(context)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
