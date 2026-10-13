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
    a = ap.parse_args()

    import ray

    def _refuse(*_a, **_k):
        raise RuntimeError("the planner reads the head's cache directly and never joins Ray")
    ray.init = ray.get_actor = _refuse
    from cache.context_ref import use_local_context_for_refs
    use_local_context_for_refs()
    from context import local_context as context
    from planner.v2 import plan as plan_night, text

    plan = plan_night(context)
    if a.json:
        if a.inputs:
            plan["inputs"] = json.loads(a.inputs)
        tmp = a.json.with_suffix(".partial")
        tmp.write_text(json.dumps(plan))
        tmp.replace(a.json)
        return 0
    print(text(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
