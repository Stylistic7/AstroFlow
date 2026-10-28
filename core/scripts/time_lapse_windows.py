"""Drive a mosaic's time-lapse (`pipelines/time_lapse.py`) through windows of nights.

    uv run python -m scripts.time_lapse_windows <mosaic_id> [--nights 15] [--no-sweep]

With the app UP and the mosaic in the cart under the time-lapse pipeline: set
the `time_lapse_nights` decision to the next window of nights (the app drains
and restarts its pass), wait until the pass is idle and every night of the
window has its PNG in the TimeLapse folder, sweep the orphans the window no
longer references (`scripts.sweep_orphans --delete`, which is what bounds the
disk to one window's intermediates), and go on to the next window. A window
whose frames do not all appear is reported and left behind — the boot log says
why — and the run goes on; re-running the driver later retries only the missing.

Side effects live here, in a CLI, the way the sweep's do: the pipeline itself
stays a pure function of the config, and a trace cannot watch a folder or
rewrite settings mid-pass.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API = "http://localhost:8000"
PIPELINE = "Siril + GraXpert + OSS Time-lapse"


def _get(path: str):
    with urllib.request.urlopen(API + path) as r:
        return json.load(r)


def _post(path: str, body: dict | None = None) -> None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req).read()


def nights_of(mosaic_id: str) -> dict:
    """`{night: runs started that night}` for the mosaic, by the pipeline's
    own rule (`pipelines.time_lapse.night_of`: the evening a run started on)."""
    from pipelines.time_lapse import night_of
    mosaics = _get("/api/mosaics")
    mosaics = mosaics if isinstance(mosaics, list) else list(mosaics.values())
    mosaic = next(m for m in mosaics if m["stable_hash"] == mosaic_id)
    targets = _get("/api/targets")
    runs_per_night: dict = {}
    for tid in mosaic["panel_target_ids"]:
        for sid in targets[tid]["stack_ids"]:
            for run in _get(f"/api/stacks/{sid}/detail")["contents"].values():
                started = datetime.datetime.strptime(run["oldest_date"], "%a, %d %b %Y %H:%M:%S %Z")
                n = night_of(started)
                runs_per_night[n] = runs_per_night.get(n, 0) + 1
    return dict(sorted(runs_per_night.items()))


def windows(runs_per_night: dict, max_nights: int, max_runs: int) -> list:
    """Consecutive windows of nights, each at most `max_nights` long and
    adding at most `max_runs` runs (one night always fits): a run added is
    about a gigabyte of native products until the sweep, so runs are what
    bound the disk, not nights."""
    out, current, runs = [], [], 0
    for night, n in runs_per_night.items():
        if current and (len(current) >= max_nights or runs + n > max_runs):
            out.append(current); current, runs = [], 0
        current.append(night); runs += n
    if current:
        out.append(current)
    return out


def current_window(mosaic_id: str) -> str | None:
    for o in _get("/api/decision_overrides")["overrides"]:
        k = o["key"]
        if k["decision_type"] == "time_lapse_nights" and k["subject_id"] == mosaic_id:
            return o["choice"]
    return None


def set_window(mosaic_id: str, first: datetime.date, last: datetime.date) -> None:
    """Sets the window decision — unless it already says so: every set
    drains and restarts the app's pass, and a driver restart must not."""
    if current_window(mosaic_id) == f"{first}..{last}":
        return
    _post("/api/decisions/override", {
        "key": {"decision_type": "time_lapse_nights", "object_type": "mosaic",
                "parent_stack_id": mosaic_id, "subject_id": mosaic_id},
        "choice": f"{first}..{last}",
    })


def pass_idle() -> bool:
    try:
        totals = (_get("/api/compute_graph/status") or {}).get("_totals") or {}
    except Exception:            # the app is restarting: not idle, try again later
        return False
    return totals.get("remaining_number", 1) == 0


def wait_for(window: list[datetime.date], folder: Path, poll_s: int = 60) -> bool:
    """True once the pass is idle and every night's PNG is there; False when
    the pass went idle without them."""
    idle_polls = 0
    while True:
        have = [n for n in window if (folder / f"{n}.png").exists()]
        if len(have) == len(window):
            return True
        idle_polls = idle_polls + 1 if pass_idle() else 0
        if idle_polls >= 3:      # idle for three polls with frames missing: it failed
            print(f"  idle, but {len(window) - len(have)} frames missing: "
                  f"{[str(n) for n in window if n not in have]}")
            return False
        print(f"  {time.strftime('%H:%M')}  {len(have)}/{len(window)} frames", flush=True)
        time.sleep(poll_s)


def sweep() -> None:
    subprocess.run([sys.executable, "-m", "scripts.sweep_orphans", "--delete"], check=True,
                   cwd="/Users/nobody/Projects/AstroFlow/core")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mosaic_id")
    parser.add_argument("--nights", type=int, default=15, help="at most this many nights per window")
    parser.add_argument("--runs", type=int, default=70, help="at most this many runs added per window")
    parser.add_argument("--no-sweep", action="store_true", help="never sweep between windows")
    parser.add_argument("--folder", default="/Volumes/AstroProcessed/TimeLapse",
                        help="where the pipeline keeps the frames (beside the products root)")
    args = parser.parse_args()

    cart = json.dumps(_get("/api/shopping_cart"))
    assert args.mosaic_id in cart and PIPELINE in cart, "put the mosaic in the cart under the time-lapse pipeline first"
    runs_per_night = nights_of(args.mosaic_id)
    nights = list(runs_per_night)
    folder = Path(args.folder) / args.mosaic_id
    plan = windows(runs_per_night, args.nights, args.runs)
    print(f"{len(nights)} nights, {nights[0]} .. {nights[-1]}, {len(plan)} windows; frames in {folder}")

    for window in plan:
        pending = [n for n in window if not (folder / f"{n}.png").exists()]
        if not pending:
            print(f"window {window[0]}..{window[-1]}: already built")
            continue
        print(f"window {window[0]}..{window[-1]} ({len(window)} nights): building", flush=True)
        set_window(args.mosaic_id, window[0], window[-1])
        time.sleep(120)              # let the drain + restart begin before judging idleness
        if not wait_for(window, folder):
            print(f"window {window[0]}..{window[-1]}: frames missing (see the boot log); going on", flush=True)
        else:
            print(f"window {window[0]}..{window[-1]}: done", flush=True)
        if not args.no_sweep:
            print("sweeping orphans", flush=True)
            sweep()
    print("all windows built; the video is", folder / "time_lapse.mp4")


if __name__ == "__main__":
    main()
