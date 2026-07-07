"""Persistent, reboot-surviving memory recorder for the live Ray app.

WHY THIS EXISTS
---------------
On macOS, Ray's own memory monitor does NOT run (`memory_monitor.cc: It is
currently supported only on Linux`), so there is no pre-emptive task kill and
no "top consumers at OOM" dump. And when the machine runs out of RAM it
swap-thrashes into a hard freeze, forcing a reboot that wipes Ray's logs in
`/tmp`. The result: the one event you most need to diagnose leaves no trace.

This script is the missing black-box recorder. It samples system memory + per
Ray-task RSS on an adaptive interval and appends one JSON line per sample to a
file under `~/Pictures/AstroFlow/` (NOT `/tmp` — survives the reboot), calling
`fsync` after every line so the last samples before a freeze are on disk. As
pressure rises it samples faster and dumps a full process list, so the run-up
to the OOM is captured densely.

It never imports the project, never touches the diskcache, never calls
`ray.init` — it only reads `ps`/psutil and writes its own log. So it cannot
perturb the app or poison the cache, and it keeps running across app restarts.

USAGE
-----
    # start recording (leave running for hours; survives app restarts):
    .venv/bin/python -m scripts.memory_watch watch

    # after the next freeze/reboot, see what was eating RAM near the peak:
    .venv/bin/python -m scripts.memory_watch report

Task attribution uses the `ray::<TaskName>` proctitle that shows up in
`ps -o command` (the same thing your `raytasks` alias greps), so each sample
tells you which op's workers were holding memory.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psutil

_GB = 1024 ** 3
_DEFAULT_DIR = Path.home() / "Pictures" / "AstroFlow" / "mem_watch"


def _default_out() -> Path:
    host = socket.gethostname().split(".")[0]
    return _DEFAULT_DIR / f"mem_watch_{host}.jsonl"


# ----------------------------- sampling -----------------------------

def _ps_snapshot() -> list[dict]:
    """One `ps` pass -> [{pid, rss_gb, pcpu, cmd}] for every process.

    `command` (not `comm`) is used because Ray rewrites it via setproctitle to
    `ray::<TaskName>` / `ray::IDLE`, which is how we attribute RSS to ops. rss
    is KiB on macOS `ps`."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,rss=,pcpu=,command="],
        capture_output=True, text=True, check=True,
    ).stdout
    procs = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, rss_kb, pcpu, cmd = parts
        try:
            procs.append({
                "pid": int(pid),
                "rss_gb": int(rss_kb) / (1024 * 1024),
                "pcpu": float(pcpu),
                "cmd": cmd,
            })
        except ValueError:
            continue
    return procs


def _ray_task(cmd: str) -> str | None:
    """Extract the op name from a `ray::<name>` proctitle, else None."""
    if not cmd.startswith("ray::"):
        return None
    tok = cmd[len("ray::"):]
    # strip trailing `()`, pid suffixes, args: keep up to first non-name char
    name = []
    for ch in tok:
        if ch.isalnum() or ch == "_":
            name.append(ch)
        else:
            break
    return "".join(name) or "IDLE"


def _sample(top_n: int, full: bool) -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    procs = _ps_snapshot()
    procs.sort(key=lambda p: p["rss_gb"], reverse=True)

    # Per-Ray-task RSS: sum + worker count across reused workers. A worker
    # holds its high-water RSS even when IDLE, so this catches the ratchet
    # that Ray's scheduler (which only knows the *reservation*) is blind to.
    by_task_rss: dict[str, float] = defaultdict(float)
    by_task_n: dict[str, int] = defaultdict(int)
    ray_total = 0.0
    for p in procs:
        t = _ray_task(p["cmd"])
        if t is None:
            continue
        by_task_rss[t] += p["rss_gb"]
        by_task_n[t] += 1
        ray_total += p["rss_gb"]

    rec = {
        "iso": datetime.now().isoformat(timespec="seconds"),
        "t": time.time(),
        "sys": {
            "total_gb": round(vm.total / _GB, 2),
            "used_gb": round((vm.total - vm.available) / _GB, 2),
            "avail_gb": round(vm.available / _GB, 2),
            "percent": vm.percent,
            "swap_used_gb": round(sw.used / _GB, 2),
            "swap_total_gb": round(sw.total / _GB, 2),
            "swap_sout_gb": round(getattr(sw, "sout", 0) / _GB, 2),
        },
        "ray_total_gb": round(ray_total, 2),
        "by_task": {
            t: {"rss_gb": round(by_task_rss[t], 2), "n": by_task_n[t]}
            for t in sorted(by_task_rss, key=by_task_rss.get, reverse=True)
        },
        "top": [
            {"pid": p["pid"], "rss_gb": round(p["rss_gb"], 2),
             "pcpu": p["pcpu"], "cmd": p["cmd"][:90]}
            for p in procs[: (len(procs) if full else top_n)]
        ],
    }
    if full:
        rec["full"] = True
    return rec


# ----------------------------- watch loop -----------------------------

def _interval_for(percent: float, base: float) -> float:
    """Sample faster as pressure climbs so the run-up to a freeze is dense."""
    if percent >= 92:
        return min(base, 1.5)
    if percent >= 85:
        return min(base, 3.0)
    if percent >= 75:
        return min(base, 6.0)
    return base


def watch(args) -> None:
    out = Path(args.out) if args.out else _default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[memory_watch] recording -> {out}")
    print(f"[memory_watch] base interval {args.interval}s, "
          f"full-snapshot/warn threshold {args.threshold}%")
    print(f"[memory_watch] host={socket.gethostname()} pid={os.getpid()}  "
          f"(Ctrl-C or `kill {os.getpid()}` to stop)")

    with open(out, "a", buffering=1) as f:
        while True:
            percent = psutil.virtual_memory().percent
            full = percent >= args.threshold
            rec = _sample(args.top, full=full)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())  # durable against a hard freeze/reboot
            if full:
                top = rec["top"][0] if rec["top"] else {}
                sys.stderr.write(
                    f"[memory_watch] HIGH {percent:.0f}%  "
                    f"used={rec['sys']['used_gb']:.1f}G "
                    f"swap={rec['sys']['swap_used_gb']:.1f}G  "
                    f"top={top.get('cmd', '?')} {top.get('rss_gb', 0):.1f}G\n"
                )
            time.sleep(_interval_for(percent, args.interval))


# ----------------------------- report -----------------------------

def _load(path: Path) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a torn last line from a hard freeze
    return recs


def report(args) -> None:
    path = Path(args.out) if args.out else _default_out()
    recs = _load(path)
    if not recs:
        print(f"[memory_watch] no samples in {path}")
        return
    span_h = (recs[-1]["t"] - recs[0]["t"]) / 3600
    print(f"# {path}")
    print(f"# {len(recs)} samples over {span_h:.2f}h "
          f"({recs[0]['iso']} .. {recs[-1]['iso']})\n")

    peak = max(recs, key=lambda r: r["sys"]["used_gb"])
    print(f"## PEAK memory: {peak['sys']['used_gb']:.1f}G used "
          f"({peak['sys']['percent']:.0f}%), "
          f"swap {peak['sys']['swap_used_gb']:.1f}G  @ {peak['iso']}")
    print(f"   ray-worker RSS total at peak: {peak['ray_total_gb']:.1f}G")
    print("   top consumers at peak:")
    for p in peak["top"][:15]:
        print(f"     {p['rss_gb']:6.1f}G  {p['pcpu']:5.1f}%cpu  {p['cmd']}")
    print("   per Ray-task at peak (rss / #workers):")
    for t, d in list(peak["by_task"].items())[:15]:
        print(f"     {d['rss_gb']:6.1f}G  x{d['n']:<3} {t}")

    # Which task name held the most RSS at ANY point, and when.
    task_max: dict[str, tuple[float, str]] = {}
    for r in recs:
        for t, d in r["by_task"].items():
            if d["rss_gb"] > task_max.get(t, (0, ""))[0]:
                task_max[t] = (d["rss_gb"], r["iso"])
    print("\n## Max RSS ever held, per Ray-task:")
    for t, (g, iso) in sorted(task_max.items(), key=lambda x: -x[1][0])[:20]:
        print(f"   {g:6.1f}G  {t:<34} @ {iso}")

    # Compact timeline, downsampled to ~40 rows.
    step = max(1, len(recs) // 40)
    print("\n## Timeline (used / swap / ray-total, top task):")
    for r in recs[::step]:
        top_task = next(iter(r["by_task"]), "-")
        print(f"   {r['iso']}  used={r['sys']['used_gb']:5.1f}G  "
              f"swap={r['sys']['swap_used_gb']:4.1f}G  "
              f"ray={r['ray_total_gb']:5.1f}G  {top_task}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="record memory samples")
    w.add_argument("--interval", type=float, default=15.0,
                   help="base seconds between samples (default 15)")
    w.add_argument("--threshold", type=float, default=85.0,
                   help="percent RAM used at which to warn + dump full process "
                        "list and sample faster (default 85)")
    w.add_argument("--top", type=int, default=20,
                   help="processes to record per normal sample (default 20)")
    w.add_argument("--out", type=str, default=None,
                   help=f"output JSONL (default {_default_out()})")
    w.set_defaults(func=watch)

    r = sub.add_parser("report", help="analyze a recorded log")
    r.add_argument("--out", type=str, default=None,
                   help=f"input JSONL (default {_default_out()})")
    r.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
