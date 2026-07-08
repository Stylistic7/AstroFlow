#!/usr/bin/env python
"""Run the test suite with process-per-file isolation — the reliable way to
parallelize Ray tests on this machine.

WHY NOT pytest-xdist: a reused xdist worker that ran Ray leaves Ray's gRPC /
raylet C++ threads alive; when the worker's interpreter finalizes they SIGABRT
(a known Ray/gRPC shutdown race — grpc#39942, ray#6025), which xdist reports as
"node down: Not properly terminated". It's intermittent and only in parallel.
A single `-n 0` pytest process is immune: it prints its results and exits
cleanly BEFORE any shutdown noise (verified: exit 0). Ray's own test suite gets
the same isolation from Bazel (one process per test target). This runner is that
model without Bazel: shard the test FILES across N independent single-process
pytest runs.

HOW IT STAYS SAFE across concurrent shards:
- ASTROFLOW_SKIP_RAY_CLEAN=1 for every shard: conftest's session-start/end
  `_ray_clean` does a machine-wide `pkill -9 ... raylet/gcs_server/...`. If N
  shards each ran it, they'd kill EACH OTHER's Ray clusters. We skip it per
  shard and bookend the whole run with a single `ray stop` instead. Each shard
  still tears down its own Ray via disable_test_mode.
- ASTROFLOW_TEST_CONCURRENCY=N: tells ray_local.enable_test_mode to size each
  private cluster to a 1/N slice of CPU/RAM (mirrors the xdist path), so N
  clusters don't oversubscribe the box.
- RAY_num_server_call_thread=1: fewer gRPC threads → smaller shutdown-race
  surface (Ray sets similar knobs in its own tests).
- Each `-n 0` run gets its own unique Ray temp dir (conftest already does this),
  so sockets/session metadata never collide.

This runs ONLY the parallel correctness phase. The `serial` (timing/benchmark)
and `exclusive` (state-isolation) groups are run separately as plain
`pytest -n 0` by `make tests` — routing them through this runner's spawned
subprocess destabilized Ray's cluster init, whereas plain `-n 0` is rock-solid.

Usage:  uv run python scripts/run_parallel_tests.py [N]
        N (or $ASTROFLOW_TEST_SHARDS) = parallel shard count (default 6).
        $INCLUDE_SLOW=1 to also run tests marked `slow`.
"""
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent  # scripts/ -> core/
VENV_RAY = CORE.parent / ".venv" / "bin" / "ray"

# Mirror pytest.ini's addopts --ignore list, plus vendored/thirdparty dirs.
IGNORE_PARTS = {"GraXpert", "ray_infra", "pix", "p2p", ".venv", "node_modules"}

N = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("ASTROFLOW_TEST_SHARDS", "6"))
_slow = "" if os.environ.get("INCLUDE_SLOW") == "1" else " and not slow"
# This runner handles ONLY the parallel correctness phase. The groups that must
# run alone — `serial` (wall-time/benchmark assertions) and `exclusive` (state
# isolation) — are run separately as plain `pytest -n 0` by `make tests`; routing
# them through this runner's spawned subprocess flaked Ray's cluster init. The
# selection here therefore excludes both, plus opt-in exhaustive/slow.
LIGHT = f"not exhaustive and not exclusive and not serial{_slow}"

BASE_ENV = {
    **os.environ,
    "PYTHONPATH": ".",
}
# NOTE: deliberately NOT setting RAY_num_server_call_thread. It was a speculative
# "shrink the gRPC shutdown race" tweak, and it turned out to *destabilize* Ray
# cluster init/teardown — the serial groups flaked (DOA hang / gcs_error) with it
# and are rock-solid without it (plain `-n 0`, verified 3/3). Keep the Ray env
# stock; the only knobs we touch are per-cluster resources in ray_local.


def find_test_files() -> list[str]:
    files = set()
    for pattern in ("test_*.py", "*_test.py"):
        for p in CORE.rglob(pattern):
            rel = p.relative_to(CORE)
            if IGNORE_PARTS & set(rel.parts):
                continue
            files.add(str(rel))
    # Largest first, then round-robin -> rough load balance across shards.
    return sorted(files, key=lambda f: (CORE / f).stat().st_size, reverse=True)


def ray_stop() -> None:
    if VENV_RAY.exists():
        subprocess.run([str(VENV_RAY), "stop", "--force"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def pytest_run(paths, marker, junit, concurrency, label):
    """Run one shard as a single-process (`-n 0`) pytest. Returns
    (label, ok, summary, output).

    ASTROFLOW_SKIP_RAY_CLEAN=1 suppresses conftest's machine-wide `_ray_clean`
    pkill — concurrent shards would kill each other's Ray clusters; the runner's
    initial `ray_stop` is their clean start instead.
    """
    env = {**BASE_ENV,
           "ASTROFLOW_TEST_CONCURRENCY": str(concurrency),
           "ASTROFLOW_SKIP_RAY_CLEAN": "1"}
    cmd = [sys.executable, "-m", "pytest", "-n", "0", "-m", marker,
           f"--junitxml={junit}", "-q", *paths]
    r = subprocess.run(cmd, cwd=CORE, env=env, capture_output=True, text=True)
    # exit 0 = passed; exit 5 = no tests collected for this marker (an empty
    # shard, e.g. a file whose tests were all deselected) — that's fine.
    ok = r.returncode in (0, 5)
    summary = _summary_line(r.stdout) or f"(exit {r.returncode})"
    return label, ok, summary, r.stdout + r.stderr


_SUMMARY_RE = re.compile(r"\d+\s+(passed|failed|error|skipped|xfailed|xpassed|deselected)")


def _summary_line(out: str) -> str:
    # pytest's real summary line (a count + keyword), NOT a traceback line that
    # merely contains "error" (e.g. `error_data = ...`).
    for line in reversed(out.splitlines()):
        s = line.strip().strip("=").strip()
        if _SUMMARY_RE.search(s) or "no tests ran" in s:
            return s
    return ""


def main() -> int:
    files = find_test_files()
    shards = [files[i::N] for i in range(N)]
    shards = [s for s in shards if s]
    print(f"parallel correctness: {len(files)} files across {len(shards)} "
          f"process-per-file shards (each pytest -n 0).\n", flush=True)

    ray_stop()  # clean slate
    t0 = time.monotonic()
    results = []

    # Parallel correctness only. Shards skip conftest's machine-wide ray-clean
    # (concurrent shards would kill each other's clusters); the ray_stop above is
    # their clean start. The serial + exclusive groups run separately as plain
    # `pytest -n 0` (see `make tests`) — routing them through this runner's
    # spawned subprocess flaked Ray's cluster init.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as ex:
        futs = [
            ex.submit(pytest_run, s, LIGHT, f"shard{i}.xml", len(shards), f"shard{i}")
            for i, s in enumerate(shards)
        ]
        for f in concurrent.futures.as_completed(futs):
            label, ok, summary, output = f.result()
            results.append((label, ok, summary, output))
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:9} {summary}", flush=True)
    print(f"  --- {len(shards)} shards wall time: {time.monotonic() - t0:.0f}s ---", flush=True)

    ray_stop()
    elapsed = time.monotonic() - t0

    failed = [(lbl, out) for (lbl, ok, _s, out) in results if not ok]
    for lbl, out in failed:
        print(f"\n===================== output: {lbl} =====================")
        print(out[-6000:])

    ok_n = sum(1 for (_l, ok, _s, _o) in results if ok)
    print(f"\n{'='*60}\nRESULT: {ok_n}/{len(results)} groups green in {elapsed:.0f}s"
          f" — {'ALL GREEN' if not failed else 'FAILURES: ' + ', '.join(l for l, _ in failed)}\n{'='*60}",
          flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
