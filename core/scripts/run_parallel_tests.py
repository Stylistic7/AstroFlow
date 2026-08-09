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
        uv run python scripts/run_parallel_tests.py exhaustive [N]
        N (or $ASTROFLOW_TEST_SHARDS) = parallel shard count
        (default 6; exhaustive default 4 — those tests peak at GBs each).
        $INCLUDE_SLOW=1 to also run tests marked `slow` (light mode only).

EXHAUSTIVE MODE: "exhaustive" only ever meant SLOW, not serial-only — these
are the top-of-the-pyramid full-pipeline tests, and they use the same private
per-process Ray clusters as everything else, so they parallelize the same
way. Sharding is per-TEST (not per-file) because the tier's wall clock is
dominated by a couple of ~15-minute mosaic tests that live in the SAME file;
round-robin scatters them across shards, so the tier costs roughly its
slowest test, not the sum. PixInsight-real tests stay safe under concurrency
via PI's own lockfile serialization (and most tests mock PI anyway).
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

_args = sys.argv[1:]
# Modes: "exhaustive" = the slow tier alone; "all" = light AND exhaustive in
# ONE worker pool (exhaustive submitted first — longest jobs lead, light
# file-shards backfill as workers free up); default = light only.
MODE = ("all" if "all" in _args
        else "exhaustive" if "exhaustive" in _args
        else "light")
_nums = [a for a in _args if a.isdigit()]
# Pool width 8 for the slow modes: measured at 4 the box sat ~75% idle /
# 86% memory free — the big mosaic tests use only 2-4 cores each. Each
# shard's private cluster is sized to 1/pool-width
# (ASTROFLOW_TEST_CONCURRENCY), so widths beyond ~8 start thinning the
# heavyweight tests' own internal parallelism instead.
_default_n = "6" if MODE == "light" else "8"
N = int(_nums[0]) if _nums else int(os.environ.get("ASTROFLOW_TEST_SHARDS", _default_n))
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


def collect_exhaustive_ids() -> list[str]:
    """Every exhaustive test id (`file::test`), for per-TEST sharding.

    Sorted by containing-file size (largest first): the heavyweight mosaic
    tests share a file, so consecutive ids round-robin onto DIFFERENT shards
    — the property that keeps the tier's wall clock near its slowest test."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "-m", "exhaustive"],
        cwd=CORE, env=BASE_ENV, capture_output=True, text=True)
    ids = [ln.strip() for ln in r.stdout.splitlines()
           if "::" in ln and not ln.startswith("=")]
    ids = [i for i in ids if not (IGNORE_PARTS & set(Path(i.split("::")[0]).parts))]
    def _size(tid: str) -> int:
        try:
            return (CORE / tid.split("::")[0]).stat().st_size
        except OSError:
            return 0
    return sorted(set(ids), key=_size, reverse=True)


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
           "ASTROFLOW_SKIP_RAY_CLEAN": "1",
           # The conftest memory gate reads GLOBAL memory, which under
           # sharding measures the sibling shards — 50% is permanently
           # exceeded while N clusters work. 85% keeps it as a real
           # brake against runaway pressure without gating on siblings.
           "ASTROFLOW_TEST_MAX_MEMORY_PERCENT": os.environ.get(
               "ASTROFLOW_TEST_MAX_MEMORY_PERCENT", "85")}
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
    # Build the job list: (paths, marker, junit, label). Every job is one
    # `pytest -n 0` subprocess with its own private Ray cluster.
    jobs: list[tuple] = []
    if MODE in ("exhaustive", "all"):
        ids = collect_exhaustive_ids()
        eshards = [s for s in (ids[i::N] for i in range(N)) if s]
        jobs += [(s, "exhaustive", f"exhshard{i}.xml", f"exhshard{i}")
                 for i, s in enumerate(eshards)]
    if MODE in ("light", "all"):
        files = find_test_files()
        lshards = [s for s in (files[i::N] for i in range(N)) if s]
        jobs += [(s, LIGHT, f"shard{i}.xml", f"shard{i}")
                 for i, s in enumerate(lshards)]
    print(f"mode={MODE}: {len(jobs)} shard jobs, pool width {N} "
          f"(each pytest -n 0, private Ray cluster).\n", flush=True)

    ray_stop()  # clean slate
    t0 = time.monotonic()
    results = []

    # Shards skip conftest's machine-wide ray-clean (concurrent shards would
    # kill each other's clusters); the ray_stop above is their clean start.
    # The serial + exclusive groups run separately as plain `pytest -n 0`
    # (see `make tests` / `make tests-all`) — routing them through this
    # runner's spawned subprocess flaked Ray's cluster init.
    #
    # max_workers=N throttles CONCURRENCY; jobs may exceed N (all-mode =
    # exhaustive shards + light shards). Submission order is the schedule:
    # exhaustive jobs were appended first, so the longest work starts
    # immediately and light shards backfill as workers free up. Cluster
    # sizing (ASTROFLOW_TEST_CONCURRENCY) uses N — the concurrent count —
    # not the total job count.
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [
            ex.submit(pytest_run, paths, marker, junit, N, label)
            for paths, marker, junit, label in jobs
        ]
        for f in concurrent.futures.as_completed(futs):
            label, ok, summary, output = f.result()
            results.append((label, ok, summary, output))
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:9} {summary}", flush=True)
    print(f"  --- {len(jobs)} shard jobs wall time: "
          f"{time.monotonic() - t0:.0f}s ---", flush=True)

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
