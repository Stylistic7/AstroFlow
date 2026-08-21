"""One sentinel actor per Ray node — the node's own telemetry source.

WHY AN ACTOR, NOT A PROBE TASK. The admission gate schedules memory-hungry
work only onto nodes it has MEASURED, and the old measurement was a probe
task with a 4 s timeout. Answering a probe needs a warm Ray worker on the
target node; a cold start imports the whole module graph (>4 s); and
`auto_memory`'s max_calls=1 keeps workers from lingering on a quiet node.
The result was a bistable trap: an idle node cannot answer, an unanswering
node is dropped from scheduling entirely ("falls out until it answers
again"), and a dropped node stays idle — measured live 2026-08-19: the MBP
at 7 executions for three hours, then 2,000+ after one lucky idle-escape
seeding, with zero code changed in between. A pinned actor imports once at
creation and answers in milliseconds forever after, so measurement no longer
depends on the node having been busy recently.

The same actor is where node-LOCAL supervision belongs, starting with the
PixInsight inactivity watchdog: a wedged PI (0% CPU, no scratch writes)
held a PixInsight slot for SEVEN HOURS overnight 2026-08-20, silently
halving cluster stacking throughput. Only a process on the same node can
cheaply watch CPU and file mtimes, so the sentinel owns the check — and the
kill, which targets the PixInsight PROCESS only, never a `ray::` worker
(see project_disk_fill_vectors: killing ray workers corrupts the run).

Grown deliberately from here: disk-fill telemetry, chip load, worker churn —
whatever a node should know about itself.
"""

import os
import subprocess
import sys
import time

import ray

# Inactivity rule per the watchdog design: a PixInsight that has used no CPU
# AND written nothing for this long is wedged. Elapsed time is deliberately
# NOT a signal — big WBPPs legitimately run for hours; they just never go
# fully silent (compute phases burn CPU, I/O phases touch scratch).
_PI_WEDGE_SECONDS = 300.0

# A PixInsight is only wedge-ELIGIBLE once it has done real work: a pooled
# instance idling between dispatches carries only ~10-20s of startup CPU and
# must not be reaped (the runner cold-starts a replacement, pure waste —
# observed 2026-08-21: every kill at ~301s, the idle-aging signature, never
# a mid-task hang). A genuinely wedged WBPP/recipe carries minutes of CPU.
_PI_MIN_WORK_CPU_SECONDS = 60.0


@ray.remote(num_cpus=0)
class NodeSentinel:
    """Pinned per node with NodeAffinity(soft=False); all methods are cheap
    local reads. State (the PI activity history) lives here so wedge
    detection needs no cross-node bookkeeping."""

    def __init__(self):
        # pid -> (last_active_wallclock, last_cpu_seconds)
        self._pi_activity: dict[int, tuple[float, float]] = {}

    def memory(self) -> tuple[str, int, int, int]:
        """(node_id, available_bytes, total_bytes, pressure_level) — the same
        contract as the old probe task, read on this node."""
        import psutil
        vm = psutil.virtual_memory()
        node_id = ray.get_runtime_context().get_node_id()
        pressure = 0
        try:
            if sys.platform == "darwin":
                r = subprocess.run(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                    capture_output=True, text=True, timeout=2)
                v = int(r.stdout.strip())
                pressure = v if v in (1, 2, 4) else 0
            elif sys.platform.startswith("linux"):
                with open("/proc/pressure/memory") as f:
                    for line in f:
                        if line.startswith("full"):
                            kv = dict(p.split("=") for p in line.split()[1:])
                            avg10 = float(kv.get("avg10", 0.0))
                            pressure = 4 if avg10 >= 10.0 else 2 if avg10 >= 1.0 else 1
                            break
        except Exception:
            pressure = 0
        return (node_id, int(vm.available), int(vm.total), pressure)

    # ---- PixInsight watchdog -------------------------------------------

    def _pi_snapshot(self) -> list[tuple[int, float, float]]:
        """[(pid, cpu_seconds, newest_scratch_mtime)] for local PI processes."""
        out = []
        try:
            r = subprocess.run(["pgrep", "-f", "PixInsight.app/Contents/MacOS/PixInsight"],
                               capture_output=True, text=True, timeout=5)
            pids = [int(p) for p in r.stdout.split()]
        except Exception:
            return out
        import psutil
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                ct = proc.cpu_times()
                cpu = float(ct.user + ct.system)
                newest = 0.0
                for f in proc.open_files():
                    p = f.path
                    if "/var/folders/" in p or "/tmp/" in p:
                        try:
                            newest = max(newest, os.path.getmtime(p))
                        except OSError:
                            pass
                out.append((pid, cpu, newest))
            except Exception:
                continue
        return out

    def pixinsight_health(self) -> list[dict]:
        """Wedge check for every local PI. A pid is ACTIVE if its cumulative
        CPU advanced or a scratch file was touched since the last look;
        wedged = inactive for `_PI_WEDGE_SECONDS`."""
        now = time.time()
        report = []
        seen = set()
        for pid, cpu, newest_write in self._pi_snapshot():
            seen.add(pid)
            last_active, last_cpu = self._pi_activity.get(pid, (now, cpu))
            if cpu > last_cpu + 0.5 or newest_write > last_active:
                last_active = now
            self._pi_activity[pid] = (last_active, cpu)
            report.append({
                "pid": pid,
                "inactive_s": round(now - last_active, 1),
                "wedged": ((now - last_active) > _PI_WEDGE_SECONDS
                           and cpu > _PI_MIN_WORK_CPU_SECONDS),
            })
        for pid in set(self._pi_activity) - seen:
            del self._pi_activity[pid]
        return report

    def kill_wedged_pixinsight(self) -> list[int]:
        """Kill every local PI the health check calls wedged — the PI process
        ONLY. The Ray task holding it fails and retries with a fresh PI,
        which is the recovery the 2026-08-20 manual intervention proved."""
        killed = []
        for h in self.pixinsight_health():
            if h["wedged"]:
                try:
                    os.kill(h["pid"], 9)
                    killed.append(h["pid"])
                    print(f"### NodeSentinel: killed WEDGED PixInsight "
                          f"pid={h['pid']} (inactive {h['inactive_s']}s)")
                except OSError:
                    pass
        return killed


def sentinel_for(node_id: str):
    """Get-or-create this node's sentinel, pinned hard to the node.

    Named + get_if_exists IN AN EXPLICIT NAMESPACE so every caller shares
    one actor per node. Detached named actors are namespace-scoped, and each
    Ray driver (main.py, every gunicorn worker, any ad-hoc shell) connects in
    its own anonymous namespace — without the explicit namespace each driver
    would silently spawn its own duplicate sentinel per node. A dead
    node's sentinel dies with it and is simply recreated on the next call
    after rejoin (soft=False placement fails fast while the node is gone —
    callers treat that like any other missed measurement)."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    return NodeSentinel.options(
        name=f"node-sentinel-{node_id}",
        # Same resolution as context._get_namespace (mock override for tests),
        # inlined so the sentinel module never imports the app graph.
        namespace=os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow"),
        get_if_exists=True,
        lifetime="detached",
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    ).remote()
