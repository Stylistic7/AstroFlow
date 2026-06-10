import os, asyncio, time, ray
from typing import Dict, Any, List


def _get_namespace() -> str:
    return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")


@ray.remote(max_restarts=-1, max_task_retries=-1)
class SchedulerActor:
    def __init__(self):
        self._jobs: List[
            Dict[str, Any]
        ] = []  # {"interval": float, "action": dict, "next": float}
        self._asiair_poller_task = None
        self._asiair_live_poller_task = None
        self._ray_metrics_task = None
        self._run_task = asyncio.get_event_loop().create_task(self._run())

    async def add(
        self,
        interval_s: float,
        action: Dict[str, Any],
        start_delay_s: float = 0.0,
    ) -> None:
        now = time.monotonic()
        self._jobs.append(
            {
                "interval": float(interval_s),
                "action": dict(action),
                "next": now + float(start_delay_s),
            }
        )

    async def start_asiair_download_poller(
        self, interval_s: float, start_delay_s: float = 0.0,
    ) -> None:
        """Run ASIAir downloads here in the scheduler, not via the StateActor.

        Downloads are pure I/O; they have nothing to do with state-machine
        consistency, so gating them on StateActor._busy meant they never ran
        while the compute graph was up (i.e. essentially always). Downloads
        now run in this actor's event loop on their own timer; only the
        drain_and_restart handoff goes through the StateActor, and only
        when there's something new to act on.
        """
        if self._asiair_poller_task is not None:
            return
        self._asiair_poller_task = asyncio.get_event_loop().create_task(
            self._asiair_poll_loop(interval_s, start_delay_s)
        )

    async def _asiair_poll_loop(self, interval_s: float, start_delay_s: float) -> None:
        try:
            await asyncio.sleep(start_delay_s)
            while True:
                try:
                    await asyncio.to_thread(_run_asiair_download_and_maybe_restart)
                except Exception as e:
                    # I/O failures should never kill the poller — it'll try
                    # again on the next tick.
                    print(f"### ASIAir poll raised, ignoring: {e}")
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            pass

    async def start_asiair_live_poller(
        self, interval_s: float = 5.0, start_delay_s: float = 5.0,
    ) -> None:
        """Poll the discovered ASIAirs for live state (capture progress,
        connected camera, last GoTo target, firmware/temp/location).

        Separate from the SMB download poller because the cadences are
        very different — file downloads happen on a 15-min cycle, but
        live state needs to refresh every few seconds during an active
        session. Runs in its own asyncio task with `to_thread`-wrapped
        socket I/O so it doesn't share the download poller's timer.
        """
        if self._asiair_live_poller_task is not None:
            return
        self._asiair_live_poller_task = asyncio.get_event_loop().create_task(
            self._asiair_live_poll_loop(interval_s, start_delay_s)
        )

    async def _asiair_live_poll_loop(
        self, interval_s: float, start_delay_s: float,
    ) -> None:
        try:
            await asyncio.sleep(start_delay_s)
            while True:
                try:
                    from asiair.live_poller import tick
                    await asyncio.to_thread(tick)
                except Exception as e:
                    print(f"### ASIAir live-poll raised, ignoring: {e}")
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            pass

    async def start_ray_metrics_poller(
        self, interval_s: float = 1.0, start_delay_s: float = 1.0,
    ) -> None:
        """1 Hz cluster CPU / memory / object-store sampler that
        feeds the Ray status chip's sparklines. Runs HERE because
        this actor is already inside the Ray cluster and has the
        `ray.*` API readily available; the gunicorn workers don't
        (the SSE worker doesn't call `ray.init`, and the main app
        worker uses sync workers, neither of which is a natural
        host for a long-running 1 Hz loop).
        """
        if self._ray_metrics_task is not None:
            return
        self._ray_metrics_task = asyncio.get_event_loop().create_task(
            self._ray_metrics_loop(interval_s, start_delay_s)
        )

    async def sample_ray_metrics_once(self) -> None:
        """Synchronously trigger one ray_metrics sample. Bypasses the
        1 Hz loop. Exists so integration tests can deterministically
        kick a sample and then immediately assert on the diskcache —
        without it, tests would have to start the poller and sleep
        for an interval, racing on timing."""
        from web_server.backend.ray_metrics import sample_once
        sample_once()

    async def get_node_id_for_introspection(self) -> str:
        """Return the Ray node ID this actor is currently scheduled on.
        Test-only — used to assert the head-pin actually placed the
        actor on the driver's node, not silently on a worker (the bug
        that left the Flask diskcache reader looking at stale data)."""
        return ray.get_runtime_context().get_node_id()

    async def _ray_metrics_loop(
        self, interval_s: float, start_delay_s: float,
    ) -> None:
        try:
            await asyncio.sleep(start_delay_s)
            while True:
                try:
                    from web_server.backend.ray_metrics import sample_once
                    sample_once()
                except Exception as e:
                    print(f"### ray_metrics sample raised, ignoring: {e}")
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            while True:
                now = time.monotonic()
                for j in self._jobs:
                    if now >= j["next"]:
                        self._dispatch(j["action"])
                        # drift-free advance
                        k = int((now - j["next"]) // j["interval"]) + 1
                        j["next"] += k * j["interval"]
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _dispatch(self, action: Dict[str, Any]) -> None:
        state = ray.get_actor("state", namespace=_get_namespace())
        state.enqueue.remote(action)

    async def stop(self) -> None:
        for task in (
            self._run_task,
            self._asiair_poller_task,
            self._asiair_live_poller_task,
            self._ray_metrics_task,
        ):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def _run_asiair_download_and_maybe_restart() -> None:
    """Download from all ASIAir devices in parallel; if any new files
    landed on disk, ask the StateActor to drain and restart the
    compute graph so they get processed."""
    from context import context
    from file_io.find_image_files import find_image_files
    from asiair.download_asiair import download_from_asiair

    devices = context.asiair_devices or []
    if not devices:
        return

    input_dirs = context.input_dirs or []
    if not input_dirs:
        return
    dest = input_dirs[0]
    files_before = sum(len(find_image_files(str(d))) for d in input_dirs)

    # Fan out one Ray task per device so downloads parallelize across
    # whichever cluster nodes have a path to the ASIAir.
    refs = [
        download_from_asiair.remote("foo", dev["ip"], dest, "H")
        for dev in devices
    ]
    ray.get(refs)

    # Tell connected web clients to refetch the ASIAir panel — the
    # newest-file mtime / active-state dot are derived from on-disk
    # Stack state, so any newly downloaded frames may shift the
    # rendering. Fires even when no new files arrived, so the panel
    # can show the dot dimming when the cycle completes without
    # producing anything new.
    try:
        from web_server.backend.announcer import announce
        announce("asiair_devices")
    except Exception as e:
        print(f"### scheduler_actor: asiair_devices announce failed: {e}")

    files_after = sum(len(find_image_files(str(d))) for d in input_dirs)
    if files_after > files_before:
        new_files = files_after - files_before
        print(f"### ASIAir download found {new_files} new files, draining graph")
        state = ray.get_actor("state", namespace=_get_namespace())
        ray.get(state.drain_and_restart.remote())
