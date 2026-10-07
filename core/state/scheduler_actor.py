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
        self._asiair_guide_stream_tasks: dict = {}  # rig key -> (ip, Task)
        self._asiair_guide_stream_supervisor_task = None
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

    async def start_asiair_guide_event_streams(
        self, start_delay_s: float = 5.0, restart_interval_s: float = 30.0,
    ) -> None:
        """Spawn one persistent guide-event-stream listener per discovered
        ASIAir. The stream parses PHD2 GuideStep notifications (every ~1s
        during guiding) into a rolling buffer the panel endpoint reads.

        Re-checks the asiair_devices list every `restart_interval_s`: if a
        new ASIAir appears, we spawn its listener; if one disappears, we
        cancel. Each listener auto-reconnects on transient TCP failures,
        so the supervisor only handles add/remove."""
        if self._asiair_guide_stream_supervisor_task is not None:
            return
        self._asiair_guide_stream_supervisor_task = (
            asyncio.get_event_loop().create_task(
                self._asiair_guide_stream_supervisor(
                    start_delay_s, restart_interval_s,
                )
            )
        )

    async def _asiair_guide_stream_supervisor(
        self, start_delay_s: float, restart_interval_s: float,
    ) -> None:
        try:
            await asyncio.sleep(start_delay_s)
            while True:
                try:
                    from context import context
                    # Keyed by rig (the device serial), so a rig that moved
                    # to another IP gets a fresh listener there. Rigs never
                    # leave the list, so a listener that gave up (the rig
                    # was off when it first connected) is respawned here.
                    wanted = {r["key"]: r["ip"] for r in (context.asiair_devices or []) if r.get("ip")}
                    for key, (ip, task) in list(self._asiair_guide_stream_tasks.items()):
                        if wanted.get(key) != ip or task.done():
                            task.cancel()
                            del self._asiair_guide_stream_tasks[key]
                            print(f"### guide stream: stopped for {key} ({ip})")
                    for key, ip in wanted.items():
                        if key in self._asiair_guide_stream_tasks:
                            continue
                        from asiair.guide_event_stream import stream_one
                        task = asyncio.get_event_loop().create_task(
                            asyncio.to_thread(stream_one, ip, key)
                        )
                        self._asiair_guide_stream_tasks[key] = (ip, task)
                        print(f"### guide stream: started for {key} ({ip})")
                except Exception as e:
                    print(f"### guide stream supervisor raised, ignoring: {e}")
                await asyncio.sleep(restart_interval_s)
        except asyncio.CancelledError:
            for _ip, task in self._asiair_guide_stream_tasks.values():
                task.cancel()
            self._asiair_guide_stream_tasks.clear()

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
                    # Tell the frontend a fresh sample landed. The sampler moved
                    # here from ray_metrics._poll_loop (9ff4073a) but the paired
                    # SSE announce was left behind in that now-orphaned loop, so
                    # the RayStatus chip stopped live-updating. announce() is a
                    # cross-process queue.put; this head-pinned actor reaches the
                    # web workers' SSE event_loop exactly like StateActor's
                    # compute_graph_status announce does.
                    from web_server.backend.announcer import announce
                    announce("ray_status")
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


_last_sun_alt: float | None = None

# Astronomical twilight: the sun this far below the horizon bounds the
# observing window at both ends.
ASTRONOMICAL_TWILIGHT_DEG = -18.0


def _sun_altitude_deg() -> float | None:
    """The sun's altitude at the observer; None when it can't be computed."""
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, get_sun
        from astropy.time import Time
        from context import context
        lat, lon, elev = context.settings.observer_location
        t = Time.now()
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        return get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg
    except Exception as e:
        print(f"### download poller: sun altitude failed (non-fatal): {e}")
        return None


def _morning_kick_due(sun_alt: float | None) -> bool:
    """One extra drain as the sun rises through -18 deg: astronomical dawn.

    By then every rig is done shooting, the newest frame on disk is
    already older than the quiesce gate, and the builder's frontier
    nudge closes the night's final runs — this kick is what turns the
    accumulated night into the morning bake. Detected as a crossing
    between consecutive polls, so it fires within one poll interval of
    dawn and never at boot: a fresh process has no previous sample, and
    its own plan already covers everything on disk. (The once-a-day
    latch this replaces lived in process memory, so every pre-noon
    restart re-fired it — a drain minutes into each morning boot.)
    """
    global _last_sun_alt
    if sun_alt is None:
        return False
    previous, _last_sun_alt = _last_sun_alt, sun_alt
    return previous is not None and previous <= ASTRONOMICAL_TWILIGHT_DEG < sun_alt


def _drain_due(new_files: int, kick: bool, sun_alt: float | None) -> bool:
    """Whether this poll's arrivals warrant a drain-and-restart now.

    Frames that land while the sun is below -18 deg wait on disk for the
    dawn kick — one re-plan for the whole night. Each mid-night drain
    closed admissions cluster-wide until the WBPP already in flight
    finished (5.2 h of a 10.3 h night, 2026-09-14) and then re-keyed the
    hot target anyway; the morning bake delivers the same master at the
    same time with the rest of the cluster free all night. Daytime
    arrivals (manual pulls, archive copies) still drain as they land, as
    does anything landing while the sun's altitude is unknown.
    """
    if kick:
        return True
    night = sun_alt is not None and sun_alt <= ASTRONOMICAL_TWILIGHT_DEG
    return new_files > 0 and not night


def _run_asiair_download_and_maybe_restart() -> None:
    """Download from all ASIAir devices in parallel; if new files landed
    on disk by day — or the once-a-day astronomical-dawn kick is due —
    ask the StateActor to drain and restart the compute graph. Night-time
    arrivals wait on disk for the kick (see `_drain_due`)."""
    from context import context
    from file_io.find_image_files import find_image_files
    from asiair.download_asiair import download_from_asiair

    # Checked before the device early-returns: the kick must fire even
    # with every rig powered off and unplugged (the usual morning state).
    sun_alt = _sun_altitude_deg()
    kick = _morning_kick_due(sun_alt)

    # The live poller's rig list: found by port scan, keyed by serial.
    rigs = list(context.asiair_devices or [])

    if not rigs and not kick:
        return

    input_dirs = context.input_dirs or []
    if not input_dirs:
        return

    new_files = 0
    if rigs:
        dest = input_dirs[0]
        files_before = sum(len(find_image_files(str(d))) for d in input_dirs)

        # Fan out one Ray task per device so downloads parallelize across
        # whichever cluster nodes have a path to the ASIAir.
        refs = [
            # The first arg is only a FALLBACK label: the downloader names
            # the per-rig folder from the frames' FOCALLEN (see
            # RIG_NAMES_BY_FOCAL_LENGTH); the rig's own name stands in when
            # a frame has no mapped focal length.
            download_from_asiair.remote(rig.get("name") or rig["ip"], rig["ip"], dest)
            for rig in rigs
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
        new_files = files_after - files_before

    if _drain_due(new_files, kick, sun_alt):
        reason = (
            f"astronomical-dawn kick, {new_files} new files" if kick
            else f"{new_files} new files"
        )
        print(f"### ASIAir poller draining graph: {reason}")
        state = ray.get_actor("state", namespace=_get_namespace())
        ray.get(state.drain_and_restart.remote())
    elif new_files > 0:
        print(f"### download poller: {new_files} new files held for the "
              f"dawn kick (sun at {sun_alt:.1f} deg)")
