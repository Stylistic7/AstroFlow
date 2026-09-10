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
        self._asiair_guide_stream_tasks: dict = {}  # hostname -> Task
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
                    devices = context.asiair_devices or []
                    # By-hostname so a re-IP'd device gets a fresh listener.
                    wanted = {
                        d.get("hostname"): d.get("ip")
                        for d in devices
                        if d.get("hostname") and d.get("ip")
                    }
                    # Cancel listeners for gone-away devices.
                    for hostname in list(self._asiair_guide_stream_tasks.keys()):
                        if hostname not in wanted:
                            task = self._asiair_guide_stream_tasks.pop(hostname)
                            task.cancel()
                            print(f"### guide stream: stopped for {hostname}")
                    # Spawn for new ones.
                    for hostname, ip in wanted.items():
                        if hostname in self._asiair_guide_stream_tasks:
                            continue
                        from asiair.guide_event_stream import stream_one
                        task = asyncio.get_event_loop().create_task(
                            asyncio.to_thread(stream_one, ip, hostname)
                        )
                        self._asiair_guide_stream_tasks[hostname] = task
                        print(f"### guide stream: started for {hostname} ({ip})")
                except Exception as e:
                    print(f"### guide stream supervisor raised, ignoring: {e}")
                await asyncio.sleep(restart_interval_s)
        except asyncio.CancelledError:
            for task in self._asiair_guide_stream_tasks.values():
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


_last_dawn_kick_date = None


def _morning_kick_due() -> bool:
    """One extra drain shortly after astronomical dawn, once per day.

    By then every rig is done shooting, the newest frame on disk is
    already older than the quiesce gate, and the builder's frontier
    nudge closes the night's final runs — this kick is what turns the
    accumulated night into the morning bake. Detected at poll
    granularity, so it fires 0-15 min after the sun crosses -18 deg.
    """
    global _last_dawn_kick_date
    import datetime as dt
    now_local = dt.datetime.now()
    if now_local.hour >= 12 or _last_dawn_kick_date == now_local.date():
        return False
    try:
        import astropy.units as u
        from astropy.coordinates import AltAz, EarthLocation, get_sun
        from astropy.time import Time
        from context import context
        lat, lon, elev = context.settings.observer_location
        t = Time.now()
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        alt = get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg
    except Exception as e:
        print(f"### morning kick: sun altitude failed (non-fatal): {e}")
        return False
    if alt > -18.0:
        _last_dawn_kick_date = now_local.date()
        return True
    return False


def _run_asiair_download_and_maybe_restart() -> None:
    """Download from all ASIAir devices in parallel; if any new files
    landed on disk — or the once-a-day astronomical-dawn kick is due —
    ask the StateActor to drain and restart the compute graph."""
    from context import context
    from file_io.find_image_files import find_image_files
    from asiair.download_asiair import download_from_asiair

    # Checked before the device early-returns: the kick must fire even
    # with every rig powered off and unplugged (the usual morning state).
    kick = _morning_kick_due()

    devices = list(context.asiair_devices or [])

    # mDNS discovery MISSES a second unit when both announce the same
    # generic name (firmware 14.39 names every device "asiair.local"):
    # 2026-09-05, FLT156 was live and serving but held no discovery
    # record while RC71 owned the only one. Union in every host that
    # answers the imager port so a discovery miss can never skip a
    # rig's download.
    from asiair.network_hosts import find_hosts_with_open_port
    known_ips = {d.get("ip") for d in devices}
    try:
        for ip in find_hosts_with_open_port(4700):
            if ip not in known_ips:
                print(f"### download poller: port-scan found undiscovered ASIAir at {ip}")
                devices.append({"ip": ip, "hostname": ""})
    except Exception as e:
        print(f"### download poller: port scan failed (non-fatal): {e}")

    if not devices and not kick:
        return

    input_dirs = context.input_dirs or []
    if not input_dirs:
        return

    new_files = 0
    if devices:
        dest = input_dirs[0]
        files_before = sum(len(find_image_files(str(d))) for d in input_dirs)

        # Fan out one Ray task per device so downloads parallelize across
        # whichever cluster nodes have a path to the ASIAir.
        refs = [
            # The first arg is only a FALLBACK label: the downloader names
            # the per-rig folder from the frames' FOCALLEN (see
            # RIG_NAMES_BY_FOCAL_LENGTH); the mDNS hostname stands in when
            # a frame has no mapped focal length.
            download_from_asiair.remote(
                (dev.get("hostname") or dev["ip"]).rstrip(".").removesuffix(".local"),
                dev["ip"], dest,
            )
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
        new_files = files_after - files_before

    if new_files > 0 or kick:
        reason = (
            f"{new_files} new files" if new_files > 0
            else "astronomical-dawn kick"
        )
        print(f"### ASIAir poller draining graph: {reason}")
        state = ray.get_actor("state", namespace=_get_namespace())
        ray.get(state.drain_and_restart.remote())
