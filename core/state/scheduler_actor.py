import os, asyncio, time, ray
from typing import Dict, Any, List

def _get_namespace() -> str:
  return os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")

@ray.remote(max_restarts=-1, max_task_retries=-1)
class SchedulerActor:
    def __init__(self):
        self._jobs: List[Dict[str, Any]] = []  # {"interval": float, "action": dict, "next": float}
        self._run_task = asyncio.get_event_loop().create_task(self._run())

    async def add(self, interval_s: float, action: Dict[str, Any], start_delay_s: float = 0.0) -> None:
        now = time.monotonic()
        self._jobs.append({"interval": float(interval_s), "action": dict(action), "next": now + float(start_delay_s)})

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
        try:
            state = ray.get_actor("state", namespace=_get_namespace())
            state.dispatch.remote(action)
        except Exception:
            pass  # no state actor yet

    async def stop(self) -> None:
        self._run_task.cancel()
        try:
            await self._run_task
        except asyncio.CancelledError:
            pass
