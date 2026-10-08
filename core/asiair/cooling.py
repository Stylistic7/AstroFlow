"""Slow-ramp cooling for an ASIAir's main camera.

The firmware chases `TargetTemp` as fast as the cooler allows (~7 C/min on
the 6200MM, measured 2026-10-07). A slow ramp keeps the firmware chasing a
one-degree step instead: the setpoint walks from the sensor's current
temperature toward the target at `rate` degrees per minute, one write per
`interval` seconds, and holds wherever the cooler is already working flat
out. Pure logic here; `scripts/asiair_cool.py` drives it over the RPC
(ZWO control names: Temperature in tenths of a degree, TargetTemp, CoolerOn,
CoolPowerPerc).
"""
from __future__ import annotations

from dataclasses import dataclass

HOLD_POWER_PERCENT = 95   # the cooler cannot keep up: do not lower the setpoint further
DONE_TOLERANCE_C = 0.5


@dataclass(frozen=True)
class Ramp:
    target_c: float
    rate_c_per_min: float = 1.0
    interval_s: float = 60.0

    @property
    def step_c(self) -> float:
        return self.rate_c_per_min * self.interval_s / 60.0

    def first_setpoint(self, temperature_c: float) -> float:
        """Start from where the sensor IS (rounded to the degree), never from
        a stale setpoint, so the first step is one step."""
        return float(round(temperature_c))

    def next_setpoint(self, setpoint_c: float, power_percent: float) -> float:
        """The setpoint for the next interval: one step toward the target,
        clamped at it; unchanged while the cooler is at `HOLD_POWER_PERCENT`."""
        if power_percent >= HOLD_POWER_PERCENT:
            return setpoint_c
        if setpoint_c > self.target_c:
            return max(self.target_c, setpoint_c - self.step_c)
        return min(self.target_c, setpoint_c + self.step_c)        # warming

    def done(self, setpoint_c: float, temperature_c: float) -> bool:
        return setpoint_c == self.target_c and abs(temperature_c - self.target_c) <= DONE_TOLERANCE_C

    def duration_min(self, temperature_c: float) -> float:
        """Nominal minutes from the sensor's temperature to the target, with
        no holds: how far ahead of the plan's start to begin."""
        return abs(self.first_setpoint(temperature_c) - self.target_c) / self.rate_c_per_min


# ------------------------------------------------------------- live stepping
# One step per rig per interval from the live poller's tick (SchedulerActor):
# `context["asiair_cooling"][rig key] = {"enabled": bool, "target_c": float}`
# is the user's request (the panel's switch + field); the progress rides on
# the rig record as `rig["cooling"]` so the panel shows it live.
import structlog

from asiair.rpc_session import AsiairRpcError, AsiairSession

log = structlog.get_logger()
_CONTROLS = ("Temperature", "CoolerOn", "CoolPowerPerc")


def step(rig: dict, request: dict, key, now: float, open_session=AsiairSession) -> "dict | None":
    """Advance one rig's ramp if an interval has passed; None when nothing
    changed. A changed target restarts the ramp from the sensor's current
    temperature. The cooler is switched ON if it was off (the sensor then
    reads ambient, which is where the ramp starts); it is never switched
    off here — the ramp ends by holding the target."""
    ramp = Ramp(float(request["target_c"]), float(request.get("rate_c_per_min", 1.0)))
    prog = dict(rig.get("cooling") or {})
    if prog.get("target_c") != ramp.target_c:
        prog = {"target_c": ramp.target_c}
    if now - prog.get("last_step_at", 0.0) < ramp.interval_s:
        return None
    live = rig.get("live") or {}
    if key is None:
        return {**prog, "phase": "no key file", "last_step_at": now}
    if not live.get("reachable") or live.get("camera_state") in (None, "close"):
        return {**prog, "phase": "waiting for camera", "last_step_at": now}
    try:
        with open_session(rig["ip"], timeout=8) as s:
            s.verify(key)
            v = {n: float(s.call("get_control_value", [n])["value"]) for n in _CONTROLS}
            temperature, power = v["Temperature"] / 10.0, v["CoolPowerPerc"]
            if not v["CoolerOn"]:
                s.call("set_control_value", ["CoolerOn", 1])
                power = 0.0
            setpoint = prog.get("setpoint_c")
            if setpoint is None:
                setpoint = ramp.first_setpoint(temperature)
            elif not ramp.done(setpoint, temperature):
                setpoint = ramp.next_setpoint(setpoint, power)
            if setpoint != prog.get("setpoint_c"):
                s.call("set_control_value", ["TargetTemp", setpoint])
    except (OSError, AsiairRpcError) as e:
        log.warning("cooling ramp: step failed", rig=rig.get("key"), error=str(e))
        return {**prog, "phase": "unreachable", "last_step_at": now}
    phase = ("done" if ramp.done(setpoint, temperature)
             else "holding (cooler flat out)" if power >= HOLD_POWER_PERCENT else "ramping")
    log.info("cooling ramp", rig=rig.get("key"), setpoint=setpoint, temperature=temperature, power=power, phase=phase)
    return {"target_c": ramp.target_c, "setpoint_c": setpoint, "temperature_c": temperature,
            "power": power, "phase": phase, "last_step_at": now}
