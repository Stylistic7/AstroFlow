"""Slow-ramp an ASIAir's main camera cooler to a target temperature.

    PYTHONPATH=. python scripts/asiair_cool.py 10.0.0.28 --target -10 --rate 1 --interval 60
    PYTHONPATH=. python scripts/asiair_cool.py 10.0.0.28 --target -10 --dry-run

Each interval: read Temperature / CoolPowerPerc, write the next setpoint
(`asiair.cooling.Ramp`), log a line. One short verified session per step,
so the imager port's idle-close never matters. Exits when the sensor sits
at the target. Ctrl-C leaves the LAST setpoint in place (the firmware
keeps cooling to it); the cooler is never switched off by this script.
Start it so that the ramp finishes before Plan mode's start: the plan's
`wait_cooling` begins exposing at whatever setpoint is current.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asiair.cooling import Ramp  # noqa: E402
from asiair.rpc_session import AsiairSession  # noqa: E402
from asiair.rsa_sign import default_key_path  # noqa: E402


def _read(ip: str, names: tuple[str, ...]) -> dict[str, float]:
    with AsiairSession(ip, timeout=8) as s:
        s.verify(default_key_path())
        return {n: float(s.call("get_control_value", [n])["value"]) for n in names}


def _write(ip: str, name: str, value: float) -> None:
    with AsiairSession(ip, timeout=8) as s:
        s.verify(default_key_path())
        s.call("set_control_value", [name, value])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("--target", type=float, required=True, help="final sensor temperature, C")
    ap.add_argument("--rate", type=float, default=1.0, help="degrees per minute (default 1)")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between setpoint writes (default 60)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule; write nothing")
    a = ap.parse_args()
    ramp = Ramp(a.target, a.rate, a.interval)

    now = _read(a.ip, ("Temperature", "TargetTemp", "CoolerOn", "CoolPowerPerc"))
    temp = now["Temperature"] / 10.0
    setpoint = ramp.first_setpoint(temp)
    print(f"sensor {temp:.1f} C, cooler {'on' if now['CoolerOn'] else 'OFF'} at {now['CoolPowerPerc']:.0f}%, "
          f"current setpoint {now['TargetTemp']:.1f} C; ramp {setpoint:.0f} -> {a.target:.0f} C at "
          f"{a.rate:g} C/min = {ramp.duration_min(temp):.0f} min")
    if a.dry_run:
        return 0
    if not now["CoolerOn"]:
        _write(a.ip, "CoolerOn", 1)
    _write(a.ip, "TargetTemp", setpoint)
    while True:
        time.sleep(a.interval)
        r = _read(a.ip, ("Temperature", "CoolPowerPerc"))
        temp, power = r["Temperature"] / 10.0, r["CoolPowerPerc"]
        if ramp.done(setpoint, temp):
            print(f"{time.strftime('%H:%M:%S')}  setpoint {setpoint:6.1f}  sensor {temp:6.1f}  power {power:3.0f}%  done")
            return 0
        nxt = ramp.next_setpoint(setpoint, power)
        if nxt != setpoint:
            _write(a.ip, "TargetTemp", nxt)
            setpoint = nxt
        print(f"{time.strftime('%H:%M:%S')}  setpoint {setpoint:6.1f}  sensor {temp:6.1f}  power {power:3.0f}%"
              + ("  (holding: cooler flat out)" if nxt == setpoint and power >= 95 else ""))


if __name__ == "__main__":
    raise SystemExit(main())
