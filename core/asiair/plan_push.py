"""Build ASIAir Plan-mode plans — coordinates AND exposures — and push them.

Plan mode's public import (the Telescopius CSV) carries coordinates only.
The device's own RPC carries everything: `get_plan` returns the plan list
in the schema below, captured from firmware 14.39 on 2026-10-06, and
`set_plan` writes it. Only the fields a human sets in the app are ours to
fill; the device computes the rest (sizes, times, progress) and we leave
them at their fresh-target values.

    plan   {id, plan_name, enable, merid_flip, mount_end_action, start_guide,
            close_cooler, wait_cooling, focuser_go_home, caa_go_home,
            shutdown_pi, start_time {type: "dusk"|"date", hour}, end_time
            {type: "dawn"}, targets: [...]}
    target {id, enable, target_name, target_coord [RA_HOURS, DEC_DEG] (JNow),
            seqs: [{id, type: "light", filter: WHEEL_SLOT, exp, repeat,
                    gain: -10000 (= "camera's current"), bin, ...}]}

COORDINATES ARE JNow. The device stores and slews to `target_coord` as
equinox-of-date coordinates; its own Telescopius CSV import converts J2000
on the way in (Panes 8-21 of Stylistic7's Cassiopeia plan read back exactly
at their J2000 values precessed to 2026-10, 0.0'). AstroFlow's centres are
J2000 plate solves, so `build_target` precesses them to the plan's night.
Until 2026-10-14 they were sent raw: every pushed panel landed one
precession (~10-15' at Dec +45..+70) south-west of its Target, and a
re-push aimed at that drifted Target compounded it (7098 Panels 2 and 4
on 2026-10-11 landed two precessions off).

A pushed plan is created DISABLED under its own name: the plan that is
armed for tonight stays armed, and enabling the new one is a tap on the
tablet. `push_plan` reads the device back after writing and, if any
pre-existing plan changed, restores the pre-push list and raises — the
write semantics of `set_plan` are reverse-engineered, not documented.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from asiair.rpc_session import AsiairSession

FILTER_SLOT_FALLBACK = {"L": 0, "R": 1, "G": 2, "B": 3, "S": 4, "H": 5, "O": 6}


@dataclass(frozen=True)
class ExposureRow:
    filter: str          # wheel slot NAME ("O"), resolved to the index at build time
    exposure_s: float
    count: int


@dataclass(frozen=True)
class PlanTarget:
    name: str
    ra_deg: float        # J2000 (ICRS), like every AstroFlow centre
    dec_deg: float
    rows: tuple[ExposureRow, ...]


def jnow(ra_deg: float, dec_deg: float, night: _dt.date) -> tuple[float, float]:
    """J2000 -> the mean equinox of `night`: what the ASIAir's goto expects."""
    import astropy.units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs").transform_to(
        FK5(equinox=Time(night.isoformat())))
    return float(c.ra.deg), float(c.dec.deg)


def wheel_slots(session: AsiairSession) -> dict[str, int]:
    """{filter name: slot index} from the device's wheel settings."""
    names = (session.call("get_wheel_setting") or {}).get("names") or []
    return {str(n): i for i, n in enumerate(names)} if names else dict(FILTER_SLOT_FALLBACK)


def device_name(name: str) -> str:
    """The name as the device will store it: it drops ':' (the name ends up
    in file names), so send what it keeps and the read-back compares equal."""
    return " ".join(name.replace(":", " ").split())


def build_target(index: int, target: PlanTarget, slots: dict[str, int], night: _dt.date) -> dict:
    missing = [r.filter for r in target.rows if r.filter not in slots]
    if missing:
        raise KeyError(f"filter(s) {missing} not in the wheel {sorted(slots)}")
    name = device_name(target.name)
    ra, dec = jnow(target.ra_deg, target.dec_deg, night)
    return {
        "id": index,
        "enable": True,
        "target_name": name,
        "target_original_name": name,
        "target_original_img": "",
        "target_coord": [round(ra / 15.0, 6), round(dec, 6)],
        "group_by_slot": True,
        "delay_first": 0,
        "delay_between_frame": 0,
        "extra_sec_per_target": 0,
        # EXECUTION ORDER, 0-based — not the list position. Stylistic7's old plan
        # ran its targets by this field (Pane 4 = 0 went first, Pane 2 = 7
        # went seventh); all-zero ties were broken alphabetically, which put
        # "7098 LBN 588" ahead of "75a1 Panel 4" (2026-10-06).
        "capture_index": index - 1,
        "obj_type": "",
        "seqs": [
            {
                "id": i,
                "type": "light",
                "filter": slots[row.filter],
                "suffix": "",
                "exp": float(row.exposure_s),
                "gain": -10000,
                "bin": 1,
                "repeat": int(row.count),
                "lapsed": 0,
                "capture_index": i,
                "enable": True,
                "autoexp": False,
            }
            for i, row in enumerate(target.rows, 1)
        ],
    }


def build_plan(plan_id: int, name: str, targets: list[PlanTarget], slots: dict[str, int],
               night: _dt.date, like: dict | None = None) -> dict:
    """A whole plan for `night` (the equinox its coordinates are precessed
    to). `like` is an existing plan whose session conventions (meridian
    flip, end action, guiding, cooler) are carried over."""
    like = like or {}
    return {
        "id": plan_id,
        "plan_name": name,
        "enable": False,
        "merid_flip": bool(like.get("merid_flip", True)),
        "mount_end_action": like.get("mount_end_action", "home"),
        "start_guide": bool(like.get("start_guide", True)),
        "close_cooler": bool(like.get("close_cooler", True)),
        "wait_cooling": bool(like.get("wait_cooling", True)),
        "focuser_go_home": bool(like.get("focuser_go_home", False)),
        "caa_go_home": bool(like.get("caa_go_home", False)),
        "shutdown_pi": False,
        "start_time": {"type": "dusk"},
        "end_time": {"type": "dawn"},
        "targets": [build_target(i, t, slots, night) for i, t in enumerate(targets, 1)],
    }


_COORD_TOL = 1e-4   # hours / degrees: the device stores ~5e-6; 1e-4 h is 5" of RA


def _essentials(plan: dict) -> dict:
    """What a human set: the fields that must survive a push untouched.
    Coordinates are kept as floats and compared with `_same`."""
    return {
        "id": plan.get("id"), "plan_name": plan.get("plan_name"), "enable": plan.get("enable"),
        "targets": [
            {
                "name": device_name(str(t.get("target_name", ""))),
                "coord": [float(c) for c in t.get("target_coord", [])],
                "enable": t.get("enable"),
                "seqs": [(s.get("filter"), float(s.get("exp", 0)), s.get("repeat"), s.get("enable")) for s in t.get("seqs", [])],
            }
            for t in plan.get("targets", [])
        ],
    }


def _same(a: dict, b: dict) -> bool:
    """`_essentials` equality with a coordinate tolerance."""
    if a is None or b is None or len(a["targets"]) != len(b["targets"]):
        return False
    if {k: v for k, v in a.items() if k != "targets"} != {k: v for k, v in b.items() if k != "targets"}:
        return False
    for ta, tb in zip(a["targets"], b["targets"]):
        if (ta["name"], ta["enable"], ta["seqs"]) != (tb["name"], tb["enable"], tb["seqs"]):
            return False
        if len(ta["coord"]) != len(tb["coord"]) or any(abs(x - y) > _COORD_TOL for x, y in zip(ta["coord"], tb["coord"])):
            return False
    return True


def push_plan(session: AsiairSession, plan: dict) -> dict:
    """Write `plan` as a NEW plan beside the existing ones and verify by
    reading back. `set_plan([plan])` is additive by id on firmware 14.39
    (measured 2026-10-06): existing plans are never re-sent — on any
    anomaly the plan we added is deleted and the pre-existing ones are
    left exactly as the device holds them."""
    before = session.call("get_plan") or []
    if any(p.get("id") == plan["id"] for p in before):
        raise ValueError(f"plan id {plan['id']} already exists on the device")
    session.call("set_plan", [plan])
    after = session.call("get_plan") or []
    kept = {p.get("id"): _essentials(p) for p in after}
    damaged = [p.get("id") for p in before if not _same(kept.get(p.get("id")), _essentials(p))]
    mine = kept.get(plan["id"])
    if damaged or mine is None or not _same(mine, _essentials(plan)):
        # Report, never "clean up": `delete_plan([{"id": N}])` wiped EVERY
        # plan on firmware 14.39 (2026-10-06), and re-sending the others
        # re-rounds their coordinates. The caller has the pre-push backup.
        raise RuntimeError(
            f"set_plan read-back mismatch (damaged pre-existing plans={damaged}, "
            f"ours present={mine is not None}); nothing further was changed"
        )
    return {"plan_id": plan["id"], "targets": len(plan["targets"]), "plans_on_device": len(after)}


def next_plan_id(session: AsiairSession) -> int:
    ids = [int(p.get("id", 0)) for p in (session.call("list_plan") or [])]
    return (max(ids) + 1) if ids else 1


def replace_plan(session: AsiairSession, plan: dict) -> dict:
    """Stylistic7's workflow (2026-10-06): the device holds ONE plan, ours.
    Delete whatever is there, write `plan` fresh, read back and verify.

    `delete_plan([{"id": N}])` removes EVERY plan on firmware 14.39, not
    just N — which is what we want here and why this must never be used
    on a rig whose plans are not ours. A fresh write has no stale targets
    to merge with (set_plan merges targets by id, so it cannot delete one),
    and the device enables a sole plan by itself; it still runs nothing
    until Start is pressed on the tablet (`is_plan_started`)."""
    before = session.call("get_plan") or []
    if any(p.get("is_plan_started") for p in before):
        raise RuntimeError("a plan is running on the device; not replacing it")
    if before:
        session.call("delete_plan", [{"id": before[0].get("id", 0)}])
        left = session.call("list_plan") or []
        if left:
            raise RuntimeError(f"delete_plan left {len(left)} plan(s) in place; nothing written")
    session.call("set_plan", [plan])
    after = session.call("get_plan") or []
    mine = next((p for p in after if p.get("id") == plan["id"]), None)
    # Compare name + targets only: the device flips `enable` on a sole plan.
    same = mine is not None and mine.get("plan_name") == plan["plan_name"] and _same(
        {"targets": _essentials(mine)["targets"]}, {"targets": _essentials(plan)["targets"]})
    if len(after) != 1 or not same:
        raise RuntimeError(f"read-back after replace does not match (plans on device: {len(after)})")
    return {"plan_id": plan["id"], "targets": len(plan["targets"]), "replaced": len(before),
            "enabled": bool(mine.get("enable")), "started": bool(mine.get("is_plan_started"))}
