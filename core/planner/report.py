"""A plan as text a human can check in a minute."""
from __future__ import annotations

from planner.model import RigPlan
from planner.sky import SLOT_MIN, Night

SUB_OVERHEAD_S = 15.0     # download + dither per sub (estimate; measure from logs)
VISIT_OVERHEAD_S = 180.0  # slew, plate solve, guiding restart per visit


def subs(block_slots: int, exposure_s: float) -> int:
    """Whole subs that fit a visit of `block_slots` slots."""
    return max(1, int((block_slots * SLOT_MIN * 60 - VISIT_OVERHEAD_S) // (exposure_s + SUB_OVERHEAD_S)))


def render(night: Night, plan: RigPlan, utc_offset_h: float) -> str:
    r = plan.rig
    lines = [f"{r.name} · {r.focal_mm:.0f} mm  ({r.image_scale:.2f}\"/px)  "
             f"dark {night.local(0, utc_offset_h)}–{night.local(night.n, utc_offset_h)}  "
             f"meridian flip: {night.local(plan.flip_slot, utc_offset_h) if plan.flip_slot is not None else 'none'}"]
    for b in plan.blocks:
        c = b.candidate
        hours = (b.end_slot - b.start_slot) / 6
        n = subs(b.end_slot - b.start_slot, c.exposure_s)
        lines.append(f"  {night.local(b.start_slot, utc_offset_h)}–{night.local(b.end_slot, utc_offset_h)}  "
                     f"{c.label:<24} {c.filter}  {n:3d} × {c.exposure_s:.0f} s  {hours:4.1f} h  "
                     f"{b.side:<4}  value {b.value:5.2f}   [{c.hours:.1f} h so far; {c.why}; urgency {c.urgency:.1f}]")
    lines.append(f"  total value {plan.score:.2f} sky-hours")
    return "\n".join(lines)
