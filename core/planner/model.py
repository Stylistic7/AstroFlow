"""The planner's vocabulary."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rig:
    key: str                  # the ASIAir serial
    name: str                 # "ASIAIR_RC71"
    focal_mm: float
    image_scale: float        # arcsec/px, derived from the focal length
    sensor: tuple[int, int]   # pixels; a rig shoots only targets on its own sensor


@dataclass
class Candidate:
    """One (panel, filter) the planner may spend time on."""
    target_id: str
    label: str                # "315e Panel 6" or the target's name
    mosaic_id: str | None
    filter: str
    ra_deg: float
    dec_deg: float
    hours: float              # integration so far
    tiers: list[list[float]]  # [[hours, weight], ...]: what each further hour is worth,
                              # spent in order (equalisation before deepening)
    why: str                  # one line: where the tiers came from
    exposure_s: float = 300.0 # one sub, from the panel's own stack (the push unit)
    kind: str = "lrgb"        # inventory.project_kind: "lrgb" or "narrowband" (R/G/B = star colour)
    rank: int = 1             # 0 for a starred cart entry (shot first wherever it is up), 1 for the rest

    @property
    def need_h(self) -> float:
        return sum(h for h, _ in self.tiers)

    def take(self, hours: float) -> float:
        """Spend `hours` of effective sky on this candidate; returns the value."""
        value = 0.0
        for tier in self.tiers:
            used = min(tier[0], hours)
            value += used * tier[1]
            tier[0] -= used
            hours -= used
        self.tiers = [t for t in self.tiers if t[0] > 1e-9]
        return value

    def worth(self, hours: float) -> float:
        """The value `take(hours)` would return, without spending."""
        value = 0.0
        for h, w in self.tiers:
            used = min(h, hours)
            value += used * w
            hours -= used
        return value


@dataclass
class Block:
    candidate: Candidate
    start_slot: int
    end_slot: int             # exclusive
    effective_h: float        # sky-hours after the efficiency discount
    value: float              # need closed × weight, minus costs, in sky-hours
    side: str                 # the pier side: "east" or "west" of the meridian at the visit's start


@dataclass
class RigPlan:
    rig: Rig
    flips: list[int] = field(default_factory=list)   # slots where the pier side changes
    blocks: list[Block] = field(default_factory=list)
    unassignable: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(b.value for b in self.blocks)
