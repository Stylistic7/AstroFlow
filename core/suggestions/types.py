from dataclasses import dataclass, asdict
from datetime import date
from enum import StrEnum

from hashing import hash_list


class SuggestionKind(StrEnum):
    MISSING_CHANNEL = "missing_channel"
    COMPLETION = "completion"


@dataclass(frozen=True)
class Suggestion:
    kind: SuggestionKind
    target_id: str
    mosaic_id: str
    ra_deg: float
    dec_deg: float
    filter: str
    current_exposure_s: float
    target_exposure_s: float
    default_exposure_per_frame_s: float
    default_frame_count: int
    panel_label: str
    reason: str

    def stable_key(self) -> str:
        return hash_list([self.mosaic_id, self.target_id, self.filter])

    def to_dict(self) -> dict:
        return {**asdict(self), "kind": str(self.kind), "stable_key": self.stable_key()}


@dataclass(frozen=True)
class PlanEntry:
    suggestion_key: str
    exposure_per_frame_s: float
    frame_count: int


@dataclass(frozen=True)
class ObservingPlan:
    for_date: date
    entries: tuple[PlanEntry, ...]
