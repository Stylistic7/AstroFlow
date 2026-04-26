"""Export an observing plan to Telescopius CSV, rich JSON, or plain text."""
import csv
import io
import json
from collections.abc import Iterable
from datetime import date

from suggestions.types import ObservingPlan, PlanEntry, Suggestion


def _hms(ra_deg: float) -> str:
    ra_h = (ra_deg % 360.0) / 15.0
    h = int(ra_h)
    m_float = (ra_h - h) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _dms(dec_deg: float) -> str:
    sign = "-" if dec_deg < 0 else "+"
    dec_abs = abs(dec_deg)
    d = int(dec_abs)
    m_float = (dec_abs - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return f"{sign}{d:02d}:{m:02d}:{s:04.1f}"


def _match(
    plan: ObservingPlan, suggestions: Iterable[Suggestion]
) -> list[tuple[PlanEntry, Suggestion]]:
    by_key = {s.stable_key(): s for s in suggestions}
    out = []
    for entry in plan.entries:
        if entry.suggestion_key in by_key:
            out.append((entry, by_key[entry.suggestion_key]))
    return out


def telescopius_csv(
    plan: ObservingPlan,
    suggestions: Iterable[Suggestion],
    panel_numbers: dict[str, int],
) -> str:
    """Telescopius target-list CSV: Pane, RA, Dec, Notes.

    `panel_numbers` maps target_id → 1-indexed panel number — same
    numbering as the polygon labels in the mosaic viewer, so the user
    can read "Panel 5" off this CSV and find polygon "5" in the image.
    """
    pairs = _match(plan, suggestions)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Pane", "RA", "Dec", "Notes"])
    for entry, s in pairs:
        n = panel_numbers.get(s.target_id)
        pane = f"Panel {n}: {s.panel_label}" if n is not None else s.panel_label
        notes = (
            f"{s.filter} — {entry.frame_count} x "
            f"{int(entry.exposure_per_frame_s)}s — {s.reason}"
        )
        writer.writerow([pane, _hms(s.ra_deg), _dms(s.dec_deg), notes])
    return buf.getvalue()


def rich_json(
    plan: ObservingPlan,
    suggestions: Iterable[Suggestion],
    panel_numbers: dict[str, int],
) -> str:
    pairs = _match(plan, suggestions)
    payload = {
        "for_date": plan.for_date.isoformat(),
        "entries": [
            {
                "panel_number": panel_numbers.get(s.target_id),
                "suggestion": s.to_dict(),
                "ra_hms": _hms(s.ra_deg),
                "dec_dms": _dms(s.dec_deg),
                "exposure_per_frame_s": entry.exposure_per_frame_s,
                "frame_count": entry.frame_count,
                "total_exposure_s": entry.exposure_per_frame_s * entry.frame_count,
            }
            for entry, s in pairs
        ],
    }
    return json.dumps(payload, indent=2)


def plain_text(
    plan: ObservingPlan,
    suggestions: Iterable[Suggestion],
    panel_numbers: dict[str, int],
) -> str:
    pairs = _match(plan, suggestions)
    lines = [f"Observing plan for {plan.for_date.isoformat()}", ""]
    for entry, s in pairs:
        n = panel_numbers.get(s.target_id)
        prefix = f"Panel {n} ({s.panel_label})" if n is not None else s.panel_label
        lines.append(
            f"- {prefix} — {entry.frame_count} x "
            f"{int(entry.exposure_per_frame_s)}s @ {s.filter} — "
            f"RA {_hms(s.ra_deg)}, Dec {_dms(s.dec_deg)} "
            f"({s.reason})"
        )
    return "\n".join(lines) + "\n"
