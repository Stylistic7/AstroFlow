from datetime import date

from suggestions.export import plain_text, telescopius_csv
from suggestions.types import ObservingPlan, PlanEntry, Suggestion, SuggestionKind


def _sug(target_id: str, filter_name: str, panel_label: str) -> Suggestion:
    return Suggestion(
        kind=SuggestionKind.MISSING_CHANNEL,
        target_id=target_id,
        mosaic_id="m1",
        ra_deg=10.0,
        dec_deg=20.0,
        filter=filter_name,
        current_exposure_s=0.0,
        target_exposure_s=3600.0,
        default_exposure_per_frame_s=300.0,
        default_frame_count=12,
        panel_label=panel_label,
        reason="r",
    )


def test_telescopius_csv_includes_panel_numbers():
    s_a = _sug("a", "H", "γ Cyg Nebula")
    s_b = _sug("b", "H", "NGC 7000")
    plan = ObservingPlan(
        for_date=date(2026, 4, 26),
        entries=(
            PlanEntry(s_a.stable_key(), 300.0, 12),
            PlanEntry(s_b.stable_key(), 300.0, 12),
        ),
    )
    # Panel numbers come from the target's 1-indexed position in the
    # owning mosaic's target_ids — same numbering the polygon labels use.
    panel_numbers = {"a": 1, "b": 2}
    csv_out = telescopius_csv(plan, [s_a, s_b], panel_numbers)
    lines = [ln.rstrip("\r") for ln in csv_out.strip().split("\n")]
    assert lines[0] == "Pane,RA,Dec,Notes"
    assert lines[1].startswith("Panel 1: γ Cyg Nebula,")
    assert lines[2].startswith("Panel 2: NGC 7000,")


def test_plain_text_includes_panel_numbers():
    s_a = _sug("a", "H", "γ Cyg Nebula")
    s_b = _sug("b", "H", "NGC 7000")
    plan = ObservingPlan(
        for_date=date(2026, 4, 26),
        entries=(
            PlanEntry(s_a.stable_key(), 300.0, 12),
            PlanEntry(s_b.stable_key(), 300.0, 12),
        ),
    )
    panel_numbers = {"a": 1, "b": 2}
    out = plain_text(plan, [s_a, s_b], panel_numbers)
    assert "Panel 1 (γ Cyg Nebula)" in out
    assert "Panel 2 (NGC 7000)" in out


def test_telescopius_csv_falls_back_when_target_unknown():
    s_a = _sug("a", "H", "γ Cyg Nebula")
    plan = ObservingPlan(
        for_date=date(2026, 4, 26),
        entries=(PlanEntry(s_a.stable_key(), 300.0, 12),),
    )
    csv_out = telescopius_csv(plan, [s_a], panel_numbers={})
    lines = [ln.rstrip("\r") for ln in csv_out.strip().split("\n")]
    # No "Panel N:" prefix when the target_id isn't in the dict.
    assert lines[1].startswith("γ Cyg Nebula,")
