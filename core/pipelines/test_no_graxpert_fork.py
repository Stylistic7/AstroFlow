"""`Hybrid: Siril + OSS Mosaic (no GraXpert)`: the A/B fork of the Siril
pipeline that runs GraXpert nowhere. Every master it builds is calibrated
with background extraction off; the GraXpert pipeline it forks keeps it on."""
from decisions.settings import ProcessingConfig
from pipelines.registry import discover_pipelines, registered_pipeline_names, registered_pipelines
from stacks.composite_stack import Stack


def _for_stack(name):
    discover_pipelines()
    fn = registered_pipelines()[(name, Stack)]
    return getattr(fn, "_function", fn)


def test_the_fork_is_offered_beside_the_pipeline_it_forks():
    names = registered_pipeline_names()
    assert "Hybrid: Siril + OSS Mosaic (no GraXpert)" in names
    assert "Hybrid: Siril + GraXpert + OSS Mosaic" in names


def test_only_the_fork_turns_background_extraction_off(monkeypatch):
    import pipelines.hybrid_siril_graxpert_oss_mosaic as siril
    seen = []
    monkeypatch.setattr(siril, "foreign_post_processed_master", lambda s, c, op: seen.append(c))
    monkeypatch.setattr(siril, "is_foreign", lambda s: True)
    for name in ("Hybrid: Siril + OSS Mosaic (no GraXpert)", "Hybrid: Siril + GraXpert + OSS Mosaic"):
        _for_stack(name)(object(), ProcessingConfig())
    assert [c.background_extraction for c in seen] == [False, True]
