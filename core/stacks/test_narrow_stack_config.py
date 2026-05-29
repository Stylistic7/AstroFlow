from dataclasses import replace

from hashing import make_hashable
from decisions.settings import ProcessingConfig, DevelopmentSettings
from stacks.stacking.pixinsight import narrow_stack_config, _StackKey


class _StubStack:
    stable_hash = "abc123"


def test_narrow_excludes_edge_erosion_and_preserves_surface():
    """Adding/changing edge_erosion_px must not change stacking's cache key
    (no re-stack), and the projected surface must be exactly the legacy four
    fields in declaration order (so the key is byte-identical to the old
    narrowed ProcessingConfig)."""
    stub = _StubStack()
    cfg = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        min_frames_in_stack=7,
        edge_erosion_px=10,
    )
    cfg_other = replace(cfg, edge_erosion_px=99)

    assert make_hashable(narrow_stack_config(stub, cfg)) == \
        make_hashable(narrow_stack_config(stub, cfg_other))

    names = [pair[0] for pair in make_hashable(narrow_stack_config(stub, cfg))]
    assert names == [
        "calibration_stacks", "development_settings",
        "min_frames_in_stack", "decisions",
    ]
    assert "edge_erosion_px" not in names


def test_narrow_returns_stackkey():
    key = narrow_stack_config(_StubStack(), ProcessingConfig())
    assert isinstance(key, _StackKey)
    assert key.calibration_stacks is None
