"""The downloader must not touch sidecars of files it did not download.

Regression for 2026-09-09: `_save_fits_overlay` rewrote a `header_overlay`
section into EVERY candidate file's sidecar on every sweep -- values recomputed
from each file's own header, so byte-identical every time, and read by nothing
in the codebase. It got the "downloaded" list from `_copy_fits_files`, which
returned all CANDIDATES restated with the destination prefix rather than what
rsync actually copied (`--ignore-existing` skips existing files), so that was
171 sidecar mtimes bumped per sweep. Each bump changed the directory's library
fingerprint, re-keyed `identify_stacks` / `identify_calibration_stacks` /
`identify_targets`, and cost a full graph traversal -- 12,545 node executions --
which then triggered the next sweep. The graph never settled.
"""
import json
from pathlib import Path

import pytest

import asiair.download_asiair as dl


def _sidecar(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".json")


def test_download_does_not_rewrite_sidecars(tmp_path):
    """The real rsync, against a destination that already has a sidecar."""
    src, dest = tmp_path / "card", tmp_path / "raw"
    src.mkdir(); dest.mkdir()
    frame = src / "Light_1.fit"
    frame.write_bytes(b"not really a fits file")
    side = _sidecar(dest / "Light_1.fit")
    side.write_text(json.dumps({"astroflow": {"image_metadata": {"stays": "put"}}}))
    before = side.stat().st_mtime_ns

    assert dl._copy_fits_files([str(frame)], str(dest))

    assert (dest / "Light_1.fit").read_bytes() == frame.read_bytes()
    assert side.stat().st_mtime_ns == before, "downloader touched an existing sidecar"
    assert json.loads(side.read_text())["astroflow"] == {"image_metadata": {"stays": "put"}}


def test_copy_reports_success_not_a_file_list(tmp_path):
    """The candidate list is not the copied list — `--ignore-existing` skips
    files already at the destination, and only rsync knows which — so the
    copier reports success and nothing else. Anything that wants to know what
    landed reads the disk. Here the destination already holds the frame, so
    NOTHING is copied, and the result is still a plain True."""
    src, dest = tmp_path / "card", tmp_path / "raw"
    src.mkdir(); dest.mkdir()
    frame = src / "Light_1.fit"
    frame.write_bytes(b"fresh")
    (dest / "Light_1.fit").write_bytes(b"already here")

    assert dl._copy_fits_files([str(frame)], str(dest)) is True
    assert (dest / "Light_1.fit").read_bytes() == b"already here"
    assert dl._copy_fits_files([], str(dest)) is True


def test_no_header_overlay_is_written_anywhere():
    """`header_overlay` was write-only -- nothing ever read it. If it comes
    back, it needs a reader and a reason."""
    import inspect
    assert "header_overlay" not in inspect.getsource(dl)
    assert not hasattr(dl, "_save_fits_overlay")


def test_search_defs_cannot_be_passed_positionally():
    """`search_defs` sat fourth positionally, behind `default_filter: str`.
    When `default_filter` was deleted with the header-overlay removal, the
    scheduler's call site kept passing its `"H"` — which slid silently into
    `search_defs`, iterated as the single character "H", and died on
    `search_def["volume"]`. Every ASIAir download failed; and because the
    poller `ray.get`s those refs BEFORE its drain-and-restart, the compute
    graph never got kicked either (boot 67, 2026-09-09).

    Keyword-only turns a stray positional into a loud TypeError on the
    driver instead of a string that happens to be iterable.
    """
    import inspect

    sig = inspect.signature(dl.download_from_asiair._function)
    assert sig.parameters["search_defs"].kind is inspect.Parameter.KEYWORD_ONLY

    sig.bind("rig", "10.0.0.28", "/dest")                    # the real call
    with pytest.raises(TypeError):
        sig.bind("rig", "10.0.0.28", "/dest", "H")           # the bug


def test_the_download_poller_calls_the_downloader_correctly():
    """Pin the actual call site, not just the signature — the signature was
    fine on its own; it was the caller that went stale."""
    import ast
    import inspect

    from state import scheduler_actor

    src = inspect.getsource(scheduler_actor)
    calls = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "remote"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "download_from_asiair"
    ]
    assert calls, "no download_from_asiair.remote(...) call found"
    for c in calls:
        assert len(c.args) == 3, (
            f"download_from_asiair.remote takes 3 positional args "
            f"(rig, ip, dest_path); call site passes {len(c.args)}"
        )
