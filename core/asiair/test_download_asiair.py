"""The downloader must not touch sidecars of files it did not download.

Regression for 2026-09-09: `_save_fits_overlay` rewrote a `header_overlay`
section into EVERY candidate file's sidecar on every sweep -- values recomputed
from each file's own header, so byte-identical every time, and read by nothing
in the codebase. Because `_copy_fits_files` returns all candidates rather than
what rsync actually copied (`--ignore-existing` skips existing files), that was
171 sidecar mtimes bumped per sweep. Each bump changed the directory's library
fingerprint, re-keyed `identify_stacks` / `identify_calibration_stacks` /
`identify_targets`, and cost a full graph traversal -- 12,545 node executions --
which then triggered the next sweep. The graph never settled.
"""
import json
from pathlib import Path

import asiair.download_asiair as dl


def _sidecar(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".json")


def test_download_does_not_rewrite_sidecars(tmp_path, monkeypatch):
    src, dest = tmp_path / "card", tmp_path / "raw"
    src.mkdir(); dest.mkdir()
    frame = src / "Light_1.fit"
    frame.write_bytes(b"not really a fits file")
    side = _sidecar(dest / "Light_1.fit")
    side.write_text(json.dumps({"astroflow": {"image_metadata": {"stays": "put"}}}))
    before = side.stat().st_mtime_ns

    monkeypatch.setattr(dl, "_copy_fits_files", lambda files, d: [str(dest / Path(f).name) for f in files])
    dl._download_group([str(frame)], str(dest))

    assert side.stat().st_mtime_ns == before, "downloader touched an existing sidecar"
    assert json.loads(side.read_text())["astroflow"] == {"image_metadata": {"stays": "put"}}


def test_no_header_overlay_is_written_anywhere():
    """`header_overlay` was write-only -- nothing ever read it. If it comes
    back, it needs a reader and a reason."""
    import inspect
    assert "header_overlay" not in inspect.getsource(dl)
    assert not hasattr(dl, "_save_fits_overlay")
