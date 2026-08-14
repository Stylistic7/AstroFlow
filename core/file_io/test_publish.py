"""Atomic publication — a file at its final path is never observed partial.

The invariant everything downstream leans on: the cache treats an existing
output as satisfied, the tile server treats it as viewable, the sweep treats
it as live. A bare cross-filesystem copy breaks it by writing incrementally
into the final content-addressed path (the "BXT loop", and the shape of the
2026-08-14 pyramid incident).
"""
import os
import threading
from pathlib import Path

import pytest

from file_io.publish import publish_atomically, publish_pair_atomically


@pytest.fixture
def d(tmp_path):
    return tmp_path


class TestPublishAtomically:
    def test_publishes_content_and_leaves_no_staging(self, d):
        src = d / "src.bin"
        src.write_bytes(b"payload" * 100)
        dest = d / "store" / "abc123.bin"
        publish_atomically(src, dest)
        assert dest.read_bytes() == b"payload" * 100
        assert [p.name for p in dest.parent.iterdir()] == ["abc123.bin"], \
            "a staging file leaked into the content-addressed directory"

    def test_consume_drops_the_source_only_when_asked(self, d):
        src = d / "s.bin"
        src.write_bytes(b"x")
        publish_atomically(src, d / "a.bin")
        assert src.exists(), "default must not consume the source"
        publish_atomically(src, d / "b.bin", consume=True)
        assert not src.exists()

    def test_same_path_is_a_no_op_not_a_destruction(self, d):
        """Callers that sometimes produce in place must not lose the file."""
        p = d / "inplace.bin"
        p.write_bytes(b"keep me")
        publish_atomically(p, p)
        assert p.read_bytes() == b"keep me"

    def test_missing_source_raises_rather_than_publishing_nothing(self, d):
        with pytest.raises(FileNotFoundError):
            publish_atomically(d / "nope.bin", d / "out.bin")

    def test_failed_copy_leaves_no_staging_and_no_dest(self, d, monkeypatch):
        src = d / "s.bin"
        src.write_bytes(b"x")
        dest = d / "out" / "x.bin"

        import file_io.publish as pub

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(pub.shutil, "copy2", boom)
        with pytest.raises(OSError):
            publish_atomically(src, dest)
        assert not dest.exists(), "a failed publish must not leave a partial dest"
        assert not list(dest.parent.iterdir()), "staging leaked after failure"

    def test_a_reader_never_sees_a_partial_file(self, d):
        """The point of the exercise: while a large publish is in flight, the
        destination either does not exist or is complete — never truncated."""
        payload = b"\xab" * (4 * 1024 * 1024)
        src = d / "big.bin"
        src.write_bytes(payload)
        dest = d / "store" / "big.bin"
        dest.parent.mkdir()
        seen = []
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                if dest.exists():
                    seen.append(dest.stat().st_size)

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        publish_atomically(src, dest)
        stop.set()
        t.join(timeout=5)
        assert all(s == len(payload) for s in seen), \
            f"observed truncated sizes: {sorted(set(seen))[:5]}"


class TestPublishPair:
    def test_payload_lands_last(self, d):
        """The payload gates visibility, so a crash between the two renames
        must leave an orphan sidecar — never a payload without its sidecar."""
        src, side = d / "p.fits", d / "p.fits.json"
        src.write_bytes(b"img")
        side.write_text("{}")
        dest, dside = d / "o" / "p.fits", d / "o" / "p.fits.json"

        order = []
        import file_io.publish as pub
        real = pub.os.replace

        def spy(a, b):
            order.append(Path(b).name)
            return real(a, b)

        pub.os.replace = spy
        try:
            publish_pair_atomically(src, dest, side, dside)
        finally:
            pub.os.replace = real
        assert order == ["p.fits.json", "p.fits"], \
            f"payload must land last, got {order}"

    def test_absent_sidecar_is_not_fatal(self, d):
        src = d / "p.fits"
        src.write_bytes(b"img")
        publish_pair_atomically(src, d / "o" / "p.fits",
                                d / "missing.json", d / "o" / "p.fits.json")
        assert (d / "o" / "p.fits").exists()
