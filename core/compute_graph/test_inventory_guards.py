"""The inventory-collapse guard — 2026-08-13.

Identification is derived from the library, which is derived from
sidecars, so an upstream break shows up here as a much smaller object set
that then silently skips cart items and empties viewer pages. That day it
published 228 stacks / 56 targets / 3 mosaics instead of ~2000 / 416 / 19
and NOTHING said so; it surfaced days later, by accident, while answering
an unrelated question.

This guard only REFUSES TO PUBLISH a collapsed derived view — it never
deletes anything. (The dangerous half of that incident, a `changed` delta
that deleted raw-frame sidecars, was removed outright rather than
guarded: see metadata/test_image_metadata.py.)
"""
import os

import pytest

from compute_graph.compute_graph import _guard_collapse


class TestInventoryCollapseGuard:
    def test_publishing_a_collapsed_inventory_raises(self):
        previous = {f"h{i}": object() for i in range(400)}
        with pytest.raises(RuntimeError, match="inventory collapse|refusing"):
            _guard_collapse("targets", previous, 56)

    def test_a_normal_pass_is_untouched(self):
        previous = {f"h{i}": object() for i in range(400)}
        _guard_collapse("targets", previous, 400)      # unchanged
        _guard_collapse("targets", previous, 412)      # grew
        _guard_collapse("targets", previous, 260)      # normal fluctuation

    def test_small_libraries_are_exempt(self):
        """A 12-object library legitimately swings; the floor keeps the
        guard from firing on a fresh install or a test fixture."""
        _guard_collapse("mosaics", {f"h{i}": object() for i in range(12)}, 1)

    def test_first_ever_publish_is_allowed(self):
        _guard_collapse("stacks", None, 0)
        _guard_collapse("stacks", {}, 1500)

    def test_env_override_permits_a_deliberate_drop(self, monkeypatch):
        previous = {f"h{i}": object() for i in range(400)}
        monkeypatch.setenv("ASTROFLOW_ALLOW_INVENTORY_COLLAPSE", "1")
        _guard_collapse("targets", previous, 3)        # must not raise
