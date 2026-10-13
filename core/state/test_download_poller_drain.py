"""The download poller drains the compute graph whenever a batch of new
frames lands (2026-10-13; the dawn-only hold of 2026-09-15 is gone)."""
from state import scheduler_actor as sa


def test_any_new_frame_drains_the_graph():
    assert sa._drain_due(new_files=1)
    assert sa._drain_due(new_files=24)


def test_an_empty_poll_leaves_the_graph_alone():
    assert not sa._drain_due(new_files=0)
