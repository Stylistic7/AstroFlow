"""run_graph admits a ready node only while Ray has room for the *custom*
resources it declares (read via `available_resources()`), so a scarce
serialized resource — e.g. the single PixInsight slot — can't accumulate a
deep FIFO backlog inside Ray that defeats the `node_sort_key` priority sort.

These are Ray-free: they pin the resource-name-agnostic plumbing (how an op's
declared resources are read across the three op flavors) and the fit check.
The end-to-end admit/retain ordering is exercised live; here we lock the
contract the loop depends on.
"""
from decorators import node
from dependency_graph.dependency_graph import _declared_custom_resources, _fits


class _CachedTask:
    """Mimics a @node / @cached.remote wrapper: resources on `_ray_resources`."""
    def __init__(self, resources):
        self._ray_resources = resources


class _RayRemoteFn:
    """Mimics a raw @ray.remote RemoteFunction: resources on `_resources`."""
    def __init__(self, resources):
        self._resources = resources


class _Bare:
    """An op that declares no resources at all."""


# --- resource reading is agnostic to op flavor and resource name -------------

def test_reads_node_or_cached_wrapper():
    assert _declared_custom_resources(_CachedTask({"PixInsight": 1})) == {"PixInsight": 1}
    # A different custom resource is read identically — no name is special-cased.
    assert _declared_custom_resources(_CachedTask({"Siril": 2})) == {"Siril": 2}


def test_falls_back_to_raw_ray_remote():
    assert _declared_custom_resources(_RayRemoteFn({"PixInsight": 1})) == {"PixInsight": 1}


def test_defaults_to_empty():
    assert _declared_custom_resources(_Bare()) == {}
    assert _declared_custom_resources(_CachedTask(None)) == {}
    assert _declared_custom_resources(_CachedTask({})) == {}


# --- the @node -> cached.remote wiring actually exposes the resources --------

def test_node_decorator_exposes_declared_resources():
    @node(resources={"PixInsight": 1})
    def _op_with_resource(x): ...

    @node
    def _bare_op(x): ...

    # `@node` is built on `cached.remote`, so the exposure works for both.
    assert _op_with_resource._ray_resources == {"PixInsight": 1}
    assert _bare_op._ray_resources == {}
    assert _declared_custom_resources(_op_with_resource) == {"PixInsight": 1}
    assert _declared_custom_resources(_bare_op) == {}


# --- the fit check used for admission ----------------------------------------

def test_fits_only_when_every_resource_is_available():
    assert _fits({"PixInsight": 1}, {"PixInsight": 1, "CPU": 8})
    assert not _fits({"PixInsight": 1}, {"PixInsight": 0, "CPU": 8})
    # A node with no custom demand always fits (the common case — never gated).
    assert _fits({}, {"PixInsight": 0})
    # Absent resource reads as zero available.
    assert not _fits({"PixInsight": 1}, {})
    # Fractional capacity (e.g. head-pinned diskcache:0.001) — plenty of room.
    assert _fits({"diskcache": 0.001}, {"diskcache": 1.0})
