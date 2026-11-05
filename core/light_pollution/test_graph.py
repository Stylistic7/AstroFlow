"""The site fold's graph: one measurement per run, one per (stack, site), one fit
per site, one table. Runs are the measurement test's synthetic frames."""
import datetime
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace

from dependency_graph.classes import canonicalize_graph
from light_pollution.graph import site_graph
from light_pollution.measure import MIN_FRAMES
from light_pollution.model import SiteModels
from light_pollution.test_measure import T0, _run

SF, OAKLAND = (37.75, -122.42), (37.78, -122.20)


@dataclass(frozen=True)
class _Run:
    """A node argument, so hashable like a Run: by its content hash."""
    content_hash: str
    metadata: dict = field(hash=False, compare=False)
    oldest_date: datetime.datetime = field(hash=False, compare=False)
    dependencies: None = None


def _light_run(tmp_path, name, position, first):
    run = _run(tmp_path, [(0.0, 0.0)] * MIN_FRAMES, name=name, first=first, position=position)
    return _Run(run.content_hash, run.metadata, T0 + datetime.timedelta(minutes=5 * first))


def _stack(stable, *runs):
    return SimpleNamespace(image_type="light", stable_hash=stable, contents={r.content_hash: r for r in runs})


def test_a_field_shot_from_two_sites_is_measured_per_site(tmp_path):
    """The same stack's runs at two sites must not share one stack node: the second site's
    dependency list would replace the first's and strand the first site's run measurements."""
    stacks = [_stack("stack01", _light_run(tmp_path, "runA", SF, 0), _light_run(tmp_path, "runB", OAKLAND, 10)),
              _stack("stack02", _light_run(tmp_path, "runC", SF, 20))]
    graph = canonicalize_graph(site_graph(stacks, {}, {}, SiteModels.empty()))
    depended = {d for deps in graph.values() for d in deps}
    terminal = [n for n in graph if n not in depended]
    assert [str(n) for n in terminal] == ["Node(fn=merge_site_models)"]
    counts = Counter(n.task._function.__name__ for n in graph)
    assert counts["measure_run_tilts"] == 3 and counts["measure_stack_tilts"] == 3 and counts["fit_site"] == 2
    stack_nodes = [n for n in graph if n.task._function.__name__ == "measure_stack_tilts"]
    assert sorted(n.args[:2] for n in stack_nodes) == [("stack01", "37.75,-122.42"), ("stack01", "37.78,-122.20"),
                                                       ("stack02", "37.75,-122.42")]
    assert {len(graph[n]) for n in stack_nodes} == {1}                     # one run under each
