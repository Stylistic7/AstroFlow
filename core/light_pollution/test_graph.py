"""The site fold's graph: one measurement per run, one fit per site over its runs in date order,
one table. Runs are the measurement test's synthetic frames."""
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


def test_one_measurement_per_run_and_one_fit_per_site_over_its_runs_in_date_order(tmp_path):
    stacks = [_stack("stack01", _light_run(tmp_path, "runC", SF, 20), _light_run(tmp_path, "runB", OAKLAND, 10)),
              _stack("stack02", _light_run(tmp_path, "runA", SF, 0))]
    graph = canonicalize_graph(site_graph(stacks, {}, {}, SiteModels.empty()))
    depended = {d for deps in graph.values() for d in deps}
    assert [str(n) for n in graph if n not in depended] == ["Node(fn=merge_site_models)"]
    counts = Counter(n.task._function.__name__ for n in graph)
    assert counts["measure_run_tilts"] == 3 and counts["fit_site"] == 2
    fits = {n.args[0].key: [d.args[0].content_hash for d in graph[n]]
            for n in graph if n.task._function.__name__ == "fit_site"}
    assert fits == {"37.75,-122.42": ["runA", "runC"], "37.78,-122.20": ["runB"]}
