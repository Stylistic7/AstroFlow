"""The library-wide fold, scheduled beside the dust epochs
(`compute_graph.scatter_spatial_groups`): one `measure_run_tilts` per light
run that states where it was shot, one `fit_site` per site (its runs in date
order — the collector's dependencies are ORDERED, they are its positional
`*tilts`), `merge_site_models` into one table, `store_site_models` into the
context, where the Siril integration and the Observing Sites panel read it.

The fit receives the site's previously stored model and keeps it unless the
refit differs materially (`model.fit_site_model`): the one deliberately
path-dependent fold here, so that a night's new runs do not re-key every
integration that used the model.
"""
from collections import Counter, defaultdict

import ray
import structlog

from clustering.quiesce import _utc
from context import context
from decorators import node
from dependency_graph import Graph
from light_pollution.measure import MIN_FRAMES, RunTilts, measure_run_tilts
from light_pollution.model import SiteModel, SiteModels, fit_site_model
from light_pollution.sites import Site, cluster_sites, site_of

log = structlog.get_logger()


@node
def fit_site(site: Site, previous: SiteModel | None, *tilts: RunTilts) -> SiteModels:
    """One site's table: its report always, its model when the data support one."""
    model, report = fit_site_model(site, [t for t in tilts if t.n_frames], previous)
    log.info("site model fitted", site=site.key, model=model.content_hash if model else None,
             reason=report.get("reason"), runs=report["runs"])
    return SiteModels.of({site.key: model} if model is not None else {}, (report,))


@node
def merge_site_models(*tables: SiteModels) -> SiteModels:
    models, report = {}, ()
    for t in tables:
        models.update(t.models)
        report += t.report
    return SiteModels.of(models, report)


def runs_by_site(stacks) -> dict[Site, list]:
    """The light runs with at least MIN_FRAMES solved frames stating a position, by site."""
    placed = []
    for stack in stacks:
        if getattr(stack, "image_type", "light") != "light":
            continue
        for run in stack.contents.values():
            positions = [p for m in run.metadata.values()
                         if m.plate_solution is not None and (p := site_of(m.header)) is not None]
            if len(positions) >= MIN_FRAMES:
                placed.append((Counter(positions).most_common(1)[0][0], run))
    sites = cluster_sites(p for p, _ in placed)
    by_site = defaultdict(list)
    for p, run in placed:
        by_site[sites[p]].append(run)
    return {s: sorted(runs, key=lambda r: (_utc(r.oldest_date), r.content_hash)) for s, runs in by_site.items()}


def site_graph(stacks, previous: SiteModels) -> Graph:
    graph = Graph({})
    collectors = []
    groups = runs_by_site(stacks)
    for site in sorted(groups, key=lambda s: s.key):
        measures = [(measure_run_tilts, r) for r in groups[site]]
        for m in measures:
            graph[m] = {}
        collector = (fit_site, site, previous.models.get(site.key))
        graph[collector] = measures
        collectors.append(collector)
    log.info("site runs scattered", sites=len(groups), runs=sum(len(r) for r in groups.values()))
    graph[merge_site_models] = collectors
    return graph


@ray.remote(num_cpus=0.1)
def scatter_site_runs(stacks: dict) -> Graph:
    """`stacks` is the identified light stacks, their values ContextRefs."""
    previous = context.get("site_models") or SiteModels.empty()
    return site_graph([v.get() if hasattr(v, "get") else v for v in stacks.values()], previous)


@ray.remote(num_cpus=0.1)
def store_site_models(table: SiteModels) -> SiteModels:
    context.site_models = table
    log.info("Stored site models", sites=len(table.report), models=len(table.models))
    return table
