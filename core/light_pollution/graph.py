"""The library-wide fold, scheduled beside the dust epochs
(`compute_graph.scatter_spatial_groups`), after the night-sky flats have
joined the calibration stacks: one `block_flat` per flat stack a run resolves
to, one `measure_run_tilts` per light run that states where it was shot (its
flat as a value dependency: the flat — the etalon correction — is applied
before anything about the glow is measured), one `fit_site` per site over its
runs in date order (a collector's dependencies are ORDERED: they are its
positional varargs), `merge_site_models` into one table, `store_site_models`
into the context, where the Siril integration and the Observing Sites panel
read it.

Each run's flat is resolved exactly as a pipeline resolves it
(`decisions.resolve.resolve_calibration`, decision overrides included), so the
measurement sees the flat the integration will divide by.

The fit receives the site's previously stored model and keeps it unless the
refit differs materially (`model.fit_site_model`): the one deliberately
path-dependent fold here, so that a night's new runs do not re-key every
integration that used the model.
"""
import inspect
from collections import Counter, defaultdict

import ray
import structlog

import light_pollution.measure as measure_module
import light_pollution.model as model_module
import light_pollution.robust as robust_module
import light_pollution.sites as sites_module
from clustering.quiesce import _utc
from context import context
from decisions.resolve import resolve_calibration
from decorators import node
from dependency_graph import Graph
from hashing import hash_string
from light_pollution.measure import MIN_FRAMES, RunTilts, block_flat, measure_run_tilts
from light_pollution.model import SiteModel, SiteModels, fit_site_model
from light_pollution.sites import Site, cluster_sites, site_of
from stacks.types import StackIdDependency

log = structlog.get_logger()

# A node's cache key hashes its own body, not the modules it calls. The science
# lives in those modules, so their source rides along as an argument (the
# `prepare_stack_for_ui._fn_hash` precedent in compute_graph): a change to the
# measurement re-measures, a change to the model refits — and nothing else does.
MEASURE_CODE = hash_string(inspect.getsource(measure_module) + inspect.getsource(robust_module) + inspect.getsource(sites_module))
MODEL_CODE = hash_string(inspect.getsource(model_module) + inspect.getsource(robust_module))


@node
def fit_site(site: Site, previous: SiteModel | None, code: str, *tilts: RunTilts) -> SiteModels:
    """One site's table: its report always, its model when the data support one.
    `code` is MODEL_CODE, in the key only."""
    model, report = fit_site_model(site, tilts, previous)
    log.info("site model fitted", site=site.key, model=model.content_hash if model else None,
             reason=report.get("reason"), runs=report["runs"], gate=report.get("gate"))
    return SiteModels.of({site.key: model} if model is not None else {}, (report,))


@node
def merge_site_models(*tables: SiteModels) -> SiteModels:
    models, report = {}, ()
    for t in tables:
        models.update(t.models)
        report += t.report
    return SiteModels.of(models, report)


def _flat_of(run, cal_stacks):
    dep = run.dependencies.get("flat") if run.dependencies else None
    if isinstance(dep, StackIdDependency):
        return cal_stacks.get(str(dep.stack_id))
    return None


def runs_by_site(stacks, cal_stacks, decisions) -> dict[Site, list]:
    """The light runs with at least MIN_FRAMES solved frames stating a
    position, by site, in date order, each with the flat stack it resolves to
    (or None)."""
    placed = []
    for stack in stacks:
        if getattr(stack, "image_type", "light") != "light":
            continue
        try:
            resolved = resolve_calibration(stack, cal_stacks, decisions)
        except Exception as e:   # a stack calibration cannot place still measures, flat-less
            log.warning("site runs: calibration unresolved", stack=str(stack), error=str(e))
            resolved = stack
        for run in resolved.contents.values():
            positions = [p for m in run.metadata.values()
                         if m.plate_solution is not None and (p := site_of(m.header)) is not None]
            if len(positions) >= MIN_FRAMES:
                placed.append((Counter(positions).most_common(1)[0][0], run, _flat_of(run, cal_stacks)))
    sites = cluster_sites(p for p, _, _ in placed)
    by_site = defaultdict(list)
    for p, run, flat in placed:
        by_site[sites[p]].append((run, flat))
    return {s: sorted(items, key=lambda rf: (_utc(rf[0].oldest_date), rf[0].content_hash)) for s, items in by_site.items()}


def site_graph(stacks, cal_stacks, decisions, previous: SiteModels) -> Graph:
    graph = Graph({})
    collectors = []
    groups = runs_by_site(stacks, cal_stacks, decisions)
    for site in sorted(groups, key=lambda s: s.key):
        measures = []
        for run, flat in groups[site]:
            measure = (measure_run_tilts, run, MEASURE_CODE)
            if flat is not None:
                flat_task = (block_flat, flat, MEASURE_CODE)
                graph[flat_task] = {}
                graph[measure] = [flat_task]          # ORDERED: the FlatBlock is the positional `*flat`
            else:
                graph[measure] = {}
            measures.append(measure)
        collector = (fit_site, site, previous.models.get(site.key), MODEL_CODE)
        graph[collector] = measures                   # ORDERED: the runs' measurements in date order
        collectors.append(collector)
    log.info("site runs scattered", sites=len(groups), runs=sum(len(r) for r in groups.values()),
             flats=len({f.content_hash for rs in groups.values() for _, f in rs if f is not None}))
    graph[merge_site_models] = collectors
    return graph


def _deref(values):
    return [v.get() if hasattr(v, "get") else v for v in values]


@ray.remote(num_cpus=0.1)
def scatter_site_runs(stacks: dict, cal_stacks: dict) -> Graph:
    """`stacks` is the identified light stacks, `cal_stacks` the calibration
    stacks with the night-sky flats merged in; values may be ContextRefs."""
    previous = context.get("site_models") or SiteModels.empty()
    cal = type(cal_stacks)(dict(zip(cal_stacks.keys(), _deref(cal_stacks.values()))))
    return site_graph(_deref(stacks.values()), cal, context.settings.decision_overrides, previous)


@ray.remote(num_cpus=0.1)
def store_site_models(table: SiteModels) -> SiteModels:
    context.site_models = table
    log.info("Stored site models", sites=len(table.report), models=len(table.models))
    return table
