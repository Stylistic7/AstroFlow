"""Binding finishes to an object's taps — the user-facing view axis.

A finish (`finishes.framework.Finish`) declares its inputs as tap display
names; the catalog binds each to the tap entry the pipeline's trace filed for
this object (`queries.product_taps`) and asks disk what is servable
(`queries.resolve_product`). A per-filter tap (a stack's "Post-Processed",
a mosaic's "Mosaic Starless") binds the entry whose filter ROLE
(`pix.tasks.filters.normalize_filter`) is the finish's role; a joined tap
("Linear SHO", filter "RGB") carries every band and binds any role.
Registration order is preference: the first registration of a name whose
required inputs are servable is the one shown. Pure reads, never a trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from catalog import queries
from finishes.registry import all_finishes
from pix.tasks.filters import normalize_filter
from products.product import Product

ABSENT, OUT_OF_DATE, LIVE = "absent", "out_of_date", "live"
_WORST = {ABSENT: 2, OUT_OF_DATE: 1, LIVE: 0}


def role_of(filter) -> str:
    """The filter role a tap entry carries: "H", "S", …, or "none" for a
    filterless master (a dark) and for the joined sentinels ("RGB")."""
    return normalize_filter(str(filter)) or "none"


def roles_of(object_type: str, object_id: str) -> set:
    return {role_of(f) for f in queries.filters_for(object_type, object_id)}


@dataclass(frozen=True)
class BoundInput:
    index: int
    tap: str
    filter: "str | None"     # the bound entry's filter; None when unbound
    path: "Path | None"      # servable path (pyramid or master); None when absent
    status: str              # live | out_of_date | absent
    optional: bool

    def product(self, b: "BoundFinish") -> "Product | None":
        if self.filter is None:
            return None
        return Product(b.object_type, b.object_id, b.pipeline, self.tap, self.filter)


@dataclass(frozen=True)
class BoundFinish:
    object_type: str
    object_id: str
    pipeline: str
    finish: object           # finishes.framework.Finish
    inputs: tuple            # BoundInput per declared input, in order
    status: str              # worst status over the REQUIRED inputs

    @property
    def name(self) -> str:
        return self.finish.name

    @property
    def role(self) -> "str | None":
        return self.finish.role

    @property
    def mtime(self) -> float:
        return max((p.stat().st_mtime for p in (i.path for i in self.inputs)
                    if p is not None and p.exists()), default=0.0)


def _matches(role: "str | None", entry_filter) -> bool:
    """Whether a finish of `role` may read a tap entry of this filter."""
    if role is None:
        return True                      # a composite binds by tap name alone
    r = role_of(entry_filter)
    return r == role or (r == "none" and role != "none")   # a join has every band


def _bind_input(taps: list, tap: str, role: "str | None", index: int,
                optional: bool, b_args: tuple) -> "BoundInput | None":
    """Bind one declared input; None when the tap is traced ambiguously or
    absent and required."""
    entries = [e for e in taps if e["display"] == tap and _matches(role, e["filter"])]
    if len(entries) != 1:
        if entries or not optional:
            return None
        return BoundInput(index, tap, None, None, ABSENT, optional)
    e = entries[0]
    path, status = queries.resolve_product(Product(*b_args, tap, e["filter"]))
    return BoundInput(index, tap, e["filter"], path, status, optional)


def bind_finish(object_type: str, object_id: str, pipeline: str, finish,
                taps: "list | None" = None, roles: "set | None" = None) -> "BoundFinish | None":
    """Bind one registration to this object's traced taps, or None when a
    required tap is missing (or ambiguous) in the trace."""
    if finish.role is not None:
        if roles is None:
            roles = roles_of(object_type, object_id)
        if finish.role not in roles:
            return None
    if taps is None:
        taps = queries.product_taps(object_type, object_id, pipeline)
    b_args = (object_type, object_id, pipeline)
    inputs = []
    for i, tap in enumerate(finish.inputs):
        bound = _bind_input(taps, tap, finish.role, i, tap in finish.optional, b_args)
        if bound is None:
            return None
        inputs.append(bound)
    status = max((i.status for i in inputs if not i.optional),
                 key=_WORST.__getitem__, default=ABSENT)
    return BoundFinish(object_type, object_id, pipeline, finish, tuple(inputs), status)


def _resolve(object_type, object_id, pipeline, name, taps, roles) -> "BoundFinish | None":
    """The first registration of `name` whose required inputs are servable;
    else the first that is at least traced (status absent)."""
    traced = None
    for f in all_finishes():
        if f.name != name:
            continue
        b = bind_finish(object_type, object_id, pipeline, f, taps, roles)
        if b is None:
            continue
        if b.status != ABSENT:
            return b
        traced = traced or b
    return traced


def resolve_finish(object_type: str, object_id: str, pipeline: str,
                   name: str) -> "BoundFinish | None":
    taps = queries.product_taps(object_type, object_id, pipeline)
    if not taps:
        return None
    return _resolve(object_type, object_id, pipeline, name, taps,
                    roles_of(object_type, object_id))


def finishes_for(object_type: str, object_id: str, pipeline: str) -> list:
    """Every finish this pipeline can show for the object, one per name, in
    registration order (= preference), traced-but-unbuilt ones included with
    status `absent`."""
    taps = queries.product_taps(object_type, object_id, pipeline)
    if not taps:
        return []
    roles = roles_of(object_type, object_id)
    out, seen = [], set()
    for f in all_finishes():
        if f.name in seen:
            continue
        seen.add(f.name)
        b = _resolve(object_type, object_id, pipeline, f.name, taps, roles)
        if b is not None:
            out.append(b)
    return out


def default_finish_for(object_type: str, object_id: str) -> "BoundFinish | None":
    """The view for a bare object URL: the first finish, in registration
    order, that some pipeline (by `_pipeline_rank`) has built."""
    pipelines = sorted(queries.pipelines_for(object_type, object_id),
                       key=queries._pipeline_rank)
    if not pipelines:
        return None
    roles = roles_of(object_type, object_id)
    taps = {p: queries.product_taps(object_type, object_id, p) for p in pipelines}
    seen = set()
    for f in all_finishes():
        if f.name in seen:
            continue
        seen.add(f.name)
        for p in pipelines:
            b = _resolve(object_type, object_id, p, f.name, taps[p], roles)
            if b is not None and b.status != ABSENT:
                return b
    return None


def band_roles(bands: int, tags: dict, filter) -> list:
    """The role carried by each band of a bound input's payload: a mono tap
    is its filter's role; a narrowband join is the zero-filled S,H,O
    layout; any other 3-band payload is R,G,B."""
    if bands == 1:
        return [role_of(filter)]
    if tags.get("COLORSPC") == "linear-sho" or "BANDS" in tags:
        return ["S", "H", "O"]
    return ["R", "G", "B"]


def stars_inputs(b: BoundFinish, tags_of) -> list:
    """The bound inputs that are STARS layers (payload `LAYER == "stars"`),
    whose samples composite into the primary's seed statistics — the
    autostretch must see the picture the finish screens them into."""
    return [i for i in b.inputs[1:]
            if i.path is not None and tags_of(i.path).get("LAYER") == "stars"]


def masters_for(target, pipeline: str) -> list:
    """The stacked masters `pipeline` made for this target — each stack with
    a `display="Master"` tap in that pipeline — in filter order. Each names
    the stack finish the stack viewer opens it in (its channel finish, which
    reads the pipeline's master), whose status and mtime it reports, so the
    target page links every master to the pipeline that made it."""
    out = []
    for sid in target.stack_ids:
        master = next((e for e in queries.product_taps("stack", sid, pipeline)
                       if e["display"] == "Master"), None)
        if master is None:
            continue
        bound = finishes_for("stack", sid, pipeline)
        out.append({"stack_id": sid, "filter": master["filter"],
                    "finish": bound[0].name if bound else None,
                    "status": bound[0].status if bound else "absent",
                    "mtime": bound[0].mtime if bound else None})
    return sorted(out, key=lambda m: queries._filter_sort_key(str(m["filter"])))


def finishes_matrix(object_type: str, object_id: str) -> "dict | None":
    """`/api/finishes` and the MCP `finishes_for_object` tool: every pipeline
    that can show the object with its finishes, plus the default view; None
    for an object the identification does not know. A pipeline with neither
    a finish nor (for a target) a master is omitted. A target's pipelines
    also list the stacked masters each made (`masters_for`). Snapshot-only:
    never traces."""
    entity = queries.resolve_entity(object_type, object_id)
    if entity is None:
        return None
    names = queries.pipelines_for(object_type, object_id)
    if object_type == "target":
        # A pipeline that only stacked this target's frames still made masters.
        names = sorted(set(names).union(*(queries.pipelines_for("stack", sid) for sid in entity.stack_ids)))
    pipelines = []
    for name in names:
        bound = finishes_for(object_type, object_id, name)
        masters = masters_for(entity, name) if object_type == "target" else []
        if not bound and not masters:
            continue     # traced taps no finish reads: nothing to pick
        pipelines.append({"name": name, "finishes": [
            {"name": b.name, "role": b.role, "status": b.status, "mtime": b.mtime,
             "inputs": [{"tap": i.tap, "filter": i.filter, "status": i.status}
                        for i in b.inputs]}
            for b in bound
        ], **({"masters": masters} if object_type == "target" else {})})
    default = default_finish_for(object_type, object_id)
    return {"pipelines": pipelines,
            "default": ({"pipeline": default.pipeline, "finish": default.name}
                        if default else None)}
