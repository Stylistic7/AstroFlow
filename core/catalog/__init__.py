"""The catalog: query the lazy product of everything AstroFlow can show.

`(object × pipeline × tap × filter) → Product → path → status`, all pure reads
over the pipeline registry, the trace, and disk (`catalog.queries`); and the
user-facing view axis, finishes bound to those taps (`catalog.finishes`).
"""
from products.product import Product, object_class, url_type_of
from catalog.queries import (
    predict_path,
    master_path,
    product_taps,
    resolve_product,
    viewable_from_master,
    pipelines_for,
    filters_for,
    resolve_entity,
)
from catalog.finishes import (
    BoundFinish,
    BoundInput,
    bind_finish,
    resolve_finish,
    finishes_for,
    default_finish_for,
    finishes_matrix,
    band_roles,
    stars_inputs,
    role_of,
)

__all__ = [
    "Product",
    "object_class",
    "url_type_of",
    "predict_path",
    "master_path",
    "product_taps",
    "resolve_product",
    "viewable_from_master",
    "pipelines_for",
    "filters_for",
    "resolve_entity",
    "BoundFinish",
    "BoundInput",
    "bind_finish",
    "resolve_finish",
    "finishes_for",
    "default_finish_for",
    "finishes_matrix",
    "band_roles",
    "stars_inputs",
    "role_of",
]
