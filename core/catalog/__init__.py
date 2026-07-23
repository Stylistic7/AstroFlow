"""The catalog: query the lazy product of everything AstroFlow can show.

`(object × pipeline × tap × filter) → Product → path → exists?`, all pure reads
over the pipeline registry, the trace, and disk. See `catalog.queries`.
"""
from products.product import Product, object_class
from catalog.queries import (
    products_for,
    existing_for,
    default_product_for,
    default_pipeline_for,
    predict_path,
    viewable_path,
    viewable_from_master,
    exists,
    pipelines_for,
    filters_for,
    resolve_entity,
    default_display_for,
)

__all__ = [
    "Product",
    "object_class",
    "products_for",
    "existing_for",
    "default_product_for",
    "default_pipeline_for",
    "predict_path",
    "viewable_path",
    "viewable_from_master",
    "exists",
    "pipelines_for",
    "filters_for",
    "resolve_entity",
    "default_display_for",
]
