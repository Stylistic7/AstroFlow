"""The canvas an image is drawn on, stamped into the image's own sidecar.

A canvas image (a placed panel, a merge, a polish, a colour composite) is
only meaningful with the `Viewbox` that maps its pixels to the sky. The web
used to look that viewbox up through a context pointer — a cache that went
stale whenever new frames re-measured the canvas before the image was
rebuilt, so the overlay was drawn on a different canvas (on 2026-10-15 the
75a1 polygons landed 180 deg rotated from the image). Stamped on the file,
the viewbox can't disagree with the pixels: they are written together.

The stamp goes on when a producer is handed a `Viewbox`, and is inherited
by a producer whose image inputs all carry the same one (merge, polish,
the colour tail).
"""
from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

from mosaics.types import Viewbox

SECTION = "canvas"


def viewbox_to_obj(vb: Viewbox) -> dict:
    p = vb.projection
    out = {f.name: getattr(vb, f.name) for f in dataclasses.fields(vb) if f.name != "projection"}
    out["projection"] = {
        "module": type(p).__module__,
        "class": type(p).__name__,
        "params": {k: float(v) for k, v in vars(p).items()},
    }
    return out


def viewbox_from_obj(d: dict) -> Viewbox:
    p = d["projection"]
    cls = getattr(importlib.import_module(p["module"]), p["class"])
    fields = {k: v for k, v in d.items() if k != "projection"}
    return Viewbox(projection=cls(**p["params"]), **fields)


def canvas_of(path: Path) -> Viewbox | None:
    """The viewbox stamped on the image at `path`, or None."""
    from file_io.sidecar import get_section
    obj = get_section(Path(path), SECTION)
    return viewbox_from_obj(obj) if obj else None


def canvas_for_call(args: tuple, kwargs: dict) -> dict | None:
    """The canvas a producer called with `args`/`kwargs` draws on: a
    `Viewbox` it was handed, else the one canvas every stamped image input
    shares (None when they disagree or none is stamped)."""
    from file_io.sidecar import get_section
    values = (*args, *kwargs.values())
    for a in values:
        if isinstance(a, Viewbox):
            return viewbox_to_obj(a)
    stamps = []
    for a in values:
        filename = getattr(a, "filename", None)
        if filename is not None and Path(filename).exists():
            stamps.append(get_section(Path(filename), SECTION))
    stamps = [s for s in stamps if s]
    if stamps and all(s["content_hash"] == stamps[0]["content_hash"] for s in stamps):
        return stamps[0]
    return None
