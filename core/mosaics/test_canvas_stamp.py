"""The canvas stamp (`mosaics/canvas_stamp.py`): a canvas image carries the
viewbox its pixels are drawn on, in its own sidecar."""
from types import SimpleNamespace

import numpy as np

from geometry.projections.stereographic import StereographicProjection
from mosaics.canvas_stamp import SECTION, canvas_for_call, canvas_of, viewbox_from_obj, viewbox_to_obj
from mosaics.types import Viewbox


def _viewbox(rotation=0.0, content_hash="vb1"):
    return Viewbox(
        content_hash=content_hash,
        projection=StereographicProjection(np.float64(0.7), 1.1, rotation),
        x_min=-0.01, x_span=0.02, y_min=-0.015, y_span=0.03,
        resolution=1e-5, num_pixels_y=3000, num_pixels_x=2000,
    )


def _same(a, b):
    """Projections compare by identity, so compare them by class and values."""
    return (viewbox_to_obj(a) == viewbox_to_obj(b)
            and type(a.projection) is type(b.projection))


def _image(tmp_path, name, canvas=None):
    from file_io.sidecar import set_section
    path = tmp_path / name
    path.write_bytes(b"x")
    if canvas is not None:
        set_section(path, SECTION, viewbox_to_obj(canvas))
    return SimpleNamespace(filename=path)


def test_round_trip_keeps_projection_and_grid():
    vb = _viewbox(rotation=np.pi)
    back = viewbox_from_obj(viewbox_to_obj(vb))
    assert _same(back, vb)
    assert vars(back.projection) == {"lon_0": 0.7, "lat_0": 1.1, "rotation": np.pi}


def test_a_viewbox_argument_is_the_canvas(tmp_path):
    vb = _viewbox()
    assert canvas_for_call(("mosaic", "S", vb), {}) == viewbox_to_obj(vb)


def test_inputs_that_agree_pass_their_canvas_on(tmp_path):
    vb = _viewbox()
    a, b = _image(tmp_path, "a.tiff", vb), _image(tmp_path, "b.tiff", vb)
    assert canvas_for_call(("mosaic", "S", a, b), {}) == viewbox_to_obj(vb)


def test_inputs_on_different_canvases_pass_nothing_on(tmp_path):
    a = _image(tmp_path, "a.tiff", _viewbox(content_hash="vb1"))
    b = _image(tmp_path, "b.tiff", _viewbox(rotation=np.pi, content_hash="vb2"))
    assert canvas_for_call(("mosaic", "S", a, b), {}) is None


def test_unstamped_inputs_pass_nothing_on(tmp_path):
    assert canvas_for_call(("mosaic", _image(tmp_path, "a.tiff")), {}) is None


def test_a_producer_on_a_canvas_publishes_it(tmp_path):
    """A producer whose input carries a canvas stamps it on its product."""
    from file_io.types import Produced, TiffFilePath
    from products import registry
    vb = _viewbox()

    def merge(mosaic, placed, *, out: TiffFilePath) -> Produced:
        from file_io.tiff_file_io import write_tiff_image
        write_tiff_image(out, np.zeros((4, 4), np.float32), {})
        return Produced(out)
    op = registry.makes_image_wrap(
        merge, filename_fn=lambda *a: TiffFilePath(tmp_path / "store" / "m.tiff"),
        canonical_call_hash=lambda *a: "m", makes_tiff=False, image_type=None)
    final = op("mosaic", _image(tmp_path, "a.tiff", vb))
    assert _same(canvas_of(final), vb)
