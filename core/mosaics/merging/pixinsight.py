import os
import shutil
import ray
import numpy as np
import structlog
from beartype import beartype
from pathlib import Path
from dataclasses import dataclass

from file_io.generic import read_header, read_image, update_header
from file_io.sidecar import get_section, set_section
from pix import Image
from pix.tasks import PhotometricMosaic
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from previews.create_tile_pyramid import default_make_tile_pyramid
from decorators import node
from decisions.settings import ProcessingConfig
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from hashing import hash_list
from mosaics.placement.panel_io import read_placed_panel, rehydrate_placed_panel

log = structlog.get_logger()


@dataclass
class _PanelInput:
    """Minimal `ImageMetadata` stand-in (filename + stable_hash) for a
    re-hydrated placed panel handed to the pairwise PI merge."""
    filename: Path
    stable_hash: str


@dataclass(frozen=True)
class MergedImage:
    pass

def _panel_count(image_metadata: ImageMetadata) -> int:
    """Number of mosaic panels contained in this intermediate.

    Placed images set `panel_count=1` in their sidecar; merges record
    the sum of their inputs. The `@makes_image` lifecycle no longer
    touches sidecars on producer body-run, so `panel_count` survives
    across reruns by default. Missing entry defaults to 1, keeping the
    chain monotonic for any pre-existing files left over from before
    this mechanism existed.
    """
    return get_section(image_metadata.filename, "panel_count") or 1

def _set_panel_count(path, count: int) -> None:
    set_section(Path(path), "panel_count", count)



def _blend_pair_numpy(image_0, image_1):
    overlap = np.where((image_0 > 0) & (image_1 > 0))
    output = np.maximum(image_0, image_1)
    output[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
    return output


@node(
    display="Merged",
    resources={"fastest_storage": 0.001},
    plate_solve=False,  # mosaic projection — non-square pixels, can't solve
)
def merge_images_subframe(
    mosaic: Mosaic, filter: str, *inputs,
) -> WithSidecar[TiffFilePath]:
    """Cheap pixel-overlap blend for the Subframe preview pipeline.
    Variadic *inputs gives ONE call per (mosaic, filter); pairwise
    blend is done inside the Ray task. Same call_hash-filename
    convention as `merge_images`: one output file per (mosaic, filter)
    in `merge_images_subframe/`."""
    assert len(inputs) >= 1
    # `*inputs` MUST flow into the call_hash — otherwise every
    # pairwise `merge_images_subframe(mosaic, filter, row, p)`
    # call in the sequential reduce produces the same filename,
    # multiple writers race on it, and downstream readers see
    # half-written zlib streams (corrupt TIFF).
    output_filename = merge_images_subframe.filename_fn(mosaic, filter, *inputs)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    paths = [i.filename.resolve(strict=True) for i in inputs]
    # bbox-on-disk panels: place each into a full-canvas accumulator at its
    # offset, max/avg-compositing overlaps (the old full-canvas _blend_pair_numpy,
    # now bbox-aware). The subframe merge output is the assembled preview mosaic.
    panels = [read_placed_panel(p) for p in paths]
    for arr, _, _ in panels:
        assert not np.isnan(np.min(arr))
    canvas_w, canvas_h = panels[0][2]
    acc = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    for arr, (x0, y0), _ in panels:
        h, w = arr.shape
        region = acc[y0:y0 + h, x0:x0 + w]
        acc[y0:y0 + h, x0:x0 + w] = _blend_pair_numpy(region, arr.astype(np.float32))
    assert not np.isnan(np.min(acc))

    stable_hash = hash_list([i.stable_hash for i in inputs])
    write_tiff_image(Path(output_filename), acc, {"FILTER": filter, "STBLHASH": stable_hash})

    new_count = sum(_panel_count(i) for i in inputs)
    _set_panel_count(output_filename, new_count)
    log.info("Merge images (subframe) complete", output=output_filename,
             panel_count=new_count, input_count=len(inputs))
    return WithSidecar(output_filename)


def _merge_images_pi_impl(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    inputs: tuple,
    output_filename: Path,
) -> WithSidecar[XisfFilePath]:
    """Pairwise PhotometricMosaic merge across all panel inputs at
    once. Returns a single merged output. The pairwise tree is done
    inside one Ray task — moving it out would create a chain of
    @node calls with identical literal args and collide on the
    call-hash filename. We accept the loss of Ray-level parallelism
    inside the merge tree (rare and bounded; mosaic panel counts are
    in the dozens, not thousands)."""
    assert len(inputs) >= 1
    output_filename = Path(output_filename)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    import tempfile
    work_dir = Path(tempfile.mkdtemp(prefix="merge_images_"))
    try:
        # Placed panels are bbox-on-disk (zstd); PhotometricMosaic needs full-canvas
        # panels sharing one coordinate frame and can only read deflate. Re-hydrate
        # each to a full-canvas deflate temp and merge those. panel_count comes from
        # the original inputs (each placed panel is one panel).
        orig_inputs = inputs
        rehydrated = []
        for k, inp in enumerate(inputs):
            temp = work_dir / f"rehydrated_{k}.tiff"
            rehydrate_placed_panel(inp.filename.resolve(strict=True), temp)
            rehydrated.append(_PanelInput(filename=temp, stable_hash=inp.stable_hash))
        inputs = tuple(rehydrated)

        if len(inputs) == 1:
            shutil.copy(inputs[0].filename, output_filename)
            return WithSidecar(output_filename)

        # Pairwise tree: take pairs, merge each pair → next level.
        current = list(inputs)
        step = 0
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                if i + 1 == len(current):
                    next_level.append(current[i])
                    continue
                a, b = current[i], current[i + 1]
                path_a = a.filename.resolve(strict=True)
                path_b = b.filename.resolve(strict=True)
                if i + 2 == len(current) and len(current) == 2:
                    # Final pair lands at the canonical output.
                    out_path = output_filename
                else:
                    out_path = work_dir / f"step_{step}_{i // 2}.xisf"
                    step += 1
                images = Image.open_many([path_a, path_b], filters=[filter, filter])
                result = PhotometricMosaic(images)
                result.save_to(out_path, mock=mock_pixinsight)
                update_header(Path(out_path), {
                    "FILTER": filter,
                    "STBLHASH": hash_list([a.stable_hash, b.stable_hash]),
                })
                # Wrap as ImageMetadata-shaped for the next pairwise step.
                from metadata.image_metadata import compute_image_metadata
                # Compute fresh metadata for the intermediate so the next
                # iteration has stable_hash + filename.
                next_level.append(_intermediate_metadata(out_path, a, b))
            current = next_level

        new_count = sum(_panel_count(i) for i in orig_inputs)
        _set_panel_count(output_filename, new_count)
        log.info("Merge images complete", output=str(output_filename),
                 panel_count=new_count, input_count=len(inputs))
        return WithSidecar(output_filename)
    finally:
        import shutil as _sh
        _sh.rmtree(work_dir, ignore_errors=True)


def _intermediate_metadata(path: Path, a, b) -> ImageMetadata:
    """Stand-in ImageMetadata for an intermediate pairwise merge
    output. We only need `.filename` and `.stable_hash` for the
    next pairwise step + `_panel_count`."""
    from dataclasses import dataclass
    @dataclass
    class _Stub:
        filename: Path
        stable_hash: str
    return _Stub(filename=path, stable_hash=hash_list([a.stable_hash, b.stable_hash]))


def _merge_images_narrow(args, kwargs):
    """Cache-key hook: project `config` to its only dependency here,
    `mock_pixinsight` (in its arg slot), so the key stays independent
    of unrelated config fields."""
    mosaic, filter, config, *inputs = args
    return (mosaic, filter, config.development_settings.mock_pixinsight, *inputs), kwargs


@node(
    display="Merged",
    cache_key_narrow=_merge_images_narrow,
    resources={"PixInsight": 1},
    auto_memory=True,
    plate_solve=False,  # mosaic projection — non-square pixels, can't solve
)
def merge_images(
    mosaic: Mosaic, filter: str, config: ProcessingConfig,
    *inputs,
) -> WithSidecar[XisfFilePath]:
    """Pairwise PhotometricMosaic merge across all panel inputs.

    Pipeline-agnostic. PI Master and PI Downscaled both call this
    with their own placed-image inputs; `*inputs` differs between
    them, so the call_hash differs, so the on-disk filename
    differs. The framework's runtime tap-path hook captures the
    actual path."""
    mock_pixinsight = config.development_settings.mock_pixinsight
    return _merge_images_pi_impl(
        mosaic, filter, mock_pixinsight, inputs,
        merge_images.filename_fn(mosaic, filter, config, *inputs),
    )
