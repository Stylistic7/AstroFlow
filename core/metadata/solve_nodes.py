"""Plate solutions as graph nodes, where a pipeline needs them.

Producers never solve their own output (decorators._metadata_wrap): a solve
costs ~45 s on one core, and inside the producer's task it holds the whole
reservation — integrate_with_siril's 12 CPUs and 12 GB, bxt_channel's GPU
slot — while the rest sits idle (2026-10-08: the Siril stacks left the
machine ~85% idle). So a pipeline asks for a solution explicitly:

  * `plate_solve_image(img)` — the app's solve of the image. After any op
    that moves pixels (a stacker, BXT, an HDR channel combine).
  * `inherit_plate_solution(img, donor)` — `donor`'s solution, for an op
    that keeps the donor's pixel grid: one with nothing to solve (a
    starless image, a star layer), or one that just doesn't move pixels
    (post-stack calibration, PixInsight post-processing). Measured
    2026-10-09 over 148 Siril masters: a re-solve after post-stack
    calibration agreed with the master's to a median 0.08 px, and the one
    13 px disagreement was a BAD re-solve (failed the solution check).
  * `solved_channel(channel, masters)` — inherit for a one-stack channel
    combine, solve a multi-stack one.

Either way the solution goes where every reader finds it: the image's
sidecar `plate_solution` section (restored by file-level metadata reads)
and the returned `ImageMetadata` (for the graph).
"""
import dataclasses
from pathlib import Path

from decorators import node
from metadata.classes import ImageMetadata


@node(num_cpus=1)
def plate_solve_image(image: ImageMetadata) -> ImageMetadata:
    """`image` with the app's plate solution; fails if it won't solve (a
    deliverable that can't be placed is a real failure). A synthetic test
    frame marked DONTSOLV legitimately has none and passes through."""
    import plate_solving
    solution = plate_solving.solve_field(Path(image.filename))
    if solution is None and image.header.get("DONTSOLV"):
        return image
    if solution is None:
        raise plate_solving.PlateSolveFailedException(f"Failed to plate-solve: {image.filename}")
    return dataclasses.replace(image, plate_solution=solution)


@node(num_cpus=0.25)
def inherit_plate_solution(image: ImageMetadata, donor: ImageMetadata) -> ImageMetadata:
    """`image` with `donor`'s plate solution, for an image on the donor's
    exact pixel grid that cannot be solved itself."""
    from file_io.sidecar import set_section
    from plate_solving import _plate_solution_to_obj
    if donor.plate_solution is None and donor.header.get("DONTSOLV"):
        return image   # a synthetic test frame: nothing to inherit, as plate_solve_image
    assert donor.plate_solution is not None, f"donor {donor.filename} carries no plate solution"
    assert (image.width_pixels, image.height_pixels) == (donor.width_pixels, donor.height_pixels), (
        f"{image.filename} is not on {donor.filename}'s grid")
    set_section(Path(image.filename), "plate_solution", _plate_solution_to_obj(donor.plate_solution))
    return dataclasses.replace(image, plate_solution=donor.plate_solution)


def solved_channel(channel: ImageMetadata, masters: list) -> ImageMetadata:
    """A combined channel's solution, decided at trace time. A one-stack
    channel is a copy of its master (`merge_target_channel`), so it inherits
    the master's solution; a multi-stack HDR combine is aligned by
    PixInsight, so it is solved. (2026-10-09: of 741 solves in a backfill
    night, ~180 were channel combines, almost all single-stack.)"""
    if len(masters) == 1:
        return inherit_plate_solution(channel, masters[0])
    return plate_solve_image(channel)
