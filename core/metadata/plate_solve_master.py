"""`plate_solve_master`: the app's plate solve of a master, as its own node.

A producer that solves inside its own task holds its whole reservation while
solve-field runs on one core (~45 s for a 61 MP master): integrate_with_siril
held 12 CPUs and 12 GB through it, and the machine sat ~85% idle. A stacker
that declares `@node(plate_solve="none")` hands its unsolved master here
instead: one CPU, nothing else held.

The solution goes where every reader finds it: `solve_field` writes the
sidecar's `plate_solution` section, which file-level reads of the master's
metadata restore, and the returned `ImageMetadata` carries it for the graph.
"""
import dataclasses
from pathlib import Path

from decorators import node
from metadata.classes import ImageMetadata


@node(num_cpus=1)
def plate_solve_master(master: ImageMetadata) -> ImageMetadata:
    """`master` with the app's plate solution; fails if it won't solve (a
    master that can't be placed is a real failure)."""
    import plate_solving
    solution = plate_solving.solve_field(Path(master.filename))
    if solution is None:
        raise plate_solving.PlateSolveFailedException(f"Failed to plate-solve: {master.filename}")
    return dataclasses.replace(master, plate_solution=solution)
