from dataclasses import dataclass
from plate_solving import PlateSolution


@dataclass(frozen=True)
class PlateSolutionAttributes:
    """Mixin for objects that have an actual WCS plate solution.

    Distinct from AstrometricAttributes, which carries derived spatial
    properties (center, footprint, scale) that any sky object can have.
    PlateSolutionAttributes means the object has a concrete pixel-to-sky
    coordinate mapping from plate solving a real image.
    """
    plate_solution: PlateSolution
