from dataclasses import dataclass
from typing import Any
from beartype import beartype


@beartype
@dataclass(frozen=True)
class AstroFlowStateAttributes:
    state: dict[str, Any]
