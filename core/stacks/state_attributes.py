from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AstroFlowStateAttributes:
  state: dict[str, Any]
