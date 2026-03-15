from dataclasses import dataclass
from typing import Any
from beartype import beartype
from frozendict import frozendict


@beartype
@dataclass(frozen=True)
class AstroFlowStateAttributes:
    state: frozendict[str, Any]
