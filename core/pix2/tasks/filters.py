"""Filter name normalization."""
import re

_FILTER_MAP = {
    "L": ["l", "lum", "luminance"],
    "R": ["r", "red"],
    "G": ["g", "green"],
    "B": ["b", "blue"],
    "S": ["s", "sii", "sulfur", "sulphur"],
    "H": ["h", "ha", "hα", "halpha", "h-alpha"],
    "O": ["o", "oiii", "oxygen", "o-iii"],
}

_CANON = {}
for canon, aliases in _FILTER_MAP.items():
    for alias in aliases:
        _CANON[alias.lower()] = canon


def normalize_filter(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"[^a-z0-9α]", "", name.lower())
    return _CANON.get(key)
