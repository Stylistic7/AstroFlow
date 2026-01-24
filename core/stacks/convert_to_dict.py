from dataclasses import is_dataclass, asdict
from collections.abc import Mapping, Sequence


def convert_to_dict(obj, omit_paths=None, transform=None):
    if omit_paths is None:
        omit_paths = []

    transform = transform or (lambda x: x)

    # Convert omit_paths into tuples, allowing wildcards
    omit_patterns = [tuple(p) if isinstance(p, list) else (p,) for p in omit_paths]

    def matches(path, pattern):
        if len(path) < len(pattern):
            return False
        return all(p == "*" or p == path[i] for i, p in enumerate(pattern))

    def should_omit(path):
        return any(matches(path, pat) for pat in omit_patterns)

    def recurse(o, path=()):
        if should_omit(path):
            return None
        if is_dataclass(o):
            o = asdict(o)
        if isinstance(o, Mapping):
            return {
                k: v
                for k, v in ((k, recurse(v, path + (k,))) for k, v in o.items())
                if v is not None
            }
        if isinstance(o, Sequence) and not isinstance(o, str):
            return [
                v
                for v in (recurse(v, path + (i,)) for i, v in enumerate(o))
                if v is not None
            ]
        return transform(o)

    return recurse(obj)
