from typing import Any
from pathlib import Path


class SidecarHasher:
    def has(self, obj: Any) -> bool:
        return isinstance(obj, Path)

    def get(self, obj: Any) -> str:
        if isinstance(obj, Path):
            with open(obj.with_suffix(".hash"), "r") as f:
                content_hash = f.read()
                print(f"got hash {content_hash}")
                return content_hash

    def set(self, obj: Any, content_hash: str) -> Any:
        print(f"setting hash {content_hash} for obj {obj}")
        if isinstance(obj, Path):
            with open(obj.with_suffix(".hash"), "w") as f:
                f.write(content_hash)
        return obj
