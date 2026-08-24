"""Generate JavaScript to open images in PixInsight."""
import re
from pathlib import Path


def sanitize_view_id(name: str) -> str:
    """A PixInsight-legal view identifier from arbitrary text.

    PI identifiers are `[A-Za-z_][A-Za-z0-9_]*`; a view assigned anything
    else is silently left at its default (`Image01`), which is exactly the
    unreadable naming this exists to avoid.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _emit(pairs: list[tuple[str, Path]]) -> str:
    lines = []
    for i, (view_id, filename) in enumerate(pairs):
        var = f"w_{i}"
        lines.append(f'var {var} = ImageWindow.open("{filename}");')
        lines.append(f'if ({var}.length > 0) {{')
        lines.append(f'  {var}[0].mainView.id = "{view_id}";')
        lines.append(f'  {var}[0].show();')
        lines.append('}')
    return '\n'.join(lines)


def generate_open_images_js(filenames: list[Path]) -> str:
    """Generate JavaScript to open image files in PixInsight."""
    from pix.image import view_id_for

    return _emit([(view_id_for(f), f) for f in filenames])


def generate_named_open_js(views: dict[str, Path]) -> str:
    """Open each file and rename its view to the mapping key.

    The recipe path names views by content hash (`view_id_for`) because it
    addresses them programmatically. This one is for humans reading the PI
    window list, so the caller supplies the name — a filter, typically.
    """
    return _emit([(sanitize_view_id(k), v) for k, v in views.items()])
