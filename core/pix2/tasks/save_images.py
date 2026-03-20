"""Generate JavaScript to save images from PixInsight."""
from pathlib import Path


def generate_save_images_js(targets: dict[str, Path]) -> str:
    """Generate JS to save views to files.

    Args:
        targets: Mapping of view_id → output file path.
    """
    lines = []
    for view_id, path in targets.items():
        path_str = str(path)
        lines.append(f'var view = View.viewById("{view_id}");')
        lines.append(f'var window = view.window;')
        lines.append(f'window.saveAs("{path_str}", false, false, true, false);')
    return "\n".join(lines)
