"""Generate JavaScript to save images from PixInsight."""
from pathlib import Path


def generate_save_images_js(targets: dict[str, Path]) -> str:
    """Generate JS to save views to files.

    Args:
        targets: Mapping of view_id → output file path.

    Skips any view that no longer exists. `Image.__init__` registers EVERY
    image as a recipe output, but a multi-step recipe consumes/renames its
    intermediates (e.g. StarAlignment renames each input `<id>` to
    `<id>_aligned`), so their original ids resolve to a null view. Saving
    those would throw `null.window`; the caller's `copy_output` still validates
    that the specific image it wants was written.
    """
    lines = []
    for view_id, path in targets.items():
        path_str = str(path)
        lines.append(f'var view = View.viewById("{view_id}");')
        lines.append('if (view !== null && !view.isNull) {')
        lines.append(f'   view.window.saveAs("{path_str}", false, false, true, false);')
        lines.append('}')
    return "\n".join(lines)
