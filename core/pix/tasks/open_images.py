"""Generate JavaScript to open images in PixInsight."""
from pathlib import Path


def generate_open_images_js(filenames: list[Path]) -> str:
    """Generate JavaScript to open image files in PixInsight."""
    from pix.image import view_id_for

    lines = []
    for i, filename in enumerate(filenames):
        view_id = view_id_for(filename)
        var = f"w_{i}"
        lines.append(f'var {var} = ImageWindow.open("{filename}");')
        lines.append(f'if ({var}.length > 0) {{')
        lines.append(f'  {var}[0].mainView.id = "{view_id}";')
        lines.append(f'  {var}[0].show();')
        lines.append('}')
    return '\n'.join(lines)
