import numpy as np
from skimage.transform import resize
import cv2


def resize_array(array: np.ndarray, new_width: int) -> np.ndarray:
    assert not np.isnan(np.min(array))
    height, width = array.shape
    new_height = (height / width) * new_width
    result = resize(array, (int(new_height), int(new_width)))
    assert not np.isnan(np.min(result)), (
        f"Failed to resize image dtype {array.dtype} min {np.min(array)} max {np.max(array)}"
    )
    return result


def autostretch(image: np.ndarray) -> np.ndarray:
    """The house display stretch (`mosaics.color.palettes.autostretch`, the
    STF the viewer seeds from) for callers whose input is not yet in
    [0, 1]: raw-ADU frames are scaled by their brightest finite pixel
    first, normalised images pass through unchanged.

    This replaced a second, older implementation (2026-09-17) whose
    midtone was solved for the UN-renormalised median — after clipping
    the shadows it stretched every image as if its background sat far
    above where it really was, which is why the S and O merged-mosaic
    thumbnails came out white while H looked right."""
    from mosaics.color.palettes import autostretch as _stf
    x = image.astype(np.float32)
    finite = x[np.isfinite(x)]
    peak = float(finite.max()) if finite.size else 1.0
    if peak > 1.0:
        x = x / peak
    return _stf(x, target_background=0.20)


def array_to_png(array: np.ndarray, debug: bool = False) -> bytes:
    if np.issubdtype(array.dtype, np.floating):
        assert not np.isnan(array).any(), "array contains NaN"

    array = (array * 255.0).astype(np.uint8)
    if debug:
        print(f"### array_to_png, array min {np.min(array)} max {np.max(array)}")
    _, png = cv2.imencode(".png", array)
    return png.tobytes()
