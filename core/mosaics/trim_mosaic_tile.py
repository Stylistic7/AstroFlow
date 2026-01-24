import os
import numpy as np
from xisf import XISF
import largestinteriorrectangle


def trim_mosaic_tile(xisf_filename):
    xisf = XISF(xisf_filename)
    im = xisf.read_image(0).squeeze()
    epsilon = 1e-5
    top, left, height, width = largestinteriorrectangle.lir(im > epsilon)
    print(
        f"Trimmed {xisf_filename} "
        "top {top} left {left} height {height} width {width}"
    )

    blank_im = np.zeros(im.shape)
    blank_im[top : top + height, left : left + width] = im[
        top : top + height, left : left + width
    ]

    file_meta = xisf.get_file_metadata()
    img_meta = xisf.get_images_metadata()[0]
    fits_meta = img_meta["FITSKeywords"]

    fits_meta["HISTORY"] = ("", "TrimMosaicTile.target: " + "foo")
    fits_meta["HISTORY"] = ("", "TrimMosaicTile.top: " + top)
    fits_meta["HISTORY"] = ("", "TrimMosaicTile.bottom: " + (top + height))
    fits_meta["HISTORY"] = ("", "TrimMosaicTile.left: " + left)
    fits_meta["HISTORY"] = ("", "TrimMosaicTile.right: " + (left + width))

    base, ext = os.path.splitext(output_filename)
    output_filename = f"{base}_trimmed{ext}"
    XISF.write(
        output_filename,
        im,
        creator_app="trim_mosaic_tile",
        image_metadata=img_meta,
        xisf_metadata=file_meta,
        codec="lz4hc",
        shuffle=True,
    )

    return output_filename


def trim_mosaic_tiles(xisf_filenames):
    return [trim_mosaic_tile(f) for f in xisf_filenames]


# keywords.push(new FITSKeyword("HISTORY", "", "TrimMosaicTile.target: " + data.targetView.fullId));
# keywords.push(new FITSKeyword("HISTORY", "", "TrimMosaicTile.top: " + data.top));
# keywords.push(new FITSKeyword("HISTORY", "", "TrimMosaicTile.bottom: " + data.bottom));
# keywords.push(new FITSKeyword("HISTORY", "", "TrimMosaicTile.left: " + data.left));
# keywords.push(new FITSKeyword("HISTORY", "", "TrimMosaicTile.right: " + data.right));
