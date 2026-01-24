from typing import Optional
from pathlib import Path
import os

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task

_javascript = """
var view_ids = {view_ids};
var output_filenames = {output_filenames};

function saveCompressedTiff(window, filename) {{
    let fileFormat = new FileFormat("TIFF");
    let f = new FileFormatInstance(fileFormat);
    if (f.isNull)
        throw new Error("Unable to create FileFormatInstance for TIFF");

    let description = new ImageDescription();
    description.bitsPerSample = 32;  // Set 32-bit per sample
    description.ieeefpSampleFormat = true;  // Enable floating-point sample format
    description.compression = true;

    if (!f.create(filename, "{tiff_output_hint}")) {{
        throw new Error("Unable to create file: " + filename);
    }}

    if (!f.setOptions(description)) {{
        throw new Error("Unable to set image options for 32-bit IEEE floating point.");
    }}

    let img = window.mainView.image;  // Retrieve the actual Image object

    if (!f.writeImage(img)) {{
        throw new Error("Failed to write image: " + filename);
    }}

    f.close();
    console.writeln("Image saved as 32-bit IEEE floating-point TIFF: " + filename);
}}

for (let i = 0; i < output_filenames.length; ++i) {{
  let filename = output_filenames[i];
  console.writeln('Saving view ' + view_ids[i] + ' to filename ' + filename);
  let window = View.viewById(view_ids[i]).window;
  let ext = filename.toLowerCase().split('.').pop();
  if (ext === 'tif' || ext === 'tiff') {{
    saveCompressedTiff(window, filename);
  }} else if (ext === 'xisf') {{
    window.saveAs(filename, false, false, true, false, "{xisf_output_hint}");
  }} else {{
    throw new Error("Unsupported extension: " + ext);
  }}
}}
"""


def SaveImages(
    images: ImageList,
    output_filenames: Optional[list[Path]] = None,
    compression: bool = True,
) -> ImageList:
    view_ids = [i.view_id for i in images]

    if output_filenames is None:
        output_filenames = [i.filename for i in images]
    else:
        output_filenames = [os.fspath(f) for f in output_filenames]

    append_task(
        ShellScriptTask(
            javascript=_javascript.format(
                view_ids=view_ids,
                output_filenames=output_filenames,
                tiff_output_hint="zip-compression" if compression else "",
                xisf_output_hint="compression-codec lz4hc+sh" if compression else "",
            )
        )
    )

    return images
