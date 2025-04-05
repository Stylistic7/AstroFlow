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

    if (!f.create(filename, "zip-compression")) {{
        throw new Error("Unable to create file: " + filePath);
    }}

    if (!f.setOptions(description)) {{
        throw new Error("Unable to set image options for 32-bit IEEE floating point.");
    }}

    let img = window.mainView.image;  // Retrieve the actual Image object

    if (!f.writeImage(img)) {{
        throw new Error("Failed to write image: " + filePath);
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
    window.saveAs(filename, false, false, true, false);
  }} else {{
    throw new Error("Unsupported extension: " + ext);
  }}
}}
"""


def SaveImages(images: ImageList) -> ImageList:
  print(f'### Entering SaveImages {images}')
  view_ids = [i.view_id for i in images]
  filenames = [i.filename for i in images]

  append_task(
      ShellScriptTask(javascript=_javascript.format(
          view_ids=view_ids, output_filenames=filenames)))

  return images
