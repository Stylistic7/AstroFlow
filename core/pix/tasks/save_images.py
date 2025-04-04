from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task

_javascript = """
var view_ids = {view_ids};
var output_filenames = {output_filenames};

function saveWithProperFormat(window, filename) {
    let ext = filename.toLowerCase().split('.').pop();

    if (ext === 'tif' || ext === 'tiff') {
        let tiff = new TIFFFormat();
        let io = new ImageOptions();
        io.compression = TIFFCompression_ZIP;
        io.bitsPerSample = 32;
        io.sampleFormat = SampleFormat_Float;
        tiff.save(window.mainView.image, filename, io);
    } else if (ext === 'xisf') {
        let xisf = new XISFFormat();
        let io = new ImageOptions();
        io.compression = true;  // Uses default codec (e.g., ZLib)
        xisf.save(window.mainView.image, filename, io);
    } else {
        throw new Error("Unsupported extension: " + ext);
    }
}

for (let i = 0; i < output_filenames.length; ++i) {{
  console.writeln('Saving view ' + view_ids[i] + ' to filename ' + output_filenames[i]);
  let window = View.viewById(view_ids[i]).window;
  saveWithProperFormat(window, output_filenames[i]);
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
