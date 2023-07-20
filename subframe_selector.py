import os
import tempfile
import subprocess
import time
import json
import math

subframe_selector_script = """
var P = new SubframeSelector;
P.routine = SubframeSelector.prototype.MeasureSubframes;
P.nonInteractive = true;
P.subframes = {light_frames};
P.fileCache = true;
P.subframeScale = {image_scale};
P.cameraGain = 0.2000;
P.cameraResolution = SubframeSelector.prototype.Bits16;
P.siteLocalMidnight = 24;
P.scaleUnit = SubframeSelector.prototype.ArcSeconds;
P.dataUnit = SubframeSelector.prototype.Normalized;
P.trimmingFactor = 0.10;
P.structureLayers = 8;
P.noiseLayers = 0;
P.hotPixelFilterRadius = 1;
P.noiseReductionFilterRadius = 0;
P.minStructureSize = 0;
P.sensitivity = 0.50;
P.peakResponse = 0.50;
P.brightThreshold = 3.00;
P.maxDistortion = 0.60;
P.allowClusteredSources = false;
P.upperLimit = 1.0000;
P.psfFit = SubframeSelector.prototype.Moffat4;
P.psfFitCircular = false;
P.maxPSFFits = 8000;
P.roiX0 = 0;
P.roiY0 = 0;
P.roiX1 = 0;
P.roiY1 = 0;
P.pedestalMode = SubframeSelector.prototype.Pedestal_Keyword;
P.pedestal = 0;
P.pedestalKeyword = "";
P.inputHints = "";
P.outputHints = "";
P.outputDirectory = "";
P.outputExtension = ".xisf";
P.outputPrefix = "";
P.outputPostfix = "";
P.outputKeyword = "SNR";
P.overwriteExistingFiles = false;
P.onError = SubframeSelector.prototype.Continue;
P.approvalExpression = "";
P.weightingExpression = "SNR";
P.sortProperty = SubframeSelector.prototype.Index;
P.graphProperty = SubframeSelector.prototype.PSFSignalWeight;
P.auxGraphProperty = SubframeSelector.prototype.Weight;
P.useFileThreads = true;
P.fileThreadOverload = 1.00;
P.maxFileReadThreads = 0;
P.maxFileWriteThreads = 0;

P.launch();
P.executeGlobal();

let measurements = "";
for (let i of P.measurements) {{
  measurements += i + "\\n";
}}

File.writeTextFile("{output_filename}", measurements);
"""


def measure_frames(light_frames, image_scale):
  with tempfile.TemporaryDirectory() as tmp_dir:
    script_filename = os.path.join(tmp_dir, "run_subframe_selector.js")
    output_filename = os.path.join(tmp_dir, "subframe_selector_result.csv")

    light_frames = [[True, str(s), "", ""] for s in light_frames]

    with open(script_filename, "w") as f:
      script = subframe_selector_script.format(
          light_frames=json.dumps(light_frames),
          image_scale=image_scale,
          output_filename=output_filename)
      f.write(script)

    print('Running PixInsight SubframeSelector on', len(light_frames),
          'light frames...')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--execute={script_filename}'
    ]).returncode

    while not os.path.exists(output_filename):
      time.sleep(5)

    with open(output_filename) as f:
      result = f.read()

      column_headings = [
          'Index', 'Approved', 'Locked', 'File', 'Weight', 'PSF Signal Weight',
          'PSF SNR', 'PSF Scale', 'PSF Scale SNR', 'PSF Count', 'M*', 'N*',
          'SNR', 'FWHM', 'Eccentricity', 'Altitude', 'Azimuth', 'Median',
          'Median Mean Deviation', 'Noise', 'Noise Ratio', 'Stars',
          'Star Residual', 'PSF Total Flux', 'PSF Total Power Flux',
          'PSF Total Mean Flux', 'PSF Total Mean Power Flux',
          'FWHM Mean Deviation', 'Eccentricity Mean Deviation',
          'Star Residual Mean Deviation', 'Median Mean Dev.'
      ]

      output_dict = {}
      for line in result.splitlines():
        line_dict = dict(zip(column_headings, line.split(',')))
        if math.isfinite(float(line_dict['PSF Signal Weight'])):
          output_dict[line_dict['File']] = line_dict

      return output_dict
