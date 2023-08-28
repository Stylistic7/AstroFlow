import os
import tempfile
import subprocess
import time
from pathlib import Path
import diskcache

import config

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

post_process_script = """
var P_sa = new StarAlignment;
P_sa.structureLayers = 5;
P_sa.noiseLayers = 0;
P_sa.hotPixelFilterRadius = 1;
P_sa.noiseReductionFilterRadius = 0;
P_sa.minStructureSize = 0;
P_sa.sensitivity = 0.50;
P_sa.peakResponse = 0.50;
P_sa.brightThreshold = 3.00;
P_sa.maxStarDistortion = 0.60;
P_sa.allowClusteredSources = false;
P_sa.localMaximaDetectionLimit = 0.75;
P_sa.upperLimit = 1.000;
P_sa.invert = false;
P_sa.distortionModel = "";
P_sa.undistortedReference = false;
P_sa.distortionCorrection = true;
P_sa.distortionMaxIterations = 20;
P_sa.distortionTolerance = 0.005;
P_sa.distortionAmplitude = 2;
P_sa.localDistortion = true;
P_sa.localDistortionScale = 256;
P_sa.localDistortionTolerance = 0.050;
P_sa.localDistortionRejection = 2.50;
P_sa.localDistortionRejectionWindow = 64;
P_sa.localDistortionRegularization = 0.010;
P_sa.matcherTolerance = 0.0500;
P_sa.ransacTolerance = 2.00;
P_sa.ransacMaxIterations = 2000;
P_sa.ransacMaximizeInliers = 1.00;
P_sa.ransacMaximizeOverlapping = 1.00;
P_sa.ransacMaximizeRegularity = 1.00;
P_sa.ransacMinimizeError = 1.00;
P_sa.maxStars = 0;
P_sa.fitPSF = StarAlignment.prototype.FitPSF_DistortionOnly;
P_sa.psfTolerance = 0.50;
P_sa.useTriangles = false;
P_sa.polygonSides = 5;
P_sa.descriptorsPerStar = 20;
P_sa.restrictToPreviews = true;
P_sa.intersection = StarAlignment.prototype.MosaicOnly;
P_sa.useBrightnessRelations = false;
P_sa.useScaleDifferences = false;
P_sa.scaleTolerance = 0.100;
P_sa.referenceImage = "";
P_sa.referenceIsFile = false;
P_sa.targets = [];
P_sa.inputHints = "";
P_sa.outputHints = "";
P_sa.mode = StarAlignment.prototype.RegisterMatch;
P_sa.writeKeywords = true;
P_sa.generateMasks = false;
P_sa.generateDrizzleData = true;
P_sa.generateDistortionMaps = false;
P_sa.inheritAstrometricSolution = true;
P_sa.frameAdaptation = false;
P_sa.randomizeMosaic = false;
P_sa.noGUIMessages = true;
P_sa.useSurfaceSplines = true;
P_sa.extrapolateLocalDistortion = true;
P_sa.splineSmoothness = 0.050;
P_sa.pixelInterpolation = StarAlignment.prototype.Auto;
P_sa.clampingThreshold = 0.30;
P_sa.outputDirectory = "";
P_sa.outputExtension = ".xisf";
P_sa.outputPrefix = "";
P_sa.outputPostfix = "_r";
P_sa.maskPostfix = "_m";
P_sa.distortionMapPostfix = "_dm";
P_sa.outputSampleFormat = StarAlignment.prototype.SameAsTarget;
P_sa.overwriteExistingFiles = false;
P_sa.onError = StarAlignment.prototype.Continue;
P_sa.useFileThreads = true;
P_sa.fileThreadOverload = 1.00;
P_sa.maxFileReadThreads = 0;
P_sa.maxFileWriteThreads = 0;
P_sa.memoryLoadControl = true;
P_sa.memoryLoadLimit = 0.85;
/*
 * Read-only properties
 *
P_sa.outputData = [ // outputImage, outputMask, totalPairMatches, inliers, overlapping, regularity, quality, rmsError, rmsErrorDev, peakErrorX, peakErrorY, H11, H12, H13, H21, H22, H23, H31, H32, H33, frameAdaptationBiasRK, frameAdaptationBiasG, frameAdaptationBiasB, frameAdaptationSlopeRK, frameAdaptationSlopeG, frameAdaptationSlopeB, frameAdaptationAvgDevRK, frameAdaptationAvgDevG, frameAdaptationAvgDevB, referenceStarX, referenceStarY, targetStarX, targetStarY, totalLocalDistortionMatches, referenceLocalDistortionX, referenceLocalDistortionY, targetLocalDistortionX, targetLocalDistortionY, targetLocalDistortionDX, targetLocalDistortionDY, localDistortionWeights, outputDistortionMap, localDistortionMinDeltaX, localDistortionMinDeltaY, localDistortionMaxDeltaX, localDistortionMaxDeltaY
];
 */

var P_sxt = new StarXTerminator;
P_sxt.ai_file = "/Applications/PixInsight/library/StarXTerminator.11.mlmodel";
P_sxt.stars = true;
P_sxt.unscreen = true;
P_sxt.overlap = 0.50;

// Open all the files. Store their ImageWindow objects in an array. Element 0 is the
// reference image.
var input_filenames = {input_filenames};
var output_path = "{output_path}";
var output_ids = {output_ids};

var windows = [];
for (let i = 0; i < input_filenames.length; ++i) {{
  let w = ImageWindow.open(input_filenames[i]);
  w[0].show();
  windows.push(w[0]);
}}

// Run StarAlignment on images 1-N, with 0 as the reference.
if (windows.length > 1) {{
  P_sa.referenceImage = windows[0].currentView.id;
  let targets = [];
  for (let i = 1; i < windows.length; ++i) {{
    targets.push([true, false, windows[i].currentView.id]);
  }}
  P_sa.targets = targets;
}}
P_sa.executeGlobal();

// Rename all the registered images to the desired output ids.
console.writeln('rename view ', windows[0].currentView.id, ' to ', output_ids[0]);
windows[0].currentView.id = output_ids[0];
for (let i = 0; i < P_sa.outputData.length; ++i) {{
  console.writeln('star alignment output', i, P_sa.outputData[i]);
  console.writeln('rename view ', P_sa.outputData[i][0], ' to ', output_ids[i + 1]);
  View.viewById(P_sa.outputData[i][0]).id = output_ids[i + 1];
}}

// Save all of the images.
for (let i = 0; i < output_ids.length; ++i) {{
  let view_id = output_ids[i];
  let window = View.viewById(view_id).window;
  window.saveAs(output_path + "/" + view_id + ".xisf", false, false, true, false);
}}


// TODO Combine RGB images, ImageSolve, run SPCC
// These color images will be added to the list of output_ids.

/*
if (rgb_channels.length) {{
  P_cc.channels = rgb_channels;
  P_cc.executeGlobal();
  let view = ImageWindow.activeWindow.currentView;
  let view_id = 'RGB';
  view.id = view_id;
  output_ids.push(view_id);

  P_solver.executeGlobal();
  P_spcc.executeOn(view);

  let window = View.viewById(view_id).window;
  window.saveAs(output_path + "/" + view_id + ".xisf", false, false, true, false);
}}
*/

// Run SXT on all images.
for (let i = 0; i < output_ids.length; ++i) {{
  let view_id = output_ids[i];
  P_sxt.executeOn(View.viewById(view_id));
}}

// Rename the base images to _starless.
for (let i = 0; i < output_ids.length; ++i) {{
  let view_id = output_ids[i];
  let view_id_starless = output_ids[i] + "_starless";
  console.writeln('Renaming view ', view_id, ' to ', view_id_starless);
  View.viewById(view_id).id = view_id_starless;
}}

// Save all images.
for (let i = 0; i < output_ids.length; ++i) {{
  let view_id = output_ids[i] + "_starless";
  let window = View.viewById(view_id).window;
  window.saveAs(output_path + "/" + view_id + ".xisf", false, false, true, false);
  window.forceClose();

  view_id = output_ids[i] + "_stars";
  window = View.viewById(view_id).window;
  window.saveAs(output_path + "/" + view_id + ".xisf", false, false, true, false);
  window.forceClose();
}}
"""


def find_reference(stacks):
  for best_rig in ['C11_F7', 'C11', 'RC71', 'TestRig']:
    for best_filter in ['H', 'O', 'S', 'L', 'G', 'R', 'B']:
      for stack in stacks:
        if stack.rig == best_rig and stack.filter == best_filter:
          return stack


@cache.memoize(tag='step_post_process_target')
def run(stacks):
  reference = find_reference(stacks)
  print('found reference', reference)
  stacks.remove(reference)
  stacks.insert(0, reference)

  input_filenames = [s.post_processed for s in stacks]

  # TODO: Push these restrictive id constraints up into the original masters.
  output_path = os.path.commonpath(input_filenames)
  output_ids = [
      os.path.splitext(os.path.basename(m))[0].replace('.', '_').replace(
          '-', 'M') + '_aligned' for m in input_filenames
  ]
  print('output_ids are', output_ids)

  output_filenames = [
      os.path.join(output_path, o + '_stars.xisf') for o in output_ids
  ]
  output_filenames += [
      os.path.join(output_path, o + '_starless.xisf') for o in output_ids
  ]

  with tempfile.TemporaryDirectory() as tmp_dir:
    sentinel_filename = os.path.join(tmp_dir, 'sentinel.txt')

    script_filename = os.path.join(tmp_dir, "run_post_processing.js")
    with open(script_filename, "w") as f:
      script = post_process_script.format(input_filenames=input_filenames,
                                          output_path=output_path,
                                          output_ids=output_ids,
                                          sentinel_filename=sentinel_filename)
      f.write(script)

    print(f'Running PixInsight post-processing on {len(input_filenames)}')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--execute={script_filename}'
    ]).returncode

    while not all(os.path.exists(f) for f in output_filenames):
      time.sleep(5)

    return output_filenames