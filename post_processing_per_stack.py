post_process_script = """
var P_bxt = new BlurXTerminator;
P_bxt.ai_file = "/Applications/PixInsight/library/BlurXTerminator.2.mlpackage";
P_bxt.correct_only = false;
P_bxt.correct_first = false;
P_bxt.nonstellar_then_stellar = false;
P_bxt.lum_only = false;
P_bxt.sharpen_stars = 0.25;
P_bxt.adjust_halos = 0.00;
P_bxt.nonstellar_psf_diameter = 0.00;
P_bxt.auto_nonstellar_psf = true;
P_bxt.sharpen_nonstellar = 0.90;

var P_sxt = new StarXTerminator;
P_sxt.ai_file = "/Applications/PixInsight/library/StarXTerminator.11.mlmodel";
P_sxt.stars = true;
P_sxt.unscreen = true;
P_sxt.overlap = 0.50;

var P_abe = new AutomaticBackgroundExtractor;
P_abe.tolerance = 1.000;
P_abe.deviation = 0.800;
P_abe.unbalance = 1.800;
P_abe.minBoxFraction = 0.050;
P_abe.maxBackground = 1.0000;
P_abe.minBackground = 0.0000;
P_abe.useBrightnessLimits = false;
P_abe.polyDegree = 4;
P_abe.boxSize = 5;
P_abe.boxSeparation = 5;
P_abe.modelImageSampleFormat = AutomaticBackgroundExtractor.prototype.f32;
P_abe.abeDownsample = 2.00;
P_abe.writeSampleBoxes = false;
P_abe.justTrySamples = false;
P_abe.targetCorrection = AutomaticBackgroundExtractor.prototype.Subtract;
P_abe.normalize = false;
P_abe.discardModel = true;
P_abe.replaceTarget = true;
P_abe.correctedImageId = "";
P_abe.correctedImageSampleFormat = AutomaticBackgroundExtractor.prototype.SameAsTarget;
P_abe.verboseCoefficients = false;
P_abe.compareModel = false;
P_abe.compareFactor = 10.00;

var P_fr = new FastRotation;
P_fr.mode = FastRotation.prototype.HorizontalMirror;
P_fr.noGUIMessages = true;

var P_ir = new IntegerResample;
P_ir.zoomFactor = -2;
P_ir.downsamplingMode = IntegerResample.prototype.Average;
P_ir.xResolution = 72.000;
P_ir.yResolution = 72.000;
P_ir.metric = false;
P_ir.forceResolution = false;
P_ir.gammaCorrection = false;
P_ir.noGUIMessages = true;

// Open all the files. Store their ImageWindow objects in an array. Element 0 is the
// reference image.
var input_filenames = {input_filenames};
var input_rigs = {input_rigs};
var output_path = "{output_path}";
var output_ids = {output_ids};

var windows = [];
for (let i = 0; i < input_filenames.length; ++i) {{
  let w = ImageWindow.open(input_filenames[i]);
  w[0].show();
  windows.push(w[0]);
}}

// Run ABE on all images.
for (let i = 0; i < windows.length; ++i) {{
  P_abe.executeOn(windows[i].currentView);
}}

// Run BXT on all images.
for (let i = 0; i < windows.length; ++i) {{
  P_bxt.executeOn(windows[i].currentView);
}}

// Run IntegerResample on all C11 images.
for (let i = 0; i < windows.length; ++i) {{
  if (input_rigs[i] == 'C11') {{
    P_ir.executeOn(windows[i].currentView);
  }}
}}

// Run horizontal mirror on all RC71 images.
for (let i = 0; i < windows.length; ++i) {{
  if (input_rigs[i] == 'RC71') {{
    P_fr.executeOn(windows[i].currentView);
  }}
}}

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
"""


def run_post_processing_for_stack(config, project_dir, date):
  print(f"Starting post-processing for date {date}")
  date_dir = os.path.join(project_dir, date.strftime('%Y-%m-%d'))
  output_dir = os.path.join(date_dir, 'post_processed')

  # Find the most recent color channels for that date.
  masters = find_masters(project_dir, date)
  if not masters:
    print(f"Skipping date {date} because there are no masters")
    return

  for master in masters:
    print(f"Found master {master} with max_date {date}")
  masters = find_most_recent_color_channels(masters)
  for master in masters:
    print(f"Using master {master}")

  # Put the reference filename in position 0.
  reference = find_reference(masters)
  assert reference
  print(f'Reference is {reference}')
  masters = [reference] + [m for m in masters if m != reference]

  if os.path.exists(output_dir):
    print(f"Post-processing output directory '{output_dir}' already exists.")
    return

  os.makedirs(output_dir, exist_ok=True)

  for master in masters:
    print('Setting output_id for master', master)
    master['output_id'] = f"{master['filter']}_{master['rig']}"

  for m in masters:
    print(f"Input master {m['stack_filename']} -> output {m['output_id']}")

  input_filenames_string = json.dumps([m['stack_filename'] for m in masters])
  print('input_filenames_string', input_filenames_string)

  input_rigs_string = json.dumps([m['rig'] for m in masters])
  print('input_rigs_string', input_rigs_string)

  output_ids_string = json.dumps([m['output_id'] for m in masters])
  print('output_ids_string', output_ids_string)

  sentinel_filename = os.path.join(output_dir, 'sentinel')

  # rgb_channels = [None] * 3
  # rgb_rig = None
  # focal_length = 0
  # pixel_size = 0
  # ra_degrees = 0
  # dec_degrees = 0
  # for master in masters:
  #   if master['filter'] == 'R':
  #     print('red channel master', master)

  #     # FIXME this is ugly
  #     rgb_rig = [rig for rig in config.rigs if rig.name == master['rig']][0]

  #     header = file_io.read_xisf(master['stack_filename'])
  #     ra_degrees = header['RA']
  #     dec_degrees = header['DEC']

  #     rgb_channels[0] = master['output_id']
  #   elif master['filter'] == 'G':
  #     rgb_channels[1] = master['output_id']
  #   elif master['filter'] == 'B':
  #     rgb_channels[2] = master['output_id']

  # if None in rgb_rig:
  #   rgb_channels = []
  # else:
  #   rgb_channels = json.dumps([[True, x] for x in rgb_channels])

  with tempfile.TemporaryDirectory() as tmp_dir:
    script_filename = os.path.join(tmp_dir, "run_post_processing.js")
    with open(script_filename, "w") as f:
      script = post_process_script.format(
          input_filenames=input_filenames_string,
          reference_basename=Path(reference['stack_filename']).stem,
          input_rigs=input_rigs_string,
          output_path=output_dir,
          output_ids=output_ids_string,
          # rgb_channels=rgb_channels,
          # focal_length=rgb_rig.focal_length,
          # pixel_size=rgb_rig.pixel_size,
          # ra_degrees=ra_degrees,
          # dec_degrees=dec_degrees,
          sentinel_filename=sentinel_filename)
      # print('script', script)
      f.write(script)

    print(f'Running PixInsight post-processing on {len(masters)} masters...')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--execute={script_filename}'
    ]).returncode

    while not os.path.exists(sentinel_filename):
      time.sleep(5)

    os.unlink(sentinel_filename)