"""AutomaticBackgroundExtractor — polynomial background modelling.

Full parameter set as reported by PixInsight 1.9.4 (`new
AutomaticBackgroundExtractor().toSource()`), so the mode is changeable at the
call site instead of hiding behind PI's defaults.

TWO DEFAULTS BITE, and both are why a naive `AutomaticBackgroundExtractor(img)`
appears to do nothing at all:

  * `targetCorrection` defaults to `Correction_None` — ABE *models* the
    background and applies NOTHING;
  * `replaceTarget` defaults to False — even when a correction is selected, it
    lands in a NEW view (`correctedImageId`), leaving the target untouched.
    Our recipe saves the target view, so the result is bit-identical input.

So an in-place correction needs BOTH `targetCorrection=` and
`replaceTarget=True`.

SUBTRACT vs DIVIDE matters physically. Sky gradients (light pollution, moon)
are ADDITIVE — photons added on top — so they subtract out. Flat-field and
vignetting residual is MULTIPLICATIVE — a throughput error that scales with
signal — so it must be DIVIDED out. Subtracting a multiplicative error flattens
the background while leaving it wrong wherever the object is bright.

`polyDegree` defaults to 4, which is high enough to absorb real large-scale
nebulosity. For a residual flat (a smooth bowl) 1-2 is plenty and much safer.
"""
from pix.tasks.process import make_process_task

# `targetCorrection` enum (PixInsight AutomaticBackgroundExtractor).
CORRECTION_NONE = 0
CORRECTION_SUBTRACT = 1
CORRECTION_DIVIDE = 2

# `modelImageSampleFormat` / `correctedImageSampleFormat` enums.
MODEL_FORMAT_F32 = 2
CORRECTED_FORMAT_SAME_AS_TARGET = 4

AutomaticBackgroundExtractor = make_process_task(
    "AutomaticBackgroundExtractor",
    {
        # --- sample selection ---
        "tolerance": 1.000,
        "deviation": 0.800,
        "unbalance": 1.800,
        "minBoxFraction": 0.050,
        "maxBackground": 1.0000,
        "minBackground": 0.0000,
        "useBrightnessLimits": False,
        "boxSize": 5,
        "boxSeparation": 5,
        # --- model ---
        "polyDegree": 4,
        "abeDownsample": 2.00,
        # --- what actually gets applied (see module docstring) ---
        "targetCorrection": CORRECTION_NONE,
        "replaceTarget": False,
        "normalize": False,
        "discardModel": False,
        # --- outputs / diagnostics ---
        "modelImageSampleFormat": MODEL_FORMAT_F32,
        "correctedImageId": "",
        "correctedImageSampleFormat": CORRECTED_FORMAT_SAME_AS_TARGET,
        "writeSampleBoxes": False,
        "justTrySamples": False,
        "verboseCoefficients": False,
        "compareModel": False,
        "compareFactor": 10.00,
    },
)
