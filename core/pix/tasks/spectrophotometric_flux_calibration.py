"""SpectrophotometricFluxCalibration.

Filter-aware: the gray-channel calibration filter is selected from the
image's FITS ``FILTER`` keyword (single-letter in this pipeline). Emission-
line filters (S/H/O) are calibrated in narrowband mode at the line center
with the physical bandwidth of the Chroma 3 nm filters; broadband ZWO
R/G/B are calibrated in broadband mode against the filter's transmission
curve. Nothing about the filter is hardcoded into the process defaults.
"""

import re

from pix.image import Image
from pix.tasks.js import Enum
from pix.tasks.process import make_process_task

# Narrowband emission-line filters: Chroma 3 nm. FITS FILTER -> line center
# (nm). Ha = 656.3, OIII = 500.7, SII = 671.6 (standard rest wavelengths).
_NARROWBAND_LINE_NM = {
    "H": 656.3,  # H-alpha
    "O": 500.7,  # OIII
    "S": 671.6,  # SII
}
_CHROMA_BANDWIDTH_NM = 3.0

# Broadband filters calibrated against a transmission curve (not a
# narrowband line). FITS FILTER -> (PixInsight grayFilterName, curve).
# Curves are PixInsight's built-in ZWO filter data, verbatim.
# L (luminance): flat / ideal response, not a ZWO bandpass.
_IDEAL_TR_CURVE = "1,1.0,500,1.0,1000,1.0,1500,1.0,2000,1.0,2500,1.0"

_ZWO_B_TR_CURVE = "418,0.003,420,0.007,422,0.012,424,0.045,426,0.098,428,0.337,430,0.695,432,0.774,434,0.815,436,0.849,438,0.873,440,0.886,442,0.899,444,0.891,446,0.891,448,0.895,450,0.909,452,0.929,454,0.932,456,0.932,458,0.935,460,0.94,462,0.945,464,0.948,466,0.944,468,0.94,470,0.939,472,0.943,474,0.946,476,0.947,478,0.947,480,0.947,482,0.947,484,0.949,486,0.951,488,0.953,490,0.954,492,0.953,494,0.952,496,0.952,498,0.955,500,0.958,502,0.961,504,0.959,506,0.957,508,0.954,510,0.948,512,0.939,514,0.93,516,0.918,518,0.914,520,0.895,522,0.828,524,0.599,526,0.311,528,0.144,530,0.065,532,0.038,534,0.016,536,0.005,538,0.004,540,0.003,542,0.002"

_ZWO_G_TR_CURVE = "502,0.003,504,0.004,506,0.005,508,0.008,510,0.018,512,0.028,514,0.062,516,0.131,518,0.275,520,0.552,522,0.741,524,0.879,526,0.946,528,0.969,530,0.965,532,0.961,534,0.957,536,0.964,538,0.972,540,0.976,542,0.978,544,0.981,546,0.982,548,0.982,550,0.982,552,0.981,554,0.976,556,0.971,558,0.969,560,0.975,562,0.978,564,0.978,566,0.977,568,0.975,570,0.973,572,0.973,574,0.973,576,0.973,578,0.97,580,0.966,582,0.962,584,0.954,586,0.946,588,0.934,590,0.876,592,0.818,594,0.586,596,0.238,598,0.132,600,0.049,602,0.025,604,0.014,606,0.008,608,0.003,610,0.003,612,0.002,614,0.002"

_ZWO_R_TR_CURVE = "592,0.003,594,0.004,596,0.006,598,0.014,600,0.023,602,0.051,604,0.078,606,0.211,608,0.443,610,0.732,612,0.918,614,0.953,616,0.96,618,0.957,620,0.964,622,0.972,624,0.979,626,0.982,628,0.978,630,0.974,632,0.972,634,0.97,636,0.97,638,0.975,640,0.978,642,0.979,644,0.98,646,0.98,648,0.981,650,0.982,652,0.982,654,0.982,656,0.982,658,0.982,660,0.984,662,0.989,664,0.988,666,0.987,668,0.986,670,0.984,672,0.982,674,0.979,676,0.978,678,0.979,680,0.98,682,0.984,684,0.985,686,0.985,688,0.985,690,0.985,692,0.985,694,0.985,696,0.973,698,0.951,700,0.916,702,0.854,704,0.775,706,0.658,708,0.53,710,0.391,712,0.267,714,0.156,716,0.115,718,0.081,720,0.05,722,0.034,724,0.02,726,0.015,728,0.011,730,0.009,732,0.007,734,0.005,736,0.003,738,0.002,740,0.001,742,0"

_BROADBAND_FILTERS: dict[str, tuple[str, str]] = {
    "R": ("ZWO R", _ZWO_R_TR_CURVE),
    "G": ("ZWO G", _ZWO_G_TR_CURVE),
    "B": ("ZWO B", _ZWO_B_TR_CURVE),
    "L": ("Ideal QE curve", _IDEAL_TR_CURVE),
}

_default_parameters = {
    "narrowbandMode": False,
    "grayFilterTrCurve": "300,0,380,0,400,1,500,1,675,1,710,0,800,0",
    "grayFilterName": "Astronomik UV-IR Block L-2",
    "redFilterTrCurve": "400,0.088,402,0.084,404,0.080,406,0.076,408,0.072,410,0.068,412,0.065,414,0.061,416,0.058,418,0.055,420,0.052,422,0.049,424,0.046,426,0.044,428,0.041,430,0.039,432,0.037,434,0.035,436,0.033,438,0.031,440,0.030,442,0.028,444,0.027,446,0.026,448,0.025,450,0.024,452,0.023,454,0.022,456,0.021,458,0.021,460,0.021,462,0.020,464,0.020,466,0.020,468,0.020,470,0.020,472,0.021,474,0.021,476,0.022,478,0.022,480,0.023,482,0.024,484,0.025,486,0.026,488,0.027,490,0.028,492,0.029,494,0.031,496,0.032,498,0.034,500,0.036,502,0.037,504,0.039,506,0.041,508,0.043,510,0.045,512,0.048,514,0.050,516,0.052,518,0.055,520,0.057,522,0.060,524,0.063,526,0.071,528,0.072,530,0.070,532,0.067,534,0.064,536,0.059,538,0.054,540,0.050,542,0.045,544,0.041,546,0.037,548,0.034,550,0.032,552,0.031,554,0.031,556,0.032,558,0.035,560,0.038,562,0.043,564,0.048,566,0.055,568,0.062,570,0.070,572,0.122,574,0.187,576,0.262,578,0.346,580,0.433,582,0.521,584,0.606,586,0.686,588,0.755,590,0.812,592,0.851,594,0.871,596,0.876,598,0.885,600,0.892,602,0.896,604,0.897,606,0.897,608,0.895,610,0.891,612,0.887,614,0.882,616,0.878,618,0.873,620,0.870,622,0.867,624,0.863,626,0.860,628,0.858,630,0.856,632,0.854,634,0.852,636,0.850,638,0.848,640,0.846,642,0.844,644,0.841,646,0.837,648,0.834,650,0.829,652,0.824,654,0.819,656,0.813,658,0.806,660,0.799,662,0.791,664,0.783,666,0.774,668,0.765,670,0.755,672,0.745,674,0.735,676,0.725,678,0.715,680,0.704,682,0.695,684,0.685,686,0.676,688,0.668,690,0.660,692,0.654,694,0.649,696,0.648,698,0.649,700,0.649",
    "redFilterName": "Sony Color Sensor R-UVIRcut",
    "greenFilterTrCurve": "400,0.089,402,0.086,404,0.082,406,0.079,408,0.075,410,0.071,412,0.066,414,0.062,416,0.058,418,0.053,420,0.049,422,0.045,424,0.042,426,0.041,428,0.042,430,0.043,432,0.044,434,0.046,436,0.047,438,0.049,440,0.051,442,0.053,444,0.055,446,0.057,448,0.059,450,0.061,452,0.064,454,0.067,456,0.069,458,0.072,460,0.075,462,0.098,464,0.130,466,0.169,468,0.215,470,0.267,472,0.323,474,0.382,476,0.443,478,0.505,480,0.566,482,0.627,484,0.684,486,0.739,488,0.788,490,0.832,492,0.868,494,0.896,496,0.915,498,0.924,500,0.921,502,0.939,504,0.947,506,0.954,508,0.961,510,0.967,512,0.973,514,0.978,516,0.982,518,0.986,520,0.989,522,0.992,524,0.994,526,0.996,528,0.997,530,0.997,532,0.995,534,0.990,536,0.986,538,0.981,540,0.977,542,0.973,544,0.969,546,0.965,548,0.960,550,0.955,552,0.949,554,0.943,556,0.936,558,0.928,560,0.919,562,0.909,564,0.898,566,0.887,568,0.874,570,0.860,572,0.845,574,0.829,576,0.812,578,0.794,580,0.775,582,0.754,584,0.733,586,0.711,588,0.688,590,0.665,592,0.640,594,0.615,596,0.589,598,0.563,600,0.537,602,0.510,604,0.483,606,0.456,608,0.430,610,0.403,612,0.377,614,0.352,616,0.328,618,0.304,620,0.282,622,0.261,624,0.242,626,0.224,628,0.225,630,0.216,632,0.207,634,0.199,636,0.192,638,0.185,640,0.179,642,0.174,644,0.169,646,0.165,648,0.161,650,0.158,652,0.156,654,0.155,656,0.154,658,0.154,660,0.155,662,0.156,664,0.158,666,0.162,668,0.165,670,0.170,672,0.176,674,0.182,676,0.189,678,0.198,680,0.207,682,0.217,684,0.228,686,0.240,688,0.240,690,0.248,692,0.257,694,0.265,696,0.274,698,0.282,700,0.289",
    "greenFilterName": "Sony Color Sensor G-UVIRcut",
    "blueFilterTrCurve": "400,0.438,402,0.469,404,0.496,406,0.519,408,0.539,410,0.557,412,0.572,414,0.586,416,0.599,418,0.614,420,0.631,422,0.637,424,0.647,426,0.658,428,0.670,430,0.682,432,0.695,434,0.708,436,0.720,438,0.732,440,0.743,442,0.753,444,0.762,446,0.770,448,0.777,450,0.783,452,0.788,454,0.791,456,0.794,458,0.796,460,0.797,462,0.798,464,0.798,466,0.799,468,0.800,470,0.801,472,0.800,474,0.798,476,0.793,478,0.785,480,0.774,482,0.760,484,0.742,486,0.707,488,0.669,490,0.633,492,0.598,494,0.565,496,0.533,498,0.502,500,0.473,502,0.446,504,0.419,506,0.394,508,0.370,510,0.348,512,0.326,514,0.306,516,0.287,518,0.268,520,0.251,522,0.235,524,0.220,526,0.205,528,0.192,530,0.179,532,0.167,534,0.156,536,0.145,538,0.136,540,0.126,542,0.118,544,0.110,546,0.102,548,0.095,550,0.089,552,0.083,554,0.077,556,0.071,558,0.066,560,0.061,562,0.057,564,0.052,566,0.048,568,0.044,570,0.039,572,0.041,574,0.039,576,0.037,578,0.035,580,0.033,582,0.032,584,0.030,586,0.029,588,0.027,590,0.026,592,0.025,594,0.024,596,0.023,598,0.022,600,0.022,602,0.021,604,0.021,606,0.020,608,0.020,610,0.020,612,0.020,614,0.020,616,0.020,618,0.021,620,0.021,622,0.022,624,0.022,626,0.023,628,0.024,630,0.025,632,0.026,634,0.027,636,0.028,638,0.030,640,0.031,642,0.033,644,0.035,646,0.036,648,0.038,650,0.040,652,0.042,654,0.045,656,0.048,658,0.051,660,0.054,662,0.057,664,0.059,666,0.061,668,0.063,670,0.065,672,0.066,674,0.068,676,0.069,678,0.070,680,0.071,682,0.072,684,0.072,686,0.073,688,0.073,690,0.073,692,0.073,694,0.073,696,0.073,698,0.073,700,0.073",
    "blueFilterName": "Sony Color Sensor B-UVIRcut",
    "grayFilterWavelength": 656.3,
    "grayFilterBandwidth": 3.0,
    "redFilterWavelength": 656.3,
    "redFilterBandwidth": 3.0,
    "greenFilterWavelength": 500.7,
    "greenFilterBandwidth": 3.0,
    "blueFilterWavelength": 500.7,
    "blueFilterBandwidth": 3.0,
    "deviceQECurve": "1,1.0,500,1.0,1000,1.0,1500,1.0,2000,1.0,2500,1.0",
    "deviceQECurveName": "Ideal QE curve",
    "broadbandIntegrationStepSize": 0.50,
    "narrowbandIntegrationSteps": 10,
    "rejectionLimit": 0.30,
    "catalogId": "GaiaDR3SP",
    "minMagnitude": 0.00,
    # Fixed magnitude limit of 14, with auto-selection disabled.
    #
    # PixInsight's auto-magnitude selection picks a bright cutoff (~8-9) when
    # the master's sky background is high (moon, light pollution), leaving too
    # few catalog stars in the cross-matchable bright tail and tripping
    # "Insufficient data: only N sample(s) available; at least 5 are required"
    # — a catastrophic failure for multi-day mosaic pipelines since one fail
    # in `post_process_stack` derails the entire mosaic-stitching path.
    #
    # mag 14 gives ~6× more catalog candidates than PixInsight's UI default of
    # 12 (each magnitude step ≈ 2.5×). Gaia DR3 BP/RP spectra are still
    # high-S/N at mag 14 (mission spec is reliable spectra to ~mag 13.5; the
    # rejection-band beyond it falls inside `rejectionLimit=0.30`). The cost
    # is negligible — local XPSD query, PSF fitting still capped at
    # `psfMaxStars=24576`, downstream quality filters (`psfMinSNR=40`,
    # `rejectionLimit`) drop low-S/N matches regardless of limit.
    "limitMagnitude": 14.00,
    "autoLimitMagnitude": False,
    "psfStructureLayers": 5,
    "saturationThreshold": 0.75,
    "saturationRelative": True,
    "saturationShrinkFactor": 0.10,
    "psfNoiseLayers": 1,
    "psfHotPixelFilterRadius": 1,
    "psfNoiseReductionFilterRadius": 0,
    "psfMinStructureSize": 0,
    "psfMinSNR": 40.00,
    "psfAllowClusteredSources": False,
    "psfType": Enum("SpectrophotometricFluxCalibration.PSFType_Auto"),
    "psfGrowth": 1.75,
    "psfMaxStars": 24576,
    "psfSearchTolerance": 4.00,
    "psfChannelSearchTolerance": 2.00,
    "generateGraphs": True,
    "generateStarMaps": False,
    "generateTextFiles": False,
    "outputDirectory": "",
}

_run_sfc = make_process_task(
    "SpectrophotometricFluxCalibration",
    _default_parameters,
    # Per-step error_patterns are checked UNCONDITIONALLY by
    # `_check_for_errors` — unlike the universal `*** Error:` catch-all,
    # which exempts any recipe containing a success_pattern step. That
    # exemption is exactly how a failed SFC slipped through on 2026-08-20
    # (stale /Volumes/Astro Gaia paths; output kept the input's old
    # PSFFLX00 stamp).
    #
    # Patterns are phrase-only (no `*** Error:` prefix): PixInsight wraps
    # long messages across lines and `re.search` does not span newlines.
    error_patterns=[
        re.compile(r"no astrometric solution"),
        re.compile(r"Invalid Gaia DR3/SP XPSD server execution"),
        re.compile(r"File I/O Error: Unable to open file"),
        # A failed marker after SFC starts is a hard failure regardless of
        # which later line printed it.
        re.compile(r"SpectrophotometricFluxCalibration: Processing view"
                   r"[\s\S]*?<\* failed \*>"),
    ],
)


def _gray_filter_overrides(filter: str | None) -> dict:
    """SFC parameter overrides selecting the gray-channel filter from the
    image's FITS FILTER keyword. Fail-fast on anything we can't calibrate
    correctly rather than silently calibrating against the wrong filter."""
    if filter is None:
        raise ValueError(
            "SpectrophotometricFluxCalibration: image has no FITS FILTER; "
            "cannot select a calibration filter."
        )
    key = filter.strip().upper()[:1]

    if key in _NARROWBAND_LINE_NM:
        line_nm = _NARROWBAND_LINE_NM[key]
        return {
            "narrowbandMode": True,
            "grayFilterWavelength": line_nm,
            "grayFilterBandwidth": _CHROMA_BANDWIDTH_NM,
            "grayFilterName": f"Chroma 3nm ({key})",
        }

    if key in _BROADBAND_FILTERS:
        gray_filter_name, tr_curve = _BROADBAND_FILTERS[key]
        return {
            "narrowbandMode": False,
            "grayFilterName": gray_filter_name,
            "grayFilterTrCurve": tr_curve,
        }

    raise ValueError(
        f"SpectrophotometricFluxCalibration: unsupported FITS FILTER "
        f"{filter!r} (expected one of S/H/O/R/G/B/L)."
    )


def SpectrophotometricFluxCalibration(image: Image, **kwargs) -> Image:
    """Spectrophotometric flux calibration with the gray filter chosen from
    ``image.filter``. Explicit ``kwargs`` still override everything.

    Fails loudly (see `_run_sfc` error_patterns) when the Gaia DR3/SP
    XPSD database is unreachable — the database file locations live in
    PixInsight's global preferences and must be updated there after any
    volume rename (2026-08-20: stale /Volumes/Astro paths made every SFC
    a silent no-op that kept the input's old PSFFLX00 stamp)."""
    return _run_sfc(image, **(_gray_filter_overrides(image.filter) | kwargs))
