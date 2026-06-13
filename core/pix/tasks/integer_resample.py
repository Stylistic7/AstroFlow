"""IntegerResample — integer factor resampling."""
from pix.tasks.js import Enum
from pix.tasks.process import make_process_task

IntegerResample = make_process_task("IntegerResample", {
    "zoomFactor": -2,
    "downsamplingMode": Enum("IntegerResample.Average"),
    "xResolution": 72.000,
    "yResolution": 72.000,
    "metric": False,
    "forceResolution": False,
    "gammaCorrection": False,
    "noGUIMessages": True,
})
