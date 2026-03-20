"""FastRotation — image rotation/mirroring."""
from pix2.tasks.js import Enum
from pix2.tasks.process import make_process_task

FastRotation = make_process_task("FastRotation", {
    "mode": Enum("FastRotation.prototype.HorizontalMirror"),
    "noGUIMessages": True,
})
