"""FastRotation — image rotation/mirroring."""
from pix.tasks.js import Enum
from pix.tasks.process import make_process_task

FastRotation = make_process_task("FastRotation", {
    "mode": Enum("FastRotation.prototype.HorizontalMirror"),
    "noGUIMessages": True,
})
