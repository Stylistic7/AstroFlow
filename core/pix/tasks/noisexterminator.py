"""NoiseXTerminator — AI-based noise reduction."""
from pix.tasks.process import make_process_task

NoiseXTerminator = make_process_task("NoiseXTerminator", {
    "ai_file": "NoiseXTerminator.2.mlpackage",
    "denoise": 0.90,
    "detail": 0.15,
})
