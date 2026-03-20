"""GradientCorrection — gradient removal."""
from pix2.tasks.process import make_process_task

GradientCorrection = make_process_task("GradientCorrection", {
    "reference": 0.50,
    "lowThreshold": 0.20,
    "lowTolerance": 0.50,
    "highThreshold": 0.05,
    "highTolerance": 0.00,
    "iterations": 15,
    "scale": 5.00,
    "smoothness": 0.40,
    "downsamplingFactor": 16,
    "protection": True,
    "protectionThreshold": 0.10,
    "protectionAmount": 0.50,
    "protectionSmoothingFactor": 16,
    "lowClippingLevel": 0.000076,
    "automaticConvergence": False,
    "convergenceLimit": 0.00001000,
    "maxIterations": 10,
    "useSimplification": False,
    "simplificationDegree": 1,
    "simplificationScale": 1024,
    "generateSimpleModel": False,
    "generateGradientModel": False,
    "generateProtectionMasks": False,
    "gridSamplingDelta": 16,
})
