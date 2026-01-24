import sys
import importlib
import importlib.metadata

# Backup original importlib.metadata.version function
_real_version = importlib.metadata.version


def _safe_version(package_name):
    if package_name == "xisf":
        return "0.0.0"  # Fake version to bypass the error
    return _real_version(package_name)  # Call the original function


# Monkey-patch importlib.metadata.version to prevent xisf import errors
importlib.metadata.version = _safe_version

# Import xisf normally after patching
xisf = importlib.import_module("xisf")

# Expose xisf's attributes so the wrapper behaves like the original module
sys.modules["my_xisf"] = xisf  # Ensure compatibility with absolute imports
globals().update(vars(xisf))
