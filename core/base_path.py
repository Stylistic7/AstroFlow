import sys
import os
from pathlib import Path


def get_base_path() -> str:
  if getattr(sys, 'frozen', False):
    # If the application is frozen (running as a standalone executable)
    return sys._MEIPASS
  else:
    # If the application is running in a development environment
    return Path(__file__).resolve().parent.parent
