import sys
import os


def get_base_path():
  if getattr(sys, 'frozen', False):
    # If the application is frozen (running as a standalone executable)
    return os.path.join(sys._MEIPASS, 'core')
  else:
    # If the application is running in a development environment
    return os.path.dirname(__file__)
