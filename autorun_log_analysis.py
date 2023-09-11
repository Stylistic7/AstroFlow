import re
import datetime
import pytz
from statemachine import StateMachine, State
from dateutil.parser import parse
from pathlib import Path
import os

import time_util
import file_io


class AutorunLogStateMachine(StateMachine):
  inactive = State(initial=True)
  running = State()
  settling = State()

  start_autorun = inactive.to(running)
  end_autorun = running.to(inactive) | settling.to(inactive)

  start_settling = running.to(settling) | settling.to.itself(internal=True)
  end_settling = settling.to(running) | running.to.itself(internal=True)

  def __init__(self):
    self.settling_periods_ = []
    self.settling_start_ = None

    super().__init__()

  def settling_periods(self):
    return self.settling_periods_

  def on_enter_settling(self, d):
    self.settling_start_ = d

  def on_exit_settling(self, d):
    self.settling_periods_.append((self.settling_start_, d))


def load_autorun_log_file(filename):
  assert os.path.exists(filename)

  autorun_start_regex = re.compile('(.*) \[Autorun\|Begin\]')
  autorun_end_regex = re.compile('(.*) \[Autorun\|End\]')

  # settling_start_regex = re.compile('(.*) \[Guide\] (Dither|Guide) Settle')
  settling_start_regex = re.compile('(.*) \[Guide\] Dither Settle')
  settling_end_regex = re.compile('(.*) \[Guide\] Settle Done')
  settling_failed_regex = re.compile('(.*) [Guide] Settle failed')

  g = AutorunLogStateMachine()

  fn_map = {
      autorun_start_regex: g.start_autorun,
      autorun_end_regex: g.end_autorun,
      settling_start_regex: g.start_settling,
      settling_end_regex: g.end_settling,
      settling_failed_regex: g.end_settling
  }

  with open(filename) as f:
    for line in f.readlines():
      for regex, fn in fn_map.items():
        match = regex.search(line)
        if match:
          fn(d=time_util.parse_local_datetime(match.groups()[0]))
          continue

  return g.settling_periods()


def load_autorun_log_files(path):
  settling_periods = []
  for filename in list(Path(path).rglob('Autorun_Log*')):
    if 'CHN' in str(filename):
      continue
    settling_periods.extend(load_autorun_log_file(filename))

  return settling_periods


def remove_frames_during_explicit_settling_periods(light_frames,
                                                   settling_periods,
                                                   tolerance=0.0,
                                                   trim_end=0.0):
  output = []
  for sub in light_frames:
    settling_found = False
    header = file_io.read_fits(sub)
    sub_start = header['DATE-OBS']
    sub_end = sub_start + datetime.timedelta(seconds=header['EXPOSURE'])
    for settling_start, settling_end in settling_periods:
      settling_end -= datetime.timedelta(seconds=trim_end)
      latest_start = max(sub_start, settling_start)
      earliest_end = min(sub_end, settling_end)
      delta = (earliest_end - latest_start).total_seconds()
      overlap = max(0, delta)
      if overlap > tolerance:
        settling_found = True
        break

    if not settling_found:
      output.append(sub)

  return output


def remove_frames_during_settling(light_frames, tolerance=0.0, trim_end=0.0):
  settling_periods = load_autorun_log_files(os.path.commonpath(light_frames))
  print(f'Found {len(settling_periods)} settling periods in log files.')
  return remove_frames_during_explicit_settling_periods(
      light_frames, settling_periods, tolerance, trim_end)
