import re
import datetime
import pytz
from statemachine import StateMachine, State
from dateutil.parser import parse
from pathlib import Path


class PhdGuideLogStateMachine(StateMachine):
  inactive = State(initial=True)
  guiding = State()
  settling = State()

  start_guiding = inactive.to(guiding)
  guiding_step = guiding.to.itself(internal=True) | settling.to.itself(
      internal=True)
  end_guiding = guiding.to(inactive) | settling.to(
      inactive) | inactive.to.itself(internal=True)

  start_settling = guiding.to(settling) | inactive.to.itself(internal=True)
  end_settling = settling.to(guiding) | guiding.to.itself(internal=True)
  settling_failed = settling.to(guiding) | guiding.to.itself(
      internal=True) | inactive.to.itself(internal=True)

  no_star_found = star_lost = inactive.to.itself(
      internal=True) | guiding.to(guiding) | settling.to(settling)

  def __init__(self):
    self.guiding_periods_ = []
    self.settling_periods_ = []

    self.settling_start_ = None

    self.guide_start_datetime_ = None
    self.current_datetime_ = None
    self.period_start_datetime_ = None
    self.exposure_length_ = None

    super().__init__()

  def set_exposure_length(self, t):
    self.exposure_length_ = t

  def guiding_periods(self):
    return self.guiding_periods_

  def settling_periods(self):
    return self.settling_periods_

  def on_enter_settling(self):
    self.settling_start_ = self.current_datetime_

  def on_exit_settling(self):
    if (self.current_datetime_ - self.settling_start_).total_seconds() > 0:
      self.settling_periods_.append(
          (self.settling_start_, self.current_datetime_))

  def on_enter_guiding(self, event, source, target, d=None):
    if event == 'guiding_step':
      return

    if d:
      self.guide_start_datetime_ = self.current_datetime_ = self.period_start_datetime_ = d
    else:
      self.period_start_datetime_ = self.current_datetime_

  def on_guiding_step(self, event, source, target, t):
    if t:
      self.current_datetime_ = self.guide_start_datetime_ + datetime.timedelta(
          seconds=t)

    if source is not self.guiding:
      self.period_start_datetime_ = self.current_datetime_

  def on_exit_guiding(self, event, source, target, d=None):
    if d:
      period = (self.period_start_datetime_, d)
    else:
      period = (self.period_start_datetime_, self.current_datetime_)

    if (period[1] - period[0]).total_seconds() > 0:
      self.guiding_periods_.append(period)


def load_phd_guide_log(filename):
  guide_start_regex = re.compile('Guiding Begins at (.*)')
  guide_end_regex = re.compile('Guiding Ends at (.*)')
  guide_step_regex = re.compile('\d+,([1-9]\d*(\.\d+)?),"Mount"')

  exposure_regex = re.compile('Exposure = (\d+) ms')

  def parse_datetime(s):
    local = pytz.timezone("America/Los_Angeles")
    naive = parse(s)
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt

  g = PhdGuideLogStateMachine()
  with open(filename) as f:
    for line in f.readlines():
      match = exposure_regex.search(line)
      if match:
        g.set_exposure_length(int(match.groups()[0]))
        continue

      match = guide_start_regex.search(line)
      if match:
        g.start_guiding(d=parse_datetime(match.groups()[0]))
        continue

      match = guide_end_regex.search(line)
      if match:
        g.end_guiding(d=parse_datetime(match.groups()[0]))
        continue

      match = guide_step_regex.search(line)
      if match:
        # print('found guiding step', match.groups()[0])
        g.guiding_step(t=float(match.groups()[0]))
        continue

      fn_map = {
          'Settling started': g.start_settling,
          'Settling complete': g.end_settling,
          'Settling failed': g.settling_failed
      }

      for s, fn in fn_map.items():
        if s in line:
          fn()

  return g.guiding_periods(), g.settling_periods()


def load_phd_guide_logs(path):
  guiding_periods = []
  settling_periods = []
  for filename in list(Path(path).rglob('PHD2_GuideLog*')):
    g, s = load_phd_guide_log(filename)
    guiding_periods.extend(g)
    settling_periods.extend(s)

  return guiding_periods, settling_periods
