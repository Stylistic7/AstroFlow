from enum import Enum


class Overlap(Enum):
  IDENTICAL = 1
  NON_OVERLAPPING = 2
  INCOMPATIBLE = 3
  LONG_EDGE = 4
  SHORT_EDGE = 5
