#!/usr/bin/env python
import sys

sys.path.append('../..')

from stack import Stack
from target import Target

stacks = [
    Stack(id='1',
          status=True,
          target='M101',
          rig='C11',
          filter='H',
          exposure_time_s=3600),
    Stack(id='2',
          status=True,
          target='M101',
          rig='C11',
          filter='S',
          exposure_time_s=3600),
    Stack(id='3',
          status=True,
          target='M101',
          rig='C11',
          filter='O',
          exposure_time_s=3600),
    Stack(id='4',
          status=True,
          target='M101',
          rig='C11',
          filter='R',
          exposure_time_s=3600),
    Stack(id='5',
          status=True,
          target='M101',
          rig='C11',
          filter='G',
          exposure_time_s=3600),
    Stack(id='6',
          status=True,
          target='M101',
          rig='C11',
          filter='B',
          exposure_time_s=3600)
]

targets = [Target(status=True, name='M101', stacks=[s.id for s in stacks])]