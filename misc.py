import sys, os

from tframe.config import Flag
from tframe.trainers import SmartTrainerHub


class DemoHub(SmartTrainerHub):
  num_blocks = Flag.integer(1, '...', is_key=None)
  hidden_dim = Flag.integer(80, '...')
  multiplier = Flag.integer(8, '...', is_key=True)
  start_at = Flag.integer(0, '...', is_key=None)
  branch_index = Flag.integer(0, '..')

DemoHub.register()

