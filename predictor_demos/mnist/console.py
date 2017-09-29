from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import console


class Modules:
  deep_conv = 'deep_conv'
  vanilla = 'vanilla'


# Decide  module to run
module_to_run = Modules.vanilla
postfix = '000'
mark = '{}_{}'.format(module_to_run, postfix)

console.execute_py('./{}.py'.format(module_to_run),
                   epoch=10,
                   batch_size=100,
                   shuffle=False,
                   mark=mark,
                   overwrite=True)




