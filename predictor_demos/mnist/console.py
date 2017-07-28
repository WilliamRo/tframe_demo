from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import console


class Modules:
  deep_conv = 'deep_conv'
  vanilla = 'vanilla'


# Decide  module to run
module_to_run = Modules.vanilla
postfix = 'l20001'
mark = '{}_{}'.format(module_to_run, postfix)

console.execute_py(r'.\{}.py'.format(module_to_run),
                   epoch=10,
                   batch_size=100,
                   shuffle=False,
                   mark=mark,
                   overwrite=True)




