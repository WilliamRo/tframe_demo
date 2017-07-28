from tframe import console


class Modules:
  vanilla = 'vanilla'
  dcgan = 'dcgan'


# Decide module to run
module_to_run = Modules.vanilla
postfix = '004'
mark = '{}_{}'.format(module_to_run, postfix)

# overwrite = False
overwrite = True

console.execute_py('./{}.py'.format(module_to_run),
                   epoch=1900,
                   batch_size=128,
                   print_cycle=20,
                   snapshot_cycle=150,
                   fix_sample_z=False,
                   shuffle=False,
                   mark=mark,
                   sample_num=25,
                   train=True,
                   overwrite=overwrite)

"""
This is console for GANs on MNIST

Sample number in training set is 60000
"""

