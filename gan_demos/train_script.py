from tframe import console


class Modules:
  mnist_vanilla_gan = 'mnist_vanilla_gan'
  mnist_dcgan = 'mnist_dcgan'


# Decide module to run
module_to_run = Modules.mnist_vanilla_gan
postfix = '000'
mark = '{}_{}'.format(module_to_run, postfix)

console.execute_py(r'.\{}.py'.format(module_to_run),
                   epoch=150,
                   batch_size=128,
                   print_cycle=50,
                   snapshot_cycle=500,
                   fix_sample_z=False,
                   shuffle=False,
                   mark=mark,
                   sample_num=25,
                   train=True,
                   overwrite=True)

"""
Sample number in training set is 60000
"""

