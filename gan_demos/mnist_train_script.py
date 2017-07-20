from tframe import console


class Modules:
  mnist_vanilla_gan = 'mnist_vanilla_gan'


console.execute_py(r'.\{}.py'.format(Modules.mnist_vanilla_gan),
                   epoch=2,
                   batch_size=128,
                   print_cycle=50,
                   snapshot_cycle=500,
                   fix_sample_z=False,
                   shuffle=False,
                   mark='mnist_vanilla_0',
                   sample_num=25,
                   train=False,
                   overwrite=False)

"""
Sample number in training set is 60000
"""

