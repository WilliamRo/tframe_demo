from tframe import console


class Modules:
  mnist_deep = 'mnist_deep'


console.execute_py(r'.\{}.py'.format(Modules.mnist_deep),
                   epoch=1,
                   batch_size=100,
                   shuffle=False,
                   mark='mnist_deep_0',
                   overwrite=True)




