import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import console

from tframe import GAN
from tframe import pedia
from tframe.layers import Linear
from tframe.layers import Activation


def main(_):
  console.suppress_logging()
  # Start
  console.start()

  # Load data
  mnist = load_mnist(r'..\data\MNIST', flatten=True, validation_size=0)

  # Define model
  gan = GAN(z_dim=100, sample_shape=[784], mark='vanilla_gan',
            output_shape=[28, 28, 1])

  gan.G.add(Linear(output_dim=128))
  gan.G.add(Activation('relu'))
  gan.G.add(Linear(output_dim=784))
  gan.G.add(Activation('sigmoid'))

  gan.D.add(Linear(output_dim=128))
  gan.D.add(Activation('relu'))
  gan.D.add(Linear(output_dim=1))
  gan.D.add(Activation('sigmoid'))

  # Build model
  gan.build(loss=pedia.cross_entropy)

  # Train model
  gan.train(training_set=mnist['train'], epoch=200, batch_size=128,
            print_cycle=50, snapshot_cycle=100)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()