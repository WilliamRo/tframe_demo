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
  # mnist = load_mnist(r'.\data\MNIST')

  # Define model
  gan = GAN(z_dim=100, sample_shape=[784], mark='vanilla_gan')

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

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
