from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import console

from tframe import Predictor
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Conv2D
from tframe.layers import Dropout
from tframe.layers import Flatten
from tframe.layers import MaxPool2D
from tframe.layers import Input

from tframe import regularizers


def main(_):
  console.suppress_logging()
  # Start
  console.start("MNIST DEEP CONV DEMO")

  # Load data
  mnist = load_mnist(r'..\..\data\MNIST', one_hot=True)

  # Define model
  reg = regularizers.L2(strength=0.1)

  model = Predictor(mark='mnist_deep')
  model.add(Input(sample_shape=[28, 28, 1]))

  model.add(Conv2D(filters=32, kernel_size=5, padding='same',
                   kernel_regularizer=reg))
  model.add(Activation('relu'))
  model.add(MaxPool2D(2, 2, 'same'))

  model.add(Conv2D(filters=64, kernel_size=5, padding='same',
                   kernel_regularizer=reg))
  model.add(Activation('relu'))
  model.add(MaxPool2D(2, 2, 'same'))

  model.add(Flatten())
  model.add(Linear(1024))
  model.add(Activation('relu'))
  model.add(Dropout())

  model.add(Linear(10))

  # Build model
  model.build(metric='accuracy', metric_name='Accuracy')

  # Train model
  model.train(training_set=mnist['train'], test_set=mnist['test'],
              epoch=5, batch_size=100, print_cycle=50)

  # End
  console.end()


if __name__ == "__main__":
  tf.app.run()