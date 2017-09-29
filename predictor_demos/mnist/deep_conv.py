from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import console
from tframe import pedia

from tframe import Classifier

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Conv2D
from tframe.layers import Dropout
from tframe.layers import Flatten
from tframe.layers import Linear
from tframe.layers import MaxPool2D
from tframe.layers import Input

from tframe import regularizers

from tframe import FLAGS


def main(_):
  console.suppress_logging()
  FLAGS.overwrite = True
  FLAGS.train = True

  # Start
  console.start("MNIST DEEP CONV DEMO")

  # Load data
  mnist = load_mnist(r'..\..\data\MNIST', one_hot=True, validation_size=5000)

  # Define model
  model = Classifier(mark='deep_conv_00')
  reg = regularizers.L2(strength=0.0)

  model.add(Input(sample_shape=[28, 28, 1]))

  model.add(Conv2D(filters=32, kernel_size=5, padding='same',
                   kernel_regularizer=reg))
  model.add(BatchNorm())
  model.add(Activation('relu'))
  model.add(MaxPool2D(2, 2, 'same'))

  model.add(Conv2D(filters=64, kernel_size=5, padding='same',
                   kernel_regularizer=reg))
  model.add(Activation('relu'))
  model.add(MaxPool2D(2, 2, 'same'))

  model.add(Flatten())
  model.add(Linear(512))
  model.add(Activation('relu'))
  model.add(Dropout())

  model.add(Linear(10))

  # Build model
  model.build(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4))

  # Train model
  if FLAGS.train:
    model.train(training_set=mnist[pedia.training],
                validation_set=mnist[pedia.validation],
                epoch=5, batch_size=100, print_cycle=50)
  else:
    model.evaluate_model(mnist[pedia.test])

  # End
  console.end()


if __name__ == "__main__":
  tf.app.run()