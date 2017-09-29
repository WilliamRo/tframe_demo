from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import console
from tframe import FLAGS
from tframe import pedia

from tframe import Classifier

from tframe.layers import Activation
from tframe.layers import BatchNorm
from tframe.layers import Linear
from tframe.layers import Input

from tframe.utils.tfdata import load_mnist


def main(_):
  console.suppress_logging()
  FLAGS.overwrite = True
  FLAGS.train = True

  # Start
  console.start()

  model = Classifier(mark='pre_bn_01')
  model.add(Input(sample_shape=[784]))

  def fc_bn_relu(bn=False):
    model.add(Linear(100))
    if bn:
      model.add(BatchNorm())
    model.add(Activation('relu'))

  fc_bn_relu(True)
  fc_bn_relu(True)

  model.add(Linear(10))

  # Build model
  model.build(loss='cross_entropy',
              optimizer=tf.train.GradientDescentOptimizer(0.01))

  mnist = load_mnist('../../data/MNIST', flatten=True, validation_size=5000,
                       one_hot=True)
  # Train or test
  if FLAGS.train:
    model.train(training_set=mnist[pedia.training],
                validation_set=mnist[pedia.validation], epoch=50,
                batch_size=128, print_cycle=50)
  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()