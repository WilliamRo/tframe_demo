from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import console
from tframe import FLAGS
from tframe import pedia

import models

from tframe.utils.tfdata import load_mnist


def main(_):
  console.suppress_logging()

  FLAGS.train = False
  FLAGS.overwrite = True
  show_false = True
  flatten = False

  # Start
  console.start('MNIST DEMO')

  # model = models.vanilla('003_post')
  model = models.deep_conv('dc_000')

  mnist = load_mnist('../../data/MNIST', flatten=flatten, validation_size=5000,
                       one_hot=True)
  # Train or test
  if FLAGS.train:
    model.train(training_set=mnist[pedia.training],
                validation_set=mnist[pedia.validation], epoch=30,
                batch_size=100, print_cycle=50)
  else:
    model.evaluate_model(mnist[pedia.test], with_false=show_false)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()