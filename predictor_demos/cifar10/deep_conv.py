from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_cifar10

from tframe import FLAGS
from tframe import console
from tframe import pedia

from predictor_demos.cifar10 import models


def main(_):
  console.suppress_logging()

  # Setting
  # FLAGS.train = False
  FLAGS.overwrite = True
  # FLAGS.shuffle = True

  # Start
  console.start('CIFAR-10 CONV DEMO')

  # Get model
  # model = models.deep_conv('dper_do0p5_reg0p2')
  model = models.ka_convnet('ka_05_bn1')

  # Train or test
  cifar10 = load_cifar10('../../data/CIFAR-10', flatten=False,
                         validation_size=5000, one_hot=True)
  if FLAGS.train:
    model.train(training_set=cifar10[pedia.training],
                validation_set=cifar10[pedia.validation], epoch=120,
                batch_size=64, print_cycle=100)
  else:
    model.evaluate_model(cifar10[pedia.test])

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
