from  __future__ import absolute_import
from  __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_cifar10

from tframe import console
from tframe import FLAGS
from tframe import pedia

from tframe.utils import imtool

import models


def main(_):
  console.suppress_logging()
  FLAGS.train = False
  FLAGS.overwrite = True

  # Start
  console.start('CIFAR-10 VANILLA GAN')

  # Get model
  model = models.vanilla('vanilla_00')

  if FLAGS.train:
    cifar10 = load_cifar10('../../data/CIFAR-10', flatten=True,
                           validation_size=0, one_hot=True)
    model.train(training_set=cifar10[pedia.training], epoch=10000,
                batch_size=128, print_cycle=50, snapshot_cycle=200)
  else:
    samples = model.generate(sample_num=16)
    console.show_status('{} samples generated'.format(samples.shape[0]))
    imtool.gan_grid_plot(samples, show=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()


