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
  FLAGS.train = True
  FLAGS.overwrite = False

  # Start
  console.start('CIFAR-10 DCGAN')

  # Get model
  model = models.dcgan('dcgan_00')

  if FLAGS.train:
    cifar10 = load_cifar10('../../data/CIFAR-10', validation_size=0,
                           one_hot=True)
    model.train(training_set=cifar10[pedia.training], epoch=20000,
                batch_size=128, print_cycle=20, snapshot_cycle=2000)
  else:
    samples = model.generate(sample_num=16)
    console.show_status('{} samples generated'.format(samples.shape[0]))
    imtool.gan_grid_plot(samples, show=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

