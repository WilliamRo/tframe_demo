from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import console
from tframe import FLAGS
from tframe import pedia

from tframe.utils import imtool

import models


def main(_):
  console.suppress_logging()
  FLAGS.train = True
  FLAGS.overwrite = True

  # Start
  console.start('MNIST VANILLA VAE')

  # Get model
  model = models.vanilla('vanilla_00')

  if FLAGS.train:
    mnist = load_mnist('../../data/MNIST', flatten=True,
                       validation_size=0, one_hot=True)
    model.train(training_set=mnist[pedia.training], epoch=1000,
                batch_size=128, print_cycle=50, snapshot_cycle=200)
  else:
    samples = model.generate(sample_num=16)
    console.show_status('{} samples generated'.format(samples.shape[0]))
    imtool.gan_grid_plot(samples, show=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
