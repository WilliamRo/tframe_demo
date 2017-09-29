from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import FLAGS
from tframe import console
from tframe import pedia

from tframe.utils import imtool

import models


def main(_):
  console.suppress_logging()
  # Start
  console.start("MNIST DCGAN DEMO")

  # Get model
  model = models.dcgan('dcgan_c00')
  # model = models.dcgan_h3_rs_nbn()

  # Train or test
  if FLAGS.train:
    mnist = load_mnist('../../data/MNIST', flatten=False, validation_size=0,
                       one_hot=True)
    model.train(training_set=mnist[pedia.training], epoch=5, batch_size=128,
                print_cycle=20, snapshot_cycle=200, D_times=1, G_times=1)
  else:
    samples = model.generate(sample_num=16)
    console.show_status('{} samples generated'.format(samples.shape[0]))
    imtool.gan_grid_plot(samples, show=True)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()