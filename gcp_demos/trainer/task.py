import os
import tensorflow as tf

import trainer.model_lib as model_lib

from tframe import console
from tframe import FLAGS
from tframe import pedia

from tframe.utils.tfdata import load_cifar10tfd


def main(_):
  console.start('GCP CIFAR-10')

  # Configuration
  if FLAGS.use_default:
    FLAGS.train = False
    FLAGS.overwrite = False
    FLAGS.smart_train = False
    FLAGS.save_best = True
    FLAGS.dont_save_until = 1

  pa = os.path.dirname
  MARK = 'dc00'
  DATA_DIR = os.path.join(pa(pa(pa(__file__))), r'data\CIFAR-10')
  LEARNING_RATE = 0.0001
  BATCH_SIZE = 32
  EPOCH = 10

  # Load data
  cifar10 = load_cifar10tfd(DATA_DIR, validation_size=5000)

  # Initialize model
  model = model_lib.deep_conv(MARK, LEARNING_RATE)

  # Train
  if FLAGS.train:
    model.train(epoch=EPOCH, batch_size=BATCH_SIZE,
                training_set=cifar10[pedia.training],
                validation_set=cifar10[pedia.validation],
                print_cycle=10)
  else: model.evaluate_model(data=cifar10[pedia.test], with_false=True)

  console.end()


if __name__ == '__main__':
  tf.app.run()