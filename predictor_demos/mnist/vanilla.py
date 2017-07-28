from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.utils.tfdata import load_mnist

from tframe import console

from tframe import Predictor
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input

from tframe import FLAGS


def main(_):
  console.suppress_logging()
  # Start
  console.start("VANILLA MNIST")

  reg = 'l2'
  kwargs = {'strength':0.001}
  # Define model
  model = Predictor(mark='vanilla00')
  model.add(Input(sample_shape=[28*28]))

  model.add(Linear(output_dim=28*28, weight_regularizer=reg, **kwargs))
  model.add(Activation('relu'))

  model.add(Linear(output_dim=10, weight_regularizer=reg, **kwargs))

  # Build model
  model.build(metric='accuracy', metric_name='Accuracy')

  # Train model
  if FLAGS.train:
    # Get data sets
    mnist = load_mnist('../../data/MNIST', one_hot=True, flatten=True)
    model.train(training_set=mnist['train'], test_set=mnist['test'],
                epoch=10, batch_size=128, print_cycle=50)
  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()
