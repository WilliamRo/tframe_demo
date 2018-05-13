import tensorflow as tf

import sys, os
abspath = os.path.abspath(__file__)
dn = os.path.dirname
demo_root = dn(dn(dn(abspath)))
sys.path.insert(0, demo_root)
sys.path.insert(0, dn(abspath))
sys.path.insert(0, dn(dn(abspath)))
del dn

from tframe import hub
from tframe import console
from tframe import SaveMode
from tframe import pedia
from tframe.utils.tfdata import load_mnist

from misc import DemoHub
import mnist_model_lib as models

hub.data_dir = os.path.join(demo_root, 'data/MNIST')


def main(_):
  console.start('mlp task')

  # Configurations
  th = DemoHub(as_global=True)
  th.num_blocks = 3
  th.multiplier = 2
  th.hidden_dim = models.INPUT_DIM * th.multiplier
  # th.actype1 = 'lrelu'   # Default: relu

  th.epoch = 10
  th.batch_size = 32
  th.learning_rate = 1e-4
  th.validation_per_round = 5
  th.print_cycle = 50

  th.train = True
  # th.smart_train = True
  # th.max_bad_apples = 4
  # th.lr_decay = 0.6

  th.early_stop = True
  th.idle_tol = 20
  th.save_mode = SaveMode.NAIVE
  # th.warm_up_thres = 1
  # th.at_most_save_once_per_round = True

  th.overwrite = True
  th.export_note = True
  th.summary = True
  # th.monitor = True
  th.save_model = True

  th.allow_growth = False
  th.gpu_memory_fraction = 0.40

  description = '0'
  th.mark = 'mlp-{}x{}-{}'.format(th.num_blocks, th.hidden_dim, description)
  # Get model
  model = models.mlp_00(th)
  # Load data
  data_set = load_mnist(th.data_dir, flatten=True, one_hot=True)
  train_set, val_set, test_set = (
    data_set[pedia.training], data_set[pedia.validation], data_set[pedia.test])

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    console.show_status('Evaluating ...')
    model.evaluate_model(train_set)
    model.evaluate_model(val_set)
    model.evaluate_model(test_set)

  # End
  console.end()


if __name__ == '__main__':
  tf.app.run()

