import tensorflow as tf
from misc import DemoHub

from tframe import Predictor
from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers import Input

from tframe.models.sl.classifier import Classifier




INPUT_SHAPE = [28, 28]
INPUT_DIM = 784


# region : MLP

def mlp_00(th):
  assert isinstance(th, DemoHub)
  # Initiate a predictor
  model = Classifier(th.mark)
  assert isinstance(model, Predictor)

  # Add layers
  model.add(Input([INPUT_DIM]))
  for i in range(th.num_blocks):
    model.add(Linear(output_dim=th.hidden_dim))
    model.add(Activation(th.actype1))
  model.add(Linear(output_dim=10))

  # Build model
  optimizer = tf.train.AdamOptimizer(th.learning_rate)
  model.build(optimizer=optimizer)

  # Return model
  return model

# endregion : MLP















