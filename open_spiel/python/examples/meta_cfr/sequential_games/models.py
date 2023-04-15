# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model definitions for optimizer network."""

import enum
from typing import Any, Callable, List, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


class ModelType(enum.Enum):
  MLP = "MLP"
  RNN = "RNN"


def _mlp_forwards(mlp_hidden_sizes: List[int]) -> hk.Transformed:
  """Returns a haiku transformation of the MLP model to be used in optimizer.

  Args:
    mlp_hidden_sizes: List containing size of linear layers.

  Returns:
    Haiku transformation of the RNN network.
  """
  def forward_fn(inputs):
    mlp = hk.nets.MLP(mlp_hidden_sizes, activation=jax.nn.relu, name="mlp")
    return mlp(inputs)
  return hk.transform(forward_fn)


def _make_rnn_network(lstm_hidden_sizes: List[int],
                      mlp_hidden_sizes: List[int]) -> hk.RNNCore:
  """Returns the RNN network.

  Args:
    lstm_hidden_sizes: List containing size of lstm layers.
    mlp_hidden_sizes: List containing size of linear layers.

  Returns:
    Returns an instance of RNN model.
  """
  layers = []
  for k, hidden_size in enumerate(lstm_hidden_sizes):
    layers += [hk.LSTM(hidden_size, name=f"lstm_layer_{k}"), jax.nn.relu]
  layers += [hk.nets.MLP(mlp_hidden_sizes, name="mlp")]
  return RNNModel(layers)


def _rnn_forwards(lstm_hidden_sizes: List[int], mlp_hidden_sizes: List[int],
                  batch_size: int) -> hk.Transformed:
  """Returns a haiku transformation of the RNN model to be used in optimizer.

  Args:
    lstm_hidden_sizes: List containing size of lstm layers.
    mlp_hidden_sizes: List containing size of linear layers.
    batch_size: Batch size.

  Returns:
    Haiku transformation of the RNN network.
  """
  def forward_fn(inputs):
    rnn = _make_rnn_network(lstm_hidden_sizes, mlp_hidden_sizes)
    initial_state = rnn.initial_state(batch_size=batch_size)
    outputs, _ = hk.dynamic_unroll(rnn, inputs, initial_state, time_major=False)
    return outputs

  return hk.transform(forward_fn)


class RNNModel(hk.RNNCore):
  """RNN model."""

  def __init__(self,
               layers: List[Union[hk.Module, Callable[[jnp.ndarray],
                                                      jnp.ndarray]]],
               name: Optional[str] = None):
    super().__init__(name=name)
    self._layers = layers

  def __call__(self, inputs, prev_state):
    x = inputs
    curr_state = [None] * len(prev_state)
    for k, layer in enumerate(self._layers):
      if isinstance(layer, hk.RNNCore):
        x, curr_state[k] = layer(x, prev_state[k])
      else:
        x = layer(x)
    return x, tuple(curr_state)

  def initial_state(self, batch_size: Optional[int]) -> Any:
    layerwise_init_state = []
    for layer in self._layers:
      if isinstance(layer, hk.RNNCore):
        layerwise_init_state.append(layer.initial_state(batch_size))
      else:
        layerwise_init_state.append(None)
    return tuple(layerwise_init_state)


class OptimizerModel:
  """Optimizer model in l2l paradigm to learn update rules of regret minimizers.

  Attributes:
    mlp_sizes: Size of mlp layers. This is a string, containing sequence of
      numbers, each number indicate size of a linear layer.
    lstm_sizes: Size of lstm layers. This is a string, containing sequence of
      numbers, each number indicate size of an lstm layer.
    initial_learning_rate: Initial value of learning rate used in learning
      rate scheduler.
    batch_size: Batch size.
    num_actions: Number of possible actions.
    num_infostates: Total number of information states.
    model_type: Type of model. For now it can be either MLP or RNN.
    use_infostate_representation: Boolean value to indicate if we use
      information state information as part of model input or not.
    rng: Jax pseudo random number generator.
    model: Neural network model we want to optimize.
    opt_update: Optax optimizer update function.
    net_params: Network parameters.
    opt_state: Optax optimizer state.
    net_apply: Network apply function.
  """

  def __init__(self,
               mlp_sizes: str,
               lstm_sizes: str,
               initial_learning_rate: float,
               batch_size: int,
               num_actions: int,
               num_infostates: int,
               model_type: str = "MLP",
               use_infostate_representation: bool = True):
    self.num_actions = num_actions
    self.num_infostates = num_infostates
    self.initial_learning_rate = initial_learning_rate
    self.batch_size = batch_size
    self.use_infostate_representation = use_infostate_representation
    self.rng = jax.random.PRNGKey(10)

    mlp_sizes_list = [
        int(mlp_size.strip()) for mlp_size in mlp_sizes.split(",")
    ]
    mlp_sizes_list.append(self.num_actions)
    lstm_sizes_list = [
        int(lstm_size.strip()) for lstm_size in lstm_sizes.split(",")
    ]

    if model_type == ModelType.MLP.value:
      self.model = _mlp_forwards(mlp_sizes_list)
    elif model_type == ModelType.RNN.value:
      self.model = _rnn_forwards(lstm_sizes_list, mlp_sizes_list,
                                 self.batch_size)
    else:
      raise ValueError(
          f"{model_type} is not a valid model, model_type should be MLP or RNN."
      )

    self.net_apply = self.model.apply
    self._net_init = self.model.init
    self.opt_update, self.net_params, self.opt_state = None, None, None

  def lr_scheduler(self, init_value: float) -> optax.Schedule:
    schedule_fn = optax.polynomial_schedule(
        init_value=init_value, end_value=0.0001, power=1., transition_steps=100)
    return schedule_fn

  def initialize_optimizer_model(self):
    """Initializes the optax optimizer and neural network model."""
    lr_scheduler_fn = self.lr_scheduler(self.initial_learning_rate)
    opt_init, self.opt_update = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(lr_scheduler_fn),
        optax.scale(-self.initial_learning_rate))

    input_size = self.num_actions
    if self.use_infostate_representation:
      input_size += self.num_infostates

    dummy_input = np.zeros(shape=[self.batch_size, 1, input_size])

    self.net_params = self._net_init(self.rng, dummy_input)
    self.opt_state = opt_init(self.net_params)
