import collections
import functools
import os
from typing import Any, Dict, Sequence, Tuple, Optional
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import checkpoints, train_state

def flatten(x):
  return x.reshape((x.shape[0], -1))

class TrainInput(collections.namedtuple(
    "TrainInput", "observation legals_mask policy value")):

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy, value = zip(*train_inputs)
    return TrainInput(
        np.array(observation, dtype=np.float32),
        np.array(legals_mask, dtype=bool),
        np.array(policy, dtype=np.float32),
        np.expand_dims(np.array(value, dtype=np.float32), 1))


class Losses(collections.namedtuple("Losses", "policy value l2")):

  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(self.policy + other.policy,
                  self.value + other.value,
                  self.l2 + other.l2)

  def __truediv__(self, n):
    return Losses(self.policy / n, self.value / n, self.l2 / n)


class ResidualBlock(nn.Module):
  filters: int
  kernel_size: int
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    residual = x
    y = nn.Conv(features=self.filters, kernel_size=(self.kernel_size, self.kernel_size), padding='SAME')(x)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = nn.relu(y)
    y = nn.Conv(features=self.filters, kernel_size=(self.kernel_size, self.kernel_size), padding='SAME')(y)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = y + residual
    y = nn.relu(y)
    return y


class PolicyHead(nn.Module):
  model_type: str
  nn_width: int
  output_size: int
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    if self.model_type == "mlp":
      x = nn.Dense(features=self.nn_width)(x)
      x = nn.relu(x)
    else:
      x = nn.Conv(features=2, kernel_size=(1, 1), padding='SAME')(x)
      x = nn.BatchNorm(use_running_average=not training)(x)
      x = nn.relu(x)
      x = flatten(x)

    policy_logits = nn.Dense(features=self.output_size)(x)
    return policy_logits


class ValueHead(nn.Module):
  model_type: str
  nn_width: int
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    if self.model_type != "mlp":
      x = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME')(x)
      x = nn.BatchNorm(use_running_average=not training)(x)
      x = nn.relu(x)
      x = flatten(x)
    
    x = nn.Dense(features=self.nn_width)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    x = nn.tanh(x)
    return x


class AlphaZeroModel(nn.Module):
  model_type: str
  input_shape: Tuple[int, ...]
  output_size: int
  nn_width: int
  nn_depth: int
  
  @nn.compact
  def __call__(self, observations, training: bool = False):
    #input_size = int(np.prod(self.input_shape))
    
    # torso
    if self.model_type == "mlp":
      x = observations
      for i in range(self.nn_depth):
        x = nn.Dense(features=self.nn_width)(x)
        x = nn.relu(x)
    elif self.model_type == "conv2d":
      x = observations.reshape((-1,) + self.input_shape)
      for i in range(self.nn_depth):
        x = nn.Conv(features=self.nn_width, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
    elif self.model_type == "resnet":
      x = observations.reshape((-1,) + self.input_shape)
      x = nn.Conv(features=self.nn_width, kernel_size=(3, 3), padding='SAME')(x)
      x = nn.BatchNorm(use_running_average=not training)(x)
      x = nn.relu(x)
      for i in range(self.nn_depth):
        x = ResidualBlock(filters=self.nn_width, kernel_size=3)(x, training)

    else:
      raise ValueError(f"Unknown model type: {self.model_type}")
    
    policy_logits = PolicyHead(model_type=self.model_type, nn_width=self.nn_width, output_size=self.output_size)(x, training)
    value_out = ValueHead(model_type=self.model_type, nn_width=self.nn_width)(x, training)
    
    return policy_logits, value_out


class TrainState(train_state.TrainState):
  batch_stats: Any


class Model(object):
  valid_model_types = ['mlp', 'conv2d', 'resnet']

  def __init__(self, model, state, path, loss_fn, update_step_fn):
    self._model = model
    self._state = state
    self._path = path
    self._loss_fn = loss_fn
    self._update_step_fn = update_step_fn

  @classmethod
  def build_model(cls, model_type, input_shape, output_size, nn_width, nn_depth,
                  weight_decay, learning_rate, path):
    if model_type not in cls.valid_model_types:
      raise ValueError(f"Invalid model type: {model_type}, "
                       f"expected one of: {cls.valid_model_types}")
    
    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    elif isinstance(input_shape, list):
      input_shape = tuple(input_shape)

    while len(input_shape) < 3:
      input_shape = input_shape + (1,)
    
    model = AlphaZeroModel(model_type = model_type, input_shape = input_shape, output_size = output_size, nn_width = nn_width, nn_depth = nn_depth)
    
    optimizer = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(0)
    input_shape_with_batch = (1, int(np.prod(input_shape)))
    variables = model.init(rng, jnp.ones(input_shape_with_batch), training=False)

    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=optimizer, batch_stats=variables.get('batch_stats', {}))
    
    def loss_fn(params, batch_stats, observations, legals_mask, policy_targets, value_targets):
      variables = {'params': params, 'batch_stats': batch_stats}
      (policy_logits, value_preds), new_model_state = model.apply(variables, observations, training=True, mutable=['batch_stats'])
      
      policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))

      policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()

      value_loss = jnp.mean(jnp.square(value_preds - value_targets))
      
      l2_reg_loss = 0.0
      for p in jax.tree_util.tree_leaves(params):
        l2_reg_loss += weight_decay * jnp.sum(jnp.square(p))

      total_loss = policy_loss + value_loss + l2_reg_loss      
      return total_loss, (policy_loss, value_loss, l2_reg_loss, new_model_state['batch_stats'])
    
    @jax.jit
    def update_step_fn(state, observations, legals_mask, policy_targets, value_targets):
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, (policy_loss, value_loss, l2_reg_loss, new_batch_stats)), grads = grad_fn(state.params, state.batch_stats, observations, legals_mask, policy_targets, value_targets)
      
      new_state = state.apply_gradients(grads=grads)
      new_state = new_state.replace(batch_stats=new_batch_stats)
      return new_state, (policy_loss, value_loss, l2_reg_loss)
    
    return cls(model, state, path, loss_fn, update_step_fn)
  
  @classmethod
  def from_checkpoint(cls, checkpoint_path, model_args=None):
    if model_args is None:
      raise ValueError("model_args must be provided for checkpoint loading")
    
    model = cls.build_model(**model_args)
    model._state = checkpoints.restore_checkpoint(checkpoint_path, model._state)
  
    return model
  
  @property
  def num_trainable_variables(self):
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(self._state.params))
  
  def print_trainable_variables(self):
    flat_params = jax.tree_util.tree_flatten(self._state.params)[0]
    for i, p in enumerate(flat_params):
      print(f"Param {i}: {p.shape}")
  
  def inference(self, observation, legals_mask):
    observation = np.array(observation, dtype=np.float32)
    legals_mask = np.array(legals_mask, dtype=bool)

    policy_logits, value = self._model.apply(
        {'params': self._state.params, 'batch_stats': self._state.batch_stats}, observation, training=False, mutable=False)
    
    policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))
    policy = jax.nn.softmax(policy_logits, axis=-1)

    return value, policy
  
  def update(self, train_inputs: Sequence[TrainInput]):
    batch = TrainInput.stack(train_inputs)
    
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(self._state, batch.observation, batch.legals_mask, batch.policy, batch.value)
    
    return Losses(policy_loss, value_loss, l2_reg_loss)
  
  def save_checkpoint(self, step):
    return checkpoints.save_checkpoint(ckpt_dir=self._path, target=self._state, step=step, keep=10)
  
  def load_checkpoint(self, path):
    self._state = checkpoints.restore_checkpoint(path, self._state)
    return self._state