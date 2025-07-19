"""
Placeholder for the NNX implementation
"""

import functools
import os
from typing import Any, Dict, Sequence, NamedTuple, Tuple, Optional
import warnings

from datetime import datetime
import flax.linen
import numpy as np
import flax
import jax
import jax.numpy as jnp
import flax.nnx as nn
import optax
import chex
from flax.training import train_state
import orbax.checkpoint as orbax

from open_spiel.python.algorithms.alpha_zero.utils import TrainInput, Losses, flatten

flax.config.update('flax_use_orbax_checkpointing', True)

activations_dict = {
    "celu": nn.celu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "glu": nn.glu,
    "hard_sigmoid": nn.hard_sigmoid,
    "hard_silu": nn.hard_silu, # Alias for hard_swish
    "hard_swish": nn.hard_swish, # Alias for hard_silu
    "hard_tanh": nn.hard_tanh,
    "leaky_relu": nn.leaky_relu,
    "log_sigmoid": nn.log_sigmoid,
    "log_softmax": nn.log_softmax,
    "logsumexp": nn.logsumexp,
    "one_hot": nn.one_hot,
    "relu": nn.relu,
    "selu": nn.selu,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "soft_sign": nn.soft_sign,
    "softmax": nn.softmax,
    "softplus": nn.softplus,
    "standardize": nn.standardize,
    "swish": nn.swish,
    "tanh": nn.tanh,
}

def get_batch_stats(layer: nn.Module):
  return nn.state(layer, nn.BatchStat)

def get_layer_parameters(layer: nn.Module):
  return nn.state(layer, nn.Param)

class Activation(nn.Module):

  def __init__(self, activation_name):
    super().__init__()
    self.activation_name = activation_name

  def __call__(self, x):
    return activations_dict[self.activation_name](x)

class MLPBlock(nn.Module):

  def __init__(self, in_features: int, out_features: int, activation: str, seed: int = 0):
    super().__init__()
    self.activation = Activation(activation)
    self.dense_layer = nn.Linear(in_features, out_features, rngs=nn.Rngs(seed))

  def __call__(self, x):
    y = self.dense_layer(x)
    y = self.activation(y)
    return y
  
class ConvBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], activation: str, seed: int = 0):
    super().__init__()
    self.conv = nn.Conv(in_features, out_features, kernel_size=kernel_size, padding="SAME", rngs=nn.Rngs(seed))
    self.activation = Activation(activation) if activation is not None else lambda x: x
    self.bn = nn.BatchNorm(out_features, rngs=nn.Rngs(seed))

  def __call__(self, x):
    y = self.conv(x)
    y = self.bn(y)
    y = self.activation(y)
    return y

class ResidualBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], activation: str, seed: int = 0):
    super().__init__()
    self.conv1 = ConvBlock(in_features, out_features, kernel_size, activation, seed) 
    self.conv2 = ConvBlock(out_features, out_features, kernel_size, None, seed) #activation's applied separately
    self.activation = Activation(activation)

  def __call__(self, x, training: bool = False):
    residual = x
    y = self.conv1(x)
    y = self.conv2(y)
    y = y + residual
    y = self.activation(y)
    return y


class PolicyHead(nn.Module):
  
  def __init__(self, in_features, nn_width, out_features, model_type, activation, seed):
    super().__init__()
    self.torso = None
    if model_type == "mlp":
      self.torso = MLPBlock(in_features, nn_width, activation, seed)
    else:
      self.torso = ConvBlock(in_features, 2, (1,1), activation, seed)

    self.policy_head = nn.Linear(nn_width, out_features, rngs=nn.Rngs(seed))
  
  def __call__(self, x):
    y = self.torso(x)
    y = flatten(y)
    policy_logits = self.policy_head(y)

    return policy_logits


class ValueHead(nn.Module):
  model_type: str
  nn_width: int

  def __init__(self, in_features, nn_width, out_features, model_type, activation, seed):
    super().__init__()
    self.torso = None
    if model_type == "mlp":
      self.torso = MLPBlock(in_features, nn_width, activation, seed)
    else:
      self.torso = ConvBlock(in_features, 1, (1,1), activation, seed)

    self.value_head = nn.Sequential(
      nn.Linear(1, out_features, rngs=nn.Rngs(seed)),
      nn.Linear(nn_width, 1, rngs=nn.Rngs(seed)),
      Activation("tanh")
    )
    
  
  def __call__(self, x):
    y = self.torso(x)
    y = flatten(y)
    policy_logits = self.value_head(y)

    return policy_logits

class AlphaZeroModel(nn.Module):
  model_type: str
  input_shape: chex.Array
  output_size: int
  nn_width: int
  nn_depth: int

  def __init__(self):
    super().__init__()
    
    @nn.split_rngs(splits=self.nn_depth)
    @nn.vmap(axis_size=self.nn_depth)
    def _create_mlp_block() -> MLPBlock:
      return MLPBlock()
    
    @nn.split_rngs(splits=self.nn_depth)
    @nn.vmap(axis_size=self.nn_depth)
    def _create_conv_block() -> ConvBlock:
      return ConvBlock()

    
    @nn.split_rngs(splits=self.nn_depth)
    @nn.vmap(axis_size=self.nn_depth)
    def _create_residual_block() -> ResidualBlock:
      return ResidualBlock()


  
  def __call__(self, observations, training: bool = False):

    @nn.split_rngs(splits=self.nn_depth)
    @nn.scan
    def scan_fn(x: jax.Array, block: Block):
      x = block(x)
      return x, None

    x, _ = scan_fn(observations, self.layers)

    # torso
    if self.model_type == "mlp":
      x = observations
      for i in range(self.nn_depth): #leave the for-loop of let it go?
        x = MLPBlock(features=self.nn_width)(x)
    elif self.model_type == "conv2d":
      x = observations.reshape((-1,) + self.input_shape)
      for i in range(self.nn_depth):
        x = ConvBlock(features=self.nn_width)(x)
    elif self.model_type == "resnet":
      x = observations.reshape((-1,) + self.input_shape)
      x = ConvBlock(features=self.nn_width)(x)
      for i in range(self.nn_depth):
        x = ResidualBlock(filters=self.nn_width, kernel_size=3)(x)

    else:
      raise ValueError(f"Unknown model type: {self.model_type}")
    
    policy_logits = PolicyHead(model_type=self.model_type, nn_width=self.nn_width, output_size=self.output_size)(x)
    value_out = ValueHead(model_type=self.model_type, nn_width=self.nn_width)(x)
    
    return policy_logits, value_out



#modifying train state
class TrainState(train_state.TrainState):
  batch_stats: nn.State
  graphdef: nn.GraphDef


class Model:
  valid_model_types = ['mlp', 'conv2d', 'resnet']

  def __init__(
    self, 
    model, 
    state, 
    path, 
    loss_fn, 
    update_step_fn,
    checkpoint_uid=None,
    checkpoint_saving_interval=None,
    max_checkpoints_to_keep=None,
    checkpoint_rotation_period=None
  ): 
    
    self._model = model
    self._state = state
    self._path = path
    self._loss_fn = loss_fn
    self._update_step_fn = update_step_fn
    
    #we're using the latest checkpointing API
    # https://flax-linen.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_str = (
      checkpoint_uid if checkpoint_uid is not None else datetime.now().strftime("%Y%m%d%H%M%S")
    )

    options = orbax.CheckpointManagerOptions(
        create=True,
        save_interval_steps=checkpoint_saving_interval,
        max_to_keep=max_checkpoints_to_keep,
        keep_period=checkpoint_rotation_period,
    )
    
    if self._path is not None:
      self._manager = orbax.CheckpointManager(
        directory=os.path.join(self._path, checkpoint_str),
        checkpointers=orbax_checkpointer,
        options=options
      )


  @classmethod
  def _create_train_state(cls, apply_fn, variables, optimizer) -> TrainState:
    return TrainState.create(
      apply_fn=apply_fn, 
      params=variables['params'], 
      tx=optimizer, 
      batch_stats=variables.get('batch_stats', {})
    )
    

  @classmethod
  def build_model(cls, model_type, input_shape, output_size, nn_width, nn_depth,
                  weight_decay, learning_rate, path, seed=0):
    
    if model_type not in cls.valid_model_types:
      raise ValueError(f"Invalid model type: {model_type}, "
                       f"expected one of: {cls.valid_model_types}")
    
    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    elif isinstance(input_shape, list):
      input_shape = tuple(input_shape)

    while len(input_shape) < 3:
      input_shape = input_shape + (1,)
    
    model = AlphaZeroModel(
      model_type = model_type, 
      input_shape = input_shape, 
      output_size = output_size, 
      nn_width = nn_width, 
      nn_depth = nn_depth
    )
    
    optimizer = nn.Optimizer(model, optax.adam(learning_rate=learning_rate))
    #todo: add l2 regualiser

    def loss_fn(params, observations, legals_mask, policy_targets, value_targets):
      model = nn.merge(state.graphdef, params, state.batch_stats)

      policy_logits, value_preds = model(observations)
      
      policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))

      policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()

      value_loss = jax.vmap(optax.l2_loss)(value_preds - value_targets).mean()
    
      total_loss = policy_loss + value_loss   

      batch_stats = get_batch_stats(model)   
      return total_loss, (policy_loss, value_loss, batch_stats)
    
    @jax.jit
    def update_step_fn(state, observations, legals_mask, policy_targets, value_targets):
        # def loss_fn(params):
        #   model = nnx.merge(state.graphdef, params, state.counts)
        #   y_pred = model(x)
        #   loss = jnp.mean((y - y_pred) ** 2)
        #   counts = nnx.state(model, Count)
        #   return loss, counts

        # grads, counts = jax.grad(loss_fn, has_aux=True)(state.params)
        # # sdg update
        # state = state.apply_gradients(grads=grads, counts=counts)
      grad_fn = nn.grad(loss_fn, has_aux=True)
      (_, (policy_loss, value_loss, l2_reg_loss, new_batch_stats)), grads = grad_fn(state.params, state.batch_stats, observations, legals_mask, policy_targets, value_targets)
      
      optimizer.update(grads)
      return new_state, (policy_loss, value_loss, l2_reg_loss)
    
    return cls(model, state, path, loss_fn, update_step_fn)
  
  @classmethod
  def from_checkpoint(cls, checkpoint_path, model_args=None):
    if model_args is None:
      raise ValueError("model_args must be provided for checkpoint loading")
    
    model = cls.build_model(**model_args)
    
    model._state = cls.load_checkpoint(checkpoint_path, model._state)
  
    return model
  
  @property
  def num_trainable_variables(self):
    return sum(np.prod(p.shape) for p in jax.tree.leaves(self._state.params))
  
  @property
  def parameters_per_layer(self):
    flat_params = flax.traverse_util.flatten_dict(self._state.params, sep='/')
    return jax.tree.map(jnp.shape, flat_params)
  
  def print_trainable_variables(self):
    flat_params, _ = jax.tree.flatten(self._state.params)
    for i, p in enumerate(flat_params):
      print(f"Param {i}: {p.shape}")
  
  def inference(self, observation, legals_mask):
    observation = jnp.array(observation, dtype=jnp.float32)
    legals_mask = jnp.array(legals_mask, dtype=jnp.bool)

    policy_logits, value = self._model.apply(
        {'params': self._state.params, 'batch_stats': self._state.batch_stats}, observation, training=False, mutable=False)
    
    policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))
    policy = nn.softmax(policy_logits, axis=-1)

    return value, policy
  
  def update(self, train_inputs: Sequence[TrainInput]):
    batch = TrainInput.stack(train_inputs)
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(self._state, batch.observation, batch.legals_mask, batch.policy, batch.value)
    
    return Losses(policy=policy_loss, value=value_loss, l2=l2_reg_loss)
  
  def save_checkpoint(self, step):
    self._manager.save(step, self._state, save_kwargs={'save_args': orbax_utils.save_args_from_target(self._state)})
    self._manager.wait_until_finished()
    return step
   
  def load_checkpoint(self, step):

    model = nn.eval_shape(lambda: create_model(0))
    state = nn.state(model)
    # Load the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(f'{path}/state', item=state)
    # update the model with the loaded state
    nn.update(model, state)

    # target = self._state
    # self._state = self._manager.restore(step, items=target)
    # self._manager.wait_until_finished()
    return self._state