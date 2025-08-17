import functools
import os
from typing import Any, Sequence, Tuple, Callable
import warnings

from datetime import datetime
import numpy as np
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import chex
import orbax
from flax.training import train_state, orbax_utils
import orbax.checkpoint

from open_spiel.python.algorithms.alpha_zero.utils import TrainInput, Losses, flatten

flax.config.update('flax_use_orbax_checkpointing', True)
warnings.warn("Pay attention that you've been using the `linen` api")


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

class Activation(nn.Module):
  """A simple `nn.Module` class wrapper for the activations
  """
  activation_name: str

  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    #if the activation not found or None just passes the identity function
    return activations_dict.get(self.activation_name, lambda x: x)(x)

class MLPBlock(nn.Module):
  features: int
  activation: str = "relu"

  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    y = nn.Dense(features=self.features)(x)
    y = Activation(self.activation)(y)
    return y
  
class ConvBlock(nn.Module):
  features: int
  kernel_size: tuple[int, int]
  activation: str = "relu"

  @nn.compact
  def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
    y = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='SAME')(x)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = Activation(self.activation)(y)
    return y

class ResidualBlock(nn.Module):
  filters: int
  kernel_size: tuple[int, int]
  activation: str = "relu"
  
  @nn.compact
  def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
    residual = x
    y = ConvBlock(self.filters, self.kernel_size, self.activation)(x, training)
    y = ConvBlock(self.filters, self.kernel_size, None)(x, training)
    y = y + residual
    y = Activation(self.activation)(y)
    return y


class PolicyHead(nn.Module):
  model_type: str
  nn_width: int
  output_size: int
  activation: str = "relu"
  
  @nn.compact
  def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:
    if self.model_type == "mlp":
      x = MLPBlock(self.nn_width, activation=self.activation)(x)
    else:
      x = ConvBlock(features = 2, kernel_size = (1, 1))(x, training)
      x = flatten(x)

    policy_logits = nn.Dense(features=self.output_size)(x)
    return policy_logits


class ValueHead(nn.Module):
  model_type: str
  nn_width: int
  activation: str = "relu"
  
  @nn.compact
  def __call__(self, x: chex.Array, training: bool = False) -> chex.Array:

    if self.model_type != "mlp":
      x = ConvBlock(features = 1, kernel_size = (1, 1), activation=self.activation)(x, training)
      x = flatten(x)
    
    x = MLPBlock(self.nn_width, self.activation)(x)
    values = MLPBlock(1, "tanh")(x)
    return values


class AlphaZeroModel(nn.Module):
  """An AlphaZero style model with a policy and value head.

  This supports three types of models: mlp, conv2d and resnet.

  All models have a shared torso stack with two output heads: policy and value.
  They have same meaning as in the AlphaGo Zero and AlphaZero papers. The resnet
  model copies the one in that paper when set with width 256 and depth 20. The
  conv2d model is the same as the resnet except uses a conv+batchnorm+relu
  instead of the res blocks. The mlp model uses dense layers instead of conv,
  and drops batch norm.

  Links to relevant articles/papers:
    https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
      access link to the AlphaGo Zero nature paper.
    https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
      has an open access link to the AlphaZero science paper.

  All are parameterized by their input (observation) shape and output size
  (number of actions), though the conv2d and resnet might only work with games
  that have spatial data (ie 3 non-batch dimensions, eg: connect four would
  work, but not poker).

  The depth is the number of blocks in the torso, where the definition of a
  block varies by model. For a resnet it's a resblock which is two conv2ds,
  batch norms and relus, and an addition. For conv2d it's a conv2d, a batch norm
  and a relu. For mlp it's a dense plus relu.

  The width is the number of filters for any conv2d and the number of hidden
  units for any dense layer.

  """
  
  model_type: str
  input_shape: Tuple[int, ...]
  output_size: int
  nn_width: int
  nn_depth: int
  activation: str = "relu"
  
  @nn.compact
  def __call__(self, observations: chex.Array, training: bool = False) -> tuple[chex.Array, chex.Array]:

    # torso
    if self.model_type == "mlp":
      x = observations
      for i in range(self.nn_depth): #leave the for-loop of let it go?
        x = MLPBlock(features=self.nn_width, activation=self.activation)(x)
    elif self.model_type == "conv2d":
      x = observations.reshape((-1,) + self.input_shape)
      for i in range(self.nn_depth):
        x = ConvBlock(features=self.nn_width, kernel_size=(3, 3), activation=self.activation)(x, training)
    elif self.model_type == "resnet":
      x = observations.reshape((-1,) + self.input_shape)
      x = ConvBlock(features=self.nn_width, kernel_size=(3, 3), activation=self.activation)(x, training)
      for i in range(self.nn_depth):
        x = ResidualBlock(filters=self.nn_width, kernel_size=(3, 3), activation=self.activation)(x, training)

    else:
      raise ValueError(f"Unknown model type: {self.model_type}")
    
    policy_logits = PolicyHead(model_type=self.model_type, nn_width=self.nn_width, output_size=self.output_size)(x, training)
    value_out = ValueHead(model_type=self.model_type, nn_width=self.nn_width)(x, training)
    
    return policy_logits, value_out


class TrainState(train_state.TrainState):
  batch_stats: Any


class Model:
  valid_model_types = ['mlp', 'conv2d', 'resnet']

  def __init__(
    self, 
    model: nn.Module, 
    state: TrainState, 
    path: str, 
    update_step_fn: Callable,
    checkpoint_uid: str = None,
  ): 
    
    self._model = model
    self._state = state
    self._path = path
    self._update_step_fn = update_step_fn
    
    #we're using the latest checkpointing API
    # https://flax-linen.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
    orbax_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.StandardCheckpointHandler())
    checkpoint_str = (
      checkpoint_uid if checkpoint_uid is not None else datetime.now().strftime("%Y%m%d%H%M%S")
    )

    options = orbax.checkpoint.CheckpointManagerOptions(create=True)
  
    self._manager = None
    if self._path is not None:
      self._manager = orbax.checkpoint.CheckpointManager(
        directory=self._path,
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
    
    optimizer = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(seed)
    input_shape_with_batch = (1, int(np.prod(input_shape)))
    variables = model.init(rng, jnp.ones(input_shape_with_batch), training=False)

    state = cls._create_train_state(model.apply, variables, optimizer)

    @jax.jit
    def update_step_fn(state, observations, legals_mask, policy_targets, value_targets):
      
      def loss_fn(params, batch_stats, observations, legals_mask, policy_targets, value_targets):
        variables = {'params': params, 'batch_stats': batch_stats}
        (policy_logits, value_preds), new_model_state = state.apply_fn(variables, observations, training=True, mutable=['batch_stats'])
        
        policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))
        policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()
        value_loss = jax.vmap(optax.l2_loss)(value_preds - value_targets).mean()
        
        l2_reg_loss = 0.0
        for p in jax.tree.leaves(params):
          l2_reg_loss += weight_decay * jnp.sum(jnp.square(p))

        total_loss = policy_loss + value_loss + l2_reg_loss      
        return total_loss, (policy_loss, value_loss, l2_reg_loss, new_model_state['batch_stats'])
    
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, (policy_loss, value_loss, l2_reg_loss, new_batch_stats)), grads = grad_fn(state.params, state.batch_stats, observations, legals_mask, policy_targets, value_targets)
      
      new_state = state.apply_gradients(grads=grads)
      new_state = new_state.replace(batch_stats=new_batch_stats)
      return new_state, (policy_loss, value_loss, l2_reg_loss)
    
    return cls(model, state, path, update_step_fn)
  
  @classmethod
  def from_checkpoint(cls, checkpoint_path, model_args=None):
    if model_args is None:
      raise ValueError("model_args must be provided for checkpoint loading")
    
    model = cls.build_model(**model_args)
    model._state = cls.load_checkpoint(checkpoint_path)
  
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
  
  def update(self, train_inputs: Sequence[TrainInput]) -> Losses:
    batch = TrainInput.stack(train_inputs)
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(self._state, batch.observation, batch.legals_mask, batch.policy, batch.value)
    
    return Losses(policy=policy_loss, value=value_loss, l2=l2_reg_loss)
  
  def save_checkpoint(self, step: int) -> int:
    self._manager.save(step, self._state, save_kwargs={'save_args': orbax_utils.save_args_from_target(self._state)})
    self._manager.wait_until_finished()
    return step
   
  def load_checkpoint(self, step: int) -> TrainState:
    target = self._state
    self._state = self._manager.restore(step, items=target)
    self._manager.wait_until_finished()
    return self._state