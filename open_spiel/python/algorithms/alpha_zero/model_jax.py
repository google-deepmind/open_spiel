import functools
import os
from typing import Any, Sequence, Tuple
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

"""implementation of the AlphaZero model, using `flax.linen` API
"""

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


def flatten(x):
  return x.reshape((x.shape[0], -1))

@chex.dataclass(frozen=True)
class TrainInput: 
  """Inputs of the model: o_t, mask_t, π(a_t|o_t), v_t
  """
  observation: chex.Array
  legals_mask: chex.Array 
  policy: chex.Array 
  value: chex.Array

  @staticmethod
  def stack(train_inputs):
    observation, legals_mask, policy, value = zip(*[
      (ti.observation, ti.legals_mask, ti.policy, ti.value) for ti in train_inputs
    ])
    return TrainInput(
        observation=jnp.array(observation, dtype=jnp.float32),
        legals_mask=jnp.array(legals_mask, dtype=jnp.bool),
        policy=jnp.array(policy, dtype=jnp.float32),
        value=jnp.expand_dims(jnp.array(value, dtype=jnp.float32), 1))

@chex.dataclass(frozen=True)
class Losses:
  """Losses: policy, value, L2
  """
  policy: chex.Array
  value: chex.Array 
  l2: chex.Array 
  
  @property
  def total(self):
    return self.policy + self.value + self.l2

  def __str__(self):
    return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, "
            "l2: {:.3f})").format(self.total, self.policy, self.value, self.l2)

  def __add__(self, other):
    return Losses(
      policy=self.policy + other.policy,
      value=self.value + other.value,
      l2=self.l2 + other.l2
    )

  def __truediv__(self, n):
    return Losses(policy=self.policy / n, value=self.value / n, l2=self.l2 / n)

class Activation(nn.Module):
  activation_name: str

  @nn.compact
  def __call__(self, x):
    return activations_dict[self.activation_name](x)

class MLPBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(features=self.features)(x)
    y = Activation("relu")(y)
    return y
  
class ConvBlock(nn.Module):
  features: int
  kernel_size: tuple[int, int]

  @nn.compact
  def __call__(self, x, training: bool = False):
    y = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='SAME')(x)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = Activation("relu")(y)
    return y

class ResidualBlock(nn.Module):
  filters: int
  kernel_size: tuple[int, int]
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    residual = x
    y = nn.Conv(features=self.filters, kernel_size=self.kernel_size, padding='SAME')(x)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = Activation("relu")(y)
    y = nn.Conv(features=self.filters, kernel_size=self.kernel_size, padding='SAME')(y)
    y = nn.BatchNorm(use_running_average=not training)(y)
    y = y + residual
    y = Activation("relu")(y)
    return y


class PolicyHead(nn.Module):
  model_type: str
  nn_width: int
  output_size: int
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    if self.model_type == "mlp":
      x = nn.Dense(features=self.nn_width)(x)
      x = Activation("relu")(x)
    else:
      x = ConvBlock(features = 2, kernel_size = (1, 1))(x, training)
      x = flatten(x)

    policy_logits = nn.Dense(features=self.output_size)(x)
    return policy_logits


class ValueHead(nn.Module):
  model_type: str
  nn_width: int
  
  @nn.compact
  def __call__(self, x, training: bool = False):
    if self.model_type != "mlp":
      x = ConvBlock(features = 1, kernel_size = (1, 1))(x)
      x = flatten(x)
    
    x = nn.Dense(features=self.nn_width)(x)
    x = Activation("relu")(x)
    x = nn.Dense(features=1)(x)
    x = Activation("tanh")(x)
    return x


class AlphaZeroModel(nn.Module):
  model_type: str
  input_shape: Tuple[int, ...]
  output_size: int
  nn_width: int
  nn_depth: int
  
  @nn.compact
  def __call__(self, observations, training: bool = False):

    # torso
    if self.model_type == "mlp":
      x = observations
      for i in range(self.nn_depth): #leave the for-loop of let it go?
        x = MLPBlock(features=self.nn_width)(x)
    elif self.model_type == "conv2d":
      x = observations.reshape((-1,) + self.input_shape)
      for i in range(self.nn_depth):
        x = ConvBlock(features=self.nn_width, kernel_size=(3, 3))(x, training)
    elif self.model_type == "resnet":
      x = observations.reshape((-1,) + self.input_shape)
      x = ConvBlock(features=self.nn_width, kernel_size=(3, 3))(x, training)
      for i in range(self.nn_depth):
        x = ResidualBlock(filters=self.nn_width, kernel_size=(3, 3))(x, training)

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

    options = orbax.checkpoint.CheckpointManagerOptions(
        create=True,
        save_interval_steps=checkpoint_saving_interval,
        max_to_keep=max_checkpoints_to_keep,
        keep_period=checkpoint_rotation_period,
    )
    
    if self._path is not None:
      self._manager = orbax.checkpoint.CheckpointManager(
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
    
    optimizer = optax.adam(learning_rate=learning_rate)
    rng = jax.random.PRNGKey(seed)
    input_shape_with_batch = (1, int(np.prod(input_shape)))
    variables = model.init(rng, jnp.ones(input_shape_with_batch), training=False)

    state = cls._create_train_state(model.apply, variables, optimizer)

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
    target = self._state
    self._state = self._manager.restore(step, items=target)
    self._manager.wait_until_finished()
    return self._state