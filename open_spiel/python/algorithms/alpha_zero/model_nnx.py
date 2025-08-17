import functools
import os
from typing import Sequence, Optional, Callable
import warnings

from datetime import datetime
import numpy as np
import flax
import jax
import jax.numpy as jnp
import flax.nnx as nn
import optax
import chex
from flax.training import train_state
import orbax.checkpoint as orbax

from open_spiel.python.algorithms.alpha_zero.utils import TrainInput, Losses, flatten, conv_output_size

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

def get_batch_stats(layer: nn.Module):
  return nn.state(layer, nn.BatchStat)

def get_layer_parameters(layer: nn.Module):
  return nn.state(layer, nn.Param)

class Activation(nn.Module):
  def __init__(self, activation_name: str) -> None:
    """A simple `nn.Module` class wrapper for the activations
    """
    self.activation_name = activation_name

  def __call__(self, x: chex.Array) -> chex.Array:
    #if the activation not found or None just passes the identity function
    return activations_dict.get(self.activation_name, lambda x: x)(x)
    
class MLPBlock(nn.Module):

  def __init__(self, in_features: int, out_features: int, activation: str, seed: int = 0):

    self.activation = Activation(activation)
    self.dense_layer = nn.Linear(in_features, out_features, rngs=nn.Rngs(seed))

  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.dense_layer(x)
    y = self.activation(y)
    return y
  
class ConvBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], activation: str, seed: int = 0):

    self.conv = nn.Conv(in_features, out_features, kernel_size=kernel_size, padding="SAME", rngs=nn.Rngs(seed))
    self.activation = Activation(activation) 
    self.bn = nn.BatchNorm(out_features, rngs=nn.Rngs(seed))

  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.conv(x)
    y = self.bn(y)
    y = self.activation(y)
    return y

class ResidualBlock(nn.Module):
  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], activation: str, seed: int = 0):

    self.conv1 = ConvBlock(in_features, out_features, kernel_size, activation, seed) 
    self.conv2 = ConvBlock(out_features, out_features, kernel_size, None, seed) #activation's applied separately
    self.activation = Activation(activation)

  def __call__(self, x: chex.Array) -> chex.Array:

    residual = x
    y = self.conv1(x)
    y = self.conv2(y)
    y = y + residual
    y = self.activation(y)
    return y


class PolicyHead(nn.Module):
  
  def __init__(self, in_features: int, nn_width: int, out_features: int, model_type: str, activation: str, seed: int = 0) -> None:

    self.torso = None
    if model_type == "mlp":
      self.torso = MLPBlock(in_features, nn_width, activation, seed)
    else:
      self.torso = nn.Sequential(
        ConvBlock(in_features, 2, (1, 1), activation, seed),
        flatten
      )

    self.policy_head = nn.Linear(nn_width, out_features, rngs=nn.Rngs(seed))
  
  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.torso(x)
    policy_logits = self.policy_head(y)

    return policy_logits


class ValueHead(nn.Module):
  model_type: str
  nn_width: int

  def __init__(self, in_features: str, nn_width, model_type, activation, seed):
    
    self.torso = lambda x: x
    
    if self.model_type != "mlp":
      self.torso = nn.Sequential(
        ConvBlock(in_features, 1, kernel_size = (1, 1), activation=activation),
        flatten
      )

    self.value_head = nn.Sequential(
      MLPBlock(1, nn_width, "relu", rngs=seed),
      MLPBlock(nn_width, 1, "tanh", rngs=seed),
    )
    
  
  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.torso(x)
    values = self.value_head(y)
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
  
  def __init__(
    self,
    model_type: str,
    input_shape: tuple[int, ...],
    output_size: int,
    nn_width: int,
    nn_depth: int,
    activation: str = "relu",
    seed: int = 0
  ) -> None:
    """

    Args:
        model_type (str): underlying model type: conv2d, mlp, or resnet
        input_shape (tuple[int, ...]): input spec
        output_size (int): action space spec
        nn_width (int): hidden layers dimensionality
        nn_depth (int): number of hidden layers
    """


    # torso
    if self.model_type == "mlp":
      self.torso = nn.Sequential(
        *[ MLPBlock(np.prod(input_shape), nn_width, activation=activation, seed=i+seed) for i in range(nn_depth)
      ])
    elif self.model_type == "conv2d":

        self.torso = nn.Sequential( 
          lambda x: x.reshape(input_shape),
          ConvBlock(input_shape, nn_width, (3, 3), activation, seed=seed), *[
            ConvBlock(nn_width, nn_width, (3, 3), activation, seed=i+seed) for i in range(1, nn_depth)
          ]
        )
    elif self.model_type == "resnet":
     self.torso = nn.Sequential( 
          lambda x: x.reshape(input_shape),
          ConvBlock(input_shape, nn_width, (3, 3), activation, seed=seed), *[
            ResidualBlock(nn_width, nn_width, (3, 3), activation, seed=i+seed) for i in range(1, nn_depth)
          ]
        )
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")
    
    
    self.policy_head = PolicyHead(nn_width, nn_width, output_size, model_type, activation, seed=seed)
    self.value_head = ValueHead(nn_width, nn_width, model_type, activation, seed=seed)

  
  def __call__(self, observations: chex.Array) -> tuple[chex.Array, chex.Array]:

    x = self.torso(observations)
    policy_logits = self.policy_head(x)
    value_out = self.value_head(x)
    
    return policy_logits, value_out



#modified train state
class TrainState(train_state.TrainState):
  batch_stats: nn.State
  graphdef: nn.GraphDef


class Model:
  valid_model_types = ['mlp', 'conv2d', 'resnet']

  def __init__(
    self, 
    model: nn.Module, 
    state: TrainState, 
    path: str, 
    update_step_fn: Callable,
    checkpoint_uid=None,
    checkpoint_saving_interval=None,
    max_checkpoints_to_keep=None,
    checkpoint_rotation_period=None
  ) -> None: 
    
    self._model = model
    self._state = state
    self._path = path
    self._update_step_fn = update_step_fn
    
    #we're using the latest checkpointing API
    # https://flax-linen.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_str = (
      checkpoint_uid if checkpoint_uid is not None else datetime.now().strftime("%Y%m%d%H%M%S")
    )

    options = orbax.CheckpointManagerOptions(create=True)
    
    self._manager = None
    if self._path is not None:
      self._manager = orbax.CheckpointManager(
        directory=os.path.join(self._path, checkpoint_str),
        checkpointers=orbax_checkpointer,
        options=options
      )


  @classmethod
  def _create_train_state(cls, model, optimizer) -> TrainState:
    graphdef, variables, batch_stats = nn.split(model, nn.Param, nn.BatchStat)

    return TrainState.create(
      apply_fn=model.forward, 
      params=variables, 
      tx=optimizer, 
      batch_stats=batch_stats,
      graphdef=graphdef
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
    state = cls._create_train_state(model, optimizer)    

    @jax.jit
    def update_step_fn(state, observations, legals_mask, policy_targets, value_targets):
      
      def loss_fn(params, observations, legals_mask, policy_targets, value_targets):
        model = nn.merge(state.graphdef, params, state.batch_stats)

        policy_logits, value_preds = model(observations)
        policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))
        policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()
        value_loss = jax.vmap(optax.l2_loss)(value_preds - value_targets).mean()
      
        total_loss = policy_loss + value_loss   

        batch_stats = get_batch_stats(model)   
        return total_loss, (policy_loss, value_loss, batch_stats)

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
    nn.display(self._model)
    flat_params, _ = jax.tree.flatten(self._state.params)
    for i, p in enumerate(flat_params):
      print(f"Param {i}: {p.shape}")
  
  def inference(self, observation, legals_mask):
    observation = jnp.array(observation, dtype=jnp.float32)
    legals_mask = jnp.array(legals_mask, dtype=jnp.bool)

    policy_logits, value = self._model(observation)
    
    policy_logits = jnp.where(legals_mask, policy_logits, jnp.full_like(policy_logits, -1e32))
    policy = nn.softmax(policy_logits, axis=-1)

    return value, policy
  
  def update(self, train_inputs: Sequence[TrainInput]):
    batch = TrainInput.stack(train_inputs)
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(self._state, batch.observation, batch.legals_mask, batch.policy, batch.value)
    
    return Losses(policy=policy_loss, value=value_loss, l2=l2_reg_loss)
  
  def save_checkpoint(self, step: int) -> int:
    self._manager.save(step, self._state)
    self._manager.wait_until_finished()
    return step
   
  def load_checkpoint(self, step: int) -> TrainState:

    model = nn.eval_shape(lambda: self._model)
    state = nn.state(model)
    self._state = self._manager.restore(step, items=self._state)
    # update the model with the loaded state
    nn.update(model, state)
    self._manager.wait_until_finished()
    return self._state