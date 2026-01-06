import os
import warnings
from typing import Callable, Optional, Sequence

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as orbax
from flax.training import train_state

from open_spiel.python.algorithms.alpha_zero.utils import Losses, TrainInput, flatten

warnings.warn("Pay attention that you've been using the `nnx` api")

activations_dict = {
  "celu": nn.celu,
  "elu": nn.elu,
  "gelu": nn.gelu,
  "glu": nn.glu,
  "hard_sigmoid": nn.hard_sigmoid,
  "hard_silu": nn.hard_silu,  # Alias for hard_swish
  "hard_swish": nn.hard_swish,  # Alias for hard_silu
  "hard_tanh": nn.hard_tanh,
  "leaky_relu": nn.leaky_relu,
  "log_sigmoid": nn.log_sigmoid,
  "log_softmax": nn.log_softmax,
  "relu": nn.relu,
  "selu": nn.selu,
  "sigmoid": nn.sigmoid,
  "silu": nn.silu,
  "soft_sign": nn.soft_sign,
  "softmax": nn.softmax,
  "softplus": nn.softplus,
  "swish": nn.swish,
  "tanh": nn.tanh,
}


def get_batch_stats(layer: nn.Module) -> nn.BatchStat:
  return nn.state(layer, nn.BatchStat)


def get_layer_parameters(layer: nn.Module) -> nn.Param:
  return nn.state(layer, nn.Param)


class Flatten(nn.Module):
  """A simple `nn.Module` class wrapper for the flattening the output"""

  def __init__(self) -> None:
    self.layer = flatten

  def __call__(self, x: chex.Array) -> chex.Array:
    return self.layer(x)


class Activation(nn.Module):
  """A simple `nn.Module` class wrapper for the activations"""

  def __init__(self, activation_name: str) -> None:
    f"""Intialises the activation function

    Args:
        activation_name (str): a name of the activation.
          The following are available: {", ".join(list(activations_dict.keys()))}
    """
    self.activation_name = activation_name

  def __call__(self, x: chex.Array) -> chex.Array:
    # if the activation not found or None just passes the identity function
    return activations_dict.get(self.activation_name, lambda x: x)(x)


class MLPBlock(nn.Module):
  def __init__(
    self, in_features: int, out_features: int, activation: str, seed: int = 0
  ) -> None:
    self.activation = Activation(activation)
    self.dense_layer = nn.Linear(in_features, out_features, rngs=nn.Rngs(seed))

  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.dense_layer(x)
    y = self.activation(y)
    return y


class ConvBlock(nn.Module):
  """Convolutional block with SAME padding + BatchNorm + Activation"""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: tuple[int, int],
    activation: str,
    seed: int = 0,
  ) -> None:
    self.conv = nn.Conv(
      in_features,
      out_features,
      kernel_size=kernel_size,
      padding="SAME",
      rngs=nn.Rngs(seed),
    )
    self.activation = Activation(activation)
    self.bn = nn.BatchNorm(out_features, rngs=nn.Rngs(seed), axis_name="batch")

  def output_shape(self, x: chex.Array) -> tuple[int, int, int]:
    # NOTE: for the given settings: padding="SAME" and strides=(1,1)
    # output size is equal to the input size, otherwise the layer should implement this method.
    # for more, see:
    # https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    return x.shape

  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.conv(x)
    y = self.bn(y)
    y = self.activation(y)
    return y


class ResidualBlock(nn.Module):
  def __init__(
    self,
    in_features: int,
    out_features: int,
    kernel_size: tuple[int, int],
    activation: str,
    seed: int = 0,
  ) -> None:
    self.conv1 = ConvBlock(in_features, out_features, kernel_size, activation, seed)
    # activation's applied separately
    self.conv2 = ConvBlock(out_features, out_features, kernel_size, None, seed)
    self.activation = Activation(activation)

  def output_shape(self, x: chex.Array) -> tuple[int, int, int]:
    return self.conv2.output_shape(x)

  def __call__(self, x: chex.Array) -> chex.Array:
    residual = x
    y = self.conv1(x)
    y = self.conv2(y)
    y = self.activation(y + residual)
    return y


class PolicyHead(nn.Module):
  """Policy network"""

  def __init__(
    self,
    input_shape: chex.Shape,
    nn_width: int,
    out_features: int,
    model_type: str,
    activation: str,
    seed: int = 0,
  ) -> None:
    *space_features, in_features = input_shape

    if model_type == "mlp":
      self.torso = MLPBlock(in_features, nn_width, activation, seed)
    else:
      self.torso = nn.Sequential(
        ConvBlock(in_features, 2, (1, 1), activation, seed),
        Flatten(),
        MLPBlock(np.prod(space_features) * 2, nn_width, activation, seed),
      )

    self.policy_head = MLPBlock(nn_width, out_features, None, seed)

  def __call__(self, x: chex.Array) -> chex.Array:
    y = self.torso(x)
    policy_logits = self.policy_head(y)
    return policy_logits


class ValueHead(nn.Module):
  """Value network"""

  def __init__(
    self,
    input_shape: chex.Shape,
    nn_width: str,
    model_type: str,
    activation: str,
    seed: int,
  ) -> None:
    *space_features, in_features = input_shape

    if model_type == "mlp":
      self.torso = MLPBlock(in_features, nn_width, activation, seed)
    else:
      self.torso = nn.Sequential(
        ConvBlock(in_features, 1, (1, 1), activation),
        Flatten(),
        MLPBlock(np.prod(space_features) * 1, nn_width, activation, seed),
      )

    self.value_head = MLPBlock(nn_width, 1, "tanh", seed)

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
    input_shape: chex.Shape,
    output_size: int,
    nn_width: int,
    nn_depth: int,
    activation: Optional[str] = "relu",
    seed: Optional[int] = 0,
  ) -> None:
    """
    Args:
      model_type (str): underlying model type: conv2d, mlp, or resnet
      input_shape (chex.Shape): input spec
      output_size (int): action space spec
      nn_width (int): hidden layers dimensionality
      nn_depth (int): number of hidden layers
      activation (str): an activation for the neural networks. Defaults to "relu"
      seed (int): random seed for the network
    """

    # torso
    if model_type == "mlp":
      self.torso = nn.Sequential(
        Flatten(),
        MLPBlock(np.prod(input_shape), nn_width, activation=activation, seed=seed),
        *[
          MLPBlock(nn_width, nn_width, activation=activation, seed=seed + _seed)
          for _seed in range(1, nn_depth)
        ],
      )
    elif model_type == "conv2d":
      self.torso = nn.Sequential(
        # by default an observation may be flat
        lambda x: x.reshape(input_shape),
        ConvBlock(input_shape[-1], nn_width, (3, 3), activation, seed=seed),
        *[
          ConvBlock(nn_width, nn_width, (3, 3), activation, seed=seed + _seed)
          for _seed in range(nn_depth)
        ],
      )
    elif model_type == "resnet":
      self.torso = nn.Sequential(
        lambda x: x.reshape(input_shape),
        ConvBlock(input_shape[-1], nn_width, (3, 3), activation, seed=seed),
        *[
          ResidualBlock(nn_width, nn_width, (3, 3), activation, seed=seed + _seed)
          for _seed in range(nn_depth)
        ],
      )
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")

    self.policy_head = PolicyHead(
      (*input_shape[:-1], nn_width), nn_width, output_size, model_type, activation, seed
    )
    self.value_head = ValueHead(
      (*input_shape[:-1], nn_width), nn_width, model_type, activation, seed
    )

  def __call__(self, observations: chex.Array) -> tuple[chex.Array, chex.Array]:
    x = self.torso(observations)
    policy_logits = self.policy_head(x)
    value_out = self.value_head(x)

    return policy_logits, value_out.squeeze(-1)


# modified train state
class TrainState(train_state.TrainState):
  batch_stats: nn.State
  graphdef: nn.GraphDef


class Model:
  valid_model_types = ["mlp", "conv2d", "resnet"]

  def __init__(self, state: TrainState, path: str, update_step_fn: Callable) -> None:
    self._state = state
    self._path = path
    self._update_step_fn = update_step_fn

    # we're using the latest checkpointing API, see:
    # https://flax.readthedocs.io/en/latest/guides/checkpointing.html

    self._checkpointer = None

    if self._path is not None:
      self._checkpointer = orbax.Checkpointer(orbax.StandardCheckpointHandler())

  @classmethod
  def _create_train_state(
    cls, model: AlphaZeroModel, optimiser: optax.GradientTransformation
  ) -> TrainState:
    graphdef, variables, batch_stats = nn.split(model, nn.Param, nn.BatchStat)

    return TrainState.create(
      apply_fn=model,
      params=variables,
      tx=optimiser,
      batch_stats=batch_stats,
      graphdef=graphdef,
    )

  @classmethod
  def build_model(
    cls,
    model_type: str,
    input_shape: chex.Shape,
    output_size: chex.Numeric,
    nn_width: int,
    nn_depth: int,
    weight_decay: float,
    learning_rate: float,
    path: str,
    seed: int = 0,
    decouple_weight_decay: bool = False,
  ) -> "Model":
    if model_type not in cls.valid_model_types:
      raise ValueError(
        f"Invalid model type: {model_type}, expected one of: {cls.valid_model_types}"
      )

    if isinstance(input_shape, int):
      input_shape = (input_shape,)
    elif isinstance(input_shape, list):
      input_shape = tuple(input_shape)

    while len(input_shape) < 3:
      input_shape = input_shape + (1,)

    model = AlphaZeroModel(
      model_type=model_type,
      input_shape=input_shape,
      output_size=output_size,
      nn_width=nn_width,
      nn_depth=nn_depth,
      seed=seed,
    )

    def mask_fn(path, _):
      # path is a tuple of segments, e.g., ('linear1', 'bias')
      names = [
        str(p.key) if isinstance(p, jax.tree_util.DictKey) else str(p) for p in path
      ]
      # Return True to APPLY decay, False to MASK it
      return ("bias" not in names) and ("bn" not in names)

    params = nn.state(model, nn.Param)
    mask = jax.tree.map_with_path(mask_fn, params)

    weights_no_bias_filter = nn.All(nn.Param, mask_fn)

    if not decouple_weight_decay:
      optimiser = optax.adam(learning_rate=learning_rate)
    else:
      optimiser = optax.adamw(
        learning_rate=learning_rate, weight_decay=weight_decay, mask=mask
      )

    state = cls._create_train_state(model, optimiser)

    @nn.jit
    @nn.vmap(in_axes=(None, 0), axis_name="batch")
    def forward(model: AlphaZeroModel, x: chex.Array) -> chex.Array:
      return model(x)

    @jax.jit
    def update_step_fn(
      state: TrainState,
      observations: chex.Array,
      legals_mask: chex.Array,
      policy_targets: chex.Array,
      value_targets: chex.Array,
    ) -> Callable:
      def loss_fn(
        params,
        batch_stats,
        observations: chex.Array,
        legals_mask: chex.Array,
        policy_targets: chex.Array,
        value_targets: chex.Array,
      ):
        model = nn.merge(state.graphdef, params, batch_stats)

        policy_logits, value_preds = forward(model, observations)
        policy_logits = jnp.where(
          legals_mask,
          policy_logits,
          jnp.full_like(policy_logits, jnp.finfo(jnp.float32).min),
        )

        policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()
        value_loss = optax.l2_loss(value_preds, value_targets).mean()

        l2_reg_loss = (
          optax.tree_utils.tree_norm(
            nn.state(model, weights_no_bias_filter), ord=2, squared=True
          )
          * weight_decay
        )

        total_loss = policy_loss + value_loss

        if not decouple_weight_decay:
          total_loss = total_loss + l2_reg_loss

        return total_loss, (
          policy_loss,
          value_loss,
          l2_reg_loss,
          get_batch_stats(model),
        )

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

      (_, (policy_loss, value_loss, l2_reg_loss, new_batch_stats)), grads = grad_fn(
        state.params,
        state.batch_stats,
        observations,
        legals_mask,
        policy_targets,
        value_targets,
      )

      new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)

      return new_state, (policy_loss, value_loss, l2_reg_loss)

    return cls(state, path, update_step_fn)

  @classmethod
  def from_checkpoint(cls, checkpoint_path, model_args=None):
    if model_args is None:
      raise ValueError("model_args must be provided for checkpoint loading")

    model = cls.build_model(**model_args)
    model._state = cls.load_checkpoint(checkpoint_path)

    return model

  @property
  def num_trainable_variables(self) -> int:
    return sum(np.prod(p.shape) for p in jax.tree.leaves(self._state.params))

  def print_trainable_variables(self) -> None:
    model = nn.merge(self._state.graphdef, self._state.params, self._state.batch_stats)
    nn.display(model)
    flat_params, _ = jax.tree.flatten(self._state.params)
    for i, p in enumerate(flat_params):
      print(f"Param {i}: {p.shape}")

  def inference(
    self, observation: chex.Array, legals_mask: chex.Array
  ) -> tuple[chex.Array, chex.Array]:
    model = nn.merge(self._state.graphdef, self._state.params, self._state.batch_stats)
    model.eval()

    observation = jnp.asarray(observation, dtype=jnp.float32)
    legals_mask = jnp.asarray(legals_mask, dtype=jnp.bool)

    policy_logits, value = model(observation)
    policy_logits = jnp.where(
      legals_mask,
      policy_logits,
      jnp.full_like(policy_logits, jnp.finfo(jnp.float32).min),
    )

    policy = nn.softmax(policy_logits)
    return value, policy

  def update(self, batch: Sequence[TrainInput] | chex.ArrayTree) -> Losses:
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(
      self._state, batch.observation, batch.legals_mask, batch.policy, batch.value
    )

    return Losses(policy=policy_loss, value=value_loss, l2=l2_reg_loss)

  def save_checkpoint(self, step: int) -> int:
    path = os.path.join(self._path, f"checkpoint-{step}")
    if self._checkpointer:
      self._checkpointer.save(path, self._state, force=True)
    else:
      print("No checkpoint path is provided. Skipping.")
    return step

  def load_checkpoint(self, step: int) -> None:
    # model = nn.eval_shape(lambda: self._model)
    if self._checkpointer:
      self._state = self._checkpointer.restore(
        os.path.join(self._path, f"checkpoint-{step}"), self._state
      )
