# Copyright 2019 DeepMind Technologies Limited
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

"""An AlphaZero model with a policy and value heads using flax.linen."""

import functools
import os
from typing import Any, Callable, Optional, Sequence
import warnings

import chex
import flax
from flax import training
from flax import traverse_util
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import orbax.checkpoint

from open_spiel.python.algorithms.alpha_zero import utils

# pylint: disable=g-bare-generic

flax.config.update("flax_use_orbax_checkpointing", True)
warnings.warn("Pay attention that you've been using the `linen` api")

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


class Activation(nn.Module):
  """A simple `nn.Module` class wrapper for the activations."""

  activation_name: str

  @nn.compact
  def __call__(self, x: chex.Array) -> chex.Array:
    # if the activation not found or None just passes the identity function
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
  """A convolution block for the AlphaZero model."""

  features: int
  kernel_size: tuple[int, int]
  activation: Optional[str] = "relu"

  @nn.compact
  def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
    y = nn.Conv(
        features=self.features, kernel_size=self.kernel_size, padding="SAME"
    )(x)
    y = nn.BatchNorm(use_running_average=not training, axis_name="batch")(y)
    y = Activation(self.activation)(y)
    return y


class ResidualBlock(nn.Module):
  filters: int
  kernel_size: tuple[int, int]
  activation: Optional[str] = "relu"

  @nn.compact
  def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
    residual = x
    y = ConvBlock(self.filters, self.kernel_size, self.activation)(x, train)
    y = ConvBlock(self.filters, self.kernel_size, None)(y, train)
    y = Activation(self.activation)(y + residual)
    return y


class PolicyHead(nn.Module):
  """A policy head for the AlphaZero model."""

  model_type: str
  nn_width: int
  output_size: int
  activation: Optional[str] = "relu"

  @nn.compact
  def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
    if self.model_type != "mlp":
      x = ConvBlock(features=2, kernel_size=(1, 1), activation=self.activation)(
          x, training
      )
      x = utils.flatten(x)

    x = MLPBlock(self.nn_width, activation=self.activation)(x)
    policy_logits = MLPBlock(self.output_size, None)(x)
    return policy_logits


class ValueHead(nn.Module):
  """A value head for the AlphaZero model."""
  model_type: str
  nn_width: int
  activation: Optional[str] = "relu"

  @nn.compact
  def __call__(self, x: chex.Array, train: bool = False) -> chex.Array:
    if self.model_type != "mlp":
      x = ConvBlock(features=1, kernel_size=(1, 1), activation=self.activation)(
          x, training
      )
      x = utils.flatten(x)

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
  input_shape: chex.Shape
  output_size: int
  nn_width: int
  nn_depth: int
  activation: Optional[str] = "relu"

  @nn.compact
  def __call__(
      self, observations: chex.Array, train: bool
  ) -> tuple[chex.Array, chex.Array]:
    # torso:
    x = observations
    if self.model_type == "mlp":
      x = utils.flatten(observations)
      for _ in range(self.nn_depth):
        x = MLPBlock(features=self.nn_width, activation=self.activation)(x)
    elif self.model_type == "conv2d":
      x = observations.reshape(self.input_shape)
      for _ in range(self.nn_depth):
        x = ConvBlock(
            features=self.nn_width,
            kernel_size=(3, 3),
            activation=self.activation,
        )(x, train)
    elif self.model_type == "resnet":
      x = observations.reshape(self.input_shape)
      x = ConvBlock(
          features=self.nn_width, kernel_size=(3, 3), activation=self.activation
      )(x, train)
      for _ in range(self.nn_depth):
        x = ResidualBlock(
            filters=self.nn_width,
            kernel_size=(3, 3),
            activation=self.activation,
        )(x, train)
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")

    policy_logits = PolicyHead(
        model_type=self.model_type,
        nn_width=self.nn_width,
        output_size=self.output_size,
    )(x, training)
    value_out = ValueHead(model_type=self.model_type, nn_width=self.nn_width)(
        x, training
    )

    return policy_logits, value_out.squeeze(-1)


class TrainState(training.train_state.TrainState):
  batch_stats: Any


class Model:
  """Basic model class."""
  valid_model_types = ["mlp", "conv2d", "resnet"]

  def __init__(
      self,
      state: TrainState,
      path: str,
      update_step_fn: Callable,
  ) -> None:
    self._state = state
    self._path = path
    self._update_step_fn = update_step_fn

    self._checkpointer = None
    if self._path is not None:
      self._checkpointer = orbax.checkpoint.PyTreeCheckpointer()

  @classmethod
  def _create_train_state(
      cls,
      apply_fn: Callable,
      variables: flax.core.FrozenDict,
      optimiser: optax.GradientTransformation,
  ) -> TrainState:
    return TrainState.create(
        apply_fn=apply_fn,
        params=variables["params"],
        tx=optimiser,
        batch_stats=variables.get("batch_stats", {}),
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
    """Builds a model."""
    if model_type not in cls.valid_model_types:
      raise ValueError(
          f"Invalid model type: {model_type}, expected one of:"
          f" {cls.valid_model_types}"
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
    )

    def mask_only_biases(params):
      flat_params = traverse_util.flatten_dict(params)
      flat_mask = {
          path: not ((path[-1] == "bias") or ("BatchNorm_0" in path))
          for path in flat_params.keys()
      }
      return traverse_util.unflatten_dict(flat_mask)

    rng = jax.random.PRNGKey(seed)
    variables = model.init(rng, jnp.ones(input_shape), False)

    if not decouple_weight_decay:
      optimiser = optax.adam(learning_rate=learning_rate)
    else:
      optimiser = optax.adamw(
          learning_rate=learning_rate,
          weight_decay=weight_decay,
          mask=mask_only_biases,
      )

    state = cls._create_train_state(model.apply, variables, optimiser)

    @jax.jit
    def update_step_fn(
        state, observations, legals_mask, policy_targets, value_targets
    ):
      def loss_fn(
          params,
          batch_stats,
          observations,
          legals_mask,
          policy_targets,
          value_targets,
      ):
        variables = {"params": params, "batch_stats": batch_stats}

        @functools.partial(
            jax.vmap,
            in_axes=(0, 0, 0, 0),
            out_axes=(0, None),
            axis_name="batch",
        )
        def _per_example_loss(
            observations, legals_mask, policy_targets, value_targets
        ):
          (policy_logits, value_preds), new_model_state = state.apply_fn(
              variables, observations, training=True, mutable=["batch_stats"]
          )

          policy_logits = jnp.where(
              legals_mask,
              policy_logits,
              jnp.full_like(policy_logits, jnp.finfo(jnp.float32).min),
          )
          policy_loss = optax.softmax_cross_entropy(
              policy_logits, policy_targets
          )
          value_loss = optax.l2_loss(value_preds, value_targets)

          return (policy_loss, value_loss), new_model_state["batch_stats"]

        (policy_loss, value_loss), new_model_state = _per_example_loss(
            observations, legals_mask, policy_targets, value_targets
        )
        policy_loss = policy_loss.mean()
        value_loss = value_loss.mean()

        l2_reg_loss = (
            optax.tree_utils.tree_norm(
                jax.tree.map(
                    lambda p, m: p * m, params, mask_only_biases(params)
                ),
                ord=2,
                squared=True,
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
            new_model_state,
        )

      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, (policy_loss, value_loss, l2_reg_loss, new_batch_stats)), grads = (
          grad_fn(
              state.params,
              state.batch_stats,
              observations,
              legals_mask,
              policy_targets,
              value_targets,
          )
      )

      new_state = state.apply_gradients(
          grads=grads, batch_stats=new_batch_stats
      )
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
    flat_params, _ = jax.tree.flatten(self._state.params)
    for i, p in enumerate(flat_params):
      print(f"Param {i}: {p.shape}")

  def inference(
      self, observation: list, legals_mask: list
  ) -> tuple[chex.Array, chex.Array]:
    """Returns the value and policy for the given state."""
    observation = jnp.asarray(observation, dtype=jnp.float32)
    legals_mask = jnp.asarray(legals_mask, dtype=jnp.bool)

    # @jax.jit  # --- with it the model is much slower than without...
    def _predict(params, batch_stats, observation: chex.Array):
      policy_logits, value = self._state.apply_fn(
          {"params": params, "batch_stats": batch_stats},
          observation,
          training=False,
          mutable=False,
      )
      policy_logits = jnp.where(
          legals_mask,
          policy_logits,
          jnp.full_like(policy_logits, jnp.finfo(jnp.float32).min),
      )
      policy = nn.softmax(policy_logits, axis=-1)
      return value, policy

    value, policy = _predict(
        self._state.params, self._state.batch_stats, observation
    )

    return value, policy

  def update(self, batch: Sequence[utils.TrainInput]) -> utils.Losses:
    self._state, (policy_loss, value_loss, l2_reg_loss) = self._update_step_fn(
        self._state,
        batch.observation,
        batch.legals_mask,
        batch.policy,
        batch.value,
    )

    return utils.Losses(policy=policy_loss, value=value_loss, l2=l2_reg_loss)

  def save_checkpoint(self, step: int, device=None) -> int:
    """Saves a checkpoint of the model."""
    jax.block_until_ready(self._state)
    path = os.path.join(self._path, f"checkpoint-{step}")

    if device is None:
      device = jax.local_devices()[0]

    sharded_state = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, jax.sharding.SingleDeviceSharding(device)),
        self._state,
    )
    if self._checkpointer:
      self._checkpointer.save(
          path,
          args=orbax.checkpoint.args.PyTreeSave(item=sharded_state),
          force=True,
      )

    return step

  def load_checkpoint(self, step: int | str, device: str = None) -> None:
    """Loads a checkpoint of the model."""
    target = self._state
    path = os.path.join(self._path, f"checkpoint-{step}")

    if device is None:
      device = jax.local_devices()[0]

    sharding = jax.sharding.SingleDeviceSharding(device)

    restore_args_tree = jax.tree.map(
        lambda _: orbax.checkpoint.type_handlers.ArrayRestoreArgs(
            sharding=sharding
        ),
        target,
    )

    if self._checkpointer:
      self._state = self._checkpointer.restore(
          path, item=orbax.checkpoint.args.PyTreeRestore(item=restore_args_tree)
      )
      jax.block_until_ready(self._state)
