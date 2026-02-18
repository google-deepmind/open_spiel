"""Common utilities for AlphaZero."""

from typing import Any, Callable

import chex
from flax import linen
from flax import nnx
import jax
import jax.numpy as jnp

AVIALABLE_APIS = ["nnx", "linen"]

# pylint: disable=g-bare-generic
# pylint: disable=g-import-not-at-top
# pylint: disable=line-too-long


def api_selector(api_version):
  """Allows to choose an implementation API.

  Args:
    api_version (str): either of {AVIALABLE_APIS}

  Returns:
    model_lib: the model library with the chosen API.

  Raises:
    ValueError: if the user used smth different
  """
  if api_version == "nnx":
    from open_spiel.python.algorithms.alpha_zero import model_nnx as model_lib
  elif api_version == "linen":
    from open_spiel.python.algorithms.alpha_zero import model_linen as model_lib
  else:
    raise ValueError(f"Only {AVIALABLE_APIS} APIs are implmented")

  return model_lib


# These should be importable modes for inference only.
# I.e., you should call the inference using a saved checkpoint to use
# the APIs interchangeably
# See examples in `model_test.py`
def linen_to_nnx(model: linen.Module, seed: int = 0) -> nnx.bridge.ToNNX:
  # NOTE: could be issues with handling_randomness
  model = nnx.bridge.ToNNX(model, rngs=nnx.Rngs(seed))
  return model


def nnx_to_linen(
    model_class: nnx.Module,
    sample_shape: tuple[int, ...],
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> tuple[Callable, nnx.bridge.ToLinen]:
  new_model = nnx.bridge.ToLinen(model_class, *args, **kwargs)
  variables = new_model.init(
      jax.random.key(seed), (1, *sample_shape), train=False
  )
  return new_model.apply, variables


def flatten(x: chex.Array) -> chex.Array:
  """Flattens the array, i.e. reshapes it in a 1D shape."""
  return x.reshape(-1)


@chex.dataclass(frozen=True)
class TrainInput:
  """Inputs of the model: o_t, mask_t, π(a_t|o_t), v_t."""

  observation: chex.Array
  legals_mask: chex.Array
  policy: chex.Array
  value: chex.Array

  @staticmethod
  def stack(train_inputs: list["TrainInput"]) -> "TrainInput":
    return jax.tree.map(lambda *x: jnp.stack(x), *train_inputs)


@chex.dataclass(frozen=True)
class Losses:
  """Losses: policy, value, L2."""

  policy: chex.Array
  value: chex.Array
  l2: chex.Array

  @property
  def total(self) -> float:
    return self.policy + self.value + self.l2

  def __str__(self) -> str:
    return (
        f"Losses(total: {self.total:.3f}, policy: {self.policy:.3f}, value:"
        f" {self.value:.3f}, l2: {self.l2:.3f})"
    )

  def __add__(self, other) -> "Losses":
    return Losses(
        policy=self.policy + other.policy,
        value=self.value + other.value,
        l2=self.l2 + other.l2,
    )

  def __truediv__(self, n) -> "Losses":
    return Losses(policy=self.policy / n, value=self.value / n, l2=self.l2 / n)
