from typing import Any, Callable

import jax.numpy as jnp
import chex
import jax
import flax.linen as linen
import flax.nnx as nnx


AVIALABLE_APIS = ["nnx", "linen"]

def api_selector(api_version):
  f"""Allows to choose an implementation API.

  Args:
    api_version (str): either of {AVIALABLE_APIS}

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
def linen_to_nnx(model: linen.Module, seed: int=0) -> nnx.bridge.ToNNX:
  # NOTE: could be issues with handling_randomness
  model = nnx.bridge.ToNNX(model, rngs=nnx.Rngs(seed))  
  return model              

def nnx_to_linen(
    model_class: nnx.Module, 
    sample_shape: tuple[int, ...], 
    seed: int = 0, 
    *args: Any,
    **kwargs: Any
  ) -> tuple[Callable, nnx.bridge.ToLinen]:
  new_model = nnx.bridge.ToLinen(model_class, *args, **kwargs)
  variables = new_model.init(jax.random.key(seed), (1, *sample_shape), train=False)
  return new_model.apply, variables

def flatten(x):
  return x.reshape(-1)

def tree_sum(tree: Any, initialiser: Any):
  """Sum of all elements in a tree. Re-used from `optax`."""
  sums = jax.tree.map(jnp.sum, tree)
  return jax.tree.reduce(lambda x, y: x+y, sums, initializer=initialiser)

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

    stacked_states = jax.tree.map(lambda *x: jnp.stack(x), *train_inputs)
    stacked_states = stacked_states.replace(value=jnp.expand_dims(stacked_states.value, 1))

    return stacked_states

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
    return (f"Losses(total: {self.total:.3f}, policy: {self.policy:.3f}, value: {self.value:.3f}, "
            f"l2: {self.l2:.3f})")

  def __add__(self, other):
    return Losses(
      policy=self.policy + other.policy,
      value=self.value + other.value,
      l2=self.l2 + other.l2
    )

  def __truediv__(self, n):
    return Losses(policy=self.policy / n, value=self.value / n, l2=self.l2 / n)
