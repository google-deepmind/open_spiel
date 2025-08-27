import jax.numpy as jnp
import chex
import jax
from typing import Any

AVIALABLE_APIS = ["nnx", "linen"]

def api_selector(api_version):
  if api_version == "nnx":
    from open_spiel.python.algorithms.alpha_zero import model_nnx as model_lib
  elif api_version == "linen":
    from open_spiel.python.algorithms.alpha_zero import model_jax as model_lib
  else:
    raise ValueError(f"Only {AVIALABLE_APIS} APIs are implmented")
  
  return model_lib

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
