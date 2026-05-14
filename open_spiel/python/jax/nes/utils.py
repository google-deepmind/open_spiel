import functools
import math

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding


def named_sharding(
  mesh: jax.sharding.Mesh, *names: str | None
) -> NamedSharding:
  return NamedSharding(mesh, P(*names))


@chex.dataclass(unsafe_hash=True)
class MeshRules:
  mlp: str | None = None
  data: str | None = None

  def __call__(self, *keys: str) -> tuple[str, ...]:
    return tuple(getattr(self, key) for key in keys)


mesh_rules = MeshRules(
  mlp="model",
  data="data",
)

SMALL_NUMBER = jnp.finfo(jnp.float32).eps


@nn.vmap(in_axes=(None, 0, 0), out_axes=0, axis_name="batch")
def batched_call(
  model: nn.Module, x: chex.Array, mask: chex.Array
) -> chex.Array:
  """Batched model call."""
  return model(x, mask)


def mask_diagonal(x: chex.Array) -> chex.Array:
  """CE-related utility to compute f_hat."""
  diag = jnp.arange(x.shape[-1])
  x = x.at[..., diag, diag].set(0.0)
  return x


def dummy_nes_batch(
  batch_size, n_players, action_sizes, rng: chex.PRNGKey
) -> dict[str, chex.Array]:
  """Quick placeholder for testing without OpenSpiel"""
  A = jnp.array(action_sizes)
  joint_shape = (batch_size, *A)

  return {
    "reward": jnp.zeros((batch_size, n_players, *A)),
    "sigma_hat": jnp.ones(joint_shape) / jnp.prod(A),
    "sigma_norm": jnp.ones(joint_shape) / jnp.prod(A),
    "epsilon_hat": jnp.zeros((batch_size, n_players)),
    "welfare": jnp.zeros(joint_shape),
    "mask": jnp.ones((batch_size, *A), dtype=jnp.bool),
  }


def compute_joint_action_size(action_shape: chex.Shape) -> int:
  """|A| = product of all players' action sizes"""
  return math.prod(action_shape)


@functools.partial(jax.jit, static_argnames=("m", "axes", "where"))
def compute_L_m_norm(
  tensor: chex.Array, m: int, axes: tuple, where: chex.Array = None
) -> chex.Array:
  if m == 2:
    # L2 norm: √(Σ x²)
    return jnp.sqrt(
      jnp.sum(jnp.square(tensor), axis=axes, keepdims=True, where=where)
    )
  elif m == 1:
    # L1 norm: Σ |x|
    return jnp.sum(jnp.abs(tensor), axis=axes, keepdims=True, where=where)
  # General L_m norm
  return jnp.power(
    jnp.sum(jnp.abs(tensor) ** m, axis=axes, keepdims=True, where=where),
    1.0 / m,
  )


def joint_mask(action_sizes: chex.Shape, max_actions: int) -> chex.Array:
  """
  Create boolean mask of shape [max_A, ..., max_A] where True indicates
  all players' actions are within their valid ranges.
  """
  N = len(action_sizes)
  # indices[p, A1, A2, ..., AN] = action index for player p
  indices = jnp.indices((max_actions,) * N)
  sizes = jnp.array(action_sizes).reshape((N,) + (1,) * N)

  # valid_per_player[p, ...] = True where player p's coordinate is valid
  valid_per_player = indices < sizes

  return jnp.all(valid_per_player, axis=0)


@functools.partial(jax.jit, static_argnames=("max_actions", "pad_value"))
def pad_game_tensor(
  G: chex.Array,
  max_actions: int,
  pad_value: float,
) -> chex.Array:
  """
  Pad payoff tensor [N, A1, ..., AN] to [N, max_actions, ..., max_actions].

  Args:
      G: Payoff tensor with natural (possibly non-cubic) shape
      max_actions: Target size for every action dimension
      pad_value: Value to pad with

  Returns:
      Padded tensor of shape [N, max_actions] * N
  """
  action_sizes = G.shape[1:]
  # Build pad_width: [(0,0)] for player dim + [(0, max-A_i)] for each action dim
  pad_width = [(0, 0)] + [(0, max_actions - a_i) for a_i in action_sizes]
  return jnp.pad(G, pad_width, mode="constant", constant_values=pad_value)


def safe_masked_mean(
  x: chex.Array, where: chex.Array, axis: tuple | int, keepdims=True
) -> chex.Array:
  """Masked mean that returns 0 when all entries are masked."""
  if where is None:
    # Fallback to the traditional mean
    return jnp.mean(x, axis=axis, keepdims=keepdims)
  
  mask_bc = jnp.broadcast_to(where, x.shape)

  x_masked = jnp.where(mask_bc, x, 0.0)
  sum_valid = jnp.sum(x_masked, axis=axis, keepdims=keepdims)
  count_valid = jnp.sum(mask_bc, axis=axis, keepdims=keepdims)

  # Avoid division by zero
  return jnp.where(count_valid > 0, sum_valid / count_valid, 0.0)


def meanvar(
  x: chex.Array, axis: tuple | int, keepdims=True, where: chex.Array = None
) -> chex.Array:
  """Meanvar aggregation function."""
  n = jnp.sum(
    jnp.ones_like(x, dtype=jnp.int32), axis=axis, keepdims=keepdims, where=where
  )
  return jnp.sum(x, axis=axis, keepdims=keepdims, where=where) / jnp.sqrt(
    jnp.maximum(n, 1)
  )
