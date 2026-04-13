import functools
import math

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp


@nn.vmap(in_axes=(None, 0), out_axes=0, axis_name="batch")
def batched_call(model: nn.Module, x: chex.Array) -> chex.Array:
  """Batched model call."""
  return model(x)


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
    "epsilon_hat": jnp.zeros((batch_size, n_players)),
    "welfare": jnp.zeros(joint_shape),
  }


def compute_joint_action_size(action_shape: tuple) -> int:
  """|A| = product of all players' action sizes"""
  return math.prod(action_shape)


@functools.partial(jax.jit, static_argnames=("m", "axes"))
def compute_L_m_norm(tensor: chex.Array, m: int, axes: tuple) -> chex.Array:
  if m == 2:
    # L2 norm: √(Σ x²)
    return jnp.sqrt(jnp.sum(jnp.square(tensor), axis=axes, keepdims=True))
  elif m == 1:
    # L1 norm: Σ |x|
    return jnp.sum(jnp.abs(tensor), axis=axes, keepdims=True)
  # General L_m norm
  return jnp.power(
    jnp.sum(jnp.abs(tensor) ** m, axis=axes, keepdims=True), 1.0 / m
  )
