# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities to generate classes of normal-form game payoff tensors.
Copied from
https://github.com/google-deepmind/nfg_transformer/blob/main/nfg_transformer/games.py
"""

import enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Callable
import jax
import jax.numpy as jnp
import chex


_MIN_SCALE = 1e-12

CE_COUNTER_EXAMPLE = jnp.array([

])

CCE_COUNTER_EXAMPLE = jnp.array([

])

def _center(
  payoff: chex.Array, player: int, joint_mask: Optional[chex.Array] = None
) -> chex.Array:
  """Returns centered zero mean payoffs."""
  if joint_mask is not None:
    count = jnp.clip(jnp.sum(joint_mask, axis=player), a_min=1.0)
    offset = jnp.sum(payoff, player, where=joint_mask, keepdims=True) / count
  else:
    offset = jnp.mean(payoff, player, keepdims=True)

  payoff -= offset
  if joint_mask is not None:
    payoff *= joint_mask
  return payoff


def _l2_norm(
  tensor: chex.Array,
  mask: Optional[chex.Array] = None,
  unit_variance: bool = False,
) -> chex.Array:
  """Returns L2 norm.

  Also called the Euclidean norm or square norm.

  Defined:
    L_2 =  SQRT( SUM_s SQUARE(T(s)) )

  Args:
    tensor: Array with shape [|S_1|,...|S_N|].
    mask: Array with shape [|S_1|,...|S_N|].
    unit_variance: Scale so elements maintain unit variance.

  Returns:
    l2_norm: Array with shape [()].
  """
  # Note: jnp.linalg.norm is only implemented for 2D arrays.
  scale = jnp.clip(
    jnp.sqrt(jnp.sum(jnp.square(tensor), where=mask)), a_min=_MIN_SCALE
  )
  if unit_variance:
    size = jnp.sum(mask) if mask is not None else tensor.size
    scale /= jnp.sqrt(size)
  return scale


def _l2_scale(
  tensor: chex.Array,
  mask: Optional[chex.Array] = None,
  min_denominator: Optional[float] = 1e-12,
  unit_variance: bool = False,
) -> chex.Array:
  """Returns L2 normalized tensor.

  Also called the Euclidean norm or square norm.

  Defined:
    T(s) = T(s) / SQRT( SUM_s SQUARE(T(s)) )

  Usually tensor will be zero-meaned before using this transform.

  Args:
    tensor: Array with shape [|S_1|,...|S_N|].
    mask: Array with shape [|S_1|,...|S_N|].
    min_denominator: Optional minimum value to normalize with. Default is 1e-12.
      Pass None to disable denominator clipping.
    unit_variance: Scale so elements maintain unit variance.

  Returns:
    normalized_tensor: Array with shape [|S_1|,...|S_N|].
  """
  scale = _l2_norm(tensor, mask=mask, unit_variance=unit_variance)
  return tensor / jnp.clip(scale, a_min=min_denominator)


def _l1_scale(
  tensor: chex.Array,
  mask: Optional[chex.Array] = None,
  min_denominator: Optional[float] = 1e-12,
  unit_variance: bool = False,
) -> chex.Array:
  """L1 normalisation: scaling by sum of absolute values."""
  scale = jnp.sum(jnp.abs(tensor), where=mask)
  scale = jnp.clip(scale, a_min=min_denominator)

  if unit_variance:
    size = jnp.sum(mask) if mask is not None else tensor.size
    scale /= jnp.sqrt(size)
  return tensor / scale


def _linf_scale(
  tensor: chex.Array,
  mask: Optional[chex.Array] = None,
  min_denominator: Optional[float] = 1e-12,
  unit_variance: bool = False,
) -> chex.Array:
  """L-infinity normalisation: scaling by max absolute value."""
  scale = jnp.max(jnp.abs(tensor), initial=0.0, where=mask)
  scale = jnp.clip(scale, a_min=min_denominator)

  if unit_variance:
    size = jnp.sum(mask) if mask is not None else tensor.size
    scale /= jnp.sqrt(size)
  return tensor / scale


def _empirical_disc_game(key, num_strategies, latent_size: int = 1):
  """Returns an instance of the empirical disc game from underlying latents."""
  assert len(num_strategies) == 2
  assert num_strategies[0] == num_strategies[1]
  k_u, k_v, k_bu, k_bv = jax.random.split(key, 4)
  u = jax.random.normal(k_u, (num_strategies[0], latent_size))
  v = jax.random.normal(k_v, (num_strategies[1], latent_size))
  u += jax.random.uniform(k_bu, (), minval=-1.0, maxval=1.0)
  v += jax.random.uniform(k_bv, (), minval=-1.0, maxval=1.0)
  logit = jnp.einsum("id,jd->ij", u, v) - jnp.einsum("jd,id->ij", u, v)
  win_rate = jax.nn.sigmoid(logit)
  return jnp.stack([win_rate, 1.0 - win_rate], axis=0)


def _random_joint_mask(key, num_strategies, p: Optional[float] = None):
  """Returns symmetrised random masks over joint actions."""
  assert len(num_strategies) == 2
  assert num_strategies[0] == num_strategies[1]
  key_p, key_m = jax.random.split(key)
  if p is None:
    p = jax.random.uniform(key_p, shape=())
  mask = jax.random.bernoulli(key_m, p=p, shape=num_strategies)
  # Ensures that at least one joint-action is observed.
  mask = mask.at[((0,) * len(num_strategies))].set(True)
  # Ensures that if payoffs (i, j) is unobserved, payoffs (j, i) isn't either.
  tril_mask = jnp.tril(mask)
  mask = jnp.logical_or(tril_mask, tril_mask.T)
  return mask


def _l_disc(
  payoffs: chex.Array,
  scale_fn: Callable,
  joint_mask: Optional[chex.Array] = None,
) -> chex.Array:
  """Returns centered and L2 normalised payoffs."""
  num_players = len(payoffs)
  payoffs_per_player = []
  for player in range(num_players):
    centered = _center(payoffs[player], player, joint_mask=joint_mask)
    scaled = scale_fn(centered, mask=joint_mask, unit_variance=True)
    payoffs_per_player.append(scaled)
  return jnp.stack(payoffs_per_player)


def l_invariant(
  key: chex.Array, scale_fn: Callable, num_strategies: Sequence[int]
) -> Tuple[chex.Array, chex.Array]:
  """Returns a sampled l1-invariant payoff tensor."""
  num_players = len(num_strategies)
  payoffs = jax.random.normal(key, shape=(num_players, *num_strategies))
  payoffs = _l_disc(payoffs, scale_fn)
  joint_mask = jnp.ones(num_strategies)
  return payoffs, joint_mask


def empirical_disc_game(
  key: chex.Array,
  num_strategies: Sequence[int],
  joint_action_keep_prob: Optional[float] = None,
  latent_size: int = 1,
) -> Tuple[chex.Array, chex.Array]:
  """Returns a sampled, optionally masked empirical disc game payoff tensor.

  Args:
    key: a `jax.random.PRNGKey` instance controlling the randomness of the
      payoff and joint-action mask sampling.
    num_strategies: the number of strategies for each player.
    joint_action_keep_prob: the probability that a joint-action is kept.
    latent_size: the dimension of the underlying latent vector.

  Returns:
    payoffs: the payoff tensor generated from the underlying randomly sampled
      latent vectors.
    mask: a tensor describing which joint action should be observed.
  """
  assert len(num_strategies) == 2
  assert num_strategies[0] == num_strategies[1]
  key_p, key_m = jax.random.split(key)
  joint_mask = _random_joint_mask(key_m, num_strategies, joint_action_keep_prob)
  payoffs = _empirical_disc_game(key_p, num_strategies, latent_size=latent_size)
  return payoffs, joint_mask


class Game(enum.Enum):
  EMPIRICAL_DISC_GAME = 0
  L1_INVARIANT = 1
  L2_INVARIANT = 2
  LINF_INVARIANT = 3


def generate_payoffs(
  game: Game,
  game_settings: Mapping[str, Any],
  num_strategies: Sequence[int],
):
  """Returns a function that generates a payoff tensor."""

  @jax.jit
  def _generate_payoff(key):
    if game == Game.L1_INVARIANT:
      payoffs, mask = l_invariant(
        key, _l1_scale, num_strategies, **game_settings
      )
    elif game == Game.L2_INVARIANT:
      payoffs, mask = l_invariant(
        key, _l2_scale, num_strategies, **game_settings
      )
    elif game == Game.LINF_INVARIANT:
      payoffs, mask = l_invariant(
        key, _linf_scale, num_strategies, **game_settings
      )
    elif game == Game.EMPIRICAL_DISC_GAME:
      payoffs, mask = empirical_disc_game(key, num_strategies, **game_settings)
    else:
      raise ValueError(f"Unrecognised game type: {game}.")
    return payoffs, mask

  return _generate_payoff
