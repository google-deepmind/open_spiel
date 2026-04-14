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
from typing import Any, Mapping, Optional, Sequence, Tuple
from absl import logging
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp


_MIN_SCALE = 1e-12


def _center(
    payoff: jnp.ndarray, player: int, joint_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
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
    tensor: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    unit_variance: bool = False,
) -> jnp.ndarray:
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
    tensor: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    min_denominator: Optional[float] = 1e-12,
    unit_variance: bool = False,
) -> jnp.ndarray:
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


def _empirical_disc_game(key, num_strategies, latent_size: int = 1):
  """Returns an instance of the empirical disc game from underlying latents."""
  assert len(num_strategies) == 2
  assert num_strategies[0] == num_strategies[1]
  k_u, k_v, k_bu, k_bv = jax.random.split(key, 4)
  u = jax.random.normal(k_u, (num_strategies[0], latent_size))
  v = jax.random.normal(k_v, (num_strategies[1], latent_size))
  u += jax.random.uniform(k_bu, (), minval=-1.0, maxval=1.0)
  v += jax.random.uniform(k_bv, (), minval=-1.0, maxval=1.0)
  logit = jnp.einsum('id,jd->ij', u, v) - jnp.einsum('jd,id->ij', u, v)
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


def _l2_disc(
    payoffs: jnp.ndarray, joint_mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
  """Returns centered and L2 normalised payoffs."""
  num_players = len(payoffs)
  payoffs_per_player = []
  for player in range(num_players):
    centered = _center(payoffs[player], player, joint_mask=joint_mask)
    scaled = _l2_scale(centered, mask=joint_mask, unit_variance=True)
    payoffs_per_player.append(scaled)
  return jnp.stack(payoffs_per_player)


def l2_invariant(
    key: jnp.ndarray, num_strategies: Sequence[int]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Returns a sampled l2-invariant payoff tensor."""
  num_players = len(num_strategies)
  payoffs = jax.random.normal(key, shape=(num_players, *num_strategies))
  payoffs = _l2_disc(payoffs)
  joint_mask = jnp.ones(num_strategies)
  return payoffs, joint_mask


def empirical_disc_game(
    key: jnp.ndarray,
    num_strategies: Sequence[int],
    joint_action_keep_prob: Optional[float] = None,
    latent_size: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
  L2_INVARIANT = 0
  EMPIRICAL_DISC_GAME = 1


def generate_payoffs(
    game: Game,
    game_settings: Mapping[str, Any],
    num_strategies: Sequence[int],
    batch_size: int,
):
  """Returns a function that generates (batches of) payoff tensors."""

  @jax.jit
  def _generate_payoff(key):
    if game == Game.L2_INVARIANT:
      payoffs, mask = l2_invariant(key, num_strategies, **game_settings)
    elif game == Game.EMPIRICAL_DISC_GAME:
      payoffs, mask = empirical_disc_game(key, num_strategies, **game_settings)
    else:
      raise ValueError(f'Unrecognised game type: {game}.')
    return payoffs, mask

  num_devices = len(jax.local_devices())
  devices = mesh_utils.create_device_mesh((num_devices, 1))
  sharding = jax.sharding.PositionalSharding(devices)
  logging.info('Mesh: %s\nSharding: %s %s', devices, sharding, sharding.shape)

  @jax.jit
  def _generate_payoffs(key):
    key, next_key = jax.random.split(key)
    keys = jax.random.split(key, batch_size)
    keys = jax.lax.with_sharding_constraint(keys, sharding)
    payoffs, masks = jax.jit(jax.vmap(_generate_payoff))(keys)
    payoffs, masks = jax.lax.with_sharding_constraint(
        (payoffs, masks),
        (
            sharding.reshape((num_devices, 1) + (1,) * len(num_strategies)),
            sharding.reshape((num_devices,) + (1,) * len(num_strategies)),
        ),
    )
    return (payoffs, masks), next_key

  return _generate_payoffs