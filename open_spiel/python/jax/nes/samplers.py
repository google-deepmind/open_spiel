import enum
import functools
import math
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pyspiel

from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.jax.nes import utils


class Data(NamedTuple):
  """Experience batch for the network."""

  reward: chex.Array  # [B, N, A1, ..., AN] or [N, A1, ..., AN]
  sigma_hat: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  epsilon_hat: chex.Array  # [B, N] or [N]
  welfare: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]

  def _broadcast(self) -> "Data":
    base_shape = self.reward.shape
    A = tuple(range(1, self.reward.ndim))
    # ε̂: [N] → [N, *A] (per-player constant across actions)
    epsilon_broadcast = jnp.broadcast_to(
      jnp.expand_dims(self.epsilon_hat, axis=A),  # Add action dims
      base_shape,
    )
    sigma_broadcast = jnp.broadcast_to(self.sigma_hat, base_shape)

    # W: [*A] → [N, *A] (same for all players)
    welfare_broadcast = jnp.broadcast_to(self.welfare, base_shape)

    return Data(
      self.reward, sigma_broadcast, epsilon_broadcast, welfare_broadcast
    )

  def stack(self) -> chex.Array:
    """Constructs [B, C=4, N, A1, ..., AN] tensor."""
    # Stack along channel dim 0: [B, 4, N, A1, ..., AN]
    return jnp.stack(
      [self.reward, self.epsilon_hat, self.sigma_hat, self.welfare], axis=1
    )


class Objective(enum.IntEnum):
  ME = 0
  MT = 1  # TODO: idk
  MU = 2
  MWME = 3
  MRE = 4
  MS = 5
  EPS_ME = 6
  EPS_MWME = 7
  EPS_MRE = 8


class GameSampler:
  """Base class for a game sampler."""

  def __init__(self, obj: Objective, m: int, z_m: float | None = None) -> None:
    self.m = m
    self.z_m = z_m
    self.obj = obj

  @functools.partial(jax.jit, static_argnums=(0,))
  def _compute_payoff_stats(self, G: chex.Array):
    action_axes = tuple(range(1, G.ndim))

    # 16(a): Compute mean μ_p = (1/|A|) Σ_a G_p(a)
    mean_payoffs = jnp.mean(G, axis=action_axes, keepdims=True)

    # 16(b): Center payoffs G̃_p(a) = G_p(a) - μ_p
    centred_payoffs = G - mean_payoffs

    # 16(c): Compute L_m norm ||G̃_p||_m = (Σ_a |G̃_p(a)|^m)^(1/m)
    norm_payoffs = utils.compute_L_m_norm(centred_payoffs, self.m, action_axes)

    # Ĝ_p(a) = G̃_p(a) · (Z_m / ||G̃_p||_m)
    scale = self.z_m / (norm_payoffs + 1e-12)

    stats = {
      "mean": mean_payoffs.squeeze(action_axes),
      "norm_raw": norm_payoffs.squeeze(action_axes),
      "scale_factor": scale.squeeze(action_axes),
    }

    G_hat = centred_payoffs * scale
    return G_hat, stats

  def _initialise_epsilon(self, G: chex.Array, rng: chex.PRNGKey) -> chex.Array:
    num_players = G.shape[0]
    if self.obj == Objective.MS:
      return jnp.full((num_players,), self.z_m)
    if self.obj in [Objective.EPS_ME, Objective.EPS_MRE, Objective.EPS_MWME]:
      return jax.random.uniform(
        rng, (num_players,), jnp.float32, -self.z_m, self.z_m
      )
    return jnp.zeros(num_players)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_epsilon(self, hat_epsilon: chex.Array, norm_payoffs: chex.Array):
    """Equation (16b): scaled + clipped \hat{ε}_p
    hat_epsilon: [N,]
    Returns \hat{ε}^{L_m}_p clipped to [-Z_m, Z_m]
    """
    # Scale target epsilon by the norm
    scaled_eps = hat_epsilon / (norm_payoffs + 1e-12)

    # Clip to [-Z_m, +Z_m] (broadcast Z_m)
    return jnp.clip(scaled_eps, -self.z_m, self.z_m)

  def _initialise_sigma(
    self, G: chex.Array, rng: chex.Array, tgt_sigma: chex.Array
  ) -> chex.Array:
    """Returns hat_sigma(a) of shape [A1, A2, ..., AN]"""
    _, *A = G.shape
    joint_A = math.prod(A)
    if self.obj in [Objective.MRE, Objective.EPS_MRE]:
      return jax.random.dirichlet(rng, alpha=jnp.ones(joint_A)).reshape(A)
    if self.obj == Objective.MT:
      return tgt_sigma
    return jnp.ones(A) / joint_A

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_sigma(self, hat_sigma: chex.Array) -> chex.Array:
    """Equation (16d): L1 unit-variance scaling for target joint distribution"""
    joint_A = math.prod(hat_sigma.shape)
    mean = 1.0 / joint_A
    z_sigma = (joint_A / jnp.sqrt(joint_A + 1.0 / joint_A)) * (
      (joint_A - 1) / joint_A
    )
    return z_sigma * (hat_sigma - mean)

  def _initialise_welfare(self, G: chex.Array) -> chex.Array:
    _, *A = G.shape
    if self.obj == Objective.MU:
      return jnp.sum(G, axis=0)  # sum over p
    if self.obj == Objective.EPS_MWME:
      return jnp.sum(G, axis=0)
    return jnp.zeros(A)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_welfare(self, W: chex.Array) -> chex.Array:
    """Equation (16c): L_m unit-variance scaling for welfare (joint)"""
    mean = jnp.mean(W, axis=tuple(range(W.ndim)), keepdims=True)
    centered = W - mean
    norm = utils.compute_L_m_norm(centered, self.m, tuple(range(W.ndim)))

    return centered / (norm + 1e-12)

  def normalise_batch(self, data: Data) -> Data:
    def _normalise_single_item(item: Data):
      # Payoffs
      G_hat, stats = self._compute_payoff_stats(item.reward)
      # Epsilon
      eps_scaled = self._scale_epsilon(item.epsilon_hat, stats["norm_raw"])
      # Sigma
      sig_scaled = self._scale_sigma(item.sigma_hat)
      # Welfare
      W_scaled = self._scale_welfare(item.welfare)

      return Data(
        reward=G_hat,
        sigma_hat=sig_scaled,
        epsilon_hat=eps_scaled,
        welfare=W_scaled,
      )._broadcast()

    return jax.vmap(_normalise_single_item)(data)


class OpenSpielGameSampler(GameSampler):
  def __init__(self, game: pyspiel.Game, obj, m, z_m=None):
    super().__init__(obj, m, z_m)
    self.game = game
    # Extract payoff tensor G_p(a)  [N, A1, ..., AN]
    self.payoff_tensor = jnp.array(
      game_payoffs_array(self.game), dtype=jnp.float32
    )
    _, *A = self.payoff_tensor.shape
    joint_A = math.prod(A)
    if self.z_m is None:
      self.z_m = jnp.array(joint_A) ** (1.0 / self.m)

  def sample_random(
    self,
    batch_size: int,
    rng: jax.make_array_from_single_device_arrays,
    target_sigma_hat: chex.Array = None,
  ) -> Data:
    """Convert any OpenSpiel normal-form / matrix game → full NES input dict."""

    G = self.payoff_tensor
    # Vectorise over the batch size with unique random keys
    batch_keys = jax.random.split(rng, batch_size)

    def _generate_single_item_random(key: jax.Array) -> Data:
      eps_key, sigma_key = jax.random.split(key, 2)
      # Payoffs
      G_hat, stats = self._compute_payoff_stats(G)

      # Epsilon
      epsilon_hat = self._initialise_epsilon(G, eps_key)
      eps_scaled = self._scale_epsilon(epsilon_hat, stats["norm_raw"])

      # Sigma
      sigma_hat = self._initialise_sigma(G, sigma_key, target_sigma_hat)
      sig_scaled = self._scale_sigma(sigma_hat)

      # Welfare
      W = self._initialise_welfare(G_hat)
      W_scaled = self._scale_welfare(W)

      return Data(
        reward=G_hat,
        sigma_hat=sig_scaled,
        epsilon_hat=eps_scaled,
        welfare=W_scaled,
      )._broadcast()

    return jax.vmap(_generate_single_item_random)(batch_keys)


class GamutGameSampler(GameSampler):
  """
  # 1. Random Game (most common baseline)
  java -jar gamut.jar --game-type random_game --num-players 2 --num-actions 3 --output-format nfg --output-file random_2p_3a.nfg

  # 2. Zero-Sum Game
  java -jar gamut.jar --game-type zero_sum_game --num-players 2 --num-actions 4 --output-format nfg --output-file zerosum_2p_4a.nfg

  # 3. Covariance Game (highly structured, common in NES experiments)
  java -jar gamut.jar --game-type covariance_game --num-players 2 --num-actions 5 --output-format nfg --output-file covariance_2p_5a.nfg

  # 4. Graphical Game (sparse payoffs)
  java -jar gamut.jar --game-type graphical_game --num-players 3 --num-actions 3 --output-format nfg --output-file graphical_3p_3a.nfg

  # 5. Polymatrix Game (common for N>2)
  java -jar gamut.jar --game-type polymatrix_game --num-players 4 --num-actions 2 --output-format nfg --output-file polymatrix_4p_2a.nfg

  # 6. 3-player non-cubic example (different action sizes)
  java -jar gamut.jar --game-type random_game --num-players 3 --num-actions 2,3,4 --output-format nfg --output-file random_3p_non_cubic.nfg
  """

  pass


class CurriculumGameSampler(GameSampler):
  pass
