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
from open_spiel.python.jax.nes import games


class Data(NamedTuple):
  """Experience batch for the network."""

  reward: chex.Array  # [B, N, A1, ..., AN] or [N, A1, ..., AN]
  sigma_hat: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  sigma_norm: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  epsilon_hat: chex.Array  # [B, N] or [N]
  welfare: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  mask: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]


def broadcast(data: Data) -> Data:
  base_shape = data.reward.shape
  A = tuple(range(1, data.reward.ndim))
  # ε̂: [N] → [N, *A] (per-player constant across actions)
  epsilon_broadcast = jnp.broadcast_to(
    jnp.expand_dims(data.epsilon_hat, axis=A),  # Add action dims
    base_shape,
  )
  sigma_hat_broadcast = jnp.broadcast_to(data.sigma_hat, base_shape)
  sigma_norm_broadcast = jnp.broadcast_to(data.sigma_hat, base_shape)

  # W: [*A] → [N, *A] (same for all players)
  welfare_broadcast = jnp.broadcast_to(data.welfare, base_shape)

  return Data(
    data.reward,
    sigma_hat_broadcast,
    sigma_norm_broadcast,
    epsilon_broadcast,
    welfare_broadcast,
    data.mask,
  )


def stack(data: Data, axis=1) -> chex.Array:
  """Constructs [B, C=4, N, A1, ..., AN] tensor."""
  # Stack along channel dim 0: [B, 4, N, A1, ..., AN]
  return jnp.stack(
    [data.reward, data.epsilon_hat, data.sigma_norm, data.welfare], axis=axis
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
  EPS_MWMRE = 9


class GameSampler:
  """Base class for a game sampler."""

  def __init__(self, obj: Objective, m: int, z_m: float | None = None) -> None:
    self.m = m
    self.z_m = z_m
    assert obj == Objective.EPS_MWMRE, (
      f"Currently only {Objective.EPS_MWMRE} objective is supported"
    )
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
    scale = self.z_m / (norm_payoffs + utils.SMALL_NUMBER)

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
    if self.obj in [
      Objective.EPS_ME,
      Objective.EPS_MRE,
      Objective.EPS_MWME,
      Objective.EPS_MWMRE,
    ]:
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
    safe_norm = jnp.maximum(norm_payoffs, 1e-3)
    scaled_eps = hat_epsilon / (safe_norm + utils.SMALL_NUMBER)

    # Clip to [-Z_m, +Z_m] (broadcast Z_m)
    return jnp.clip(scaled_eps, -self.z_m, self.z_m)

  def _initialise_sigma(
    self, G: chex.Array, rng: chex.Array, tgt_sigma: chex.Array
  ) -> chex.Array:
    """Returns hat_sigma(a) of shape [A1, A2, ..., AN]"""
    _, *A = G.shape
    joint_A = math.prod(A)
    if self.obj in [Objective.MRE, Objective.EPS_MRE, Objective.EPS_MWMRE]:
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
    if self.obj in [Objective.EPS_MWME, Objective.EPS_MWMRE]:
      return jnp.sum(G, axis=0)
    return jnp.zeros(A)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_welfare(self, W: chex.Array) -> chex.Array:
    """Equation (16c): L_m unit-variance scaling for welfare (joint)"""
    mean = jnp.mean(W, axis=tuple(range(W.ndim)), keepdims=True)
    centered = W - mean
    norm = utils.compute_L_m_norm(centered, self.m, tuple(range(W.ndim)))

    return centered / (norm + utils.SMALL_NUMBER)

  def normalise_batch(self, data: Data) -> Data:
    def _normalise_single_item(item: Data):
      # Payoffs
      G_hat, stats = self._compute_payoff_stats(item.reward)
      # Epsilon
      eps_scaled = self._scale_epsilon(item.epsilon_hat, stats["norm_raw"])
      # Sigma
      sigma_scaled = self._scale_sigma(item.sigma_hat)
      # Welfare
      W_scaled = self._scale_welfare(item.welfare)

      return Data(
        reward=G_hat,
        sigma_hat=item.epsilon_hat,
        sigma_norm=sigma_scaled,
        epsilon_hat=eps_scaled,
        welfare=W_scaled,
        mask=data.mask,  # stays unchanged
      )

    return jax.vmap(_normalise_single_item)(data)


class OpenSpielGameSampler(GameSampler):
  """A simple sampler for NFGs in OpenSpiel."""

  def __init__(
    self, game: pyspiel.Game, obj: Objective, m: int, z_m: float = None
  ) -> None:
    super().__init__(obj, m, z_m)
    self.game = game
    # Extract payoff tensor G_p(a)  [N, A1, ..., AN]
    self.payoff_tensor = jnp.array(
      game_payoffs_array(self.game), dtype=jnp.float32
    )
    _, *A = self.payoff_tensor.shape
    # For these types of games, mask is always identity
    self.mask = jnp.ones(A, dtype=jnp.bool)
    joint_A = math.prod(A)
    if self.z_m is None:
      self.z_m = jnp.array(joint_A) ** (1.0 / self.m)

  def sample_random(
    self,
    batch_size: int,
    rng: chex.PRNGKey,
    target_sigma_hat: chex.Array = None,
  ) -> Data:
    """Convert any OpenSpiel normal-form / matrix game → NES input Data."""

    G = self.payoff_tensor
    # Vectorise over the batch size with unique random keys
    batch_keys = jax.random.split(rng, batch_size)

    @jax.jit
    def _generate_single_item_random(key: jax.Array) -> Data:
      eps_key, sigma_key = jax.random.split(key, 2)
      # Payoffs
      G_hat, stats = self._compute_payoff_stats(G)

      # Epsilon
      epsilon_hat = self._initialise_epsilon(G, eps_key)
      eps_scaled = self._scale_epsilon(epsilon_hat, stats["norm_raw"])

      # Sigma
      sigma_hat = self._initialise_sigma(G, sigma_key, target_sigma_hat)
      sigma_scaled = self._scale_sigma(sigma_hat)

      # Welfare
      W = self._initialise_welfare(G)
      W_scaled = self._scale_welfare(W)

      return Data(
        reward=G_hat,
        sigma_hat=sigma_hat,
        sigma_norm=sigma_scaled,
        epsilon_hat=eps_scaled,
        welfare=W_scaled,
        mask=self.mask,
      )

    return jax.vmap(_generate_single_item_random)(batch_keys)


class RandomGameSampler(GameSampler):
  def __init__(
    self,
    game: games.Game,
    num_strategies: int,
    game_settings: dict,
    obj: Objective,
    m: int,
    z_m: float = None,
  ) -> None:
    super().__init__(obj, m, z_m)
    self.game = game
    self.payoff_tensor_fn = games.generate_payoffs(
      self.game, game_settings, num_strategies
    )
    sample_key = jax.random.key(0)
    sample_tensor, mask = self.payoff_tensor_fn(sample_key)
    _, *A = sample_tensor.shape
    joint_A = utils.compute_joint_action_size(A)
    if self.z_m is None:
      self.z_m = jnp.array(joint_A) ** (1.0 / self.m)

  def sample_random(
    self,
    batch_size: int,
    rng: chex.Array,
    target_sigma_hat: chex.Array = None,
  ) -> Data:
    """Convert any OpenSpiel normal-form / matrix game → full NES input dict."""
    # Vectorise over the batch size with unique random keys
    batch_keys = jax.random.split(rng, batch_size)

    @jax.jit
    def _generate_single_item_random(rng: jax.Array) -> Data:
      payoff_rng, eps_rng, sigma_rng = jax.random.split(rng, 3)
      # Payoffs
      G, mask = self.payoff_tensor_fn(payoff_rng)
      G_hat, stats = self._compute_payoff_stats(G)

      # Epsilon
      epsilon_hat = self._initialise_epsilon(G, eps_rng)
      eps_scaled = self._scale_epsilon(epsilon_hat, stats["norm_raw"])

      # Sigma
      sigma_hat = self._initialise_sigma(G, sigma_rng, target_sigma_hat)
      sigma_scaled = self._scale_sigma(sigma_hat)

      # Welfare
      W = self._initialise_welfare(G_hat)
      W_scaled = self._scale_welfare(W)

      return Data(
        reward=G_hat,
        sigma_hat=sigma_hat,
        sigma_norm=sigma_scaled,
        epsilon_hat=eps_scaled,
        welfare=W_scaled,
        mask=mask,
      )

    return jax.vmap(_generate_single_item_random)(batch_keys)


class MixedGameSampler(GameSampler):
  pass
