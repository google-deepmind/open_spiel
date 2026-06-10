import enum
import functools
import dataclasses
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import pyspiel

from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.jax.nes import utils
from open_spiel.python.jax.nes import games

Game = pyspiel.Game | games.Game | list[pyspiel.Game | games.Game]


class Data(NamedTuple):
  """Experience batch for the network."""

  payoffs: chex.Array  # [B, N, A1, ..., AN] or [N, A1, ..., AN]
  strategy_base: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  strategy_norm: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  epsilon_target: chex.Array  # [B, N] or [N]
  welfare: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  strat_mask_per_player: list[chex.Array]  # [[[A_p]] or [[B, A_p]] p={1,...,N}
  joint_mask: chex.Array  # [B, A1, ..., AN] or [A1, ..., AN]
  original_action_sizes: chex.Shape | None = None


class Mechanism(NamedTuple):
  """Mehanism design batch."""

  context: chex.Array


def broadcast(data: Data) -> Data:
  """Helper to broadcast Data to [N, A_1, ..., A_N]. """

  base_shape = data.payoffs.shape
  A = tuple(range(1, data.payoffs.ndim))
  # ε̂: [N] → [N, *A] (per-player constant across actions)
  epsilon_broadcast = jnp.broadcast_to(
    jnp.expand_dims(data.epsilon_target, axis=A),  # Add action dims
    base_shape,
  )
  strat_base_broadcast = jnp.broadcast_to(data.strategy_base, base_shape)
  strat_norm_broadcast = jnp.broadcast_to(data.strategy_norm, base_shape)

  # W: [*A] → [N, *A] (same for all players)
  welfare_broadcast = jnp.broadcast_to(data.welfare, base_shape)

  return Data(
    data.payoffs,
    strat_base_broadcast,
    strat_norm_broadcast,
    epsilon_broadcast,
    welfare_broadcast,
    data.strat_mask_per_player,
    data.joint_mask,
    data.original_action_sizes,
  )


def stack(data: Data, axis=1) -> chex.Array:
  """Constructs [B, C=4, N, A1, ..., AN] tensor."""

  # Stack along channel dim 0: [B, 4, N, A1, ..., AN]
  return jnp.stack(
    [data.payoffs, data.epsilon_target, data.strategy_norm, data.welfare],
    axis=axis,
  )


def dummy_nes_batch(
  batch_size, n_players, action_sizes, rng: chex.PRNGKey
) -> Data:
  """Quick placeholder for testing without calling game samplers."""

  A = jnp.array(action_sizes)
  joint_shape = (batch_size, *A)

  return Data(
    payoffs=jnp.zeros((batch_size, n_players, *A)),
    strategy_base=jnp.ones(joint_shape) / jnp.prod(A),
    strategy_norm=jnp.ones(joint_shape) / jnp.prod(A),
    epsilon_target=jnp.zeros((batch_size, n_players)),
    welfare=jnp.zeros(joint_shape),
    strat_mask_per_player=[
      jnp.ones((batch_size, a), dtype=jnp.bool) for a in A
    ],
    joint_mask=jnp.ones((batch_size, *A), dtype=jnp.bool),
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
  EPS_MWMRE = 9  # Placeholder


class GameSampler:
  """Base class for a game sampler."""

  def __init__(self, obj: Objective, m: int, z_m: float | None = None) -> None:
    self.m = m
    self.z_m = z_m
    self.obj = obj
    self.max_actions = None

  @functools.partial(jax.jit, static_argnums=(0,))
  def _compute_payoff_stats(self, payoffs: chex.Array):
    action_axes = tuple(range(1, payoffs.ndim))

    # 16(a): Compute mean μ_p = (1/|A|) Σ_a G_p(a)
    mean_payoffs = jnp.mean(payoffs, axis=action_axes, keepdims=True)

    # 16(b): Center payoffs G̃_p(a) = G_p(a) - μ_p
    centred_payoffs = payoffs - mean_payoffs

    # 16(c): Compute L_m norm ||G̃_p||_m = (Σ_a |G̃_p(a)|^m)^(1/m)
    norm_payoffs = utils.compute_L_m_norm(centred_payoffs, self.m, action_axes)

    # Ĝ_p(a) = G̃_p(a) · (Z_m / ||G̃_p||_m)
    scale = self.z_m / (norm_payoffs + utils.SMALL_NUMBER)

    stats = {
      "mean": mean_payoffs.squeeze(action_axes),
      "norm_raw": norm_payoffs.squeeze(action_axes),
      "scale_factor": scale.squeeze(action_axes),
    }

    payoffs_norm = centred_payoffs * scale
    return payoffs_norm, stats

  def _initialise_epsilon(
    self, payoffs: chex.Array, rng: chex.PRNGKey
  ) -> chex.Array:
    num_players = payoffs.shape[0]
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

  def _initialise_strategy(
    self, payoffs: chex.Array, rng: chex.Array, strategy_proposal: chex.Array
  ) -> chex.Array:
    """Returns hat_sigma(a) of shape [A1, A2, ..., AN]"""
    _, *A = payoffs.shape
    joint_A = utils.compute_joint_action_size(A)
    if self.obj in [Objective.MRE, Objective.EPS_MRE, Objective.EPS_MWMRE]:
      return jax.random.dirichlet(rng, alpha=jnp.ones(joint_A)).reshape(A)
    if self.obj == Objective.MT:
      return strategy_proposal
    return jnp.ones(A) / joint_A

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_strategy(self, strat_base: chex.Array) -> chex.Array:
    """Equation (16d): L1 unit-variance scaling for target joint distribution"""
    joint_A = utils.compute_joint_action_size(strat_base.shape)
    mean = 1.0 / joint_A
    z_sigma = (joint_A / jnp.sqrt(joint_A + 1.0 / joint_A)) * (
      (joint_A - 1) / joint_A
    )
    return z_sigma * (strat_base - mean)

  def _initialise_welfare(self, payoffs: chex.Array) -> chex.Array:
    _, *A = payoffs.shape
    if self.obj == Objective.MU:
      return jnp.sum(payoffs, axis=0)  # sum over p
    if self.obj in [Objective.EPS_MWME, Objective.EPS_MWMRE]:
      return jnp.sum(payoffs, axis=0)
    return jnp.zeros(A)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _scale_welfare(self, welfare: chex.Array) -> chex.Array:
    """Equation (16c): L_m unit-variance scaling for welfare (joint)"""
    mean = jnp.mean(welfare, axis=tuple(range(welfare.ndim)), keepdims=True)
    centered = welfare - mean
    norm = utils.compute_L_m_norm(centered, self.m, tuple(range(welfare.ndim)))

    return centered / (norm + utils.SMALL_NUMBER)

  def normalise_batch(self, data: Data) -> Data:
    def _normalise_single_item(item: Data):
      # Payoffs
      payoff_norm, stats = self._compute_payoff_stats(item.payoffs)
      # Epsilon
      epsilon_scaled = self._scale_epsilon(
        item.epsilon_target, stats["norm_raw"]
      )
      # Sigma
      strategy_norm = self._scale_strategy(item.strategy_base)
      # Welfare
      welfare_scaled = self._scale_welfare(item.welfare)

      return Data(
        payoffs=payoff_norm,
        strategy_base=item.strategy_base,
        strategy_norm=strategy_norm,
        epsilon_target=epsilon_scaled,
        welfare=welfare_scaled,
        strat_mask_per_player=item.strat_mask_per_player,  # stays unchanged
        joint_mask=item.joint_mask,  # stays unchanged
      )

    return jax.vmap(_normalise_single_item)(data)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _sample_single_random(
    self, rng: chex.PRNGKey, strategy_proposal: chex.Array
  ) -> Data:
    raise NotImplementedError()

  def sample_random(
    self,
    batch_size: int,
    rng: chex.PRNGKey,
    strategy_proposal: chex.Array = None,
  ) -> Data:
    """Convert any OpenSpiel normal-form / matrix game → NES input Data."""

    # Vectorise over the batch size with unique random keys
    batch_keys = jax.random.split(rng, batch_size)

    return jax.vmap(self._sample_single_random)(batch_keys, strategy_proposal)


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

    self.num_players, *self.A = self.payoff_tensor.shape

    # For these types of games, masks are always identity
    self.strat_mask_per_player = [jnp.ones(a, dtype=jnp.bool) for a in self.A]
    self.joint_mask = utils.make_joint_mask_from_strat_masks(
      self.strat_mask_per_player
    )

    joint_A = utils.compute_joint_action_size(self.A)
    self.max_actions = max(self.A)
    if self.z_m is None:
      self.z_m = jnp.array(joint_A, dtype=jnp.float32) ** (1.0 / self.m)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _sample_single_random(
    self, rng: chex.PRNGKey, strategy_proposal: chex.Array
  ) -> Data:
    eps_key, strat_key = jax.random.split(rng, 2)
    # Payoffs
    payoffs = self.payoff_tensor
    payoffs_norm, stats = self._compute_payoff_stats(payoffs)

    # Epsilon
    epsilon_hat = self._initialise_epsilon(payoffs, eps_key)
    eps_scaled = self._scale_epsilon(epsilon_hat, stats["norm_raw"])

    # Sigma
    strategy_base = self._initialise_strategy(
      payoffs_norm, strat_key, strategy_proposal
    )
    strategy_norm = self._scale_strategy(strategy_base)

    # Welfare
    welfare = self._initialise_welfare(payoffs)
    welfare_scaled = self._scale_welfare(welfare)

    return Data(
      payoffs=payoffs_norm,
      strategy_base=strategy_base,
      strategy_norm=strategy_norm,
      epsilon_target=eps_scaled,
      welfare=welfare_scaled,
      strat_mask_per_player=self.strat_mask_per_player,
      joint_mask=self.joint_mask,
    )


class RandomGameSampler(GameSampler):
  def __init__(
    self,
    game: games.Game,
    num_strategies: int,
    game_settings: dict,
    obj: Objective,
    m: int,
    z_m: float = None,
    seed: int = 0,
  ) -> None:
    super().__init__(obj, m, z_m)
    self.game = game
    self.payoff_tensor_fn = games.generate_payoffs(
      self.game, game_settings, num_strategies
    )
    sample_key = jax.random.key(seed)
    sample_tensor, _ = self.payoff_tensor_fn(sample_key)
    self.num_players, *self.A = sample_tensor.shape
    self.max_actions = max(self.A)

    joint_A = utils.compute_joint_action_size(self.A)
    if self.z_m is None:
      self.z_m = jnp.array(joint_A, dtype=jnp.float32) ** (1.0 / self.m)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _sample_single_random(
    self, rng: chex.PRNGKey, strategy_proposal: chex.Array
  ) -> Data:
    payoff_rng, eps_rng, mask_rng, count_rng, strat_rng = jax.random.split(
      rng, 5
    )
    # Payoffs
    payoffs, _ = self.payoff_tensor_fn(payoff_rng)
    payoffs_norm, stats = self._compute_payoff_stats(payoffs)

    action_counts = jax.random.randint(
      count_rng, (self.num_players,), minval=2, maxval=self.max_actions + 1
    )

    strat_mask_per_player = utils.rand_choice_strat_mask(
      mask_rng, self.A, action_counts
    )
    joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)

    # Epsilon
    epsilon_hat = self._initialise_epsilon(payoffs, eps_rng)
    eps_scaled = self._scale_epsilon(epsilon_hat, stats["norm_raw"])

    # Sigma
    strategy_base = self._initialise_strategy(
      payoffs_norm, strat_rng, strategy_proposal
    )
    strategy_norm = self._scale_strategy(strategy_base)

    # Welfare
    W = self._initialise_welfare(payoffs)
    welfare_scaled = self._scale_welfare(W)

    return Data(
      payoffs=payoffs_norm,
      strategy_base=strategy_base,
      strategy_norm=strategy_norm,
      epsilon_target=eps_scaled,
      welfare=welfare_scaled,
      strat_mask_per_player=strat_mask_per_player,
      joint_mask=joint_mask,
    )


@dataclasses.dataclass
class MixedGameConfig:
  """Configuration for a single game source in the mixed sampler."""

  # For random games
  game_type: games.Game = None
  num_strategies: tuple[int, ...] | None = None
  game_settings: dict = dataclasses.field(default_factory=dict)

  # For OpenSpiel games
  open_spiel_name: str | None = None

  # Sampling weight relative to other configs
  weight: float = 1.0

  def __post_init__(self) -> None:
    chex.assert_not_both_none(self.game_type, self.open_spiel_name)


class MultiGameSampler(GameSampler):
  """
  Samples a batch of games with varying sizes and pads the game tensor
    to a common shape.

  Each game in the batch might have a different number of actions per player.
  All games are padded to [N, max_actions, ..., max_actions] with zeros,
  and a boolean mask indicates valid entries.
  """

  def __init__(
    self,
    game_configs: list[MixedGameConfig],
    max_actions: int,
    obj: Objective = Objective.EPS_MWMRE,
    m: int = 2,
    z_m: float | None = None,
  ) -> None:
    super().__init__(obj, m, z_m=z_m)

    self._samplers: list[OpenSpielGameSampler | RandomGameSampler] = []
    self.num_players = None
    for gc in game_configs:
      if gc.game_type is not None:
        sampler = RandomGameSampler(
          game=gc.game_type,
          num_strategies=gc.num_strategies,
          game_settings=gc.game_settings,
          obj=obj,
          z_m=z_m,
          m=m,
        )

      elif gc.open_spiel_name is not None:
        sampler = OpenSpielGameSampler(
          game=pyspiel.load_game(gc.open_spiel_name), obj=obj, z_m=z_m, m=m
        )
      else:
        # No good sampler - go next.
        continue

      if self.num_players is None:
        self.num_players = sampler.num_players
      else:
        assert self.num_players == sampler.num_players, (
          "All games have to have the same number of players"
        )

      self._samplers.append(sampler)

    self.max_actions = (
      max_actions
      if max_actions is not None
      else max(sampler.max_actions for sampler in self._samplers)
    )

    if self.z_m is None:
      self.z_m = jnp.array(
        self.max_actions**self.num_players, dtype=jnp.float32
      ) ** (1.0 / self.m)

    weights = jnp.array([gc.weight for gc in game_configs])
    self.sampling_probs = weights / jnp.sum(weights)

  def _sample_single_random(
    self, rng: chex.PRNGKey, idx: int, strategy_proposal: chex.Array = None
  ) -> Data:
    sampler = self._samplers[idx]
    raw = sampler._sample_single_random(rng, strategy_proposal)
    return self._pad_data(raw)

  def _pad_data(self, data: Data) -> Data:
    action_sizes = data.joint_mask.shape
    payoffs = utils.pad_game_tensor(data.payoffs, self.max_actions, 0.0)
    mask_pad = utils.joint_mask(action_sizes, self.max_actions)

    def _pad_stategy(strat, normalise):
      sigma_pad = utils.pad_game_tensor(
        jnp.expand_dims(strat, 0), self.max_actions, 0.0
      ).squeeze(0)
      sigma_pad = jnp.where(mask_pad, sigma_pad, 0.0)
      if normalise:
        sigma_sum = jnp.sum(sigma_pad, where=mask_pad)
        sigma_pad = jnp.where(
          mask_pad, sigma_pad / (sigma_sum + utils.SMALL_NUMBER), 0.0
        )
      return sigma_pad

    def _pad_player_mask(mask: chex.Array) -> chex.Array:
      """Pad a single player's strategy mask to max_actions."""
      pad_width = (0, self.max_actions - mask.shape[0])
      return jnp.pad(mask, pad_width, mode="constant", constant_values=False)

    strat_masks_pad = [_pad_player_mask(m) for m in data.strat_mask_per_player]

    welfare = utils.pad_game_tensor(
      jnp.expand_dims(data.welfare, 0), self.max_actions, 0.0
    ).squeeze(0)

    return Data(
      payoffs=payoffs,
      strategy_base=_pad_stategy(data.strategy_base, True),
      strategy_norm=_pad_stategy(data.strategy_norm, False),
      epsilon_target=data.epsilon_target,
      welfare=welfare,
      strat_mask_per_player=strat_masks_pad,
      joint_mask=mask_pad,
    )

  def sample_random(
    self,
    batch_size: int,
    rng: chex.PRNGKey,
    idx: int = None,
    strategy_proposal: chex.Array = None,
  ) -> Data:
    keys = jax.random.split(rng, batch_size)

    if idx is None:
      idx_rng, rng = jax.random.split(rng)
      source_idx = jax.random.choice(
        idx_rng, len(self._samplers), p=self.sampling_probs
      )
      idx = source_idx

    return jax.vmap(
      functools.partial(
        self._sample_single_random, idx=idx, strategy_proposal=strategy_proposal
      )
    )(keys)
