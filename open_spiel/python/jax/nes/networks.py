import enum

import chex
import flax.nnx as nn
import jax.nn.initializers as init
import jax.numpy as jnp

from open_spiel.python.jax.nes import utils

"""Implements modules from NES paper: 'Turbocharging
  Solution Concepts: Solving NEs, CEs and CCEs with
  Neural Equilibrium Solver.' See the paper:
  https://arxiv.org/abs/2210.09257
"""

class Mode(enum.Enum):
  CE = 0
  CCE = 1


BASE_KERNEL_INIT = init.variance_scaling(
  1.0, mode="fan_in", distribution="normal"
) #unit variance based on the number of inputs


class EquivariantPayoffPooling(nn.Module):
  """Payoff-to-Payoff pooling — exactly Appendix C (18a)-(18l).
  All pools are broadcast back to the original input shape [C, N, A1, ..., AN].
  """

  def __init__(self, num_players: int):
    self.num_players = num_players

    # We build the list of pooling functions once in __init__
    self.pools = []

    # (18a) identity
    self.pools.append(lambda x: x)

    # (18b) per-player mean (joint-action mean)
    self.pools.append(
      lambda x: jnp.mean(x, axis=tuple(range(2, x.ndim)), keepdims=True)
    )

    # (18c) player + joint-action mean
    self.pools.append(
      lambda x: jnp.mean(x, axis=(1,) + tuple(range(2, x.ndim)), keepdims=True)
    )

    # (18d) own-action mean for each player p
    for p in range(self.num_players):
      own_axis = 1 + p
      self.pools.append(lambda x: jnp.mean(x, axis=own_axis, keepdims=True))

    # (18e)–(18h) opponent-specific
    for q in range(num_players):
      q_act_axis = 2 + q  # Actions start at axis 2: AAAA!

      # (18e) Depends on p and a_q.
      self.pools.append(
        lambda x, qa=q_act_axis: jnp.mean(
          x, axis=tuple(i for i in range(2, x.ndim) if i != qa), keepdims=True
        )
      )

      # (18f) Depends on p and a_{-q}.
      self.pools.append(
        lambda x, qa=q_act_axis: jnp.mean(x, axis=qa, keepdims=True)
      )

      # (18g) Depends on a_q only.
      self.pools.append(
        lambda x, qa=q_act_axis: jnp.mean(
          x,
          axis=(1,) + tuple(i for i in range(2, x.ndim) if i != qa),
          keepdims=True,
        )
      )

      # (18h) Depends on a_{-q} only. REDUCE player (axis 1) AND a_q.
      self.pools.append(
        lambda x, qa=q_act_axis: jnp.mean(x, axis=(1, qa), keepdims=True)
      )

    # (18i)-(18l) Cross-player pools
    for q in range(num_players):
      for p in range(num_players):
        if p == q:
          continue
        # Use a helper factory to strictly bind p and q without closure issues
        self.pools.extend(self._build_cross_player_pools(p, q))

    self.num_pools = len(self.pools)

  def _build_cross_player_pools(self, p: int, q: int) -> list[chex.Array]:
    """Helper to build 18i-18l for a specific (p, q) pair."""

    def slice_and_broadcast(x, reduce_axes):
      # x_q shape: [C, A_0, ..., A_{N-1}]
      x_q = x[:, q]
      mean_q = jnp.mean(x_q, axis=reduce_axes, keepdims=True)
      # Add player dim back: [C, 1, A_0, ..., A_{N-1}]
      mean_q = jnp.expand_dims(mean_q, axis=1)
      return jnp.broadcast_to(mean_q, x.shape)

    p_act = 1 + p  # SLICED!
    q_act = 1 + q

    def pool_18i(x):
      return slice_and_broadcast(x, reduce_axes=p_act)

    def pool_18j(x):
      all_actions = tuple(range(1, x.ndim - 1))
      reduce_a_minus_q = tuple(i for i in all_actions if i != q_act)
      return slice_and_broadcast(x, reduce_axes=reduce_a_minus_q)

    def pool_18k(x):
      return slice_and_broadcast(x, reduce_axes=q_act)

    def pool_18l(x):
      all_actions = tuple(range(1, x.ndim - 1))
      return slice_and_broadcast(x, reduce_axes=all_actions)

    return [pool_18i, pool_18j, pool_18k, pool_18l]

  def __call__(self, x: chex.Array) -> chex.Array:
    """x shape: [C, N, A1, ..., AN]"""
    original_shape = x.shape

    def broadcast(y: chex.Array) -> chex.Array:
      """Helper to maintained the fixed tensor size."""
      return jnp.broadcast_to(y, original_shape)

    pooled_list = [broadcast(pool(x)) for pool in self.pools]

    return jnp.concatenate(pooled_list, axis=0)  # [C * num_pools, N, *A]


class EquivariantPayoffToPayoff(nn.Module):
  """Equavarian Payoff to Payoffs layer, Eq. 12"""

  def __init__(
    self,
    num_players: int,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.pooling = EquivariantPayoffPooling(num_players)
    self.linear = nn.Linear(
      in_channels * self.pooling.num_pools,
      out_channels,
      rngs=rngs,
      kernel_init=BASE_KERNEL_INIT,
    )
    self.bn = nn.BatchNorm(out_channels, rngs=rngs, axis_name='batch')
    self.act = nn.relu

  def __call__(self, x: chex.Array) -> chex.Array:
    pooled = self.pooling(x)
    # [N, *A, C * num_pools]
    pooled_t = jnp.moveaxis(pooled, 0, -1)
    out_t = self.act(self.bn(self.linear(pooled_t)))
    return jnp.moveaxis(out_t, -1, 0)


class PayoffsToDuals(nn.Module):
  """Transforms payoffs to dual variables.

  For CCE: α_p(a'_p) - one value per player-action pair
  For CE: α_p(a'_p, a''_p) - symmetric matrix per player

  Following paper Section 4.4:
  - CCE duals: use pools that sum over at least -p
  - CE duals: generate two CCE duals and take outer product, with zero diagonal
  """

  def __init__(
    self,
    mode: Mode,
    payoff_channels: int,
    dual_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.mode = mode
    self.dual_channels = dual_channels
    self.linear = nn.Linear(
      payoff_channels,
      dual_channels,
      use_bias=False,
      rngs=rngs,
      kernel_init=BASE_KERNEL_INIT,
    )
    if mode == Mode.CE:
      self.linear_aux = nn.Linear(
        payoff_channels,
        dual_channels,
        use_bias=False,
        rngs=rngs,
        kernel_init=BASE_KERNEL_INIT,
      )

  def _marginalise_payoffs(self, alpha: chex.Array) -> chex.Array:
    """Helper for marginalisation: [C_dual, N, A1, ..., AN] → [C_dual, N, A]"""

    # Detect if this is a cubic game (all players have the same number of actions)
    A_sizes = tuple(alpha.shape[2:])
    is_cubic = len(set(A_sizes)) == 1
    max_A = max(A_sizes)

    m_duals = []
    for p in range(alpha.shape[1]):
      # Slice for this player → [C_dual, A1, A2, ..., AN]
      alpha_p_slice = alpha[:, p]
      own_axis = p + 1

      # Reduce everything except the own-action dimension
      reduce_axes = tuple(
        i for i in range(1, alpha_p_slice.ndim) if i != own_axis
      )
      #  [C_dual, A_p]
      alpha_p = jnp.mean(alpha_p_slice, axis=reduce_axes, keepdims=False)
      if is_cubic:
        m_duals.append(alpha_p)
      else:
        # Pad to max_A for non-cubic games
        if alpha_p.shape[-1] < max_A:
          pad_width = [(0, 0)] * (alpha_p.ndim - 1) + [(0, max_A - alpha_p.shape[-1])]
          alpha_p = jnp.pad(alpha_p, pad_width, mode='constant')
        m_duals.append(alpha_p)
      
    # Stack along the player dimension -> [C, N, max_A]
    return jnp.stack(m_duals, axis=1)

  def __call__(self, x: chex.Array) -> chex.Array:
    # x is [C, N, *A]

    # NOTE: currently, we marginalise after the projection
    # could be moved to before.

    x_t = jnp.moveaxis(x, 0, -1)  # [N, *A, C]

    if self.mode == Mode.CCE:
      alpha_cce = jnp.moveaxis(self.linear(x_t), -1, 0)  # [C_dual, N, *A]
      return self._marginalise_payoffs(alpha_cce)

    elif self.mode == Mode.CE:
      # Outer product

      alpha_med = jnp.moveaxis(self.linear(x_t), -1, 0)
      alpha_rec = jnp.moveaxis(self.linear_aux(x_t), -1, 0)

      alpha1 = self._marginalise_payoffs(alpha_med)  # [C_dual, N, A]
      alpha2 = self._marginalise_payoffs(alpha_rec)  # [C_dual, N, A]

      # Outer product (Eq. 13 generalisation)
      # same as jnp.einsum('cni,cnj->cnij', alpha1, alpha2)
      alpha_ce = alpha1[:, :, :, None] * alpha2[:, :, None, :]
      alpha_ce = utils.mask_diagonal(alpha_ce)

      return alpha_ce

    raise ValueError(f"Unknown mode {self.mode}")


class EquivariantDualPoolingCCE(nn.Module):
  """Equivariant Payoff-to-Payoff poolings based on Appendix C, Equations of (18a)-(18l)."""

  def __init__(self, num_players: int):
    self.pools = [
      lambda x: x,  # (19a) identity
      lambda x: jnp.mean(x, axis=1, keepdims=True),  # (19b) mean over players
      lambda x: jnp.mean(x, axis=2, keepdims=True),  # (19c) mean over actions
      lambda x: jnp.mean(
        x, axis=(1, 2), keepdims=True
      ),  # (19d) mean over player + actions
    ]

    # given that there are N `num_players`
    self.num_pools = len(self.pools)

  def __call__(self, x: chex.Array) -> chex.Array:
    # x is a tensor of shape [C, N, A1, ..., AN]
    # returned shape has to be [B, C * num_pools, N, *A]
    return jnp.concatenate(
      [jnp.broadcast_to(pool(x), x.shape) for pool in self.pools], axis=0
    )  # [C * num_pools, N, *A]


class EquivariantDualPoolingCE(nn.Module):
  """Equivariant Payoff-to-Payoff poolings based on Appendix C, Equations of (18a)-(18l)."""

  def __init__(self, num_players: int):
    self.pools = [
      lambda x: x,  # (20a) identity
      lambda x: jnp.swapaxes(x, 2, 3),  # (20b) swap a' ↔ a''
      lambda x: jnp.mean(x, axis=2, keepdims=True),  # (20c) mean over a'
      lambda x: jnp.mean(x, axis=3, keepdims=True),  # (20d) mean over a''
      lambda x: jnp.mean(x, axis=(2, 3), keepdims=True),  # (20e) mean over both
      lambda x: jnp.mean(x, axis=(1, 2), keepdims=True),  # (20f) mean over p,a'
      lambda x: jnp.mean(
        x, axis=(1, 3), keepdims=True
      ),  # (20g) mean over p,a''
      lambda x: jnp.mean(
        x, axis=(1, 2, 3), keepdims=True
      ),  # (20h) mean over all
    ]

    self.num_pools = len(self.pools)

  def __call__(self, x: chex.Array) -> chex.Array:
    # x is a tensor of shape [C, N, A1, ..., AN]
    # returned shape has to be [C * num_pools, N, *A]
    return jnp.concatenate(
      [jnp.broadcast_to(pool(x), x.shape) for pool in self.pools], axis=0
    )  # [C * num_pools, N, *A]


class EquivariantDualToDual(nn.Module):
  """Equavariant Dual variables processing layer, Eq. 14-15"""

  def __init__(
    self,
    num_players: int,
    in_channels: int,
    out_channels: int,
    mode: Mode,
    last_activation: bool,
    *,
    rngs: nn.Rngs,
  ) -> None:
    
    self.pooling = (
      EquivariantDualPoolingCCE(num_players)
      if mode == Mode.CCE
      else EquivariantDualPoolingCE(num_players)
    )
    self.linear = nn.Linear(
      in_channels * self.pooling.num_pools,
      out_channels,
      rngs=rngs,
      kernel_init=BASE_KERNEL_INIT,
    )
    self.bn = nn.BatchNorm(out_channels, rngs=rngs, axis_name='batch')
    base_act = nn.relu

    if last_activation:
      self.act = (
        lambda x: nn.softplus(utils.mask_diagonal(x))
        if mode == Mode.CE
        else nn.softplus(x)
      )
    else:
      self.act = (
        lambda x: base_act(utils.mask_diagonal(x))
        if mode == Mode.CE
        else base_act(x)
      )

  def __call__(self, x: chex.Array) -> chex.Array:
    pooled = self.pooling(x)
    pooled_t = jnp.moveaxis(pooled, 0, -1)
    out_t = self.act(self.bn(self.linear(pooled_t)))
    return jnp.moveaxis(out_t, -1, 0)


class NeuralEquilibriumModel(nn.Module):
  """This is a simple model for Neural Equlibrium Solver.
  This implementation follows the paper 'Turbocharging
  Solution Concepts: Solving NEs, CEs and CCEs with
  Neural Equilibrium Solver.' For more details, see
  https://arxiv.org/abs/2210.09257

  The model encompasses the scheme depicted on Figure 2
  of the above paper and consists of 4 main modules:

  Input tensor: [C_in, N, A_1, ..., A_N]  (Joint Action Space)
    ↓
  1. EquivariantPayoffToPayoff (Eq. 12, applied sequentially)
    Maintains the full joint space to learn cross-player interactions.
    Output: [C_mid, N, A_1, ..., A_N]
    ↓
  2. PayoffsToDuals (Marginalisation & Projection)
    Collapses the joint space down to the individual space (summing over -p),
    then applies the linear projection to get base individual regrets.
    A = A_1 = ... = A_N
    Output: [C_dual, N, A]
    ↓
  3. CE Expansion (Equation 13 - Only if mode == Mode.CE)
    Bootstraps unconditional regrets into conditional pairwise rules.
    CCE path stays: [C_dual, N, A]
    CE path expands: [C_dual, N, A, A]
    ↓
  4. EquivariantDualToDual (Eq. 14 for CCE, Eq. 15 for CE)
    Refines the dual variables entirely within the individual action space.
    ↓
  Final non-negative duals α (via Softplus)
    CCE: [C_dual, N, A]
    CE:  [C_dual, N, A, A]
  """

  def __init__(
    self,
    num_players: int,
    dual_channels: int,
    mode: Mode,
    payoff_channel_list: list[int] = [64, 128, 256, 128],
    dual_channel_list: list[int] = [32, 32],
    *,
    rngs: nn.Rngs,
  ):
    # Payoff-to-payoff layers (5 layers in paper)
    payoff_layers = []
    in_ch = 4  # [payoffs, target_joint, target_epsilon, welfare]
    for i, out_ch in enumerate(payoff_channel_list):
      payoff_layers.append(
        EquivariantPayoffToPayoff(
          num_players,
          in_ch,
          out_ch,
          rngs=rngs,
        )
      )
      in_ch = out_ch

    self.eq_payoff_layers = nn.Sequential(*payoff_layers)

    # Payoffs to duals projection
    self.payoff_to_dual_proj = PayoffsToDuals(
      mode, payoff_channel_list[-1], dual_channels, rngs=rngs
    )

    # Dual-to-dual layers (2 layers in paper)
    # For CCE: [C_dual, N, A] -> ... -> [1, N, A]
    # For CE: [C_dual, N, A, A] -> ... -> [1, N, A, A]
    dual_layers = []
    in_ch = dual_channels

    for i, out_ch in enumerate(dual_channel_list):
      is_last = i == len(dual_channel_list) - 1
      dual_layers.append(
        EquivariantDualToDual(
          num_players, in_ch, out_ch, mode, last_activation=is_last, rngs=rngs
        )
      )
      in_ch = out_ch

    self.dual_layers = nn.Sequential(*dual_layers)

    # Final projection to single channel (if needed)
    self.final_proj = (
      nn.Conv(
        dual_channel_list[-1], 1, kernel_size=(1,), padding="SAME", rngs=rngs
      )
      if dual_channel_list[-1] != 1
      else lambda x: x
    )

  def __call__(self, game_tensor: chex.Array) -> chex.Array:
    """NES forward pass.

    Args:
        game_tensor (chex.Array): game tensor or shape [4, N, *A]

    Returns:
        alpha (chex.Array): dual variables of the shape dependent on
          the type of equilibrium:
          1. [C, N, |Ap|] for CCE
          2. [C, N, |Ap|, |Ap|] for CE
    """
    # game_tensor:
    x = game_tensor

    # Payoff processing
    x = self.eq_payoff_layers(x)

    # Project to dual space
    x = self.payoff_to_dual_proj(x)

    # Dual processing
    alpha = self.dual_layers(x)

    # Final projection if needed
    if x.shape[0] != 1:
      # Move channels to end for conv, then back
      alpha = jnp.moveaxis(alpha, 0, -1)
      # Ensure that the values stay positive 
      # Even if we take the projection 
      alpha = nn.softplus(self.final_proj(alpha))
      alpha = jnp.moveaxis(alpha, -1, 0)

    # Returning
    # For CCE: [1, N, A]
    # For CE: [1, N, A, A]
    return alpha
