import enum
from typing import Callable

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
)  # unit variance based on the number of inputs


class EquivariantPooling(nn.Module):
  """Base class for all equivariant poolings."""

  def __init__(self, pool_fns: list[Callable]) -> None:
    self.pools = pool_fns
    self.num_pools = len(pool_fns)

  def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
    """x shape: [C, N, A1, ..., AN]"""

    return jnp.concatenate(
      [pool(x, mask) for pool in self.pools], axis=0
    )  # [C * num_pools, N, *A]


class EquivariantPayoffPooling(EquivariantPooling):
  """Payoff-to-Payoff poolings from Appendix C (18a)-(18l)."""

  def __init__(self, num_players: int) -> None:
    # We build the list of pooling functions once in __init__
    pools = []

    # (18a) identity
    pools.append(lambda x, m: x)

    # (18b) per-player mean (joint-action mean) φ_{a_1,...,a_N} g(p, ...)
    pools.append(
      lambda x, mask: utils.reduce(
        x,
        tuple(range(2, x.ndim)),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        where=mask,
        broadcast=True,
      )
    )

    # (18c) player + joint-action mean φ_{p, a_1,...,a_N} g(p, ...)
    pools.append(
      lambda x, mask: utils.reduce(
        x,
        (1,) + tuple(range(2, x.ndim)),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        where=mask,
        broadcast=True,
      )
    )

    # (18d) own-action mean for each player p φ_p g(p, ...)
    pools.append(
      lambda x, mask: utils.reduce(
        x,
        (1,),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        where=mask,
        broadcast=True,
      )
    )

    # (18e)–(18h) opponent-specific
    for q in range(num_players):
      q_act_axis = 2 + q  # Actions start at axis 2: AAAA!

      # (18e) Depends on p and a_q: φ_{p, a_q} g(p, ...)
      pools.append(
        lambda x, m, qa=q_act_axis: utils.reduce(
          x,
          axis=(1, qa),
          reduction=utils.Reduction.MEAN,
          include_identity=False,
          include_all=True,
          include_other=False,
          where=m,
          broadcast=True,
        )
      )

      # (18f)  φ_{p, a_{-q}} g(p, ...)
      pools.append(
        lambda x, m, qa=q_act_axis: utils.reduce(
          x,
          axis=(1,) + tuple(i for i in range(2, x.ndim) if i != qa),
          reduction=utils.Reduction.MEAN,
          include_identity=False,
          include_all=True,
          include_other=False,
          where=m,
          broadcast=True,
        )
      )

      # (18g) Depends on a_q only.
      pools.append(
        lambda x, m, qa=q_act_axis: utils.reduce(
          x,
          axis=(qa,),
          reduction=utils.Reduction.MEAN,
          include_identity=False,
          include_all=True,
          include_other=False,
          where=m,
          broadcast=True,
        )
      )

      # (18h) Depends on a_{-q} only. REDUCE player (axis 1) and a_q.
      pools.append(
        lambda x, m, qa=q_act_axis: utils.reduce(
          x,
          axis=tuple(i for i in range(2, x.ndim) if i != qa),
          reduction=utils.Reduction.MEAN,
          include_identity=False,
          include_all=True,
          include_other=False,
          where=m,
          broadcast=True,
        )
      )

    # (18i)-(18l) Cross-player pools
    for q in range(num_players):
      for p in range(num_players):
        if p == q:
          continue
        # Use a helper factory to strictly bind p and q without closure issues
        pools.extend(self._build_cross_player_pools(p, q))

    super().__init__(pools)

  def _build_cross_player_pools(self, p: int, q: int):
    """Build (18i)-(18l) for a specific (p, q) pair, p ≠ q.

    All pools take player q's payoff, reduce over axes related to p,
    then broadcast back to [C, N, A1, ..., AN].
    """
    pools = []

    def _slice_reduce_broadcast(x, mask, reduce_axes):
      """Slice to player q, reduce, broadcast back."""
      # x: [C, N, A1, ..., AN]
      # x_q: [C, A1, ..., AN]
      x_q = x[:, q]

      # Apply mask if provided (mask has shape [A1, ..., AN])
      if mask is not None:
        where = mask
      else:
        where = None

      # Reduce
      reduced = utils.reduce(
        x_q,
        reduce_axes,
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=False,
        where=where,
      )  # shape depends on reduce_axes, but broadcast=True ensures [C, A1, ..., AN]

      # Add player dimension back: [C, 1, A1, ..., AN]
      expanded = jnp.expand_dims(reduced, axis=1)

      # Broadcast to [C, N, A1, ..., AN]
      return jnp.broadcast_to(expanded, x.shape)

    p_act = 1 + p
    q_act = 1 + q

    # (18i) Reduce over a_p
    # φ_{a_p} g(q, ...)
    pools.append(lambda x, m: _slice_reduce_broadcast(x, m, (p_act,)))

    # (18j) Reduce over a_{-p}
    pools.append(
      lambda x, m: _slice_reduce_broadcast(
        x, m, tuple(i for i in range(1, x.ndim - 1) if i != p_act)
      )
    )

    # (18k) Reduce over a_q
    # φ_{a_p} g(q, ...)
    pools.append(lambda x, m: _slice_reduce_broadcast(x, m, (q_act,)))

    # (18l) Reduce over a_{-q}
    # φ_{a_{-q}} g(q, ...)
    pools.append(
      lambda x, m: _slice_reduce_broadcast(
        x, m, tuple(i for i in range(1, x.ndim - 1) if i != q_act)
      )
    )

    return pools


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
    self.bn = nn.BatchNorm(out_channels, rngs=rngs, axis_name="batch")
    self.act = nn.relu

  def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
    pooled = self.pooling(x, mask)
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
      rngs=rngs,
      kernel_init=BASE_KERNEL_INIT,
    )
    if mode == Mode.CE:
      self.linear_aux = nn.Linear(
        payoff_channels,
        dual_channels,
        rngs=rngs,
        kernel_init=BASE_KERNEL_INIT,
      )

  def _marginalise_payoffs(
    self, alpha: chex.Array, joint_mask: chex.Array
  ) -> chex.Array:
    """Helper for marginalisation: [C_dual, N, A1, ..., AN] → [C_dual, N, A]"""

    # Detect if this is a cubic game (all players have the same number of actions)
    A_sizes = tuple(alpha.shape[2:])
    is_cubic = len(set(A_sizes)) == 1
    max_A = max(A_sizes)

    m_duals = []
    for p in range(alpha.shape[1]):
      # [C_dual, A1, A2, ..., AN]
      alpha_p_slice = alpha[:, p]
      own_axis = p + 1

      # Reduce everything except the own-action dimension
      reduce_axes = tuple(
        i for i in range(1, alpha_p_slice.ndim) if i != own_axis
      )
      #  [C_dual, A_p]
      alpha_p = utils.reduce(
        alpha_p_slice,
        axis=reduce_axes,
        reduction=utils.Reduction.MEAN,
        where=joint_mask,
        include_identity=False,
        include_other=False,
        broadcast=False,
      ).squeeze()

      if is_cubic:
        m_duals.append(alpha_p)
      else:
        # Pad to max_A for non-cubic games
        if alpha_p.shape[-1] < max_A:
          pad_width = [(0, 0)] * (alpha_p.ndim - 1) + [
            (0, max_A - alpha_p.shape[-1])
          ]
          alpha_p = jnp.pad(alpha_p, pad_width, mode="constant")
        m_duals.append(alpha_p)

    # Stack along the player dimension -> [C, N, max_A]
    return jnp.stack(m_duals, axis=1)

  def __call__(self, x: chex.Array, joint_mask: chex.Array) -> chex.Array:
    # x is [C, N, *A]

    # NOTE: currently, we marginalise after the projection
    # could be moved to before.

    x_t = jnp.moveaxis(x, 0, -1)  # [N, *A, C]

    if self.mode == Mode.CCE:
      alpha_cce = jnp.moveaxis(self.linear(x_t), -1, 0)  # [C_dual, N, *A]
      return self._marginalise_payoffs(alpha_cce, joint_mask)

    elif self.mode == Mode.CE:
      # Outer product

      alpha_med = jnp.moveaxis(self.linear(x_t), -1, 0)
      alpha_rec = jnp.moveaxis(self.linear_aux(x_t), -1, 0)

      alpha1 = self._marginalise_payoffs(
        alpha_med, joint_mask
      )  # [C_dual, N, A]
      alpha2 = self._marginalise_payoffs(
        alpha_rec, joint_mask
      )  # [C_dual, N, A]

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
      lambda x: utils.reduce(
        x,
        (1,),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (19b) mean over players
      lambda x: utils.reduce(
        x,
        (2,),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (19c) mean over actions
      lambda x: utils.reduce(
        x,
        (1, 2),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (19d) mean over player + actions
    ]

    # given that there are N `num_players`
    self.num_pools = len(self.pools)

  def __call__(self, x: chex.Array) -> chex.Array:
    # x is a tensor of shape [C, N, A1, ..., AN]
    # returned shape has to be [B, C * num_pools, N, *A]
    return jnp.concatenate(
      [pool(x) for pool in self.pools], axis=0
    )  # [C * num_pools, N, *A]


class EquivariantDualPoolingCE(nn.Module):
  """Equivariant Payoff-to-Payoff poolings based on Appendix C, Equations of (18a)-(18l)."""

  def __init__(self, num_players: int):
    self.pools = [
      lambda x: x,  # (20a) identity
      lambda x: jnp.swapaxes(x, 2, 3),  # (20b) swap a' ↔ a''
      lambda x: utils.reduce(
        x,
        (2,),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20c) mean over a'
      lambda x: utils.reduce(
        x,
        (2,),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20d) mean over a''
      lambda x: utils.reduce(
        x,
        (2, 3),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20e) mean over both
      lambda x: utils.reduce(
        x,
        (1, 2),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20f) mean over p,a'
      lambda x: utils.reduce(
        x,
        (1, 3),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20g) mean over p,a''
      lambda x: utils.reduce(
        x,
        (1, 2, 3),
        utils.Reduction.MEAN,
        include_identity=False,
        include_all=True,
        include_other=False,
        broadcast=True,
      ),  # (20h) mean over all
    ]

    self.num_pools = len(self.pools)

  def __call__(self, x: chex.Array) -> chex.Array:
    # x is a tensor of shape [C, N, A1, ..., AN]
    # returned shape has to be [C * num_pools, N, *A]
    return jnp.concatenate(
      [pool(x) for pool in self.pools], axis=0
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
    self.bn = nn.BatchNorm(out_channels, rngs=rngs, axis_name="batch")
    base_act = nn.relu

    if last_activation:
      self.act = (
        lambda x: utils.mask_diagonal(nn.softplus(x))
        if mode == Mode.CE
        else nn.softplus(x)
      )
    else:
      self.act = (
        lambda x: utils.mask_diagonal(base_act(x))
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
  ) -> None:
    self._mode = mode

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

    self.eq_payoff_layers = nn.List(payoff_layers)

    # Payoffs to duals projection
    self.payoff_to_dual_proj = PayoffsToDuals(
      mode, payoff_channel_list[-1], dual_channels, rngs=rngs
    )

    # Dual-to-dual layers (2 layers in paper)
    # For CCE: [C_dual, N, A] -> ... -> [1, N, A]
    # For CE: [C_dual, N, A, A] -> ... -> [1, N, A, A]
    dual_layers = []
    in_ch = dual_channels

    if dual_channel_list[-1] != 1:
      dual_channel_list += [1]

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
      nn.Linear(
        dual_channel_list[-1], 1, rngs=rngs, kernel_init=BASE_KERNEL_INIT
      )
      if dual_channel_list[-1] != 1
      else lambda x: x
    )

  def _mask_alpha(self, alpha: chex.Array, mask: chex.Array) -> chex.Array:
    """Zero out alphas for padded actions using the joint mask."""
    N = alpha.shape[1]
    alpha_mask = []

    if self._mode == Mode.CCE:
      # alpha: [C, N, A]; mask: [A1, ..., AN]
      # Per-player valid actions: does any valid joint action use this a_p?
      for p in range(N):
        alpha_mask.append(mask[p])
    else:  # CE
      # alpha: [C, N, A, A]; mask: [A1, ..., AN]
      for p in range(N):
        valid = mask[p]
        # Outer product for (dev, rec)
        ce_valid = jnp.logical_and(valid[:, None], valid[None, :])  # [A, A]
        diag = jnp.arange(ce_valid.shape[0])
        ce_valid = ce_valid.at[diag, diag].set(False)
        alpha_mask.append(ce_valid)

    alpha_mask = jnp.stack(alpha_mask)
    alpha_mask = jnp.broadcast_to(alpha_mask, alpha.shape)
    return jnp.where(alpha_mask, alpha, 0.0)

  def __call__(
    self,
    game_tensor: chex.Array,
    strat_mask_per_player: list[chex.Array],
    joint_mask: chex.Array | None = None,
  ) -> chex.Array:
    """NES forward pass.

    Args:
        game_tensor (chex.Array): game tensor or shape [4, N, *A]

    Returns:
        alpha (chex.Array): dual variables of the shape dependent on
          the type of equilibrium:
          1. [C, N, |Ap|] for CCE
          2. [C, N, |Ap|, |Ap|] for CE
    """
    # Payoff processing
    x = game_tensor

    if joint_mask is None:
      joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)

    # Can't be standard `nnx.Sequential` as it accepts mask
    for layer in self.eq_payoff_layers:
      x = layer(x, joint_mask)

    # Project to dual space
    x = self.payoff_to_dual_proj(x, joint_mask)

    # Dual processing
    alpha = self.dual_layers(x)

    # Returning
    # For CCE: [1, N, A]
    # For CE: [1, N, A, A]

    # Mask padded duals
    alpha = self._mask_alpha(alpha, strat_mask_per_player)

    return alpha.squeeze(0)
