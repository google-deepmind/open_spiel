import flax.nnx as nn
import jax.numpy as jnp
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils
import chex


class Payoff2PayoffPooling(networks.EquivariantPooling):
  def __init__(self, num_players: int) -> None:
    pools = []

    # identity mean
    pools.append(
      lambda x, m: utils.meanvar(
        x, axis=tuple(range(x.ndim)), keepdims=True, where=m
      )
    )

    # meanvar over player axis only
    pools.append(lambda x, m: utils.meanvar(x, axis=1, keepdims=True, where=m))

    # meanvar over own action for each player p
    for p in range(num_players):
      act_axis = 2 + p
      # over a_p
      pools.append(
        lambda x, m, ax=act_axis: utils.meanvar(
          x, axis=ax, keepdims=True, where=m
        )
      )

      # over p, a_p
      pools.append(
        lambda x, m, ax=act_axis: utils.meanvar(
          x, axis=(1, ax), keepdims=True, where=m
        )
      )

      # over a_{-p} (all except ax)
      pools.append(
        lambda x, m: utils.meanvar(
          x,
          axis=tuple(i for i in range(2, x.ndim) if i != act_axis),
          keepdims=True,
          where=m,
        )
      )

      # over p, a_{-p} (all except ax)
      pools.append(
        lambda x, m: utils.meanvar(
          x,
          axis=(1,) + tuple(i for i in range(2, x.ndim) if i != act_axis),
          keepdims=True,
          where=m,
        )
      )

    super().__init__(pools)


class Outcome2OutcomePooling(networks.EquivariantPooling):
  """Equivariant poolings for [B, C, N, O]
  (player permutation or/and outcome permutation)
  """

  def __init__(self) -> None:
    pools = [
      # identity
      lambda x: x,
      # over players
      lambda x: utils.meanvar(x, axis=1, keepdims=True),
      # over outcomes
      lambda x: utils.meanvar(x, axis=2, keepdims=True),
      # over players and outcomes
      lambda x: utils.meanvar(x, axis=(1, 2), keepdims=True),
    ]
    super().__init__(pools)


class PayoffToPayoffLayer(nn.Module):
  """Equavarian Payoff to Payoffs layer, Eq. 12"""

  def __init__(
    self,
    num_players: int,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.pooling = Payoff2PayoffPooling(num_players)
    self.linear = nn.Linear(
      in_channels * self.pooling.num_pools,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.gelu

  def __call__(self, x: chex.Array, mask: chex.Array) -> chex.Array:
    pooled = self.pooling(x, mask)
    # [N, *A, C * num_pools]
    pooled_t = jnp.moveaxis(pooled, 0, -1)
    out_t = self.act(self.linear(pooled_t))
    return jnp.moveaxis(out_t, -1, 0)


class Outcome2OutcomeLayer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    last_activation: bool,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.pool = Outcome2OutcomePooling()
    self.linear = nn.Linear(
      in_channels * self.pool.num_pools,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.softplus if last_activation else nn.gelu

  def __call__(self, x: chex.Array) -> chex.Array:
    # x: [B, C, N, O]
    pooled = self.pool(x, mask=None)
    pooled_t = jnp.moveaxis(pooled, 0, -1)  # [B, N, O, C*pools]
    out_t = self.act(self.linear(pooled_t))
    return jnp.moveaxis(out_t, -1, 0)  # [B, C', N, O]


class PayoffOutcome2PayoffOutcomeLayer(nn.Module):
  """PO2O: Payoff-Outcome -> Outcome."""

  def __init__(
    self,
    num_players: int,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    strategy_pools = []
    for p in range(num_players):
      act_axis = 2 + p
      # over a_p
      strategy_pools.append(
        lambda x, m, ax=act_axis: utils.meanvar(
          x, axis=ax, keepdims=True, where=m
        )
      )

      # over a_{-p} (all except ax)
      strategy_pools.append(
        lambda x, m: utils.meanvar(
          x,
          axis=tuple(i for i in range(2, x.ndim - 1) if i != act_axis),
          keepdims=True,
          where=m,
        )
      )

    # 1. Pool over strategy axes (all action dims)
    self.pool_strat = networks.EquivariantPooling(
      [
        lambda x, m: x,
        lambda x, m: utils.meanvar(
          x, axis=tuple(range(2, x.ndim - 1)), keepdims=True, where=m
        ),
      ]
      + strategy_pools
    )
    self.linear_strat = nn.Linear(
      in_channels * self.pool_strat.num_pools,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    # 2. Pool over player axis
    self.pool_player = networks.EquivariantPooling(
      [
        lambda x, m: x,
        lambda x, m: utils.meanvar(x, axis=1, keepdims=True, where=m),
      ]
    )
    self.linear_player = nn.Linear(
      out_channels * 2,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    # 3. Pool over outcome axis
    self.pool_outcome = networks.EquivariantPooling(
      [
        lambda x, m: x,
        lambda x, m: utils.meanvar(x, axis=-1, keepdims=True, where=m),
      ]
    )
    self.linear_outcome = nn.Linear(
      out_channels * 2,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )

    self.act = nn.gelu

  def _apply_linear(self, x: chex.Array, linear: nn.Module) -> chex.Array:
    # x: [C*pools, N, A1, ..., AN, O] → move channels to end
    x_t = jnp.moveaxis(x, 0, -1)
    out_t = linear(x_t)
    return jnp.moveaxis(out_t, -1, 0)

  def __call__(self, x: chex.Array, mask: chex.Array = None) -> chex.Array:
    # x: [B, C, N, A1, ..., AN, O]
    # Pool over strategies
    x = self.pool_strat(x, mask)
    x = self._apply_linear(x, self.linear_strat)
    x = self.act(x)
    # Pool over players
    x = self.pool_player(x, mask)
    x = self._apply_linear(x, self.linear_player)
    x = self.act(x)
    # Pool over outcomes
    x = self.pool_outcome(x, mask)
    x = self._apply_linear(x, self.linear_outcome)
    x = self.act(x)


class PayoffOutcome2OutcomeLayer(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    *,
    rngs: nn.Rngs,
  ) -> None:
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.pooling = networks.EquivariantPooling(
      [
        lambda x, m: utils.meanvar(x, axis=1, keepdims=True, where=m),
        lambda x, m: utils.meanvar(x, axis=-1, keepdims=True, where=m),
      ]
    )
    self.linear = nn.Linear(
      in_channels * 4,
      out_channels,
      rngs=rngs,
      kernel_init=networks.BASE_KERNEL_INIT,
    )
    self.act = nn.gelu

  def __call__(self, x: chex.Array, mask: chex.Array = None) -> chex.Array:
    # x: [C, N, A1, ..., AN, O]
    # Collapsing action axes
    pooled_t = utils.meanvar(x, (2, x.ndim - 1), keepdims=False, where=mask)
    pooled_t = self.pooling(pooled_t, mask=None)
    pooled_t = jnp.moveaxis(pooled_t, 0, -1)
    out_t = self.act(self.bn(self.linear(pooled_t)))
    return jnp.moveaxis(out_t, -1, 0)


class InverseEquilibriumGenerator(nn.Module):
  """
  Input:  [2, N, A1, ..., AN]  (target_sigma broadcast + noise)
  Output: [N, A1, ..., AN]     (induced game payoffs)
  """

  def __init__(
    self, num_players: int, channel_list: list[int], *, rngs: nn.Rngs
  ):
    layers = []
    in_ch = 2  # target_sigma + noise
    for out_ch in channel_list:
      layers.append(
        PayoffOutcome2OutcomeLayer(num_players, in_ch, out_ch, rngs=rngs)
      )
      in_ch = out_ch
    self.layers = nn.Sequential(*layers)
    self.final = nn.Linear(in_ch, 1, rngs=rngs)  # identity activation

  def __call__(self, target_sigma, noise, mask=None):
    # Broadcast target_sigma to match noise shape
    # target_sigma: [A1, ..., AN] → [B, 1, N, A1, ..., AN]
    target_expanded = jnp.expand_dims(target_sigma, axis=0)
    x = jnp.concatenate([target_expanded, noise], axis=0)  # [2, N, A1, ..., AN]
    for layer in self.layers:
      x = layer(x, mask)
    # Final projection: [C, N, A1, ..., AN] → [1, N, A1, ..., AN]
    x_t = jnp.moveaxis(x, 0, -1)
    out = self.final(x_t)  # identity, no activation
    return jnp.moveaxis(out, -1, 0).squeeze(1)  # [N, A1, ..., AN]


class ContractDesignGenerator(nn.Module):
  """
  Input:  dict with {
      'costs': [N, A1, ..., AN],
      'transition': [A1, ..., AN, O],
      'principal_payoff': [O,]
  }
  Output: [N, O]  (contracts v_p(o), non-negative)
  """

  def __init__(
    self,
    num_players: int,
    po2po_channels: int,
    o2o_channels: int,
    num_po2po: int,
    num_o2o: int,
    *,
    rngs: nn.Rngs,
  ):
    
    # PO2PO layers
    self.po2po_layers = nn.List(
      [
        PayoffOutcome2PayoffOutcomeLayer(
          num_players, 3 if i == 0 else po2po_channels, po2po_channels, rngs
        )
        for i in range(num_po2po)
      ]
    )
    # PO2O: collapse actions
    self.po2o = PayoffOutcome2OutcomeLayer(po2po_channels, o2o_channels, rngs)
    # O2O layers
    self.o2o_layers = nn.Sequential(
      Outcome2OutcomeLayer(
        o2o_channels,
        o2o_channels,
        last=(i == num_o2o - 1),
        rngs=rngs,
      )
      for i in range(num_o2o)
    )

  def __call__(self, costs, transition, principal_payoff, mask=None):
    # Broadcast all to [3, N, A1, ..., AN, O]

    N, *A = costs.shape
    Oc = principal_payoff.shape[-1]

    # Expand dimensions
    costs_exp = jnp.expand_dims(costs, axis=-1)  # [N, A..., 1]
    costs_exp = jnp.broadcast_to(costs_exp, (N,) + A + (Oc,))

    trans_exp = jnp.expand_dims(transition, axis=0)  # [1, A..., O]
    trans_exp = jnp.broadcast_to(trans_exp, (N,) + A + (Oc,))

    principal_exp = jnp.reshape(principal_payoff, (1,) + (1,) * len(A) + (Oc,))
    principal_exp = jnp.broadcast_to(principal_exp, (N,) + A + (Oc,))

    x = jnp.stack(
      [costs_exp, trans_exp, principal_exp], axis=0
    )  # [3, N, A..., O]

    # PO2PO layers
    for layer in self.po2po_layers:
      x = layer(x, mask)

    # PO2O: collapse to [C, N, O]
    x = self.po2o(x, mask)

    # O2O layers
    x = self.o2o_layers(x)

    # Squeeze channel dim: [1, N, O] → [N, O]
    return x.squeeze(0)


class SchedulingGenerator(nn.Module):
  #TODO: ...
  def __init__(self, num_players, channel_list, max_delta=1.0, *, rngs):
    self.num_players = num_players
    self.max_delta = max_delta
    
    # Hidden layers
    self.hidden = nn.Sequential([
      PayoffToPayoffLayer(num_players, 1, channel_list[0], rngs),
      *[PayoffToPayoffLayer(num_players, c, c, rngs) for c in channel_list[1:]],
    ])
    self.final = nn.Linear(channel_list[-1], 1, rngs=rngs)
    
    # Scalar gate for residual: taxes = gate * (-max_delta * sigmoid(...))
    # Initialize gate near 0 so taxes start near 0
    self.gate = nn.Param(jnp.array(0.01))

  def __call__(self, base_payoffs: chex.Array, mask: chex.Array=None) -> chex.Array:
    x = jnp.expand_dims(base_payoffs, axis=0)
    x = self.hidden(x, mask)
    x_t = jnp.moveaxis(x, 0, -1)
    taxes_raw = self.final(x_t)
    taxes_raw = jnp.squeeze(taxes_raw, axis=-1)
    
    taxes = -self.max_delta * nn.sigmoid(taxes_raw)
    return self.gate.value * taxes  # start near zero, grow during training