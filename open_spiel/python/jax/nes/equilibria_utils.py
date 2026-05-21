import functools
from typing import Callable
import chex
import string
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.nes import utils


def _map_per_player(fn, *, payoffs=None, **per_player_kwargs) -> dict:
  """Map `fn(player, payoff=..., **kwargs)` over all players.

  Args:
      fn: Per-player function. First positional arg is `player`.
      payoffs: Optional [N, A1, ..., AN] array.
      **per_player_kwargs: Optional sequences of length N.
  """
  if payoffs is not None:
    num_players = len(payoffs)
    payoff_list = tuple(payoffs[p] for p in range(num_players))
  else:
    for v in per_player_kwargs.values():
      if v is not None:
        num_players = len(v)
        break
    else:
      raise ValueError("Cannot infer num_players from arguments")
    payoff_list = (None,) * num_players

  kwarg_tuples = {
    k: tuple(v) if v is not None else (None,) * num_players
    for k, v in per_player_kwargs.items()
  }

  return tuple(
    fn(
      player=p,
      payoff=payoff_list[p],
      **{k: v[p] for k, v in kwarg_tuples.items()},
    )
    for p in range(num_players)
  )


@functools.partial(jax.jit, static_argnames=("player",))
def cce_gain_per_player(
  player: int,
  *,
  payoff: chex.Array | None = None,
  ce_gain: chex.Array | None = None,
  wsce_gain: chex.Array | None = None,
) -> chex.Array:
  """Computes PER PLAYER CCE deviation (gain).

  The gain is defined as:
    A_p^CCE(a_p', a) = G_p(a_p', a_{-p}) - G_p(a_p, a_{-p}),
    where:
      - a_p' is a recommended action
      - G_p is a payoff for a player `p`.

  Args:
    player: Integer player index p.
    payoff: Optional array with shape [A1, ..., AN].
    ce_gain: Optional array with shape [A_p', A_p'', A1, ..., AN].
    wsce_gain: Optional array with shape
      [A_p', A_p'', A1, ..., A_{p-1}, A_{p+1}, ..., AN].

  Returns:
    cce_gain: Array with shape [A_p', A1, ..., AN].
  """

  if payoff is not None:
    # Shape: [    1,|A_1|,...,|A_p-1|,|A_p|,|A_p+1|,...,|A_N|]
    payoff = jnp.expand_dims(payoff, 0)
    # The swap is changing shapes:
    # From:  [    1,|A_1|,...,|A_p-1|,|A_p|,|A_p+1|,...,|A_N|]
    # To:    [|A_p|,|A_1|,...,|A_p-1|,    1,|A_p+1|,...,|A_N|]
    dev_term = jnp.swapaxes(payoff, 0, player + 1)
    return dev_term - payoff

  if ce_gain is not None:
    # The sum is changing shapes:
    # From:  [|A'_p|,|A"_p|, |A_1|,...,|A_N|]
    # To:    [|A'_p|,        |A_1|,...,|A_N|]
    return jnp.sum(ce_gain, axis=1)

  if wsce_gain is not None:
    # [A_p', A1, ..., AN]
    return jnp.moveaxis(wsce_gain, 1, player + 1)


def cce_gain(
  *,
  payoffs: chex.Array | None = None,
  ce_gain_per_player: list[chex.Array] | None = None,
  wsce_gain_per_player: list[chex.Array] | None = None,
) -> tuple[chex.Array, ...]:
  """Computes CCE deviation gain for every player.

  Args:
    payoffs: Optional array with shape [N, A1, ..., AN].
    ce_gain_per_player: Optional sequence of arrays with shape
      [[A_p', A_p'', A1, ..., AN]]_{p=1:N}.
    wsce_gain_per_player: Optional sequence of arrays with shape
      [[A_p', A_p'', A1, ..., A_{p-1}, A_{p+1}, ..., AN]]_{p=1:N}.

  Returns:
    cce_gain_per_player: Tuple of arrays with shape
      [[A_p', A1, ..., AN]]_{p=1:N}.
  """

  return _map_per_player(
    cce_gain_per_player,
    payoffs=payoffs,
    ce_gain=ce_gain_per_player,
    wsce_gain=wsce_gain_per_player,
  )


@functools.partial(jax.jit, static_argnames=("player",))
def expected_cce_gain_per_player(
  *,
  player: int | None = None,
  payoff: chex.Array | None = None,
  cce_gain: chex.Array | None = None,
  ce_gain: chex.Array | None = None,
  cce_dual_grad: chex.Array | None = None,
  correlated_joint_strategy: chex.Array | None = None,
  player_marg_per_player: list[chex.Array] | None = None,
  strat_mask: chex.Array | None = None,
) -> chex.Array:
  """Computes PER PLAYER expected CCE gain.

  Must supply exactly one of {payoff, cce_gain, ce_gain, cce_dual_grad}.
  Must supply exactly one of {correlated_joint_strategy, player_marg_per_player}.

  Args:
    player: Player index p (required for payoff/cce_gain/ce_gain).
    payoff: Array with shape [A1, ..., AN].
    cce_gain: Array with shape [A_p', A1, ..., AN].
    ce_gain: Array with shape [A_p', A_p'', A1, ..., AN].
    cce_dual_grad: Array with shape [A_p'] (direct dual gradient).
    correlated_joint_strategy: Array with shape [A1, ..., AN].
    player_marg_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.
    strat_mask: Optional per-action mask with shape [Ap].

  Returns:
    expected_cce_gain: Array with shape [A_p'].
  """

  if payoff is not None:
    num_players = payoff.ndim
    inds = string.ascii_lowercase[:num_players]
    dev_inds = inds[:player] + "d" + inds[player + 1 :]

    if correlated_joint_strategy is not None:
      dev = jnp.einsum(
        f"{dev_inds},{inds}->d", payoff, correlated_joint_strategy
      )
      rec = jnp.einsum(f"{inds},{inds}->", payoff, correlated_joint_strategy)
      expected_cce_gain = dev - rec
    else:
      inds_ = ",".join(inds)
      dev = jnp.einsum(
        f"{dev_inds},{inds_}->d", payoff, *player_marg_per_player
      )
      rec = jnp.einsum(f"{inds},{inds_}->", payoff, *player_marg_per_player)
      expected_cce_gain = dev - rec

  elif cce_gain is not None:
    num_players = cce_gain.ndim - 1
    inds = string.ascii_lowercase[:num_players]

    if correlated_joint_strategy is not None:
      expected_cce_gain = jnp.einsum(
        f"d{inds},{inds}->d", cce_gain, correlated_joint_strategy
      )
    else:
      inds_ = ",".join(inds)
      expected_cce_gain = jnp.einsum(
        f"d{inds},{inds_}->d", cce_gain, *player_marg_per_player
      )

  elif ce_gain is not None:
    num_players = ce_gain.ndim - 2
    inds = string.ascii_lowercase[:num_players]

    if correlated_joint_strategy is not None:
      expected_cce_gain = jnp.einsum(
        f"dr{inds},{inds}->d", ce_gain, correlated_joint_strategy
      )
    else:
      inds_ = ",".join(inds)
      expected_cce_gain = jnp.einsum(
        f"dr{inds},{inds_}->d", ce_gain, *player_marg_per_player
      )

  elif cce_dual_grad is not None:
    expected_cce_gain = -cce_dual_grad

  if strat_mask is not None:
    expected_cce_gain *= strat_mask

  return expected_cce_gain


def expected_cce_gain(
  *,
  payoffs: chex.Array | None = None,
  cce_gain_per_player: list[chex.Array] | None = None,
  ce_gain_per_player: list[chex.Array] | None = None,
  cce_dual_grad_per_player: list[chex.Array] | None = None,
  correlated_joint_strategy: chex.Array | None = None,
  player_marg_per_player: list[chex.Array] | None = None,
  strat_mask_per_player: list[chex.Array] | None = None,
) -> tuple[chex.Array, ...]:
  """Computes the expected CCE gain for every player.

  Args:
    payoffs: Array with shape [N, A1, ..., AN].
    cce_gain_per_player: Sequence of arrays with shape
      [[A_p', A1, ..., AN]]_{p=1:N}.
    ce_gain_per_player: Sequence of arrays with shape
      [[A_p', A_p'', A1, ..., AN]]_{p=1:N}.
    cce_dual_grad_per_player: Sequence of arrays with shape [[A_p']]_{p=1:N}.
    correlated_joint_strategy: Array with shape [A1, ..., AN].
    player_marg_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.
    strat_mask_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.

  Returns:
    expected_cce_gain_per_player: Tuple of arrays with shape [[A_p']]_{p=1:N}.
  """

  return _map_per_player(
    functools.partial(
      expected_cce_gain_per_player,
      correlated_joint_strategy=correlated_joint_strategy,
      player_marg_per_player=player_marg_per_player,
    ),
    payoffs=payoffs,
    cce_gain=cce_gain_per_player,
    ce_gain=ce_gain_per_player,
    cce_dual_grad=cce_dual_grad_per_player,
    strat_mask=strat_mask_per_player,
  )


def cce_logit(
  cce_dual_per_player: list[chex.Array],
  *,
  payoffs: chex.Array | None = None,
  max_cce_gain: float | None = None,
) -> tuple[chex.Array, chex.Array]:
  """Returns the CCE logit for primal recovery.

  Computes:
    l(a) = - sum_p sum_{a_p'} alpha_p(a_p') * [G_p(a_p', a_{-p}) - G_p(a)]
            + sum_p max_cce_gain * sum_{a_p'} alpha_p(a_p')   [optional]

  Args:
    cce_dual_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.
    payoffs: Array with shape [N, A1, ..., AN].
    max_cce_gain: Optional scalar offset added per player.

  Returns:
    cce_logit: Array with shape [A1, ..., AN].
  """
  cce_dual_per_player = tuple(cce_dual_per_player)
  num_players = len(cce_dual_per_player)

  def _calc_logit(
    player: int, dual: chex.Array, payoff: chex.Array
  ) -> chex.Array:
    # dual: [Ap]
    # payoff: [A1, ..., AN]
    dual_sum = jnp.sum(dual)  # scalar
    logit = -payoff * dual_sum  # [A1, ..., AN]

    inds = string.ascii_lowercase[:num_players]
    pind = inds[player]
    oinds = inds[:player] + inds[player + 1 :]

    # sum_{a_p'} dual[a_p'] * G_p(a_p', a_{-p})  -> shape [A1, ..., 1, ..., AN]
    contracted = jnp.einsum(f"{inds},{pind}->{oinds}", payoff, dual)
    logit += jnp.expand_dims(contracted, axis=player)

    return logit

  cce_logit_per_player = jax.tree.map(
    _calc_logit,
    tuple(range(num_players)),
    cce_dual_per_player,
    tuple(payoffs) if payoffs is not None else (None,) * num_players,
  )
  logit = -sum(cce_logit_per_player)

  if max_cce_gain is not None:
    logit += sum(max_cce_gain * jnp.sum(dual) for dual in cce_dual_per_player)

  return logit


@functools.partial(jax.jit, static_argnames=("player",))
def ce_gain_per_player(
  player: int,
  *,
  payoff: chex.Array | None = None,
  cce_gain: chex.Array | None = None,
) -> chex.Array:
  """Computes CE deviation gain for a single player.

  The gain is defined as:
    A_p^CE(a_p', a_p'', a) = G_p(a_p', a_{-p}) - G_p(a_p'', a_{-p}),
    where:
      - a_p' is a deviation action
      - a_p'' is a recommended action
      - G_p is a per player payoff


  Args:
    player: Integer player index p.
    payoff: Optional array with shape [A1, ..., AN].
    cce_gain: Optional array with shape [A_p', A1, ..., AN].

  Returns:
    ce_gain: Array with shape [A_p', A_p'', A1, ..., AN].
  """

  if payoff is not None:
    # dev_term: broadcast payoff[dev, a_{-p}] to [A_p', 1, A1, ..., AN]
    dev_term = jnp.expand_dims(
      cce_gain_per_player(player, payoff=payoff) + payoff, 1
    )
    # rec_term: broadcast payoff[rec, a_{-p}] to [1, A_p'', A1, ..., AN]
    rec_term = jnp.expand_dims(
      cce_gain_per_player(player, payoff=payoff) + payoff, 0
    )
    return dev_term - rec_term

  if cce_gain is not None:
    # cce_gain[dev, a] = payoff[dev, a_{-p}] - payoff[a]
    # ce_gain[dev, rec, a] = payoff[dev, a_{-p}] - payoff[rec, a_{-p}]
    # = (cce_gain[dev, a] + payoff[a]) - (cce_gain[rec, a] + payoff[a])
    # = cce_gain[dev, a] - cce_gain[rec, a]
    dev_term = jnp.expand_dims(cce_gain, 1)  # [A_p', 1, A1, ..., AN]
    rec_term = jnp.expand_dims(cce_gain, 0)  # [1, A_p'', A1, ..., AN]
    return dev_term - rec_term


def ce_gain(
  *,
  payoffs: chex.Array | None = None,
  cce_gain_per_player: list[chex.Array] | None = None,
) -> tuple[chex.Array, ...]:
  """Returns CE deviation gain for every player.

  Args:
    payoffs: Optional array with shape [N, A1, ..., AN].
    cce_gain_per_player: Optional sequence of arrays with shape
      [[A_p', A1, ..., AN]]_{p=1:N}.

  Returns:
    ce_gain_per_player: Tuple of arrays with shape
      [[A_p', A_p'', A1, ..., AN]]_{p=1:N}.
  """

  return _map_per_player(
    ce_gain_per_player,
    payoffs=payoffs,
    cce_gain=cce_gain_per_player,
  )


@functools.partial(jax.jit, static_argnames=("player",))
def expected_ce_gain_per_player(
  player: int,
  *,
  payoff: chex.Array | None = None,
  ce_gain: chex.Array | None = None,
  correlated_joint_strategy: chex.Array | None = None,
  player_marg_per_player: list[chex.Array] | None = None,
) -> chex.Array:
  """Returns the expected CE gain for a single player.

  Computes for each (dev, rec) pair:
    sum_{a: a_p = rec} joint(a) * [G_p(dev, a_{-p}) - G_p(rec, a_{-p})]

  Must supply exactly one of {payoff, ce_gain}.
  Must supply exactly one of {correlated_joint_strategy, player_marg_per_player}.

  Args:
    player: Player index p.
    payoff: Array with shape [A1, ..., AN].
    ce_gain: Array with shape [A_p', A_p'', A1, ..., AN].
    correlated_joint_strategy: Array with shape [A1, ..., AN].
    player_marg_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.

  Returns:
    expected_ce_gain: Array with shape [A_p', A_p''].
  """

  if payoff is not None:
    num_players = payoff.ndim
    inds = string.ascii_lowercase[:num_players]
    dev_inds = inds[:player] + "d" + inds[player + 1 :]
    rec_inds = inds[:player] + "r" + inds[player + 1 :]

    if correlated_joint_strategy is not None:
      # dev_term[dev, rec] = sum_{a_{-p}} payoff[dev, a_{-p}] * joint[rec, a_{-p}]
      dev_term = jnp.einsum(
        f"{dev_inds},{rec_inds}->dr", payoff, correlated_joint_strategy
      )
      # rec_term[rec] = sum_{a} payoff[a] * joint[a]
      rec_term = jnp.einsum(
        f"{rec_inds},{rec_inds}->r", payoff, correlated_joint_strategy
      )
    else:
      inds_ = ",".join(inds)
      # For product marginals:
      # dev_term[dev, rec] = marg_p(rec) * sum_{a_{-p}} payoff[dev, a_{-p}] * prod_{q!=p} marg_q
      dev_marg = jnp.einsum(
        f"{dev_inds},{inds_}->D", payoff, *player_marg_per_player
      )
      rec_marg = jnp.einsum(
        f"{rec_inds},{inds_}->R", payoff, *player_marg_per_player
      )
      dev_term = dev_marg[:, None]  # [D, 1] broadcasts to [D, R]
      rec_term = rec_marg

    expected_ce_gain = dev_term - rec_term[None, :]

  elif ce_gain is not None:
    num_players = ce_gain.ndim - 2
    inds = string.ascii_lowercase[:num_players]

    if correlated_joint_strategy is not None:
      ce_gain_collapsed = jnp.mean(ce_gain, axis=player + 2)
      joint_moved = jnp.moveaxis(
        correlated_joint_strategy, player, 0
      )  # [R, A_{-p}...]
      expected_ce_gain = jnp.einsum(
        "dr...,r...->dr", ce_gain_collapsed, joint_moved
      )
    else:
      num_players = ce_gain.ndim - 2
      inds = string.ascii_lowercase[:num_players]
      inds_ = ",".join(inds)
      expected_ce_gain = jnp.einsum(
        f"dr{inds},{inds_}->dr", ce_gain, *player_marg_per_player
      )
  return expected_ce_gain


def expected_ce_gain(
  *,
  payoffs: chex.Array | None = None,
  ce_gain_per_player: list[chex.Array] | None = None,
  correlated_joint_strategy: chex.Array | None = None,
  player_marg_per_player: list[chex.Array] | None = None,
) -> tuple[chex.Array, ...]:
  """Returns the expected CE gain for every player.

  Args:
    payoffs: Array with shape [N, A1, ..., AN].
    ce_gain_per_player: Sequence of arrays with shape
      [[A_p', A_p'', A1, ..., AN]]_{p=1:N}.
    correlated_joint_strategy: Array with shape [A1, ..., AN].
    player_marg_per_player: Sequence of arrays with shape [[Ap]]_{p=1:N}.

  Returns:
    expected_ce_gain_per_player: Tuple of arrays with shape
      [[A_p', A_p'']]_{p=1:N}.
  """

  return _map_per_player(
    functools.partial(
      expected_ce_gain_per_player,
      correlated_joint_strategy=correlated_joint_strategy,
      player_marg_per_player=player_marg_per_player,
    ),
    payoffs=payoffs,
    ce_gain=ce_gain_per_player,
  )


def ce_logit(
  ce_dual_per_player: list[chex.Array],
  *,
  payoffs: chex.Array | None = None,
  max_ce_gain: float | None = None,
) -> tuple[chex.Array, chex.Array]:
  """Returns the CE logit for primal recovery.

  Computes:
    l(a) = - sum_p [ sum_{dev} alpha_p(dev, a_p) * G_p(dev, a_{-p})
                      - beta_p(a_p) * G_p(a) ]
            + sum_p max_ce_gain * sum_{dev,rec} alpha_p(dev, rec)   [optional]
  where beta_p(rec) = sum_{dev} alpha_p(dev, rec).

  Args:
    ce_dual_per_player: Sequence of arrays with shape
      [[A_p', A_p'']]_{p=1:N}.
    payoffs: Array with shape [N, A1, ..., AN].
    max_ce_gain: Optional scalar offset added per player.

  Returns:
    ce_logit: Array with shape [A1, ..., AN].
  """
  ce_dual_per_player = tuple(ce_dual_per_player)
  num_players = len(ce_dual_per_player)

  def _calc_logit(
    player: int, dual: chex.Array, payoff: chex.Array
  ) -> chex.Array:
    # dual: [A_p', A_p'']
    # payoff: [A1, ..., AN]
    beta_p = jnp.sum(dual, axis=0)  # [Ap'']

    inds = string.ascii_lowercase[:num_players]
    dev_inds = inds[:player] + "d" + inds[player + 1 :]

    # term1[a] = sum_{dev} dual[dev, a_p] * G_p(dev, a_{-p})
    dual_labels = "d" + inds[player]
    term1 = jnp.einsum(f"{dual_labels},{dev_inds}->{inds}", dual, payoff)

    # term2[a] = beta_p(a_p) * G_p(a)
    term2 = payoff * beta_p.reshape(
      [1 if i != player else -1 for i in range(payoff.ndim)]
    )

    return term1 - term2

  ce_logit_per_player = jax.tree.map(
    _calc_logit,
    tuple(range(num_players)),
    ce_dual_per_player,
    tuple(payoffs) if payoffs is not None else (None,) * num_players,
  )
  logit = -sum(ce_logit_per_player)

  if max_ce_gain is not None:
    logit += sum(max_ce_gain * jnp.sum(dual) for dual in ce_dual_per_player)

  return logit


def compute_cce_gap(
  payoffs: chex.Array,  # [N, A1, ..., AN]
  sigma: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  *,
  joint_mask: chex.Array | None = None,
  strat_mask_per_player: list[chex.Array] | None = None,
) -> chex.Array:
  """CCE gap: sum_p [max_{a_p'} (dev_payoff(a_p') - eq_pay - epsilon_p)]^+.

  Delegates to expected_cce_gain for deviation gain computation.

  Args:
      payoffs: Player payoffs with shape [N, A1, ..., AN].
      sigma: Joint strategy with shape [A1, ..., AN].
      epsilon: Per-player slack with shape [N].
      joint_mask: Valid joint action mask with shape [A1, ..., AN].
          If None and strat_mask_per_player is provided, derived automatically.
      strat_mask_per_player: Per-player valid action masks.
          Used to derive joint_mask when joint_mask is not provided.

  Returns:
      gap: Scalar CCE gap.
  """
  if joint_mask is None and strat_mask_per_player is not None:
    joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)

  # Reuse expected_cce_gain: computes E[G_p(dev, a_{-p}) - G_p(a)] for each dev
  gains = expected_cce_gain(
    payoffs=payoffs,
    correlated_joint_strategy=sigma,
    strat_mask_per_player=strat_mask_per_player,
  )  # tuple of [Ap] per player

  gap = jnp.array(0.0, dtype=payoffs.dtype)
  for p, gain_p in enumerate(gains):
    best_gain = jnp.max(gain_p)
    slack = best_gain - epsilon[p]
    gap = gap + jnp.maximum(slack, 0.0)

  return gap


def compute_ce_gap(
  payoffs: chex.Array,  # [N, A1, ..., AN]
  sigma: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  *,
  joint_mask: chex.Array | None = None,
  strat_mask_per_player: list[chex.Array] | None = None,
) -> chex.Array:
  """CE gap: sum_p [max_{dev,rec} (E[G_p(dev,a_{-p})|a_p=rec] - E[G_p(a)|a_p=rec] - epsilon_p)]^+.

  Delegates to expected_ce_gain for conditional deviation gain computation.

  Args:
      payoffs: Player payoffs with shape [N, A1, ..., AN].
      sigma: Joint strategy (correlated) with shape [A1, ..., AN].
      epsilon: Per-player slack with shape [N].
      joint_mask: Valid joint action mask with shape [A1, ..., AN].
          If None and strat_mask_per_player is provided, derived automatically.
      strat_mask_per_player: Per-player valid action masks.
          Used to derive joint_mask when joint_mask is not provided.

  Returns:
      gap: Scalar CE gap.
  """
  if joint_mask is None and strat_mask_per_player is not None:
    joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)

  # Reuse expected_ce_gain: computes E[G_p(dev, a_{-p}) - G_p(rec, a_{-p}) | rec]
  # for each (dev, rec) pair
  gains = expected_ce_gain(
    payoffs=payoffs,
    correlated_joint_strategy=sigma,
  )  # tuple of [Ap', Ap''] per player

  gap = jnp.array(0.0, dtype=payoffs.dtype)
  for p, gain_p in enumerate(gains):
    best_gain = jnp.max(gain_p)
    slack = best_gain - epsilon[p]
    gap = gap + jnp.maximum(slack, 0.0)

  return gap


def _build_cce_gains(
  payoffs: chex.Array, valid_idx: chex.Array
) -> list[chex.Array]:
  """Vectorised CCE deviation gains. Returns list of (Ap, n_valid) arrays."""
  N = payoffs.shape[0]
  action_shape = payoffs.shape[1:]
  joint_size = int(np.prod(action_shape))
  flat = payoffs.reshape(N, joint_size)
  grid = np.array(np.unravel_index(np.arange(joint_size), action_shape))
  gains = []

  for p in range(N):
    Ap = action_shape[p]
    # All deviated indices for player p: shape (Ap, joint_size)
    dev_grid = np.broadcast_to(grid, (Ap, N, joint_size)).copy()
    dev_grid[:, p, :] = np.arange(Ap)[:, None]
    dev_flat = np.ravel_multi_index(
      tuple(dev_grid.transpose(1, 0, 2)), action_shape
    )
    # Gain: G_p(dev, a_{-p}) - G_p(a), then mask to valid
    gain_p = flat[p, dev_flat] - flat[p][None, :]
    gains.append(gain_p[:, valid_idx])

  return gains


def _build_ce_gains(
  payoffs: chex.Array, valid_idx: chex.Array
) -> list[chex.Array]:
  """Vectorised CE deviation gains. Returns list of (Ap, Ap, n_valid) arrays."""
  N = payoffs.shape[0]
  action_shape = payoffs.shape[1:]
  joint_size = int(np.prod(action_shape))
  flat = payoffs.reshape(N, joint_size)
  grid = np.array(np.unravel_index(np.arange(joint_size), action_shape))
  gains = []

  for p in range(N):
    Ap = action_shape[p]
    dev_grid = np.broadcast_to(grid[None, :, :], (Ap, N, joint_size)).copy()
    dev_grid[:, p, :] = np.arange(Ap)[:, None]
    dev_flat = np.ravel_multi_index(
      tuple(dev_grid.transpose(1, 0, 2)), action_shape
    )
    dev_payoffs = flat[p, dev_flat]

    mask = (grid[p][None, :] == np.arange(Ap)[:, None]).astype(float)
    diff = dev_payoffs[:, None, :] - dev_payoffs[None, :, :]
    gain_p = diff * mask[None, :, :]
    gain_p = gain_p * (1.0 - np.eye(Ap)[:, :, None])
    gains.append(gain_p[:, :, valid_idx])

  return gains


def mwmre_solver(
  payoffs: chex.Array,
  hat_sigma: chex.Array,
  eps_hat: chex.Array,
  joint_mask: chex.Array = None,
  mu: float = 1.0,
  rho: float = 1.0,
  eps_plus: float | chex.Array = None,
  mode: str = "CE",
  verbose: bool = False,
) -> dict:
  """Solve exact ε-MWMRE CE or CCE via convex optimization.
  Primal objective:
      max_{σ ≥ 0}  μ·Σ_a σ(a)·W(a)  -  ρ·KL(σ || σ̂)
      s.t. Σ_a σ(a) = 1
      and (C)CE incentive constraints
      KL(σ || σ̂) = Σ_a σ(a)·log(σ(a)/σ̂(a))
                = Σ_a σ(a)·log σ(a) - Σ_a σ(a)·log σ̂(a)
                = -cp.entr(σ) - σ^T·log(σ̂)

  Therefore, network adopts the dual objective:
    μ · welfare^T σ
    +  ρ·cp.sum(cp.entr(σ))
    +  ρ·σ^T·log(σ̂)
    - ρ ∑_p(ε_p^+ - ε_p)ln( 1/e (ε_p^+ - ε_p) / (ε_p^+ - ε_p))
  """

  N, *A = payoffs.shape

  joint_size = int(np.prod(A))

  # --- Defaults and flattening ---
  payoffs_flat = np.asarray(payoffs).reshape((N, joint_size))
  target_flat = np.asarray(hat_sigma).flatten()

  if joint_mask is None:
    joint_mask = np.ones(A, dtype=bool)
  mask_flat = np.asarray(joint_mask).flatten()
  valid_idx = np.where(mask_flat)[0]
  n_valid = len(valid_idx)

  # Ensure eps_hat doesn't exceed eps_plus (avoids log(negative))
  eps_hat = np.minimum(eps_hat, eps_plus - utils.SMALL_NUMBER)

  # --- Variables: only over valid joints ---
  sigma_valid = cp.Variable(n_valid, nonneg=True)
  epsilon = cp.Variable(N, nonneg=True)

  constraints = [cp.sum(sigma_valid) == 1]

  for p in range(N):
    constraints.append(epsilon[p] <= eps_plus)

  # --- Equilibrium constraints ---
  if mode == "CCE":
    gains = _build_cce_gains(np.asarray(payoffs), valid_idx)
    for p, gain_p in enumerate(gains):
      # gain_p shape: (Ap, n_valid)
      constraints.append(gain_p @ sigma_valid <= epsilon[p])
  else:
    gains = _build_ce_gains(np.asarray(payoffs), valid_idx)
    for p, gain_p in enumerate(gains):
      # gain_p shape: (Ap, Ap, n_valid) -> flatten first two dims
      constraints.append(
        gain_p.reshape(-1, n_valid) @ sigma_valid <= epsilon[p]
      )

  # --- Objective ---
  welfare = payoffs_flat.sum(axis=0)[valid_idx]
  welfare_term = mu * (welfare @ sigma_valid)

  target_valid = np.clip(target_flat[valid_idx], utils.SMALL_NUMBER, 1.0)
  target_valid = target_valid / target_valid.sum()

  # KL(sigma || target) over valid entries
  kl_term = -cp.sum(cp.entr(sigma_valid)) - sigma_valid @ np.log(target_valid)

  # Epsilon penalty: rho * sum_p [ entr(diff) + diff + diff*log(target_diff) ]
  eps_term = 0
  for p in range(N):
    diff = eps_plus - epsilon[p]
    target_diff = eps_plus - eps_hat[p]
    eps_term += rho * (
      cp.entr(diff + utils.SMALL_NUMBER)  # -diff*log(diff), concave
      + diff  # linear
      + diff
      * np.log(target_diff + utils.SMALL_NUMBER)  # linear (constant coeff)
    )

  objective = welfare_term - kl_term + eps_term

  # --- Solve ---
  prob = cp.Problem(cp.Maximize(objective), constraints)
  prob.solve(solver=cp.ECOS, verbose=verbose, abstol=1e-6, reltol=1e-6)

  if sigma_valid.value is None:
    return {"sigma": None, "status": prob.status, "error": "Solver failed"}

  # --- Reconstruct full sigma ---
  sigma_full = np.zeros(joint_size)
  sigma_full[valid_idx] = sigma_valid.value
  sigma_star = sigma_full.reshape(A)

  # --- Metrics ---
  actual_welfare = float(np.sum(sigma_star * payoffs.sum(axis=0)))
  actual_kl = float(
    np.sum(
      sigma_star
      * (
        np.log(sigma_star + utils.SMALL_NUMBER)
        - np.log(hat_sigma + utils.SMALL_NUMBER)
      )
    )
  )

  solver_gap = 0.5 * jnp.abs(sigma_star - hat_sigma).sum()
  welfare_gap = jnp.sum((sigma_star - hat_sigma) * payoffs.sum(axis=0))
  eps_gap = jnp.max(jnp.abs(eps_hat - epsilon.value))

  return {
    "solver_gap": solver_gap,
    "welfare_gap": welfare_gap,
    "eps_gap": eps_gap,
    "sigma": sigma_star,
    "status": prob.status,
    "objective_value": float(prob.value),
    "welfare": actual_welfare,
    "kl_to_hat": actual_kl,
  }
