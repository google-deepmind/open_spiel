import functools
import chex
import string
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.nes import utils


def _map_per_player(
  fn, *, payoffs: chex.Array = None, **per_player_kwargs: chex.Array
) -> tuple:
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
  correlated_joint_strategy: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  *,
  joint_mask: chex.Array | None = None,
  strat_mask_per_player: list[chex.Array] | None = None,
) -> chex.Array:
  """CCE gap: sum_p [max_{a_p'} (dev_payoff(a_p') - eq_pay - epsilon_p)]^+.

  Delegates to expected_cce_gain for deviation gain computation.

  Args:
      payoffs: Player payoffs with shape [N, A1, ..., AN].
      correlated_joint_strategy: Joint strategy with shape [A1, ..., AN].
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
    correlated_joint_strategy=correlated_joint_strategy,
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
  correlated_joint_strategy: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  *,
  joint_mask: chex.Array | None = None,
  strat_mask_per_player: list[chex.Array] | None = None,
) -> chex.Array:
  """CE gap:
    sum_p [max_{dev,rec}(E[G_p(dev,a_{-p})|a_p=rec]-E[G_p(a)|a_p=rec] - epsilon_p)]^+.

  Delegates to expected_ce_gain for conditional deviation gain computation.

  Args:
      payoffs: Player payoffs with shape [N, A1, ..., AN].
      correlated_joint_strategy: Joint strategy (correlated) with shape [A1, ..., AN].
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
    correlated_joint_strategy=correlated_joint_strategy,
  )  # tuple of [Ap', Ap''] per player

  gap = jnp.array(0.0, dtype=payoffs.dtype)

  for p, gain_p in enumerate(gains):
    # Mask invalid actions so they don't affect the maximum.
    if strat_mask_per_player is not None:
      valid = strat_mask_per_player[p]
      valid_2d = jnp.logical_and(valid[:, None], valid[None, :])
      min_val = jnp.finfo(gain_p.dtype).min
      gain_p = jnp.where(valid_2d, gain_p, min_val)

    best_gain = jnp.max(gain_p)
    slack = best_gain - epsilon[p]
    gap = gap + jnp.maximum(slack, 0.0)

  return gap


def _build_cce_gains(
  payoffs: chex.Array, 
  valid_idx: chex.Array,
  strat_mask_per_player:  list[chex.Array]
) -> list[chex.Array]:
  """Vectorised CCE deviation gains using cce_gain.
    Returns list of (n_valid_p, n_valid) arrays -- one row per *valid*
    deviation action for that player.
  """
  action_shape = payoffs.shape[1:]
  joint_size = utils.compute_joint_action_size(action_shape)
  # Use the module's cce_gain utility instead of manual indexing
  cce_gains = cce_gain(payoffs=payoffs)  # tuple of [Ap', A1, ..., AN]
  gains = []
  for p, gain_p in enumerate(cce_gains):
    # Flatten joint action dimensions -> [Ap', joint_size]
    gain_flat = np.asarray(gain_p).reshape(gain_p.shape[0], joint_size)
    # Drop padded/invalid deviation actions (rows) to avoid infeasibility
    gain_flat = gain_flat[strat_mask_per_player[p]]
    # Mask to valid joints only
    gains.append(gain_flat[:, valid_idx])
  return gains


def _build_ce_gains(
  payoffs: chex.Array, 
  valid_idx: chex.Array, 
  strat_mask_per_player:  list[chex.Array]
) -> list[chex.Array]:
  """Vectorised CE deviation gains using ce_gain.
    Returns list of (n_valid_p, n_valid_p, n_valid) arrays -- both the dev and
    rec axes are restricted to valid actions, using the *same* per-player
    mask, so the diagonal (dev == rec, same real action) still lines up after
    filtering.
  """
  action_shape = payoffs.shape[1:]
  joint_size = utils.compute_joint_action_size(action_shape)

  ce_gains = ce_gain(payoffs=payoffs)  # tuple of [Ap', Ap'', A1, ..., AN]

  gains = []
  for p, gain_p in enumerate(ce_gains):
    # Flatten joint action dimensions -> [Ap', Ap'', joint_size]
    gain_flat = np.asarray(gain_p).reshape(
      gain_p.shape[0], gain_p.shape[1], joint_size
    )
    mask_p = strat_mask_per_player[p]
    # Drop padded/invalid dev actions (rows) and rec actions (cols); same
    # rationale as _build_cce_gains, plus a padded "recommendation"
    gain_flat = gain_flat[mask_p][:, mask_p]
    # Mask to valid joints only
    gains.append(gain_flat[:, :, valid_idx])

  return gains


def mwmre_solver(
  payoffs: chex.Array,
  welfare: chex.Array,
  strat_pred: chex.Array,
  epsilon_target: chex.Array,
  strat_mask_per_player: list[chex.Array],
  welfare_coeff: float = 1.0,
  entropy_coeff: float = 1.0,
  epsilon_max: float | chex.Array = None,
  mode: str = "CE",
  verbose: bool = False,
) -> dict:
  """Solve exact ε-MWMRE CE or CCE via convex optimisation.

  Primal objective:
    max_{σ ≥ 0}  μ·Σ_a σ(a)·W(a) - KL(σ || σ̂)
                 - ρ·Σ_p(ε_p⁺-ε_p)·ln(1/e · (ε_p⁺-ε_p)/(ε_p⁺-ε̂_p))
    s.t. Σ_a σ(a) = 1
    and (C)CE incentive constraints
    KL(σ || σ̂) = Σ_a σ(a)·log(σ(a)/σ̂(a))
               = Σ_a σ(a)·log σ(a) - Σ_a σ(a)·log σ̂(a)
                = -cp.entr(σ) - σ^T·log(σ̂)

  Args:
    payoffs: Player payoffs with shape [N, A1, ..., AN].
    welfare: Player welfare with shape [N, A1, ..., AN].
    strat_pred: Joint strategy with shape [A1, ..., AN].
    epsilon_target: Per-player slack with shape [N].
    strat_mask_per_player: Per-player valid action masks.
    welfare_coeff: Objective scalar coefficient
    ent_coeff: Objective scalar coefficient
    epsilon_max: Objective scalar coefficient
    mode: str: CCE or CE
    verbose: Where to verbose the `cvxpy` solver

  Returns:
    report: A dict with scalar solver metrics and gaps.
  """

  N, *A = payoffs.shape

  joint_size = utils.compute_joint_action_size(A)

  payoffs_flat = np.asarray(payoffs).reshape((N, joint_size))
  target_flat = np.asarray(strat_pred).flatten()

  joint_mask = utils.make_joint_mask_from_strat_masks(strat_mask_per_player)
  mask_flat = np.asarray(joint_mask).flatten()
  valid_idx = np.where(mask_flat)[0]
  n_valid = len(valid_idx)

  valid_actions_per_player = [np.where(m)[0] for m in strat_mask_per_player]

  # Ensure eps_hat doesn't exceed eps_plus (avoids log(negative))
  epsilon_target = np.minimum(epsilon_target, epsilon_max - utils.SMALL_NUMBER)

  # Variables: only over valid joints
  strategy = cp.Variable(n_valid, nonneg=True)
  # epsilon_p ranges over [-epsilon_max, epsilon_max] (Table 3)
  # negative epsilon represents a strict/robust
  # equilibrium margin and is required to reproduce e.g. the "MS" and
  # eps-ME/eps-MRE parameterisations.
  epsilon = cp.Variable(N)

  constraints = [cp.sum(strategy) == 1]

  for p in range(N):
    constraints.append(epsilon[p] <= epsilon_max)
    constraints.append(epsilon[p] >= -epsilon_max)

  # --- Equilibrium constraints ---
  if mode == "CCE":
    gains = _build_cce_gains(
      np.asarray(payoffs), valid_idx, strat_mask_per_player
    )
    for p, gain_p in enumerate(gains):
      # gain_p shape: (n_valid_p, n_valid)
      constraints.append(gain_p @ strategy <= epsilon[p])
  else:
    gains = _build_ce_gains(
      np.asarray(payoffs), valid_idx, strat_mask_per_player
    )
    for p, gain_p in enumerate(gains):
      for dev in range(gain_p.shape[0]):
        for rec in range(gain_p.shape[1]):
          if dev == rec:
            # same real action (dev/rec share the same filtered order)
            continue 

          coeffs = gain_p[dev, rec, :].copy()

          # back to the real action id before comparing against a[p] below.
          rec_action = valid_actions_per_player[p][rec]

          for i, flat_idx in enumerate(valid_idx):
            a = np.unravel_index(flat_idx, A)
            if a[p] != rec_action:
              coeffs[i] = 0.0

          constraints.append(coeffs @ strategy <= epsilon[p])

    # for p, gain_p in enumerate(gains):
    #   # gain_p shape: (Ap, Ap, n_valid) -> flatten first two dims
    #   constraints.append(gain_p.reshape(-1, n_valid) @ strategy <= epsilon[p])

  if welfare is not None:
    welfare_flat = np.asarray(welfare).reshape(joint_size)[valid_idx]
  else:
    welfare_flat = payoffs_flat.sum(axis=0)[valid_idx]
  welfare_term = welfare_coeff * (welfare_flat @ strategy)

  target_valid = np.clip(target_flat[valid_idx], utils.SMALL_NUMBER, 1.0)
  target_valid = target_valid / target_valid.sum()

  # KL(sigma || target) = sum(sigma * log(sigma/target)) = -entr(sigma) - sigma^T log(target)
  kl_term = -cp.sum(cp.entr(strategy)) - strategy @ np.log(target_valid)

  # Epsilon penalty
  eps_term = 0
  for p in range(N):
    diff = epsilon_max - epsilon[p]
    target_diff = epsilon_max - epsilon_target[p]

    # Clamp target_diff to avoid log(0) — it's a constant parameter
    target_diff_safe = float(np.maximum(target_diff, utils.SMALL_NUMBER))

    # cp.entr(diff) = -diff*log(diff) is concave in diff, hence concave in epsilon
    # Add SMALL_NUMBER to avoid log(0) gradient singularity when epsilon -> epsilon_max.
    diff_safe = diff + utils.SMALL_NUMBER

    eps_term += entropy_coeff * (
      cp.entr(diff_safe)  # -diff*log(diff), concave
      + diff  # linear in epsilon
      + diff * np.log(target_diff_safe)  # linear in epsilon
    )

  objective = welfare_term - kl_term + eps_term

  prob = cp.Problem(cp.Maximize(objective), constraints)
  prob.solve(solver=cp.ECOS, verbose=verbose, abstol=1e-3, reltol=1e-3)

  if strategy.value is None:
    return {"stat_pred": None, "status": prob.status, "error": "Solver failed"}

  # Reconstruct full sigma
  strategy_target = np.zeros(joint_size)
  strategy_target[valid_idx] = strategy.value
  strategy_target = strategy_target.reshape(A)

  # Metrics
  welfare_full = np.asarray(welfare) if welfare is not None else payoffs.sum(axis=0)
  actual_welfare = float(np.sum(strategy_target * welfare_full))
  actual_kl = float(
    np.sum(
      strategy_target
      * (
        np.log(strategy_target + utils.SMALL_NUMBER)
        - np.log(strat_pred + utils.SMALL_NUMBER)
      )
    )
  )

  solver_gap = 0.5 * jnp.abs(strategy_target - strat_pred).sum()
  welfare_gap = jnp.sum((strategy_target - strat_pred) * welfare_full)
  eps_gap = jnp.max(jnp.abs(epsilon_target - epsilon.value))

  return {
    "solver_gap": solver_gap,
    "welfare_gap": welfare_gap,
    "eps_gap": eps_gap,
    "strategy_target": strategy_target,
    "status": prob.status,
    "objective_value": float(prob.value),
    "welfare": actual_welfare,
    "kl_to_hat": actual_kl,
  }
