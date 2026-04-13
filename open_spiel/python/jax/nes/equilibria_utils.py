import functools

import chex
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnames=("player",))
def player_contribution_cce(
  alpha_p: chex.Array,  # [A_p]
  G_p: chex.Array,  # [A1, ..., AN]
  player: int,  # p
) -> tuple[chex.Array, float]:
  """
  CCE: Σ_{dev} α_p(dev) * [G_p(dev, a_{-p}) - G_p(a_p, a_{-p})]

  Returns:
      contribution: [A1, ..., AN] - adds to logits l(a)
      sum_alpha: scalar - for epsilon computation
  """
  # Contract alpha[dev] with G_p[dev, ...] -> [A_{-p}]
  weighted_dev = jnp.tensordot(alpha_p, G_p, axes=(0, player))

  # Insert singleton at player axis for broadcasting: [A1, ..., 1, ..., AN]
  weighted_dev = jnp.expand_dims(weighted_dev, axis=player)

  # Sum of alphas for current action weighting
  sum_alpha = jnp.sum(alpha_p)

  # Deviation contribution: weighted_dev - G_p * sum_alpha
  contribution = weighted_dev - G_p * sum_alpha

  return contribution, sum_alpha


@functools.partial(jax.jit, static_argnames=("player",))
def player_contribution_ce(
  alpha_p: chex.Array,  # [A_p, A_p] - axis 0: dev, axis 1: rec
  G_p: chex.Array,  # [A1, ..., AN]
  player: int,  # p
) -> tuple[chex.Array, float]:
  """
  CE: Σ_{dev} α_p(dev, a_p) * [G_p(dev, a_{-p}) - G_p(a_p, a_{-p})]

  Note: alpha_p[dev, rec] where rec is the recommended action (a_p)
  """
  # Contract dev axis (0) with player axis -> [A_rec, A_{-p}]
  weighted_dev = jnp.tensordot(alpha_p, G_p, axes=(0, player))

  # Move rec axis to player's position: [A1, ..., A_rec, ..., AN]
  weighted_dev = jnp.moveaxis(weighted_dev, 0, player)

  # beta_p(rec) = Σ_{dev} alpha_p(dev, rec)
  beta_p = jnp.sum(alpha_p, axis=0)  # [A_p]

  # Broadcast beta to match G_p shape
  contribution = weighted_dev - G_p * beta_p.reshape(
    [1 if i != player else -1 for i in range(G_p.ndim)]
  )

  return contribution, jnp.sum(alpha_p)


def compute_cce_gap(
  payoffs: chex.Array,  # [N, A1, ..., AN]
  sigma: chex.Array,  # [A1, ..., AN]  (your recovered joint strategy)
  epsilon: chex.Array,  # [N]            (ˆε_p vector)
) -> chex.Array:
  N = payoffs.shape[0]

  gap = jnp.array(0.0, dtype=payoffs.dtype)

  for p in range(N):
    # 1. Equilibrium payoff for player p under σ
    eq_pay = jnp.sum(sigma * payoffs[p])

    # 2. Best unilateral deviation payoff
    # marginal_opp = sum of σ over player p's action
    marginal_opp = jnp.sum(sigma, axis=p, keepdims=True)
    axes_to_reduce = tuple(i for i in range(N) if i != p)
    dev_payoffs = jnp.sum(marginal_opp * payoffs[p], axis=axes_to_reduce)

    best_dev = jnp.max(dev_payoffs)

    # 3. Positive slack for this player
    slack = best_dev - eq_pay - epsilon[p]
    gap = gap + jnp.maximum(slack, 0.0)

  return gap


def compute_ce_gap(
  payoffs: chex.Array,  # [N, A1, ..., AN]
  sigma: chex.Array,  # [A1, ..., AN]  recovered joint strategy
  epsilon: chex.Array,  # [N]           ˆε_p vector
) -> chex.Array:
  """CE Gap"""
  N = payoffs.shape[0]
  total_gap = jnp.array(0.0, dtype=payoffs.dtype)

  for p in range(N):
    Ap = payoffs.shape[p + 1]  # true action size for player p

    # Marginal probability of recommended action curr for player p
    marginal_curr = jnp.sum(
      sigma, axis=p, keepdims=True
    )  # [1, ..., 1, Ap, 1, ...]

    # For every possible curr and dev, compute conditional gain
    player_gap = jnp.array(0.0, dtype=payoffs.dtype)

    for curr in range(Ap):
      # Indicator for actions where a_p == curr
      indicator_curr = jnp.zeros_like(sigma)
      # Set 1 where player p's action == curr
      idx = list(range(N))
      idx[p] = curr
      indicator_curr = indicator_curr.at[tuple(idx)].set(
        1.0
      )  # this is slow, better way below

      # More efficient: use broadcasting
      # Conditional expected payoff for curr
      eq_pay_curr = jnp.sum(
        payoffs[p] * (sigma * (jnp.arange(Ap) == curr)[None, ...])
      )

      # Best deviation payoff when recommended curr
      best_dev_pay = -jnp.inf
      for dev in range(Ap):
        if curr == dev:
          continue
        # Deviation payoff when a_p = curr but deviates to dev
        dev_pay = (
          payoffs[p]
          .at[curr]
          .set(
            payoffs[p][
              tuple([dev if i == p else slice(None) for i in range(N)])
            ]
          )
        )
        cond_dev_pay = jnp.sum(dev_pay * sigma) / (marginal_curr + 1e-12)
        best_dev_pay = jnp.maximum(best_dev_pay, cond_dev_pay)

      slack = best_dev_pay - eq_pay_curr - epsilon[p]
      player_gap += jnp.maximum(slack, 0.0) * marginal_curr.squeeze()

    total_gap += player_gap

  return total_gap


def _remove_dominated_strategies(payoffs: chex.Array) -> tuple:
  N = payoffs.shape[0]
  action_sizes = list(payoffs.shape[1:])
  kept = [np.arange(s) for s in action_sizes]

  for p in range(N):
    Ap = action_sizes[p]
    dominated = []
    for a in range(Ap):
      payoff_a = payoffs[p][
        tuple(slice(None) if i != p else a for i in range(N))
      ]
      is_dominated = False
      for b in range(Ap):
        if a == b:
          continue
        payoff_b = payoffs[p][
          tuple(slice(None) if i != p else b for i in range(N))
        ]
        # Strict dominance check (with tolerance)
        if np.all(payoff_b >= payoff_a + 1e-8) and np.any(
          payoff_b > payoff_a + 1e-8
        ):
          is_dominated = True
          break
        if is_dominated:
          dominated.append(a)
    kept[p] = np.delete(kept[p], dominated)
    action_sizes[p] = len(kept[p])

  # Actually reduce the payoffs array
  reduced_payoffs = payoffs[np.ix_(range(N), *kept)]
  return reduced_payoffs, kept


def solve_cce_lp(
  payoffs: chex.Array,
  eps: float = 1e-8,
  verbose: bool = False,
) -> dict:
  """Exact CCE as LP (hybrid: simple + dominated elimination + ECOS)."""
  # payoffs, _ = _remove_dominated_strategies(payoffs)
  N = payoffs.shape[0]
  action_sizes = payoffs.shape[1:]
  joint_size = int(np.prod(action_sizes))

  sigma = cp.Variable(joint_size, nonneg=True)
  flat_payoffs = payoffs.reshape((N, joint_size))

  constraints = [cp.sum(sigma) == 1]

  for p in range(N):
    Ap = action_sizes[p]
    for dev in range(Ap):
      dev_payoffs = np.zeros(joint_size)
      idx = 0
      for a in np.ndindex(*action_sizes):
        dev_a = list(a)
        dev_a[p] = dev
        dev_idx = np.ravel_multi_index(tuple(dev_a), action_sizes)
        dev_payoffs[idx] = flat_payoffs[p, dev_idx]
        idx += 1
      gain = dev_payoffs - flat_payoffs[p]
      constraints.append(gain @ sigma <= eps)

  prob = cp.Problem(cp.Minimize(0), constraints)
  prob.solve(
    solver=cp.ECOS, verbose=verbose, abstol=1e-9, reltol=1e-9, max_iters=10000
  )

  if sigma.value is None:
    return {"sigma": None, "status": prob.status, "error": "Solver failed"}

  return {
    "sigma": sigma.value.reshape(action_sizes),
    "status": prob.status,
    "value": prob.value,
  }


def solve_ce_lp(
  payoffs: chex.Array,
  eps: float = 1e-8,
  verbose: bool = False,
) -> dict:
  """Exact CE as LP (pairwise per-player constraints)."""
  # payoffs, _ = _remove_dominated_strategies(payoffs)
  N = payoffs.shape[0]
  action_sizes = payoffs.shape[1:]
  joint_size = int(np.prod(action_sizes))

  sigma = cp.Variable(joint_size, nonneg=True)
  flat_payoffs = payoffs.reshape((N, joint_size))

  constraints = [cp.sum(sigma) == 1]

  for p in range(N):
    Ap = action_sizes[p]
    for curr in range(Ap):
      # Indicator for actions where player p plays "curr"
      indicator_curr = np.zeros(joint_size)
      idx = 0
      for a in np.ndindex(*action_sizes):
        if a[p] == curr:
          indicator_curr[idx] = 1.0
        idx += 1

      for dev in range(Ap):
        if curr == dev:
          continue
        # Deviation payoff when conditioning on curr
        dev_payoffs = np.zeros(joint_size)
        idx = 0
        for a in np.ndindex(*action_sizes):
          if a[p] == curr:
            dev_a = list(a)
            dev_a[p] = dev
            dev_idx = np.ravel_multi_index(tuple(dev_a), action_sizes)
            dev_payoffs[idx] = flat_payoffs[p, dev_idx]
          idx += 1

        # Linearized CE constraint:
        # sum_{a: a_p=curr} sigma(a) * (G(dev) - G(curr)) <= eps * sum_{a: a_p=curr} sigma(a)
        gain = dev_payoffs - flat_payoffs[p] * indicator_curr
        constraints.append(gain @ sigma <= eps * (indicator_curr @ sigma))

  prob = cp.Problem(cp.Minimize(0), constraints)
  prob.solve(
    solver=cp.ECOS, verbose=verbose, abstol=1e-9, reltol=1e-9, max_iters=10000
  )

  if sigma.value is None:
    return {"sigma": None, "status": prob.status, "error": "Solver failed"}

  return {
    "sigma": sigma.value.reshape(action_sizes),
    "status": prob.status,
    "value": prob.value,
  }


def solve_mwme_lp(
  payoffs: chex.Array,
  hat_sigma: chex.Array,
  mu: float = 1.0,
  rho: float = 1.0,
  eps: float = 1e-8,
  verbose: bool = False,
  enforce_equilibrium: bool = True,
) -> dict:
  """MWME: Maximize μ·Welfare − ρ·KL(σ || ˆσ)  [hybrid etalon version]"""
  # payoffs, _ = _remove_dominated_strategies(payoffs)

  N = payoffs.shape[0]
  action_sizes = payoffs.shape[1:]
  joint_size = int(np.prod(action_sizes))

  sigma = cp.Variable(joint_size, nonneg=True)
  flat_payoffs = payoffs.reshape((N, joint_size))
  welfare = flat_payoffs.sum(axis=0)
  flat_hat = np.clip(hat_sigma.flatten(), 1e-12, None)

  # Entropy-based KL (stable formulation)
  kl_term = cp.sum(cp.multiply(sigma, cp.log(sigma + 1e-12) - np.log(flat_hat)))

  objective = mu * (welfare @ sigma) - rho * kl_term

  constraints = [cp.sum(sigma) == 1]

  if enforce_equilibrium:
    for p in range(N):
      Ap = action_sizes[p]
      for dev in range(Ap):
        dev_payoffs = np.zeros(joint_size)
        idx = 0
        for a in np.ndindex(*action_sizes):
          dev_a = list(a)
          dev_a[p] = dev
          dev_idx = np.ravel_multi_index(tuple(dev_a), action_sizes)
          dev_payoffs[idx] = flat_payoffs[p, dev_idx]
          idx += 1
        gain = dev_payoffs - flat_payoffs[p]
        constraints.append(gain @ sigma <= eps)

  prob = cp.Problem(cp.Maximize(objective), constraints)
  prob.solve(
    solver=cp.ECOS, verbose=verbose, abstol=1e-9, reltol=1e-9, max_iters=10000
  )

  if sigma.value is None:
    return {"sigma": None, "status": prob.status, "error": "Solver failed"}

  sigma_star = sigma.value.reshape(action_sizes)
  actual_welfare = np.sum(sigma_star * payoffs.sum(axis=0))
  actual_kl = np.sum(
    sigma_star * (np.log(sigma_star + 1e-12) - np.log(hat_sigma + 1e-12))
  )

  return {
    "sigma": sigma_star,
    "status": prob.status,
    "objective_value": prob.value,
    "welfare": float(actual_welfare),
    "kl_to_hat": float(actual_kl),
  }
