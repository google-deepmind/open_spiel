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

    # Best unilateral deviation payoff
    # marginal_opp = sum of σ over player p's action
    marginal_opp = jnp.sum(sigma, axis=p, keepdims=True)
    axes_to_reduce = tuple(i for i in range(N) if i != p)
    dev_payoffs = jnp.sum(marginal_opp * payoffs[p], axis=axes_to_reduce)
    
    best_dev = jnp.max(dev_payoffs)

    # Positive slack for this player
    slack = best_dev - eq_pay - epsilon[p]
    # No deviation can improve my payoff by more than ϵ_p
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

    G_p = payoffs[p]
    # Marginal probability of each recommendation P(a_p = rec)
    marginal_rec = jnp.sum(sigma, axis=tuple(i for i in range(sigma.ndim) if i != p))
    
    # Move player axis to front: [Ap, A_{-p}]
    sigma_moved = jnp.moveaxis(sigma, p, 0)
    G_moved = jnp.moveaxis(G_p, p, 0)
    
    # Conditional expected payoff: E[G_p | a_p = rec]
    # = sum_{a_{-p}} sigma(rec, a_{-p}) * G_p(rec, a_{-p}) / marginal_rec[rec]
    eq_pay_numerator = jnp.sum(sigma_moved * G_moved, axis=tuple(range(1, sigma_moved.ndim)))
    eq_pay_cond = eq_pay_numerator / (marginal_rec + 1e-12)
    
    # Deviation payoff matrix: [Ap_dev, Ap_rec]
    # dev_payoffs[dev, rec] = E[G_p(dev, a_{-p}) | a_p = rec]
    dev_numerator = jnp.sum(
        sigma_moved[None, ...] * G_moved[:, None, :],
        axis=-1
    )
    dev_payoffs_cond = dev_numerator / (marginal_rec[None, :] + 1e-12)
    
    # Best deviation for each recommendation
    best_dev_cond = jnp.max(dev_payoffs_cond, axis=0)
    
    # Gain for each rec: best_dev - current_payoff - epsilon
    gain_per_rec = best_dev_cond - eq_pay_cond - epsilon[p]
    
    # Weight by marginal probability and add to total
    player_gap = jnp.sum(marginal_rec * jnp.maximum(gain_per_rec, 0.0))
    total_gap = total_gap + player_gap

    total_gap += player_gap

  return total_gap


def mwme_lp_solver_gap(
  payoffs: chex.Array,
  hat_sigma: chex.Array,
  mu: float = 1.0,
  rho: float = 1.0,
  eps: float = 1e-8,
  verbose: bool = False,
  enforce_equilibrium: bool = True,
) -> dict:
  """MWME: Maximize μ·Welfare − ρ·KL(σ || ˆσ)  [hybrid etalon version]"""

  N = payoffs.shape[0]
  action_sizes = payoffs.shape[1:]
  joint_size = int(np.prod(action_sizes))

  sigma = cp.Variable(joint_size, nonneg=True)
  flat_payoffs = payoffs.reshape((N, joint_size))
  welfare = flat_payoffs.sum(axis=0)
  flat_hat = np.clip(hat_sigma.flatten(), 1e-12, None)

  # Entropy-based KL (stable formulation)
  # kl_term = cp.sum(cp.multiply(sigma, cp.log(sigma + 1e-12) - np.log(flat_hat)))
  # KL(sigma || hat) = sum(sigma * log(sigma/hat))
  #                 = sum(sigma * log(sigma)) - sum (sigma * log(hat))
  #                 = -entr(sigma) - sigma @ log(hat)
  kl_term = -cp.sum(cp.entr(sigma)) - sigma @ np.log(flat_hat)

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
