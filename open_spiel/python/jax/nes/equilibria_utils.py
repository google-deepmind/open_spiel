import functools

import chex
import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.nes import utils


@functools.partial(jax.jit, static_argnames=("player",))
def player_contribution_cce(
  alpha_p: chex.Array,  # [A_p]
  G_p: chex.Array,  # [A1, ..., AN],
  player: int,  # p
) -> tuple[chex.Array, float]:
  """
  CCE: Σ_{dev} α_p(dev) * [G_p(dev, a_{-p}) - G_p(a_p, a_{-p})]

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
  G_p: chex.Array,  # [A1, ..., AN],
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
  sigma: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  joint_mask: chex.Array = None,  # [A1, ..., AN]
) -> chex.Array:
  """
  CCE Gap: sum_p [max_{a_p'} (dev_payoff(a_p') - eq_pay - epsilon_p)]^+
  Where dev_payoff(a_p') = sum_{a_{-p}} sigma_{-p}(a_{-p}) * G_p(a_p', a_{-p})
  """
  N = payoffs.shape[0]
  gap = jnp.array(0.0, dtype=payoffs.dtype)

  for p in range(N):
    # Equilibrium payoff for player p
    eq_pay = jnp.sum(sigma * payoffs[p], where=joint_mask)
    # Marginal over opponents: sum over player p's action dimension
    marginal_opp = jnp.sum(sigma, axis=p, keepdims=False, where=joint_mask)

    # Move player p axis to front for payoffs
    payoffs_moved = jnp.moveaxis(payoffs[p], p, 0)  # [Ap, A_{-p}...]

    # Best unilateral deviation payoff
    # marginal_opp = sum of σ over player p's action

    # Flatten a_{-p} dimensions
    payoffs_flat = payoffs_moved.reshape(
      payoffs_moved.shape[0], -1
    )  # [Ap, prod(A_{-p})]
    marginal_flat = marginal_opp.reshape(-1)  # [prod(A_{-p})]

    if joint_mask is not None:
      mask_moved = jnp.moveaxis(joint_mask, p, 0)
      # Zero out invalid entries
      payoffs_flat = jnp.where(
        mask_moved.reshape(payoffs_moved.shape[0], -1), payoffs_flat, 0.0
      )

    # Deviation payoffs: dot product for each a_p'
    dev_payoffs = payoffs_flat @ marginal_flat  # [Ap]

    # Best deviation
    best_dev = jnp.max(dev_payoffs)

    # Slack ϵ_p
    slack = best_dev - eq_pay - epsilon[p]
    # No deviation can improve my payoff by more than ϵ_p
    gap = gap + jnp.maximum(slack, 0.0)

  return gap


def compute_ce_gap(
  payoffs: chex.Array,  # [N, A1, ..., AN]
  sigma: chex.Array,  # [A1, ..., AN]
  epsilon: chex.Array,  # [N]
  joint_mask: chex.Array = None,  # [A1, ..., AN]
) -> chex.Array:
  """
  CE Gap:

  For each player p and each deviation pair (a_p', a_p''):
    A_p = sum_{a: a_p = a_p''} sigma(a) * [G_p(a_p', a_{-p}) - G_p(a_p'', a_{-p})]

  CE Gap = sum_p [max_{a_p', a_p''} A_p(a_p', a_p'') - epsilon_p]^+
  """
  N = payoffs.shape[0]
  gap = jnp.array(0.0, dtype=payoffs.dtype)

  for p in range(N):
    
    marginal_rec = jnp.sum(
      sigma,
      axis=tuple(i for i in range(sigma.ndim) if i != p),
      where=joint_mask,
    )  
    
    # Move player p axis to front: [Ap, A_{-p}...]
    payoffs_moved = jnp.moveaxis(payoffs[p], p, 0)
    sigma_moved = jnp.moveaxis(sigma, p, 0)
    
    mask_moved = None
    if joint_mask is not None:
      mask_moved = jnp.moveaxis(joint_mask, p, 0)

    # Flatten a_{-p} dimensions, [Ap, prod(A_{-p})]
    A_p = payoffs_moved.shape[0]

    # Conditional expected payoff: E[U_p(a) | a_p = rec]
    eq_num = jnp.sum(
        sigma_moved * payoffs_moved,
        axis=tuple(range(1, sigma_moved.ndim)),
        where=mask_moved,
    )  # [Ap]
    eq_cond = eq_num / (marginal_rec + utils.SMALL_NUMBER)  # [Ap]

    sigma_flat = sigma_moved.reshape(A_p, -1)
    payoffs_flat = payoffs_moved.reshape(A_p, -1)

    mask_flat = None
    if mask_moved is not None:
      mask_flat = mask_moved.reshape(A_p, -1)
      sigma_flat = jnp.where(mask_flat, sigma_flat, 0.0)
      payoffs_flat = jnp.where(mask_flat, payoffs_flat, 0.0)

    # Conditional deviation payoff: E[U_p(dev, a_{-p}) | a_p = rec]
    # = sum_{a_{-p}} sigma(rec, a_{-p}) * U_p(dev, a_{-p}) / marginal_rec[rec]
    dev_num = jnp.sum(
        sigma_flat[None, ...] * payoffs_flat[:, None, ...],
        axis=-1,  # Sum over ALL a_{-p} dims
        where=mask_flat[None, ...] if mask_flat is not None else None,
    )

    dev_cond = dev_num / (marginal_rec[None, :] + utils.SMALL_NUMBER)  # [Ap_dev, Ap_rec]
    
    # Best conditional deviation gain for each recommendation
    gain_per_rec = jnp.max(dev_cond, axis=0) - eq_cond - epsilon[p]  # [Ap]
    
    # Weight by marginal probability and sum
    gap += jnp.sum(marginal_rec * jnp.maximum(gain_per_rec, 0.0))

    # Another version, doesn't seem to work
    # A[dev, rec] = sum_{a_{-p}} sigma(rec, a_{-p}) * G_moved[dev, a_{-p}]
    #               - sum_{a_{-p}} sigma(rec, a_{-p}) * G_moved[rec, a_{-p}]


    # term1 = jnp.sum(sigma_flat[None, :, :] * payoffs_flat[:, None, :], axis=-1)
    # term2 = jnp.sum(sigma_flat * payoffs_flat, axis=-1)
    # dev_payoffs = term1 - term2[None, :]  # [Ap_dev, Ap_rec]

    # best_dev = jnp.max(dev_payoffs)

    # # Slack
    # slack = best_dev - epsilon[p]
    # gap = gap + jnp.maximum(slack, 0.0)

  return gap


def _build_cce_gains(payoffs: chex.Array, valid_idx: chex.Array) -> list[chex.Array]:
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
    dev_flat = np.ravel_multi_index(tuple(dev_grid.transpose(1, 0, 2)), action_shape)
    # Gain: G_p(dev, a_{-p}) - G_p(a), then mask to valid
    gain_p = flat[p, dev_flat] - flat[p][None, :]
    gains.append(gain_p[:, valid_idx])

  return gains



def _build_ce_gains(payoffs: chex.Array, valid_idx: chex.Array) -> list[chex.Array]:
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
    dev_flat = np.ravel_multi_index(tuple(dev_grid.transpose(1, 0, 2)), action_shape)
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
        constraints.append(gain_p.reshape(-1, n_valid) @ sigma_valid <= epsilon[p])

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
          cp.entr(diff + utils.SMALL_NUMBER)                       # -diff*log(diff), concave
          + diff                                                   # linear
          + diff * np.log(target_diff + utils.SMALL_NUMBER)        # linear (constant coeff)
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
    actual_kl = float(np.sum(
        sigma_star * (np.log(sigma_star + utils.SMALL_NUMBER) - np.log(hat_sigma + utils.SMALL_NUMBER))
    ))

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


