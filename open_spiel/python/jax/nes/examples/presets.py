import dataclasses

import chex
import jax.numpy as jnp

from open_spiel.python.jax.nes import samplers, utils

Objective = samplers.Objective

# | # | Test                                 | What It Proves                                   |
# | - | ------------------------------------ | ------------------------------------------------ |
# | 1 | Training convergence on `8×8`        | The unsupervised loss works                      |
# | 2 | In-distribution accuracy (`8×8`)     | NES matches iterative solvers                    |
# | 3 | Zero-shot on `4×4`, `16×16`, `32×32` | Shape-independent generalization                 |
# | 4 | MWME vs. MRE vs. ε variants          | Flexible selection framework                     |
# | 5 | CCE vs. CE duals                     | Both solution concepts are learnable             |
# | 6 | 2×2 canonical games                  | Qualitative correctness & polytope visualization |


@dataclasses.dataclass
class MWMREConfig:
  """Configuration for ε-MWMRE family of equilibrium solvers."""

  strategy_target: chex.Array | None
  """σ̂: Target joint strategy for entropy regularization."""

  epsilon_target: float
  """ε̂: Target slack for each player."""

  welfare_coeff: float = 1.0
  """μ: Weight on welfare term W(a) = Σ_p G_p(a). 
  μ > 0 for welfare-aware, μ = 0 for welfare-agnostic."""

  entropy_coeff: float = 10.0
  """ρ: Weight on KL-divergence penalty KL(σ || σ̂).
    ρ > 0 always (required for convexity).
  """

  epsilon_max: float = 10.0
  """ε⁺: Upper bound / budget on per-player slack ε_p.
    Sets the feasible region for incentive constraints.
  """

  @property
  def objective_name(self) -> Objective:
    """Name of the specific objective variant."""

    has_welfare = self.welfare_coeff > 0
    has_eps = self.epsilon_target > 0.0

    if has_welfare and has_eps:
      return Objective.EPS_MWMRE
    elif has_welfare and not has_eps:
      return Objective.MWME
    elif not has_welfare and has_eps:
      return Objective.EPS_MRE
    else:
      return Objective.MRE if self.strategy_target is not None else Objective.ME

  def validate(self) -> None:
    """Check parameter consistency."""
    assert self.entropy_coeff > 0, "ρ must be > 0 for convexity"
    assert self.epsilon_max >= 0, "ε⁺ must be non-negative"

    if self.strategy_target == "none":
      assert self.welfare_coeff > 0, "Need μ > 0 if no entropy target"

    if self.objective_name in (Objective.MWME, Objective.ME):
      assert self.epsilon_target == 0.0, f"{self.objective_name} requires ε̂ = 0"


def mwme(
  action_size: chex.Shape,
  welfare_coeff: float = 1.0,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
) -> MWMREConfig:
  """Max Welfare, Min Entropy: μ>0, ρ>0, ε̂=0, σ̂=uniform."""
  return MWMREConfig(
    welfare_coeff=welfare_coeff,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=jnp.ones(action_size)
    / utils.compute_joint_action_size(action_size),
    epsilon_target=0.0,
  )


def mre(
  strategy_target: chex.Array,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
) -> MWMREConfig:
  """Min Relative Entropy: μ=0, ρ>0, ε̂=0, σ̂=arbitrary."""
  return MWMREConfig(
    welfare_coeff=0.0,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=strategy_target,
    epsilon_target=0.0,
  )


def me(
  action_size: chex.Shape,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
) -> MWMREConfig:
  """Max Entropy: μ=0, ρ>0, ε̂=0, σ̂=uniform."""
  return MWMREConfig(
    welfare_coeff=0.0,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=jnp.ones(action_size)
    / utils.compute_joint_action_size(action_size),
    epsilon_target=0,
  )


def eps_mwme(
  action_size: chex.Shape,
  welfare_coeff: float = 1.0,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
  epsilon_target: float = 10,
) -> MWMREConfig:
  """ε-approx MWME: μ>0, ρ>0, ε̂>0, σ̂=uniform."""
  return MWMREConfig(
    welfare_coeff=welfare_coeff,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=jnp.ones(action_size)
    / utils.compute_joint_action_size(action_size),
    epsilon_target=epsilon_target,
  )


def eps_mre(
  strategy_target: chex.Array,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
  epsilon_target: float = 10.0,
) -> MWMREConfig:
  """ε-approx MRE: μ=0, ρ>0, ε̂>0, σ̂=arbitrary."""
  return MWMREConfig(
    welfare_coeff=0.0,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=strategy_target,
    epsilon_target=epsilon_target,
  )


def eps_mwmre(
  strategy_target: chex.Array,
  welfare_coeff: float = 1.0,
  entropy_coeff: float = 10.0,
  epsilon_max: float = 10.0,
  epsilon_target: float = 10.0,
) -> MWMREConfig:
  """Full ε-MWMRE: μ>0, ρ>0, ε̂>0, σ̂=arbitrary."""
  return MWMREConfig(
    welfare_coeff=welfare_coeff,
    entropy_coeff=entropy_coeff,
    epsilon_max=epsilon_max,
    strategy_target=strategy_target,
    epsilon_target=epsilon_target,
  )
