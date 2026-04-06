import pyspiel
import jax.numpy as jnp
import chex
import jax
import flax.nnx as nn
import enum

class Welfare(enum.StrEnum):
   UTIL = "utilitarian"
   NEUTRAL = "neutral"
   MAX = "max"


@nn.vmap(in_axes=(None, 0), out_axes=0, axis_name='batch')
def batched_call(model: nn.Module, x: chex.Array) -> chex.Array:
  """Batched model call."""
  return model(x)

def mask_diagonal(x: chex.Array) -> chex.Array:
  """CE-related utility to compute f_hat."""
  diag = jnp.arange(x.shape[-1])
  x = x.at[..., diag, diag].set(0.0)
  return x

# def compute_metrics():
#   """Useful metrics"""
#   # Metrics
#   welfare = jnp.sum(sigma * W)
#   rel_entropy = jnp.sum(sigma * (jnp.log(sigma + 1e-12) - log_hat_sigma))
  
#   # CCE Regret: max_{a'_p} E_{a_{-p}~σ}[G_p(a'_p, a_{-p})] - E_a[G_p(a)]
#   def compute_regret(p: int):
#       eq_pay_p = jnp.sum(sigma * G[p])
      
#       # Marginal of opponents: sum over player p axis
#       marginal_opp = jnp.sum(sigma, axis=p, keepdims=True)
      
#       # Expected payoff for each deviation a'_p
#       axes_opp = tuple(i for i in range(N) if i != p)
#       dev_payoffs = jnp.sum(marginal_opp * G[p], axis=axes_opp)  # [Ap]
      
#       return jnp.max(dev_payoffs) - eq_pay_p
  
#   regrets = jax.vmap(compute_regret)(jnp.arange(N))
  
#   return {
#     "welfare": welfare,
#     "rel_entropy": rel_entropy,
#     "cce_regret": regrets,
#   }


def dummy_nes_batch(batch_size, n_players, action_sizes) -> dict[str, chex.Array]:
  """Quick placeholder for testing without OpenSpiel"""
  A = jnp.array(action_sizes)
  joint_shape = (batch_size, *A)

  return {
      "returns": jnp.zeros((batch_size, n_players, *A)),        
      "hat_sigma": jnp.ones(joint_shape) / jnp.prod(A),
      "hat_epsilon": jnp.zeros((batch_size, n_players)),
      "welfare": jnp.zeros(joint_shape),
  }

def to_nes_batch(
    game,                                      # pyspiel game object (matrix game)
    batch_size: int = 1,
    target_epsilon: float = 0.0,               # ˆε_p target (usually 0 or small)
    welfare_type: Welfare = Welfare.MAX,         # "utilitarian", "neutral", "max_welfare"
    rng: jax.Array = None            # for random targets if you want
) -> dict[str, jnp.ndarray]:
    """Convert any OpenSpiel normal-form / matrix game → full NES input dict.
    """
    # 1. Extract payoff tensor G_p(a)  [N, A1, ..., AN]
    payoff_array = pyspiel.utils.game_payoffs_array(game)   # NumPy array
    G = jnp.array(payoff_array, dtype=jnp.float32)          # [N, A1, ..., AN]

    N = G.shape[0]
    action_sizes = G.shape[1:]                              # tuple (A1, A2, ...)
    joint_A = int(jnp.prod(jnp.array(action_sizes)))

    # 2. Target joint distribution ˆσ(a)  [A1, ..., AN]
    if rng is None:
        hat_sigma = jnp.ones(action_sizes) / joint_A        # uniform (default)
    else:
        # optional: sample a random valid distribution
        key = jax.random.split(rng, 1)[0]
        hat_sigma = jax.random.dirichlet(key, alpha=jnp.ones(joint_A)).reshape(action_sizes)

    # 3. Target approximation ˆε_p  [N]
    hat_epsilon = jnp.full((N,), target_epsilon, dtype=jnp.float32)

    # 4. Welfare function W(a)  [A1, ..., AN]
    if welfare_type == Welfare.UTIL:
        W = jnp.sum(G, axis=0)                              # sum of all players' payoffs
    elif welfare_type == Welfare.NEUTRAL:
        W = jnp.zeros(action_sizes)
    elif welfare_type == Welfare.MAX:
        W = jnp.max(G, axis=0)                              # max payoff across players
    else:
        raise ValueError(f"Unknown welfare_type {welfare_type}")

    # Add batch dimension to everything
    batch = {
        "returns": G[jnp.newaxis].repeat(batch_size, axis=0),           # [B, N, A1, ..., AN]
        "hat_sigma": hat_sigma[jnp.newaxis].repeat(batch_size, axis=0),
        "hat_epsilon": hat_epsilon[jnp.newaxis].repeat(batch_size, axis=0),
        "welfare": W[jnp.newaxis].repeat(batch_size, axis=0),
    }

    return batch

def game_to_payoff_tensor(game: pyspiel.Game) -> chex.Array:
  """Convert OpenSpiel game → payoff tensor [N, A1, ..., AN]"""
  payoffs = pyspiel.utils.game_payoffs_array(game)   # shape: (num_players, *action_sizes)
  # payoffs[p, a1, a2, ..., aN] = payoff for player p
  return jnp.array(payoffs, dtype=jnp.float32)


@jax.jit
def compute_joint_action_size(action_shape: chex.Array) -> int:
  """|A| = product of all players' action sizes"""
  return int(jnp.prod(action_shape))


@jax.jit
def unit_variance_scale_G(G: chex.Array, m: int = 2) -> chex.Array:
  """Equation (16a): L_m unit-variance scaling for payoffs (per player)
  G: [B, N, A1, A2, ..., AN]
  Returns G^{L_m}
  """
  N, *A = G.shape
  joint_A = compute_joint_action_size(jnp.array(A))

  # Zero-mean per player
  mean_p = jnp.mean(G, axis=tuple(range(1, G.ndim)), keepdims=True)  # [B, N, 1...]
  centered = G - mean_p

  # L_m norm per player
  if m == 2:
      norm_p = jnp.linalg.norm(centered, axis=tuple(range(2, G.ndim)), keepdims=True)
      Z_m = jnp.sqrt(float(joint_A))
  else:
      # General m-norm: (sum |x_i|^m)^{1/m}
      norm_p = jnp.sum(jnp.abs(centered)**m, axis=tuple(range(2, G.ndim)), keepdims=True)**(1.0/m)
      # Z_m not explicitly given for m≠2 in the paper; we fall back to L2 style for stability
      Z_m = float(joint_A)**(1.0/m)

  return Z_m * (centered / (norm_p + 1e-12))


@jax.jit
def unit_variance_scale_epsilon(
    hat_epsilon: chex.Array,
    G: chex.Array,
    m: int = 2
) -> chex.Array:
  """Equation (16b): scaled + clipped \hat{ε}_p
  hat_epsilon: [B, N]
  Returns \hat{ε}^{L_m}_p clipped to [-Z_m, Z_m]
  """
  B, N = hat_epsilon.shape
  A_shape = G.shape[2:]
  joint_A = compute_joint_action_size(jnp.array(A_shape))

  # Compute the same per-player L_m norm used for G
  mean_p = jnp.mean(G, axis=tuple(range(2, G.ndim)), keepdims=True)
  centered = G - mean_p
  if m == 2:
      norm_p = jnp.linalg.norm(centered, axis=tuple(range(2, G.ndim)), keepdims=True)
      Z_m = jnp.sqrt(float(joint_A))
  else:
      norm_p = jnp.sum(jnp.abs(centered)**m, axis=tuple(range(2, G.ndim)), keepdims=True)**(1.0/m)
      Z_m = float(joint_A)**(1.0/m)

  # Scale target epsilon by the norm
  scaled_eps = hat_epsilon[..., None, None, ...] * norm_p.squeeze(axis=tuple(range(2, G.ndim)))

  # Clip to [-Z_m, +Z_m] (broadcast Z_m)
  return jnp.clip(scaled_eps.squeeze(), -Z_m, Z_m)


@jax.jit
def unit_variance_scale_W(W: chex.Array, m: int = 2) -> chex.Array:
  """Equation (16c): L_m unit-variance scaling for welfare (joint)"""
  mean = jnp.mean(W, axis=tuple(range(1, W.ndim)), keepdims=True)
  centered = W - mean
  joint_A = compute_joint_action_size(jnp.array(W.shape[1:]))

  if m == 2:
    norm = jnp.linalg.norm(centered, axis=tuple(range(1, W.ndim)), keepdims=True)
    Z_m = jnp.sqrt(float(joint_A))
  else:
    norm = jnp.sum(jnp.abs(centered)**m, axis=tuple(range(1, W.ndim)), keepdims=True)**(1.0/m)
    Z_m = float(joint_A)**(1.0/m)

  return Z_m * (centered / (norm + 1e-12))


@jax.jit
def unit_variance_scale_sigma(hat_sigma: chex.Array) -> chex.Array:
  """Equation (16d): L1 unit-variance scaling for target joint distribution"""
  joint_A = compute_joint_action_size(jnp.array(hat_sigma.shape[1:]))
  mean = 1.0 / joint_A

  # Z_σ from Eq. (17a)
  Z_sigma = (joint_A / jnp.sqrt(joint_A + 1.0 / joint_A)) * ((joint_A - 1) / joint_A)

  return Z_sigma * (hat_sigma - mean)


@jax.jit
def apply_unit_variance_scaling(batch: dict[str, chex.Array], m: int = 2) -> dict[str, chex.Array]:
  """Apply the full Appendix B scaling to an entire batch.
  Returns a new dict with all four tensors scaled exactly as in the paper.
  """
  G_scaled = unit_variance_scale_G(batch["G"], m=m)
  W_scaled = unit_variance_scale_W(batch["W"], m=m)
  hat_eps_scaled = unit_variance_scale_epsilon(batch["hat_epsilon"], batch["G"], m=m)
  hat_sigma_scaled = unit_variance_scale_sigma(batch["hat_sigma"])

  return {
      "G": G_scaled,
      "hat_sigma": hat_sigma_scaled,
      "hat_epsilon": hat_eps_scaled,
      "W": W_scaled,
  }


class PayoffTransform:
  """Input normalisation transformations"""

  def __init__(self, m: int) -> None:
    self.m = m

  def __call__(
      self, G: chex.Array, 
      hat_epsilon: chex.Array, 
      welfare: chex.Array, 
      hat_sigma: chex.Array
  ) -> chex.Array:
    return jnp.concatenate([
        unit_variance_scale_G(G, self.m),
        unit_variance_scale_epsilon(hat_epsilon, G, self.m),
        unit_variance_scale_W(welfare, self.m),
        unit_variance_scale_sigma(hat_sigma),
      ], axis = 0
    )
  
class PayoffInit:
  """Intialisation of
    - hat_epsilon
    - welfare
    - hat_sigma
    of Possible Neural Equilibrium Solver solution 
    parameterisations according to the Table 3.
  """
  def __init__(self):
    pass
   
  def __call__(self, *args, **kwds):
    pass