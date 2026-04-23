import jax.numpy as jnp
from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import nes
import chex


def mechanism_loss(
    generator: networks.MechanismGenerator,
    deb: nes.DifferentiableEquilibriumBlock,   # your frozen NES
    batch: dict,                           # contains "omega" and "target_sigma"
    mu: float = 1.0,
    rho: float = 0.1,                      # KL weight
) -> tuple[chex.Array, dict]:
  """End-to-end loss: Generator → DEB → distance to target equilibrium."""
  omega = batch["omega"]                 # [B, context_dim]
  target_sigma = batch["target_sigma"]   # [B, A1, ..., AN]

  # 1. Generate induced game
  G = generator(omega)                   # [B, N, A1, ..., AN]

  # 2. Run frozen NES (DEB)
  sigma_star = deb(G)                    # [B, A1, ..., AN]

  # 3. Loss = KL(σ* || target) + optional welfare regularizer
  kl_loss = jnp.sum(
      sigma_star * (jnp.log(sigma_star + 1e-12) - jnp.log(target_sigma + 1e-12)),
      axis=tuple(range(1, sigma_star.ndim))
  ).mean()

  # Optional: encourage high welfare
  welfare = jnp.sum(sigma_star * jnp.sum(G, axis=1))   # utilitarian welfare
  welfare_loss = -mu * welfare.mean()

  total_loss = kl_loss + welfare_loss

  aux = {
      "kl_loss": kl_loss,
      "welfare": welfare.mean(),
      "induced_G": G,
      "recovered_sigma": sigma_star,
  }

  return total_loss, aux