import functools
from typing import NamedTuple

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
import pyspiel

from open_spiel.python.jax.nes import model, utils


class Data(NamedTuple):
  """Experience batch for the network."""

  returns: chex.Array
  hat_sigma: chex.Array
  hat_epsilon: chex.Array
  welfare: chex.Array

@functools.partial(jax.jit, static_argnames=("mode",))
def recover_primals(
  alpha: chex.Array,
  data: Data, # Assumes a NamedTuple or Dataclass
  mu: float,
  rho: float,
  epsilon_plus: float,
  mode: model.Mode = model.Mode.CCE,
) -> tuple:
  """
  hat{varepsilon}-MWMRE dual loss (Equation 7).
  """

  G = data.returns        # [N, A1, ..., AN]
  hat_sigma = data.hat_sigma  # [A1, ..., AN]
  hat_epsilon = data.hat_epsilon  # [N]
  W = data.welfare        # [A1, ..., AN]

  N = G.shape[0]

  # Safely reduce the dual channel dimension
  if alpha.shape[0] != 1:
    alpha = jnp.sum(alpha, axis=0)
  else:
    alpha = alpha.squeeze(0)  # [N, A] or [N, A, A]
  log_hat_sigma = jnp.log(hat_sigma + 1e-12)
  def compute_player_contribution(p: int):
    """Compute deviation and sum_alpha for player p."""
    alpha_p = alpha[p]  # [Ap] for CCE, [Ap, Ap] for CE
    G_p = G[p]          # [A1, ..., AN]
    # Move player axis to front for matrix multiplication
    G_moved = jnp.moveaxis(G_p, p, 0)  # [Ap, A1, ..., Ap-1, Ap+1, ..., AN]
    Ap = G_moved.shape[0]
    
    # Reshape to 2D matrix [Ap, prod(other_dims)]
    G_matrix = G_moved.reshape(Ap, -1)
    
    if mode == model.Mode.CCE.value:
        # alpha_p: [Ap], G_matrix: [Ap, -1]
        # Compute: alpha_p @ G_matrix -> [-1]

        weighted_flat = alpha_p @ G_matrix  # matmul
        
        # Reshape back to [A1, ..., Ap-1, Ap+1, ..., AN]
        other_shape = G_p.shape[:p] + G_p.shape[p+1:]
        weighted_sum = weighted_flat.reshape(other_shape)
        
        # Expand dims at position p for broadcasting: [..., 1, ...]
        weighted_payoff = jnp.expand_dims(weighted_sum, axis=p)
        
        # Sum of duals
        sum_alpha_p = jnp.sum(alpha_p)
        
        # Deviation: weighted_payoff - G_p * sum_alpha_p
        deviation = weighted_payoff - G_p * sum_alpha_p
        
    else:  # Mode.CE
        # alpha_p: [Ap, Ap] (deviation, recommended)
        # We need: sum_{deviation} alpha[deviation, recommended] * G[deviation, ...]
        # This is: alpha_p.T @ G_matrix
        # alpha_p.T: [Ap, Ap], G_matrix: [Ap, -1] -> [Ap, -1]
        weighted_flat = alpha_p.T @ G_matrix 
        
        # Reshape to [Ap, A1, ..., Ap-1, Ap+1, ..., AN]
        other_shape = (Ap,) + G_p.shape[:p] + G_p.shape[p+1:]
        weighted_sum = weighted_flat.reshape(other_shape)
        
        # Move the Ap (recommended action) dimension back to position p
        weighted_payoff = jnp.moveaxis(weighted_sum, 0, p)  # [A1, ..., AN]
        
        # Sum of duals over both (deviation, recommended)
        sum_alpha_p = jnp.sum(alpha_p)
        
        # beta_p(recommended) = sum over deviation (axis 0)
        beta_p = jnp.sum(alpha_p, axis=0)  # [Ap]
        
        # Broadcast beta_p to full shape [A1, ..., AN]
        beta_shape = [1] * N
        beta_shape[p] = Ap
        beta_broadcast = beta_p.reshape(beta_shape)
        
        # Deviation: weighted_payoff - G_p * beta_broadcast
        deviation = weighted_payoff - G_p * beta_broadcast
       
    return deviation, sum_alpha_p

  deviations = []
  sum_alphas = []
  for p in range(N):
    dev, sum_a = compute_player_contribution(p)
    deviations.append(dev)
    sum_alphas.append(sum_a)

  deviations = jnp.stack(deviations)
  sum_alphas = jnp.stack(sum_alphas)

  # Sum deviations over all players: Σ_p [...]
  deviation_term = jnp.sum(deviations, axis=0)  # [A1, ..., AN]

  # Compute logits l(a)
  logits = mu * W - (1.0 / rho) * deviation_term

  log_sum_exp = jax.nn.logsumexp(log_hat_sigma + logits)
  sigma = hat_sigma * jnp.exp(logits - log_sum_exp)

  epsilon = (hat_epsilon - epsilon_plus) * jnp.exp(-sum_alphas / rho) + epsilon_plus

  return logits, sigma, epsilon, sum_alphas, log_sum_exp

class NESSolver:
  """Neural Equilibrium Solver wrapped in an OpenSpiel API."""

  def __init__(
    self,
    game: pyspiel.Game,
    mode: model.Mode,
    network_config: dict[str, int | list[int]],
    mu: float = 1.0,
    rho: float = 1.0,
    epsilon_plus: float = 10.0,
    norm: int = 2,
    batch_size: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-7,
    network_train_steps: int = 1000,
    gradient_clipping: float | None = None,
    seed: int = 42,
  ) -> None:
    """Initializes the NES optimizer for a specific game."""

    self._game = game
    self._mode = mode
    self._batch_size = batch_size
    self._network_train_steps = network_train_steps
    self._mu = mu
    self._rho = rho
    self._eps_plus = epsilon_plus

    self._num_players = game.num_players()
    self._num_actions = game.num_distinct_actions()

    self._normaliser = utils.PayoffTransform(m=norm)
    self._initialiser = None

    self._iteration = 1
    self._learning_rate = learning_rate
    self._rngkey = jax.random.key(seed)

    self._backend = jax.default_backend()
    self._last_loss_value = None
    self._last_data = None

    self._network = model.NeuralEquilibriumModel(
      num_players=self._num_players,
      num_actions=self._max_actions,
      dual_channels=network_config["dual_channels"],
      mode=self.mode,
      payoff_channel_list=network_config["payoff_channel_list"],
      dual_channel_list=network_config["dual_channel_list"],
      rngs=nn.Rngs(seed),
    )

    def mask_fn(path, _):
      # path is a tuple of segments, e.g., ('linear1', 'bias')
      names = [
        str(p.key) if isinstance(p, jax.tree_util.DictKey) else str(p)
        for p in path
      ]
      # Return True to APPLY decay, False to MASK it
      return ("bias" not in names) and ("bn" not in names)

    params = nn.state(model, nn.Param)
    mask = jax.tree.map_with_path(mask_fn, params)

    optimiser = optax.adamw(
      learning_rate=learning_rate, weight_decay=weight_decay, mask=mask
    )

    if gradient_clipping:
      optimiser = optax.chain(
        optimiser, optax.clip_by_global_norm(gradient_clipping)
      )

    self._optimizer = nn.Optimizer(self._network, optimiser, wrt=nn.Param)

    self._graphdef_q_network_opt = nn.graphdef((self._network, self._optimizer))
    self._graphdef_network = nn.graphdef(self._network)

    self._jittable_update = self._get_jitted_update()

    self.optimizer = nn.Optimizer(
      self.model, optax.adam(learning_rate), wrt=nn.Param
    )

  def _get_jitted_update(self):
    def _dual_loss(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the Q-network."""
      alpha = utils.batched_call(network, data)

      # Recovering primal variables
      _, _, epsilon, sum_alphas, log_sum_exp = recover_primals(
        alpha, data, self._mu, self._rho, self._eps_plus, self._mode
      )

      # Total loss: Equation (7)
      loss = (
        log_sum_exp
        + self._eps_plus * jnp.sum(sum_alphas)
        - self._rho * jnp.sum(epsilon)
      )

      return loss.mean()

    @jax.jit
    def update(
      policy_model: nn.Module,
      optimiser: nn.Optimizer,
      batch: Data,
    ) -> chex.Array:
      main_loss, grads = nn.value_and_grad(_dual_loss)(policy_model, Data)
      optimiser.update(policy_model, grads)
      return main_loss

    return update

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def select_action(self) -> chex.Array:
    """Selecting player's strategy sigma."""

  def solve(self) -> dict:
    """
    Executes the optimization loop to find the equilibrium.
    Returns the optimal dual variables and the recovered primal policy.
    """
    print(
      f"Starting NES {self.mode.name} Solver for {self.max_iterations} iterations..."
    )

    # Optimization Loop
    for i in range(self._network_train_steps):
      loss = self._jittable_update(self.model, self.optimizer, self.game_tensor)

    # Final Forward Pass to get optimized outputs
    optimal_duals = utils.batched_call(self._network, self.game_tensor)

    optimal_policy = recover_primals(optimal_duals, self.game_tensor, ...)

    return {
      "duals": optimal_duals,
      # "policy": optimal_policy
    }
