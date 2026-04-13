import functools
from typing import Callable

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
import pyspiel

from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils
from open_spiel.python.jax.nes import samplers
from open_spiel.python.jax.nes import equilibria_utils as eu

Data = samplers.Data

 
@functools.partial(jax.vmap, in_axes=(0, 0, None, None, None, None))
@functools.partial(jax.jit, static_argnames=("mode",))
def recover_primals(
  alpha: chex.Array,
  data: Data, # Assumes a NamedTuple
  mu: float,
  rho: float,
  epsilon_plus: float,
  mode: networks.Mode = networks.Mode.CCE,
) -> tuple:
  """
  ε̂-MWMRE dual loss (Equation 7).
  """
  G = data.reward        # [N, A1, ..., AN]
  N = G.shape[0]

  hat_sigma = data.sigma_hat  # [A1, ..., AN]
  hat_epsilon = data.epsilon_hat  # [N]
  W = data.welfare        # [A1, ..., AN]

  # Safely reduce the dual channel dimension
  if alpha.shape[0] != 1:
    alpha = jnp.sum(alpha, axis=0)
  else:
    alpha = alpha.squeeze(0)  # [N, A] or [N, A, A]

  if mode == networks.Mode.CCE.value:
    contrib_fn = eu.player_contribution_cce
  else:
    contrib_fn = eu.player_contribution_ce

  deviations = []
  sum_alphas = []
  for p in range(N):
    dev, sum_a = contrib_fn(alpha[p], G[p], p)
    deviations.append(dev)
    sum_alphas.append(sum_a)

  deviations = jnp.stack(deviations)
  sum_alphas = jnp.stack(sum_alphas)

  # Sum deviations over all players: Σ_p [...]
  deviation_term = jnp.sum(deviations, axis=0)  # [A1, ..., AN]

  # Compute logits l(a)
  logits = mu * W - (1.0 / rho) * deviation_term

  log_hat_sigma = jnp.log(hat_sigma + 1e-12)
  log_sum_exp = jax.nn.logsumexp(log_hat_sigma + logits)
  sigma = hat_sigma * jnp.exp(logits - log_sum_exp)

  epsilon = (hat_epsilon - epsilon_plus) * jnp.exp(-sum_alphas / rho) + epsilon_plus

  return logits, sigma, epsilon, sum_alphas, log_sum_exp

class NESSolver:
  """Neural Equilibrium Solver wrapped in an OpenSpiel API."""

  def __init__(
    self,
    game: pyspiel.Game,
    mode: networks.Mode,
    network_config: dict[str, int | list[int]],
    mu: float = 1.0,
    rho: float = 1.0,
    epsilon_plus: float = None,
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

    self._num_players = game.num_players()
    self._num_actions = game.num_distinct_actions()

    # Temp: TODO: get rid of
    self._sampler = samplers.OpenSpielGameSampler(
      game, samplers.Objective.EPS_MWME, m=norm, z_m=epsilon_plus 
    )
    self._eps_plus = self._sampler.z_m

    self._iteration = 0
    self._learning_rate = learning_rate
    self._rngkey, init_key = jax.random.split(jax.random.key(seed), 2)

    self._backend = jax.default_backend()
    self._last_loss_value = None
    self._last_data = self._sampler.sample_random(self._batch_size, init_key)

    self._network = networks.NeuralEquilibriumModel(
      num_players=self._num_players,
      dual_channels=network_config["dual_channels"],
      mode=self._mode,
      payoff_channel_list=network_config["payoff_channel_list"],
      dual_channel_list=network_config["dual_channel_list"],
      rngs=nn.Rngs(seed),
    )

    params = nn.state(self._network, nn.Param)
    def mask_fn(path, _):
      # path is a tuple of segments, e.g., ('linear1', 'bias')
      names = [
        str(p.key) if isinstance(p, jax.tree_util.DictKey) else str(p)
        for p in path
      ]
      # Return True to APPLY decay, False to MASK it
      return ("bias" not in names) and ("bn" not in names)

    mask = jax.tree.map_with_path(mask_fn, params)

    optimiser = optax.adamw(
      learning_rate=learning_rate, weight_decay=weight_decay,
    )

    if gradient_clipping:
      optimiser = optax.chain(
        optax.clip_by_global_norm(gradient_clipping),  # 1. Clip the raw gradients first
        optimiser                                      # 2. Then apply AdamW
      )

    self._optimizer = nn.Optimizer(self._network, optimiser, wrt=nn.Param)

    self._graphdef_network_opt = nn.graphdef((self._network, self._optimizer))
    self._graphdef_network = nn.graphdef(self._network)
    self._batch_stats = nn.BatchStat(self._network)
    self.broadcast_fn = jax.vmap(samplers.broadcast)

    self._update_fn = self._get_jitted_update()

  def _get_jitted_update(self) -> Callable:
    

    def _dual_loss(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the Q-network."""
      network.train()
      alpha = utils.batched_call(network, samplers.stack(self.broadcast_fn(data), axis=1))

      # Recovering primal variables
      logits, sigma, epsilon, sum_alphas, log_sum_exp = recover_primals(
        alpha, data, self._mu, self._rho, self._eps_plus, self._mode.value
      )

      # Total loss: Equation (7)
      loss = (
        log_sum_exp
        + self._eps_plus * jnp.sum(sum_alphas)
        - self._rho * jnp.sum(epsilon)
      )

      aux = {
          "logits": logits,
          "sigma": sigma,
          "epsilon": epsilon,
      }
      return loss.mean(), aux

    @jax.jit
    def update(
        network_opt_state: nn.State,
        data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:
      
      # 1. Merge the graphdef with your incoming functional states
      policy_model, optimiser = nn.merge(
          self._graphdef_network_opt, network_opt_state
      )
      
      (main_loss, aux), grads = nn.value_and_grad(_dual_loss, has_aux=True)(policy_model, data)

      optimiser.update(policy_model, grads)

      new_network_opt_state = nn.state((policy_model, optimiser))

      return main_loss, (new_network_opt_state, aux)

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
    network_state = nn.state((self._network, self._optimizer))
    # Optimization Loop
    for i_seed in range(self._network_train_steps):
      batch = self._sampler.sample_random(self._batch_size, jax.random.key(i_seed), None)
      loss_val, new_state = self._update_fn(
        network_state, batch
      )
      nn.update((self._network, self._optimizer), new_state[0])
      aux = new_state[1]
      self._iteration += 1

    if self._mode == networks.Mode.CCE:
      print("GAP", jax.vmap(eu.compute_cce_gap)(batch.reward, aux["sigma"], aux["epsilon"]).mean() )
    elif self._mode == networks.Mode.CE:
      print("GAP", jax.vmap(eu.compute_ce_gap)(batch.reward, aux["sigma"], aux["epsilon"]).mean() )

    return {}
    # TODO: Final Forward Pass to get optimized outputs
    # optimal_duals = utils.batched_call(self._network, self.game_tensor)

    # optimal_policy = recover_primals(optimal_duals, self.game_tensor, ...)

    # return {
    #   "duals": optimal_duals,
    #   # "policy": optimal_policy
    # }
