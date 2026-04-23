import functools
from typing import Callable, Optional

import chex
import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
import pyspiel
import etils.epath
import orbax.checkpoint as ocp


from open_spiel.python.jax.nes import networks
from open_spiel.python.jax.nes import utils
from open_spiel.python.jax.nes import samplers
from open_spiel.python.jax.nes import games
from open_spiel.python.jax.nes import equilibria_utils as eu

# Shortcut for the data structrure
Data = samplers.Data
Game = pyspiel.Game | games.Game | list[pyspiel.Game] | list[games.Game]


@functools.partial(jax.jit, static_argnames=("mode", "action_sizes"))
def recover_primals(
  alpha: chex.Array,
  data: Data,
  mu: float,
  rho: float,
  epsilon_plus: float,
  action_sizes: chex.Shape,
  mode: networks.Mode = networks.Mode.CCE,
) -> tuple:
  """
  ε̂-MWMRE dual loss (Equation 7).
  """
  G = data.reward  # [N, A1, ..., AN]
  N = G.shape[0]

  hat_sigma = data.sigma_hat  # [A1, ..., AN]
  hat_epsilon = data.epsilon_hat  # [N]
  W = data.welfare  # [A1, ..., AN]

  # Safely reduce the dual channel dimension
  if alpha.shape[0] != 1:
    alpha = jnp.sum(alpha, axis=0)
  else:
    alpha = alpha.squeeze(0)  # [N, A] or [N, A, A]

  if mode == networks.Mode.CCE.value:
    get_alpha_p = lambda p: eu.mask_alpha_cce(alpha, action_sizes)[
      p, : action_sizes[p]
    ]  # noqa: E731
    contrib_fn = eu.player_contribution_cce
  else:
    get_alpha_p = lambda p: eu.mask_alpha_ce(alpha, action_sizes)[
      p, : action_sizes[p], : action_sizes[p]
    ]  # noqa: E731
    contrib_fn = eu.player_contribution_ce

  deviations = []
  sum_alphas = []
  for p in range(N):
    alpha_p = get_alpha_p(p)
    dev, sum_a = contrib_fn(alpha_p, G[p], p)
    deviations.append(dev)
    sum_alphas.append(sum_a)

  deviations = jnp.stack(deviations)
  sum_alphas = jnp.stack(sum_alphas)

  # Sum deviations over all players: Σ_p [...]
  deviation_term = jnp.sum(deviations, axis=0)  # [A1, ..., AN]

  # Compute logits l(a)

  logits = mu * W - (1.0 / rho) * deviation_term

  log_hat_sigma = jnp.log(hat_sigma + utils.SMALL_NUMBER)
  log_sum_exp = jax.nn.logsumexp(log_hat_sigma + logits)
  sigma = hat_sigma * jnp.exp(logits - log_sum_exp)

  epsilon = (hat_epsilon - epsilon_plus) * jnp.exp(
    -sum_alphas / rho
  ) + epsilon_plus

  return logits, sigma, epsilon, sum_alphas, log_sum_exp


class DifferentiableEquilibriumBlock(nn.Module):
  """Stratefy retrieval NES block with backprop support."""

  def __init__(
    self,
    network: networks.NeuralEquilibriumModel,
    mu: float,
    rho: float,
    epsilon_plus: float,
    mode: networks.Mode,
  ) -> None:
    self._mu = mu
    self._rho = rho
    self._eps_plus = epsilon_plus
    self._mode = mode
    self._network = network

  def __call__(self, data: Data) -> chex.Array:
    alpha = self._network(
      samplers.stack(samplers.broadcastn(data), axis=1), data.mask
    )
    _, sigma, _, _, _ = recover_primals(
      alpha, data, self._mu, self._rho, self._eps_plus, self._mode.value
    )
    return sigma


class NESolver:
  """Neural Equilibrium Solver wrapped in an OpenSpiel API."""

  def __init__(
    self,
    game: Game,
    mode: networks.Mode,
    network_config: dict[str, int | list[int]],
    mu: float = 1.0,
    rho: float = 1.0,
    epsilon_plus: float = None,
    norm: int = 2,
    batch_size: int = 100,
    learning_rate: Callable | float = 1e-4,
    weight_decay: float = 1e-7,
    network_train_steps: int = 100,
    gradient_clipping: float | None = None,
    seed: int = 42,
    game_kwargs: dict = {},
    allow_checkpointing: Optional[bool] = False,
  ) -> None:
    """Initializes the NES optimizer for a specific game."""

    self._game = game
    self._mode = mode
    self._batch_size = batch_size
    self._network_train_steps = network_train_steps
    self._mu = mu
    self._rho = rho

    if isinstance(game, pyspiel.Game):
      self._num_players = game.num_players()
      self._num_actions = tuple(
        game.num_distinct_actions() for _ in range(self._num_players)
      )

      self._sampler = samplers.OpenSpielGameSampler(
        game, samplers.Objective.EPS_MWMRE, m=norm, z_m=epsilon_plus
      )
    elif isinstance(game, games.Game):
      assert "num_strategies" in game_kwargs, (
        "Found missing `num_strategies` argument in `game_kwargs`."
      )

      num_strategies = game_kwargs["num_strategies"]
      self._num_players = len(num_strategies)
      self._num_actions = num_strategies

      self._num_players = len(num_strategies)
      game_settings = game_kwargs.get("game_settings", {})

      self._sampler = samplers.RandomGameSampler(
        game,
        num_strategies,
        game_settings,
        samplers.Objective.EPS_MWMRE,
        m=norm,
        z_m=epsilon_plus,
      )
    else:
      raise ValueError("Unsupported game type!")

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

    def mask_fn(params):
      # Not applying decay to biases and batchnorm params. See:
      # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
      return jax.tree.map_with_path(
        lambda path, _: not ("bias" in str(path) or "bn" in str(path)), params
      )

    optimiser = optax.adamw(
      learning_rate=learning_rate, weight_decay=weight_decay, mask=mask_fn
    )

    self._checkpointer = None
    if allow_checkpointing:
      self._checkpointer = ocp.StandardCheckpointer()

    if gradient_clipping:
      optimiser = optax.chain(
        # Clip the raw gradients first
        optax.clip_by_global_norm(gradient_clipping),
        optimiser,
      )

    self._optimizer = nn.Optimizer(self._network, optimiser, wrt=nn.Param)

    self._graphdef_network_opt = nn.graphdef((self._network, self._optimizer))
    self._graphdef_network = nn.graphdef(self._network)
    self.broadcast_fn = jax.vmap(samplers.broadcast)

    self._update_fn = self._get_jitted_update()
    self._infer_fn = self._get_jitted_inference()

  def _get_jitted_inference(self) -> Callable:
    def _recover(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the NES-network."""
      network.eval()

      alpha = utils.batched_call(
        network, samplers.stack(self.broadcast_fn(data), axis=1), data.mask
      )

      # Recovering primal variables
      logits, sigma, epsilon, sum_alphas, log_sum_exp = jax.vmap(
        recover_primals, in_axes=(0, 0, None, None, None, None, None)
      )(
        alpha,
        data,
        self._mu,
        self._rho,
        self._eps_plus,
        self._num_actions,
        self._mode.value,
      )
      # Total loss: Equation (7)
      loss = (
        log_sum_exp
        + self._eps_plus * jnp.sum(sum_alphas)
        - self._rho * jnp.sum(epsilon)
      )

      primals = {
        "logits": logits,
        "sigma": sigma,
        "epsilon": epsilon,
      }
      return loss, (primals, alpha)

    @jax.jit
    def infer(
      network_state: nn.State,
      data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:
      policy_model = nn.merge(self._graphdef_network, network_state)
      return _recover(policy_model, data)

    return infer

  def _get_jitted_update(self) -> Callable:
    """Generate jittable update function."""

    def _dual_loss(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the NES-network."""
      network.train()
      alpha = utils.batched_call(
        network, samplers.stack(self.broadcast_fn(data), axis=1), data.mask
      )

      # Recovering primal variables
      logits, sigma, epsilon, sum_alphas, log_sum_exp = jax.vmap(
        recover_primals, in_axes=(0, 0, None, None, None, None, None)
      )(
        alpha,
        data,
        self._mu,
        self._rho,
        self._eps_plus,
        self._num_actions,
        self._mode.value,
      )

      # Total loss: Equation (7)
      loss = (
        log_sum_exp
        + self._eps_plus * jnp.sum(sum_alphas)
        - self._rho * jnp.sum(epsilon)
      )

      primals = {
        "logits": logits,
        "sigma": sigma,
        "epsilon": epsilon,
      }
      return loss.mean(), primals

    @jax.jit
    def update(
      network_opt_state: nn.State,
      data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:
      policy_model, optimiser = nn.merge(
        self._graphdef_network_opt, network_opt_state
      )

      (main_loss, primals), grads = nn.value_and_grad(_dual_loss, has_aux=True)(
        policy_model, data
      )

      optimiser.update(policy_model, grads)

      new_network_opt_state = nn.state((policy_model, optimiser))

      return main_loss, (new_network_opt_state, primals)

    return update

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey."""
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def save(
    self, checkpoint_dir: etils.epath.Path, save_optimiser: bool = True
  ) -> None:
    """Saves a current NES network state.

    Args:
      checkpoint_dir (etils.epath.Path): target checkpoint location.
      save_optimiser (bool, optional): whether save only the optimiser or just
        the network's weights. Defaults to True.
    """
    assert self._checkpointer, (
      "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    )
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = etils.epath.Path(checkpoint_dir)
    if save_optimiser:
      self._checkpointer.save(
        checkpoint_dir / "optimiser",
        nn.state((self._network, self._optimizer)),
        force=True,
      )
    else:
      self._checkpointer.save(checkpoint_dir / "state", nn.state(self._network))
    self._checkpointer.wait_until_finished()

  def load(
    self, checkpoint_dir: etils.epath.Path, load_optimiser: bool = True
  ) -> None:
    """Restores the NES network state.

    Args:
      checkpoint_dir (etils.epath.Path): target checkpoint dir.
      load_optimiser (bool, optional): whether load only the optimiser or just
        the network's weights. Defaults to True.
    """
    assert self._checkpointer, (
      "Checkpointing disallowed. Set `allow_checkpointing` in the contructor"
    )
    if isinstance(checkpoint_dir, str):
      checkpoint_dir = etils.epath.Path(checkpoint_dir)
    checkpoint_dir = etils.epath.Path(checkpoint_dir)

    if load_optimiser:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "optimiser",
        nn.state((self._network, self._optimizer)),
      )
      nn.update((self._network, self._optimizer), state_restored)

    else:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "state", nn.state(self._q_network)
      )
      nn.update(self._q_network, state_restored)
    self._checkpointer.wait_until_finished()

  def _logging_callback(
    self,
    reward: chex.Array,
    sigma: chex.Array,
    epsilon: chex.Array,
    solve: bool,
  ) -> dict:
    if self._mode == networks.Mode.CCE:
      regret = jax.vmap(eu.compute_cce_gap)(reward, sigma, epsilon)
    else:  # CE gap
      regret = jax.vmap(eu.compute_ce_gap)(reward, sigma, epsilon)

    solver_gap = {}
    if solve:
      solver_gap = eu.mwme_lp_solver_gap(
        reward[0], sigma[0], self._mu, self._rho
      )
      solver_gap["solver_gap"] = jnp.abs(solver_gap["sigma"] - sigma).mean()

    solver_gap.update({"regret": regret.mean()})
    return solver_gap

  def solve(self) -> dict:
    """
    Executes the optimization loop to find the equilibrium.
    Returns the optimal dual variables and the recovered primal policy.
    """
    network_state = nn.state((self._network, self._optimizer))
    # Optimisation Loop
    for i_seed in range(self._network_train_steps):
      batch = self._sampler.sample_random(
        self._batch_size, jax.random.key(i_seed), None
      )
      loss_val, (new_state, primals) = self._update_fn(network_state, batch)
      nn.update((self._network, self._optimizer), new_state)
      self._iteration += 1

      # TODO: do actual logging
      logs = self._logging_callback(
        batch.reward, primals["sigma"], primals["epsilon"], solve=False
      )
      logs.update({"loss": loss_val})
      print(logs)

    batch = self._sampler.sample_random(
      1, jax.random.key(self._network_train_steps), None
    )
    loss_val, (primals, duals) = self._infer_fn(nn.state(self._network), batch)
    logs = self._logging_callback(
      batch.reward, primals["sigma"], primals["epsilon"], solve=True
    )
    print("AFTER_TRAINING\n")
    logs.pop("sigma")
    print(logs)

    return {
      "duals": duals,
      "policy": primals["sigma"],
    }
