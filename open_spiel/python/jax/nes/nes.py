import functools
from typing import Callable, Optional

import absl.logging

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
Game = samplers.Game


@functools.partial(jax.jit, static_argnames=("mode",))
def recover_primals(
  alpha: chex.Array,
  data: Data,
  mu: float,
  rho: float,
  epsilon_plus: float,
  mode: networks.Mode = networks.Mode.CCE,
) -> tuple:
  """
  ε̂-MWMRE dual loss (Equation 7, [1]).
  L_dual = log_sum_exp + ε⁺ · Σ_p Σ_a α_p(a) - ρ · Σ_p ε_p
  """
  G = data.reward  # [N, A1, ..., AN]
  N = G.shape[0]

  hat_sigma = data.sigma_hat  # [A1, ..., AN]
  hat_epsilon = data.epsilon_hat # [N]
  W = data.welfare  # [A1, ..., AN]

  if mode == networks.Mode.CCE.value:
    contrib_fn = eu.player_contribution_cce
  else:  # noqa: E731
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

  W_masked = jnp.where(data.mask, W, 0.0)

  logits = mu * W_masked -  deviation_term

  log_hat_sigma = jnp.log(hat_sigma + utils.SMALL_NUMBER)

  log_sum_exp = jax.nn.logsumexp(log_hat_sigma + logits, where=data.mask)
  sigma = jnp.where(data.mask, hat_sigma * jnp.exp(logits - log_sum_exp), 0.0)
  sigma = sigma / (sigma.sum() + utils.SMALL_NUMBER)

  epsilon = (hat_epsilon - epsilon_plus) * jnp.exp(
    -sum_alphas / rho
  ) + epsilon_plus


  return logits, sigma, epsilon, sum_alphas, log_sum_exp


class DifferentiableEquilibriumBlock(nn.Module):
  """NES block with backprop support."""

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
    ).squeeze(1)
    _, sigma, _, _, _ = recover_primals(
      alpha, data, self._mu, self._rho, self._eps_plus, self._mode.value
    )
    return sigma


def objective_from_coefficients(
  mu: float,
  eps_hat: float | chex.Array,
  uniform_sigma: bool,
) -> samplers.Objective:
  # Simplified version

  if mu:
    if uniform_sigma:
      if eps_hat == 0:
        return samplers.Objective.MWME
      elif eps_hat > 0:
        return samplers.Objective.EPS_MWME
    else:
      if eps_hat == 0:
        return samplers.Objective.EPS_MRE
  else:
    if uniform_sigma:
      if eps_hat == 0:
        return samplers.Objective.ME
    else:
      if eps_hat == 0:
        return samplers.Objective.MRE
      elif eps_hat > 0:
        return samplers.Objective.EPS_MRE


class NESolver:
  """Neural Equilibrium Solver wrapped in an OpenSpiel API."""

  def __init__(
    self,
    game: Game,
    mode: networks.Mode,
    network_config: dict[str, int | list[int]],
    objective: samplers.Objective | None = samplers.Objective.EPS_MWMRE,
    mu: float = 1.0,
    rho: float = 1.0,
    epsilon_plus: float = None,
    norm: int = 2,
    batch_size: int = 100,
    eval_batch_size: int = 1,
    learning_rate: Callable | float = 1e-4,
    weight_decay: float = 1e-7,
    network_train_steps: int = 100,
    log_every: int = -1,
    save_every: int = -1,
    gradient_clipping: float | None = None,
    seed: int = 42,
    game_kwargs: Optional[dict] = {},
    allow_checkpointing: Optional[bool] = False,
  ) -> None:
    """Initializes the NES optimizer for a specific game."""

    self._game = game
    self._mode = mode
    self._batch_size = batch_size
    self._eval_batch_size = eval_batch_size

    self._network_train_steps = network_train_steps
    self._mu = mu
    self._rho = rho

    if game in pyspiel.registered_names():
      _game = pyspiel.load_game(game)
      self._num_players = _game.num_players()
      self._num_actions = tuple(
        _game.num_distinct_actions() for _ in range(self._num_players)
      )

      self._sampler = samplers.OpenSpielGameSampler(
        _game, obj=objective, m=norm, z_m=epsilon_plus
      )
    elif isinstance(game, games.Game):
      assert "num_strategies" in game_kwargs, (
        "Found missing `num_strategies` argument in `game_kwargs`."
      )

      num_strategies = game_kwargs["num_strategies"]
      self._num_players = len(num_strategies)
      self._num_actions = num_strategies

      game_settings = game_kwargs.get("game_settings", {})

      self._sampler = samplers.RandomGameSampler(
        game,
        num_strategies,
        game_settings,
        obj=objective,
        m=norm,
        z_m=epsilon_plus,
      )
    elif isinstance(game, list):
      assert "max_actions" in game_kwargs, (
        "Found missing `max_actions` argument in `game_kwargs`."
      )
      game_configs = []
      for g in game:
        if g in pyspiel.registered_names():
          game_configs.append(samplers.MixedGameConfig(open_spiel_name=g))
        elif isinstance(g, games.Game):
          assert "num_strategies" in game_kwargs, (
            "Found missing `num_strategies` argument in `game_kwargs`."
          )

          num_strategies = game_kwargs["num_strategies"]
          game_settings = game_kwargs.get("game_settings", {})
          game_configs.append(
            samplers.MixedGameConfig(
              game_type=g,
              game_settings=game_settings,
              num_strategies=num_strategies,
            )
          )

      self._sampler = samplers.MultiGameSampler(
        game_configs=game_configs,
        max_actions=game_kwargs["max_actions"],
        obj=objective,
        m=norm,
        z_m=epsilon_plus,
      )

      self._num_players = self._sampler.num_players
      self._num_actions = tuple(
        game_kwargs["max_actions"] for _ in range(self._num_players)
      )

    else:
      raise ValueError("Unsupported game type!")
    
    self._rngkey, init_key = jax.random.split(jax.random.key(seed), 2)
    self._last_data = self._sampler.sample_random(self._batch_size, init_key)

    self._eps_plus = epsilon_plus if epsilon_plus is not None else self._sampler.z_m

    self._iteration = 0
    self._learning_rate = learning_rate

    self._backend = jax.default_backend()
    self._last_loss_value = None

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
    # TODO: allow checkpointing
    self._save_every = save_every
    self._log_every = log_every
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

    def _dual_loss_fn(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the NES-network."""

      alpha = utils.batched_call(
        network, samplers.stack(self.broadcast_fn(data), axis=1), data.mask
      ).squeeze(1)
      # Recovering primal variables
      logits, sigma, epsilon, sum_alphas, log_sum_exp = jax.vmap(
        recover_primals, in_axes=(0, 0, None, None, None, None)
      )(
        alpha,
        data,
        self._mu,
        self._rho,
        self._eps_plus,
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
      return loss.mean(), (primals, alpha)

    self._loss_fn = _dual_loss_fn
    self._update_fn = self._get_jitted_update()
    self._infer_fn = self._get_jitted_inference()

  def _get_jitted_inference(self) -> Callable:
    @jax.jit
    def infer(
      network_state: nn.State,
      data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:

      policy_model = nn.merge(self._graphdef_network, network_state)
      policy_model.eval()

      _, (primals, duals) = self._loss_fn(policy_model, data)
      return primals, duals

    return infer

  def _get_jitted_update(self) -> Callable:
    """Generate jittable update function."""

    @jax.jit
    def update(
      network_opt_state: nn.State,
      data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:
      policy_model, optimiser = nn.merge(
        self._graphdef_network_opt, network_opt_state,
      )
      policy_model.train()
      (loss, (primals, _)), grads = nn.value_and_grad(
        self._loss_fn, has_aux=True
      )(policy_model, data)

      optimiser.update(policy_model, grads)

      new_network_opt_state = nn.state((policy_model, optimiser))

      return loss, (new_network_opt_state, primals)

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

    if load_optimiser:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "optimiser",
        nn.state((self._network, self._optimizer)),
      )
      nn.update((self._network, self._optimizer), state_restored)

    else:
      state_restored = self._checkpointer.restore(
        checkpoint_dir / "state", nn.state(self._network)
      )
      nn.update(self._network, state_restored)
    self._checkpointer.wait_until_finished()

  def _logging_callback(
    self,
    reward: chex.Array,
    sigma: chex.Array,
    epsilon: chex.Array,
    mask: chex.Array,
  ) -> dict:
    batch_size = reward.shape[0]

    if self._mode == networks.Mode.CCE:
      regret = jax.vmap(eu.compute_cce_gap)(reward, sigma, epsilon, mask)
    else:  # CE gap
      regret = jax.vmap(eu.compute_ce_gap)(reward, sigma, epsilon, mask)

    data = []
    for i in range(batch_size):

      solver_data = eu.mwmre_solver(
        reward[i],
        sigma[i],
        epsilon[i],
        mask[i],
        self._mu,
        self._rho,
        self._eps_plus,
        self._mode.name,
      )
      data.append({
        "solver_gap": solver_data["solver_gap"],
        "welfare_gap": solver_data["welfare_gap"],
        "eps_gap": solver_data["eps_gap"]
      })

    mean_batch_data = jax.tree.map(
      lambda *args: jnp.mean(jnp.stack(args), axis=0), *data
    )

    mean_batch_data.update({f"{self._mode.name}_gap": regret.mean()})
    return mean_batch_data

  def evaluate(
    self, batch_size: int, sampler: Optional[samplers.GameSampler] = None
  ) -> dict:
    sampler = sampler if sampler else self._sampler
    batch = sampler.sample_random(
      batch_size, jax.random.key(self._network_train_steps), None
    )
    primals, duals = self._infer_fn(nn.state(self._network), batch)
    log = self._logging_callback(
      batch.reward, primals["sigma"], batch.epsilon_hat, batch.mask
    )
    for k, v in log.items():
      print(f"\t{k}:\t{v}")
    print()
    
    return log

  def solve(self, logger=None) -> dict:
    """
    Executes the optimization loop to find the equilibrium.
    Returns the optimal dual variables and the recovered primal policy.
    """
    logs = []

    network_state = nn.state((self._network, self._optimizer))

    # Optimisation Loop
    for i_seed in range(self._network_train_steps):
      
      batch = self._sampler.sample_random(
        self._batch_size, jax.random.key(i_seed), None
      )

      loss_val, (network_state, primals) = self._update_fn(network_state, batch)
      nn.update((self._network, self._optimizer), network_state)

      self._iteration += 1

      # TODO: do actual logging
      if self._iteration % self._log_every == 0:
        # Cheap evaluation during training
        print(f"It. {self._iteration} | Loss {loss_val}")

        log = self.evaluate(self._batch_size)
        logs.append(log)

    # Peform expensive evaluation after the training
    logs.append(self.evaluate(self._eval_batch_size))

    return logs
