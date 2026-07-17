import functools
from typing import Callable, Optional

import chex
import etils.epath
import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pyspiel
import trackio

from open_spiel.python.jax.nes import equilibria_utils as eu
from open_spiel.python.jax.nes import games, networks, samplers, utils
from open_spiel.python.utils import data_logger

"""Neural Equilibrium Solver agent implemented in Jax.

  See the paper
  [1] "Turbocharging Solution Concepts: Solving NEs, CEs and 
  CCEs with Neural Equilibrium Solvers"

  https://arxiv.org/abs/2210.09257 for more details.
"""

# Shortcut for data structrures
Data = samplers.Data
Game = samplers.Game
Objective = samplers.Objective


@functools.partial(jax.jit, static_argnames=("mode",))
def recover_primals(
  duals: chex.Array,
  data: Data,
  welfare_coeff: float,
  entropy_coeff: float,
  epsilon_max: float,
  mode: int,
) -> tuple:
  """ε̂-MWMRE dual loss (Equation 7, [1]).
  L_dual = log_sum_exp + ε⁺ · Σ_p Σ_a α_p(a) - ρ · Σ_p ε_p

    Args:
      duals: Player payoffs with shape [N, A] or [N, A, A].
      data: Batch of network data.
      welfare_coeff: A coefficient of the ME objective.
      entropy_coeff: A coefficient of the MRE objective.
      epsilon_max: MRE cap.
      mode: CE or CCE.

  Returns:
      Recovered primal variabls: A tuple consisting of
        logits, strategy, epsilon, sum_duals, log_sum_exp
  """
  payoffs = data.payoffs  # [N, A1, ..., AN]

  strategy = data.strategy_base  # [A1, ..., AN]
  epsilon_target = data.epsilon_target  # [N]
  W = data.welfare  # [A1, ..., AN]

  if mode == networks.Mode.CCE.value:
    deviation_term = eu.cce_logit(duals, payoffs=payoffs)  # [A1, ..., AN]
  else:  # CE
    deviation_term = eu.ce_logit(duals, payoffs=payoffs)  # [A1, ..., AN]
  sum_duals = jnp.sum(duals, axis=tuple(range(1, duals.ndim)))  # [N]

  W_masked = jnp.where(data.joint_mask, W, 0.0)

  logits = welfare_coeff * W_masked - deviation_term

  log_strat_base = jnp.log(strategy + utils.SMALL_NUMBER)

  log_sum_exp = jax.nn.logsumexp(log_strat_base + logits, where=data.joint_mask)
  strategy = jnp.where(
    data.joint_mask, strategy * jnp.exp(logits - log_sum_exp), 0.0
  )

  epsilon = (epsilon_target - epsilon_max) * jnp.exp(
    -sum_duals / entropy_coeff
  ) + epsilon_max

  return logits, strategy, epsilon, sum_duals, log_sum_exp


class DifferentiableEquilibriumBlock(nn.Module):
  """NES block with backprop support."""

  def __init__(
    self,
    network: networks.NeuralEquilibriumModel,
    welfare_coeff: float,
    entropy_coeff: float,
    epsilon_max: float,
    mode: networks.Mode,
  ) -> None:
    self._mu = welfare_coeff
    self._rho = entropy_coeff
    self._eps_cap = epsilon_max
    self._mode = mode
    self._network = network

  def __call__(self, data: Data) -> chex.Array:
    duals = self._network(
      samplers.stack(samplers.broadcast(data), axis=1),
      data.strat_mask_per_player,
    )
    # Recovering primal variables
    _, strategy, _, _, _ = recover_primals(
      duals,
      data,
      self._mu,
      self._rho,
      self._eps_cap,
      self._mode.value,
    )
    return strategy


class NESolver:
  """Neural Equilibrium Solver wrapped in an OpenSpiel API."""

  def __init__(
    self,
    game: Game,
    mode: networks.Mode,
    network_config: dict[str, int | list[int]],
    objective: samplers.Objective | None = samplers.Objective.EPS_MWMRE,
    welfare_coeff: float = 1.0,
    entropy_coeff: float = 1.0,
    epsilon_max: float = None,
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
    """Initialises the NES optimizer for a specific game."""

    self._game = game
    self._mode = mode
    self._batch_size = batch_size
    self._eval_batch_size = eval_batch_size

    self._network_train_steps = network_train_steps
    self._mu = welfare_coeff
    self._rho = entropy_coeff

    if game in pyspiel.registered_names():
      _game = pyspiel.load_game(game)
      self._num_players = _game.num_players()
      self._num_actions = tuple(
        _game.num_distinct_actions() for _ in range(self._num_players)
      )

      self._sampler = samplers.OpenSpielGameSampler(
        _game, obj=objective, m=norm, z_m=epsilon_max
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
        z_m=epsilon_max,
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
        z_m=epsilon_max,
      )

      self._num_players = self._sampler.num_players
      self._num_actions = tuple(
        game_kwargs["max_actions"] for _ in range(self._num_players)
      )

    else:
      raise ValueError("Unsupported game type!")

    self._rngkey, init_key = jax.random.split(jax.random.key(seed), 2)
    self._last_data = self._sampler.sample_random(self._batch_size, init_key)

    self._eps_cap = (
      epsilon_max if epsilon_max is not None else self._sampler.z_m
    )

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
      learning_rate=learning_rate, weight_decay=weight_decay, mask=None
    )

    self._checkpointer = None
    # TODO: allow checkpointing
    self._save_every = save_every
    self._log_every = log_every
    if allow_checkpointing:
      self._checkpointer = ocp.StandardCheckpointer()

    self._logger = None

    if gradient_clipping:
      optimiser = optax.chain(
        # Clip the raw gradients first
        optax.adaptive_grad_clip(gradient_clipping),
        optimiser,
      )

    self._optimizer = nn.Optimizer(self._network, optimiser, wrt=nn.Param)
    self._exact_solver = utils.AsyncExactSolver()

    self._graphdef_network_opt = nn.graphdef((self._network, self._optimizer))
    self._graphdef_network = nn.graphdef(self._network)
    self.broadcast_fn = jax.vmap(samplers.broadcast)

    def _dual_loss_fn(
      network: nn.Module,
      data: Data,
    ) -> chex.Array:
      """Loss function for the NES-network."""

      broadcasted = self.broadcast_fn(data)
      stacked = samplers.stack(broadcasted, axis=1)

      duals = utils.batched_call(
        network,
        stacked,
        data.strat_mask_per_player,
      )
      # Recovering primal variables
      logits, strategy, epsilon, sum_duals, log_sum_exp = jax.vmap(
        recover_primals, in_axes=(0, 0, None, None, None, None)
      )(
        duals,
        data,
        self._mu,
        self._rho,
        self._eps_cap,
        self._mode.value,
      )

      # Total loss: Equation (7)
      loss = (
        log_sum_exp
        + self._eps_cap * jnp.sum(sum_duals, axis=-1)
        - self._rho * jnp.sum(epsilon, axis=-1)
      )

      primals = {
        "logits": logits,
        "strategy": strategy,
        "epsilon": epsilon,
      }
      return loss.mean(), (primals, duals)

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

      loss, (primals, duals) = self._loss_fn(policy_model, data)
      return loss, (primals, duals)

    return infer

  def _get_jitted_update(self) -> Callable:
    """Generate jittable update function."""

    @jax.jit
    def update(
      network_opt_state: nn.State,
      data: Data,
    ) -> tuple[chex.Array, nn.State, nn.State]:
      policy_model, optimiser = nn.merge(
        self._graphdef_network_opt, network_opt_state, copy=True
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

  def _logging_callback(self, batch: Data, primals: dict) -> dict:
    # Fast JAX-computed gap (always available immediately)
    if self._mode == networks.Mode.CCE:
      regret = jax.vmap(eu.compute_cce_gap)(
        batch.payoffs,
        primals["strategy"],
        batch.epsilon_target,
        joint_mask=batch.joint_mask,
      )
    else:
      regret = jax.vmap(eu.compute_ce_gap)(
        batch.payoffs,
        primals["strategy"],
        batch.epsilon_target,
        joint_mask=batch.joint_mask,
      )

    # Expensive exact solves: run in parallel via thread pool
    futures = self._exact_solver.submit_batch(
      eu.mwmre_solver,
      batch.payoffs,
      batch.welfare,
      primals["strategy"],
      batch.epsilon_target,
      batch.joint_mask,
      self._mu,
      self._rho,
      self._eps_cap,
      self._mode.name,
    )

    # Block here only for logging (still much faster than sequential)
    exact_stats = self._exact_solver.collect(futures, timeout=30.0)

    mean_batch_data = exact_stats
    mean_batch_data[f"{self._mode.name}_gap"] = regret.mean()
    return mean_batch_data

  def evaluate(
    self,
    step: int,
    batch_size: int,
    sampler: Optional[samplers.GameSampler] = None,
  ) -> dict:
    sampler = sampler if sampler is not None else self._sampler
    batch = sampler.sample_random(batch_size, jax.random.key(step), None)
    loss, (primals, duals) = self._infer_fn(nn.state(self._network), batch)
    log = self._logging_callback(batch, primals)
    log["eval_loss"] = loss
    for k, v in log.items():
      print(f"\t{k}:\t{v}")
    print()

    return log

  def solve(self, logging_path: etils.epath.Path = None) -> dict:
    """
    Executes the optimization loop to find the equilibrium.
    Returns the optimal dual variables and the recovered primal policy.
    """
    # Logger setup
    trackio.init(
      project="NES",
    )
    logger = None
    if logging_path is not None:
      logger = data_logger.DataLoggerJsonLines(
        logging_path, "NES_learner", True
      )

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

      if self._log_every > 0 and self._iteration % self._log_every == 0:
        # Cheap evaluation during training

        print(f"It. {self._iteration} | Loss {loss_val}")

        log = self.evaluate(self._iteration, self._batch_size)
        trackio.log({k: v.item() for k, v in log.items()}, step=self._iteration)
        if logger is not None:
          logger.write({k: v.item() for k, v in log.items()})

        logs.append(log)

    # Peform expensive evaluation after the training
    logs.append(self.evaluate(self._iteration, self._eval_batch_size))
    trackio.finish()

    return logs
