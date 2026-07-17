# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic AlphaZero implementation.

This implements the AlphaZero training algorithm. It spawns N actors which feed
trajectories into a replay buffer which are consumed by a learner. The learner
generates new weights, saves a checkpoint, and tells the actors to update. There
are also M evaluators running games continuously against a standard MCTS+Solver,
though each at a different difficulty (i.e. number of simulations for MCTS).

Due to the multi-process nature of this algorithm the logs are written to files,
one per process. The learner logs are also output to stdout. The checkpoints are
also written to the same directory.

Links to relevant articles/papers:
  https://deepmind.com/blog/article/alphago-zero-starting-scratch has an open
    access link to the AlphaGo Zero nature paper.
  https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
    has an open access link to the AlphaZero science paper.
"""

import datetime
import functools
import itertools
import json
import threading
import queue
import os
import sys
import tempfile
import time
import traceback
from typing import Any

import chex
import jax.numpy as jnp
import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import replay_buffer as buffer_lib
from open_spiel.python.algorithms.alpha_zero import utils
import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import stats


# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


@chex.dataclass(frozen=True)
class TrajectoryState:
  """A particular point along a trajectory."""

  observation: chex.Array
  current_player: chex.Array
  legals_mask: chex.Array
  action: chex.Array
  policy: chex.Array
  value: chex.Array


@chex.dataclass(frozen=True)
class Trajectory:
  states: list[TrajectoryState]
  returns: chex.Array


@chex.dataclass(frozen=True)
class Config:
  """A config for the model/experiment."""

  game: str
  path: str
  learning_rate: float
  weight_decay: float
  decouple_weight_decay: bool
  train_batch_size: int
  replay_buffer_size: int
  replay_buffer_reuse: bool
  max_steps: int
  checkpoint_freq: int
  actors: int

  evaluators: int
  evaluation_window: int
  eval_levels: int
  uct_c: float
  max_simulations: int

  policy_alpha: float
  policy_epsilon: float
  temperature: float
  temperature_drop: float

  nn_model: str
  nn_width: int
  nn_depth: int
  observation_shape: chex.Shape
  output_size: int
  verbose: bool
  quiet: bool

  nn_api_version: str = "nnx"


def _init_model_from_config(config: Config):
  return utils.api_selector(config.nn_api_version).Model.build_model(
      config.nn_model,
      config.observation_shape,
      config.output_size,
      config.nn_width,
      config.nn_depth,
      config.weight_decay,
      config.learning_rate,
      config.path,
      decouple_weight_decay=config.decouple_weight_decay,
  )


def watcher(fn):
  """A decorator to fn/processes that gives a logger and logs exceptions."""

  @functools.wraps(fn)
  def _watcher(*, config, num=None, **kwargs):
    """Wrap the decorated function."""
    name = fn.__name__
    if num is not None:
      name += "-" + str(num)
    with file_logger.FileLogger(config.path, name, config.quiet) as logger:
      print("{} started".format(name))
      logger.print("{} started".format(name))
      try:
        return fn(config=config, logger=logger, **kwargs)
      except Exception as e:
        logger.print(
            "\n".join([
                "",
                " Exception caught ".center(60, "="),
                traceback.format_exc(),
                "=" * 60,
            ])
        )
        print("Exception caught in {}: {}".format(name, e))
        raise
      finally:
        logger.print("{} exiting".format(name))
        print("{} exiting".format(name))

  return _watcher


def _init_bot(
    config: Config, game: Any, evaluator_: mcts.Evaluator, evaluation: bool
) -> mcts.MCTSBot:
  """Initialises a MCTS bot."""
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      evaluator_,
      solve=False,
      dirichlet_noise=noise,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=config.verbose,
      dont_return_chance_node=True,
  )


def _play_game(
    logger: Any,
    game_num: int,
    game: Any,
    bots: tuple[mcts.MCTSBot, mcts.MCTSBot],
    temperature: float,
    temperature_drop: int,
) -> Trajectory:
  """Play one game, return the trajectory."""
  trajectory_states = []
  actions = []
  state = game.new_initial_state()
  random_state = np.random.RandomState()
  logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.opt_print("Initial state:\n{}".format(state))
  while not state.is_terminal():
    if state.is_chance_node():
      # For chance nodes, rollout according to chance node's probability
      # distribution
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      action = random_state.choice(action_list, p=prob_list)
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      state.apply_action(action)
    else:
      root = bots[state.current_player()].mcts_search(state)
      policy = np.zeros(game.num_distinct_actions())
      for c in root.children:
        policy[c.action] = c.explore_count
      policy = policy ** (1 / temperature)
      policy /= policy.sum()
      if len(actions) >= temperature_drop:
        action = root.best_child().action
      else:
        action = np.random.choice(len(policy), p=policy)

      trajectory_states.append(
          TrajectoryState(
              observation=state.observation_tensor(),
              current_player=state.current_player(),
              legals_mask=state.legal_actions_mask(),
              action=action,
              policy=policy,
              value=root.total_reward / root.explore_count,
          )
      )
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      logger.opt_print(
          f"Player {state.current_player()} sampled action: {action_str}"
      )
      state.apply_action(action)
  logger.opt_print("Next state:\n{}".format(state))

  trajectory = Trajectory(states=trajectory_states, returns=state.returns())

  logger.print(
      "Game {}: Returns: {}; Actions: {}".format(
          game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)
      )
  )
  return trajectory


@watcher
def actor(
    *,
    config: Config,
    game: pyspiel.Game,
    model: Any,
    logger: Any,
    out_queue: queue.Queue,
    stop_event: threading.Event
) -> None:
  """An actor process runner that generates games and returns trajectories."""

  logger.print("Initialising bots")
  # Per-thread evaluator (own LRU cache) referencing shared model
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  
  bots = [
      _init_bot(config, game, az_evaluator, False),
      _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count(1):
    if stop_event.is_set():
      return
    traj = _play_game(
        logger, game_num, game, bots,
        config.temperature, config.temperature_drop,
    )
    out_queue.put(traj)


@watcher
def evaluator(
    *, 
    game: pyspiel.Game, 
    model: Any,
    config: Config, 
    logger: Any, 
    out_queue: queue.Queue,
    stop_event: threading.Event
) -> None:
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = buffer_lib.Buffer(config.evaluation_window, force_cpu=True)

  logger.print("Initialising bots")
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  random_evaluator = mcts.RandomRolloutEvaluator()

  for game_num in itertools.count():
    if stop_event.is_set():
      return
    az_player = game_num % 2
    difficulty = (game_num // 2) % config.eval_levels
    max_simulations = int(config.max_simulations * (10 ** (difficulty / 2)))
    bots = [
        _init_bot(config, game, az_evaluator, True),
        mcts.MCTSBot(
            game,
            config.uct_c,
            max_simulations,
            random_evaluator,
            solve=True,
            verbose=config.verbose,
            dont_return_chance_node=True,
        ),
    ]
    if az_player == 1:
      bots = list(reversed(bots))

    trajectory = _play_game(
        logger, game_num, game, bots, temperature=1, temperature_drop=0
    )
    result = jnp.asarray(trajectory.returns[az_player])
    results.append(result)
    out_queue.put((difficulty, result))

    logger.print(
        f"AZ: {trajectory.returns[az_player]},      MCTS:"
        f" {trajectory.returns[1 - az_player]},     AZ avg/{len(results)}:"
        f" {jnp.mean(results.data):.3f}"
    )


@watcher
def learner(
    *,
    game: pyspiel.Game,
    model: Any,
    config: Config,
    trajectory_queue: queue.Queue, 
    eval_queue: queue.Queue,
    eval_lock: threading.Lock,
    logger: Any,
) -> None:
  """A learner that consumes the replay buffer and trains the network."""
  logger.also_to_stdout = True

  replay_buffer = buffer_lib.Buffer(config.replay_buffer_size)
  learn_rate = config.replay_buffer_size // config.replay_buffer_reuse

  data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

  stage_count = config.eval_levels
  value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
  value_predictions = [stats.BasicStats() for _ in range(stage_count)]
  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
  # evaluation is yet on cpu
  evals = [
      buffer_lib.Buffer(config.evaluation_window, force_cpu=True)
      for _ in range(config.eval_levels)
  ]
  total_trajectories = 0

  def drain_queue(q):
    while True:
      try:
        yield q.get_nowait()
      except queue.Empty:
        break

  def trajectory_generator():
    """Yield trajectories from the shared actor queue."""
    while True:
      yield trajectory_queue.get()

  def collect_trajectories() -> tuple[int, int]:
    """Collects the trajectories from actors into the replay buffer."""

    logger.print("Collecting trajectories")
    num_trajectories = 0
    num_states = 0
    for trajectory in trajectory_generator():
      num_trajectories += 1
      num_states += len(trajectory.states)
      game_lengths.add(len(trajectory.states))
      game_lengths_hist.add(len(trajectory.states))

      # we learn from perspective of only the first player,
      # rather than rotating
      game_outcome = trajectory.returns[0]
      if game_outcome > 0:
        outcomes.add(0)
      elif game_outcome < 0:
        outcomes.add(1)
      else:
        outcomes.add(2)

      for s in trajectory.states:
        replay_buffer.append(
            utils.TrainInput(
                observation=jnp.asarray(s.observation, dtype=jnp.float32),
                legals_mask=jnp.asarray(s.legals_mask, dtype=jnp.bool),
                policy=jnp.asarray(s.policy, dtype=jnp.float32),
                value=jnp.asarray(game_outcome, dtype=jnp.float32),
            )
        )

      for stage in range(stage_count):
        # Scale for the length of the game
        index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
        n = trajectory.states[index]
        # Replaced by the MSE to get a smoother estimate of the value learning
        value_accuracies[stage].add(
            (n.value - trajectory.returns[n.current_player]) ** 2
        )
        value_predictions[stage].add(abs(n.value))

      if num_states >= learn_rate:
        break
    return num_trajectories, num_states

  def learn(step: int | str) -> tuple[str, utils.Losses]:
    """Sample from the replay buffer, update weights and save a checkpoint."""
    losses = []
    for _ in range(len(replay_buffer) // config.train_batch_size):
      data = replay_buffer.sample(config.train_batch_size)
      losses.append(model.update(data))

    # Always save a checkpoint, either for keeping or for loading the weights to
    # the actors. We only allow numbers, so use -1 as the "latest".
    # with eval_lock:  # blocks all actor inference
    save_path = model.save_checkpoint(
        step if step % config.checkpoint_freq == 0 else -1
    )

    # for an unlucky case when the agent hasn't collected enough transitions
    if losses:
      losses = sum(losses, utils.Losses(policy=0, value=0, l2=0)) / len(losses)
    else:
      losses = utils.Losses(policy=0, value=0, l2=0)

    logger.print(losses)
    logger.print("Checkpoint saved:", save_path)
    return save_path, losses

  last_time = time.time() - 60
  for step in itertools.count(1):
    for value_accuracy in value_accuracies:
      value_accuracy.reset()
    for value_prediction in value_predictions:
      value_prediction.reset()
    game_lengths.reset()
    game_lengths_hist.reset()
    outcomes.reset()

    num_trajectories, num_states = collect_trajectories()
    total_trajectories += num_trajectories
    now = time.time()
    seconds = now - last_time
    last_time = now

    logger.print("Step:", step)
    logger.print((
        f"Collected {num_states:5} states from {num_trajectories:3} games, "
        f"{(num_states / seconds):.1f} states/s. "
        f"{(num_states / (config.actors * seconds)):.1f} states/(s*actor), "
        f"game length: {(num_states / num_trajectories):.1f}"
    ))
    logger.print(
        f"Buffer size: {len(replay_buffer)}. States seen:"
        f" {replay_buffer.total_seen}"
    )

    save_path, losses = learn(step)

    for difficulty, outcome in drain_queue(eval_queue):
      evals[difficulty].append(outcome)

    batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
    batch_size_stats.add(1)
    data_log.write({
        "step": step,
        "total_states": replay_buffer.total_seen,
        "states_per_s_actor": num_states / (config.actors * seconds),
        "total_trajectories": total_trajectories,
        "trajectories_per_s": num_trajectories / seconds,
        "queue_size": 0,  # Only available in C++.
        "game_length": game_lengths.as_dict,
        "game_length_hist": game_lengths_hist.data,
        "outcomes": outcomes.data,
        "value_accuracy": [v.as_dict for v in value_accuracies],
        "value_prediction": [v.as_dict for v in value_predictions],
        "eval": {
            "count": evals[0].total_seen,
            "results": [np.mean(e.data).item() if e else 0 for e in evals],
        },
        "batch_size": batch_size_stats.as_dict,
        "batch_size_hist": [0, 1],
        "loss": {
            "policy": float(losses.policy),
            "value": float(losses.value),
            "l2reg": float(losses.l2),
            "sum": float(losses.total),
        },
        "cache": {  # Null stats because it's hard to report between workers.
            "size": 0,
            "max_size": 0,
            "usage": 0,
            "requests": 0,
            "requests_per_s": 0,
            "hits": 0,
            "misses": 0,
            "misses_per_s": 0,
            "hit_rate": 0,
        },
    })
    logger.print()

    if config.max_steps > 0 and step >= config.max_steps:
      break

def alpha_zero(config: Config) -> None:
  # NOTE: a single device accelearation is currently supported

  """Start all the worker threads for a full alphazero setup."""
  game = pyspiel.load_game(config.game)
  config = config.replace(
      observation_shape=game.observation_tensor_shape(),
      output_size=game.num_distinct_actions(),
  )

  print("Starting game", config.game)
  if game.num_players() != 2:
    sys.exit("AlphaZero can only handle 2-player games.")
  game_type = game.get_type()
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    raise ValueError("Game must have terminal rewards.")
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Game must have sequential turns.")

  path = config.path
  if not path:
    path = tempfile.mkdtemp(
        prefix="az-{}-{}-".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game
        )
    )
    config = config.replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit(f"{path} isn't a directory")
  print(f"Writing logs and checkpoints to: {path}")
  print(
      f"Model type: {config.nn_model}(width={config.nn_width},"
      f" depth={config.nn_depth})"
  )

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config.__dict__, indent=2, sort_keys=True) + "\n")

  print("Initializing model")
  model = _init_model_from_config(config)
  print(
      f"Model type: {config.nn_model}({config.nn_width}, {config.nn_depth})"
  )
  print("Model size:", model.num_trainable_variables, "variables")

  save_path = model.save_checkpoint(0)
  print("Initial checkpoint:", save_path)

  # Shared queues
  trajectory_queue = queue.Queue()
  eval_queue = queue.Queue()
  stop_event = threading.Event()
  eval_lock = threading.Lock()
  # Spawn actor threads, and each gets its own Game instance.
  actors = [
    threading.Thread(
        target=actor,
        kwargs={
            "config": config,
            "game": pyspiel.load_game(config.game),  # per-thread
            "model": model,
            "out_queue": trajectory_queue,
            "num": i,
            "stop_event": stop_event
        },
        daemon=True,  # so they die with main process
    )
    for i in range(config.actors)
  ]

  evaluators = [
    threading.Thread(
        target=evaluator,
        kwargs={
            "config": config,
            "game": pyspiel.load_game(config.game),
            "model": model,
            "out_queue": eval_queue,
            "num": i,
            "stop_event": stop_event
        },
        daemon=True,
    )
    for i in range(config.evaluators)
  ]

  for t in actors + evaluators:
    t.start()

  try:
    learner(
        game=game,
        config=config,
        model=model,
        trajectory_queue=trajectory_queue,
        eval_queue=eval_queue,
        eval_lock=eval_lock
      )
  except (KeyboardInterrupt, EOFError):
    print("Caught a KeyboardInterrupt, stopping early.")
  finally:
    stop_event.set()
    # Drain queues to unblock waiting threads, then join
    for q in (trajectory_queue, eval_queue):
      while not q.empty():
        try:
          q.get_nowait()
        except queue.Empty:
          break
    for t in actors + evaluators:
      t.join(timeout=1.0)

