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
though each at a different difficulty (ie number of simulations for MCTS).

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
import os
import sys
import tempfile
import time
import traceback

import numpy as np

import multiprocessing
from open_spiel.python.algorithms import mcts

from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero.utils import api_selector, TrainInput, Losses
from open_spiel.python.algorithms.alpha_zero import replay_buffer as buffer_lib

import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats

import chex
import jax.numpy as jnp

# Set the start method to 'spawn' for CUDA compatibility and monkey-patch
# to prevent OpenSpiel from overriding it.
# See:
# https://github.com/google-deepmind/open_spiel/issues/1326
try:
    # jax and TF aren't compatible with the `fork` strategy due to the inner multitheading
    # see:
    # https://github.com/jax-ml/jax/issues/1805
    multiprocessing.set_start_method("spawn", force=True)

    # Monkey-patch to prevent OpenSpiel from changing the start method.
    def _do_nothing_set_start_method(method, force=False):
        pass

    multiprocessing.set_start_method = _do_nothing_set_start_method
except RuntimeError:
    # This may be raised in child processes where the context is already set.
    pass

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
  return api_selector(config.nn_api_version).Model.build_model(
    config.nn_model,
    config.observation_shape,
    config.output_size,
    config.nn_width,
    config.nn_depth,
    config.weight_decay,
    config.learning_rate,
    config.path,
    decouple_weight_decay=config.decouple_weight_decay
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
        logger.print("\n".join([
            "",
            " Exception caught ".center(60, "="),
            traceback.format_exc(),
            "=" * 60,
        ]))
        print("Exception caught in {}: {}".format(name, e))
        raise
      finally:
        logger.print("{} exiting".format(name))
        print("{} exiting".format(name))
  return _watcher


def _init_bot(config, game, evaluator_, evaluation):
  """Initializes a bot."""
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
      dont_return_chance_node=True
    )


def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
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
      policy = policy**(1 / temperature)
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
          value=root.total_reward / root.explore_count
        )
      )
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      logger.opt_print(f"Player {state.current_player()} sampled action: {action_str}")
      state.apply_action(action)
  logger.opt_print("Next state:\n{}".format(state))

  trajectory = Trajectory(
    states=trajectory_states,
    returns=state.returns()
  )

  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory


def update_checkpoint(logger, queue, model, az_evaluator):
  """Read the queue for a checkpoint to load, or an exit signal."""
  path = None
  while True:  # Get the last message, ignore intermediate ones.
    try:
      path = queue.get_nowait()
    except spawn.Empty:
      break
  if path:
    logger.print("Inference cache:", az_evaluator.cache_info())
    logger.print("Loading checkpoint", path)
    model.load_checkpoint(path)
    az_evaluator.clear_cache()
  elif path is not None:  # Empty string means stop this process.
    return False
  return True


@watcher
def actor(*, config: Config, game, logger, queue):
  """An actor process runner that generates games and returns trajectories."""
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  bots = [
    _init_bot(config, game, az_evaluator, False),
    _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count(1):
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return
    queue.put(_play_game(logger, game_num, game, bots, config.temperature,
                         config.temperature_drop))


@watcher
def evaluator(*, game, config, logger, queue):
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = buffer_lib.Buffer(config.evaluation_window, force_cpu=True)

  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  random_evaluator = mcts.RandomRolloutEvaluator()

  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
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
          dont_return_chance_node=True
        )
    ]
    if az_player == 1:
      bots = list(reversed(bots))

    trajectory = _play_game(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0)
    results.append(jnp.asarray(trajectory.returns[az_player]))
    queue.put((difficulty, jnp.asarray(trajectory.returns[az_player])))

    logger.print(f"AZ: {trajectory.returns[az_player]},\
      MCTS: {trajectory.returns[1 - az_player]},\
      AZ avg/{len(results)}: {jnp.mean(results.data):.3f}"
    )


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger):
  """A learner that consumes the replay buffer and trains the network."""
  logger.also_to_stdout = True

  replay_buffer = buffer_lib.Buffer(config.replay_buffer_size)
  learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print(f"Model type: {config.nn_model}({config.nn_width}, {config.nn_depth})")
  logger.print("Model size:", model.num_trainable_variables, "variables")
  
  save_path = model.save_checkpoint(0)
  logger.print("Initial checkpoint:", save_path)
  broadcast_fn(str(save_path))

  data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

  stage_count = 7
  value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
  value_predictions = [stats.BasicStats() for _ in range(stage_count)]
  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
  #evaluation is yet on cpu
  evals = [buffer_lib.Buffer(config.evaluation_window, force_cpu=True) for _ in range(config.eval_levels)]
  total_trajectories = 0

  def trajectory_generator():
    """Merge all the actor queues into a single generator."""
    while True:
      found = 0
      for actor_process in actors:
        try:
          yield actor_process.queue.get_nowait()
        except spawn.Empty:
          pass
        else:
          found += 1
      if found == 0:
        time.sleep(0.01)  # 10ms

  def collect_trajectories():
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
        replay_buffer.append(TrainInput(
            observation=jnp.asarray(s.observation, dtype=jnp.float32), 
            legals_mask=jnp.asarray(s.legals_mask,dtype=jnp.bool),
            policy=jnp.asarray(s.policy, dtype=jnp.float32),
            value=game_outcome
          ) 
        )
            
      for stage in range(stage_count):
        # Scale for the length of the game
        index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
        n = trajectory.states[index]
        # Let's leave it out while we're testing
        # value_accuracies[stage].add(int(
        #   (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
        # ))
        value_accuracies[stage].add((n.value-trajectory.returns[n.current_player])**2)
        value_predictions[stage].add(abs(n.value))

      if num_states >= learn_rate:
        break
    return num_trajectories, num_states

  def learn(step):
    """Sample from the replay buffer, update weights and save a checkpoint."""
    losses = [] 

    for _ in range(len(replay_buffer) // config.train_batch_size):
      data = replay_buffer.sample(config.train_batch_size)
      losses.append(model.update(data))

    # Always save a checkpoint, either for keeping or for loading the weights to
    # the actors. We only allow numbers, so use -1 as "latest".
    save_path = model.save_checkpoint(step if step % config.checkpoint_freq == 0 else -1)
    
    # for an unlucky case when the agent didn't collect enough transitions
    if len(losses):
      losses = sum(losses, Losses(policy=0, value=0, l2=0)) / len(losses)
    else:
      losses = Losses(policy=0, value=0, l2=0)

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
    logger.print(
        ("Collected {:5} states from {:3} games, {:.1f} states/s. "
         "{:.1f} states/(s*actor), game length: {:.1f}").format(
             num_states, num_trajectories, num_states / seconds,
             num_states / (config.actors * seconds),
             num_states / num_trajectories))
    logger.print(f"Buffer size: {len(replay_buffer)}. States seen: {replay_buffer.total_seen}")

    save_path, losses = learn(step)

    for eval_process in evaluators:
      while True:
        try:
          difficulty, outcome = eval_process.queue.get_nowait()
          evals[difficulty].append(outcome)
        except spawn.Empty:
          break

    batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
    batch_size_stats.add(1)
    data_log.write({
      "step": step,
      "total_states": replay_buffer.total_seen.item(),
      "states_per_s": num_states / seconds,
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
        "count": evals[0].total_seen.item(),
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
      "cache": {  # Null stats because it's hard to report between processes.
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

    broadcast_fn(str(save_path))


def alpha_zero(config: Config):
  # NOTE: a single device accelearation is currently supported

  """Start all the worker processes for a full alphazero setup."""
  game = pyspiel.load_game(config.game)
  config = config.replace(
      observation_shape=game.observation_tensor_shape(),
      output_size=game.num_distinct_actions())

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
    path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
    config = config.replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit(f"{path} isn't a directory")
  print(f"Writing logs and checkpoints to: {path}")
  print(f"Model type: {config.nn_model}(width={config.nn_width}, depth={config.nn_depth})")

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config.__dict__, indent=2, sort_keys=True) + "\n")

  actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                        "num": i})
            for i in range(config.actors)]
  
  evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                "num": i})
                for i in range(config.evaluators)]

  def broadcast(msg):
    for proc in actors+evaluators:
      proc.queue.put(msg)

  try:
    learner(game=game, config=config, actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators, broadcast_fn=broadcast)
  except (KeyboardInterrupt, EOFError):
    print("Caught a KeyboardInterrupt, stopping early.")
  finally:
    broadcast("")
    # for actor processes to join we have to make sure that their q_in is empty,
    # including backed up items
    for proc in actors:
      while proc.exitcode is None:
        while not proc.queue.empty():
          proc.queue.get_nowait()
        proc.join(JOIN_WAIT_DELAY)
    for proc in evaluators:
      proc.join()
