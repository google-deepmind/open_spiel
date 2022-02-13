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

# Lint as: python3
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

import collections
import datetime
import functools
import itertools
import json
import os
import random
import sys
import tempfile
import time
import traceback

import numpy as np

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import file_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


class TrajectoryState(object):
  """A particular point along a trajectory."""

  def __init__(self, observation, current_player, legals_mask, action, policy,
               value):
    self.observation = observation
    self.current_player = current_player
    self.legals_mask = legals_mask
    self.action = action
    self.policy = policy
    self.value = value


class Trajectory(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self):
    self.states = []
    self.returns = None

  def add(self, information_state, action, policy):
    self.states.append((information_state, action, policy))


class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)


class Config(collections.namedtuple(
    "Config", [
        "game",
        "path",
        "learning_rate",
        "weight_decay",
        "train_batch_size",
        "replay_buffer_size",
        "replay_buffer_reuse",
        "max_steps",
        "checkpoint_freq",
        "actors",
        "evaluators",
        "evaluation_window",
        "eval_levels",

        "uct_c",
        "max_simulations",
        "policy_alpha",
        "policy_epsilon",
        "temperature",
        "temperature_drop",

        "nn_model",
        "nn_width",
        "nn_depth",
        "observation_shape",
        "output_size",

        "quiet",
    ])):
  """A config for the model/experiment."""
  pass


def _init_model_from_config(config):
  return model_lib.Model.build_model(
      config.nn_model,
      config.observation_shape,
      config.output_size,
      config.nn_width,
      config.nn_depth,
      config.weight_decay,
      config.learning_rate,
      config.path)


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
      verbose=False,
      dont_return_chance_node=True)


def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
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
      trajectory.states.append(
          TrajectoryState(state.observation_tensor(), state.current_player(),
                          state.legal_actions_mask(), action, policy,
                          root.total_reward / root.explore_count))
      action_str = state.action_to_string(state.current_player(), action)
      actions.append(action_str)
      logger.opt_print("Player {} sampled action: {}".format(
          state.current_player(), action_str))
      state.apply_action(action)
  logger.opt_print("Next state:\n{}".format(state))

  trajectory.returns = state.returns()
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
def actor(*, config, game, logger, queue):
  """An actor process runner that generates games and returns trajectories."""
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Initializing bots")
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  bots = [
      _init_bot(config, game, az_evaluator, False),
      _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count():
    if not update_checkpoint(logger, queue, model, az_evaluator):
      return
    queue.put(_play_game(logger, game_num, game, bots, config.temperature,
                         config.temperature_drop))


@watcher
def evaluator(*, game, config, logger, queue):
  """A process that plays the latest checkpoint vs standard MCTS."""
  results = Buffer(config.evaluation_window)
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
            verbose=False,
            dont_return_chance_node=True)
    ]
    if az_player == 1:
      bots = list(reversed(bots))

    trajectory = _play_game(logger, game_num, game, bots, temperature=1,
                            temperature_drop=0)
    results.append(trajectory.returns[az_player])
    queue.put((difficulty, trajectory.returns[az_player]))

    logger.print("AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}".format(
        trajectory.returns[az_player],
        trajectory.returns[1 - az_player],
        len(results), np.mean(results.data)))


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger):
  """A learner that consumes the replay buffer and trains the network."""
  logger.also_to_stdout = True
  replay_buffer = Buffer(config.replay_buffer_size)
  learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
  logger.print("Initializing model")
  model = _init_model_from_config(config)
  logger.print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                           config.nn_depth))
  logger.print("Model size:", model.num_trainable_variables, "variables")
  save_path = model.save_checkpoint(0)
  logger.print("Initial checkpoint:", save_path)
  broadcast_fn(save_path)

  data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

  stage_count = 7
  value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
  value_predictions = [stats.BasicStats() for _ in range(stage_count)]
  game_lengths = stats.BasicStats()
  game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
  outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
  evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
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
    num_trajectories = 0
    num_states = 0
    for trajectory in trajectory_generator():
      num_trajectories += 1
      num_states += len(trajectory.states)
      game_lengths.add(len(trajectory.states))
      game_lengths_hist.add(len(trajectory.states))

      p1_outcome = trajectory.returns[0]
      if p1_outcome > 0:
        outcomes.add(0)
      elif p1_outcome < 0:
        outcomes.add(1)
      else:
        outcomes.add(2)

      replay_buffer.extend(
          model_lib.TrainInput(
              s.observation, s.legals_mask, s.policy, p1_outcome)
          for s in trajectory.states)

      for stage in range(stage_count):
        # Scale for the length of the game
        index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
        n = trajectory.states[index]
        accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
        value_accuracies[stage].add(1 if accurate else 0)
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
    # the actors. It only allows numbers, so use -1 as "latest".
    save_path = model.save_checkpoint(
        step if step % config.checkpoint_freq == 0 else -1)
    losses = sum(losses, model_lib.Losses(0, 0, 0)) / len(losses)
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
    logger.print("Buffer size: {}. States seen: {}".format(
        len(replay_buffer), replay_buffer.total_seen))

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
        "total_states": replay_buffer.total_seen,
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
            "count": evals[0].total_seen,
            "results": [sum(e.data) / len(e) if e else 0 for e in evals],
        },
        "batch_size": batch_size_stats.as_dict,
        "batch_size_hist": [0, 1],
        "loss": {
            "policy": losses.policy,
            "value": losses.value,
            "l2reg": losses.l2,
            "sum": losses.total,
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

    broadcast_fn(save_path)


def alpha_zero(config: Config):
  """Start all the worker processes for a full alphazero setup."""
  game = pyspiel.load_game(config.game)
  config = config._replace(
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
  if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    raise ValueError("Game must be deterministic.")

  path = config.path
  if not path:
    path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
    config = config._replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit("{} isn't a directory".format(path))
  print("Writing logs and checkpoints to:", path)
  print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                    config.nn_depth))

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

  actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                         "num": i})
            for i in range(config.actors)]
  evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                 "num": i})
                for i in range(config.evaluators)]

  def broadcast(msg):
    for proc in actors + evaluators:
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
