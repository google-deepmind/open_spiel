# Copyright 2022 DeepMind Technologies Limited
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

"""An example of use of PPO.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
"""

# pylint: disable=g-importing-member
import collections
from datetime import datetime
import logging
import os
import random
import sys
import time
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import pyspiel
from open_spiel.python.pytorch.ppo import PPO
from open_spiel.python.pytorch.ppo import PPOAgent
from open_spiel.python.pytorch.ppo import PPOAtariAgent
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import ObservationType
from open_spiel.python.vector_env import SyncVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name",
                    os.path.basename(__file__).rstrip(".py"),
                    "the name of this experiment")
flags.DEFINE_string("game_name", "atari", "the id of the OpenSpiel game")
flags.DEFINE_float("learning_rate", 2.5e-4,
                   "the learning rate of the optimizer")
flags.DEFINE_integer("seed", 1, "seed of the experiment")
flags.DEFINE_integer("total_timesteps", 10_000_000,
                     "total timesteps of the experiments")
flags.DEFINE_integer("eval_every", 10, "evaluate the policy every N updates")
flags.DEFINE_bool("torch_deterministic", True,
                  "if toggled, `torch.backends.cudnn.deterministic=False`")
flags.DEFINE_bool("cuda", True, "if toggled, cuda will be enabled by default")

# Atari specific arguments
flags.DEFINE_string("gym_id", "BreakoutNoFrameskip-v4",
                    "the id of the environment")
flags.DEFINE_bool(
    "capture_video", False,
    "whether to capture videos of the agent performances (check out `videos` folder)"
)

# Algorithm specific arguments
flags.DEFINE_integer("num_envs", 8, "the number of parallel game environments")
flags.DEFINE_integer(
    "num_steps", 128,
    "the number of steps to run in each environment per policy rollout")
flags.DEFINE_bool(
    "anneal_lr", True,
    "Toggle learning rate annealing for policy and value networks")
flags.DEFINE_bool("gae", True, "Use GAE for advantage computation")
flags.DEFINE_float("gamma", 0.99, "the discount factor gamma")
flags.DEFINE_float("gae_lambda", 0.95,
                   "the lambda for the general advantage estimation")
flags.DEFINE_integer("num_minibatches", 4, "the number of mini-batches")
flags.DEFINE_integer("update_epochs", 4, "the K epochs to update the policy")
flags.DEFINE_bool("norm_adv", True, "Toggles advantages normalization")
flags.DEFINE_float("clip_coef", 0.1, "the surrogate clipping coefficient")
flags.DEFINE_bool(
    "clip_vloss", True,
    "Toggles whether or not to use a clipped loss for the value function, as per the paper"
)
flags.DEFINE_float("ent_coef", 0.01, "coefficient of the entropy")
flags.DEFINE_float("vf_coef", 0.5, "coefficient of the value function")
flags.DEFINE_float("max_grad_norm", 0.5,
                   "the maximum norm for the gradient clipping")
flags.DEFINE_float("target_kl", None, "the target KL divergence threshold")


def setup_logging():
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter(
      "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
  handler.setFormatter(formatter)
  root.addHandler(handler)


def make_single_atari_env(gym_id,
                          seed,
                          idx,
                          capture_video,
                          run_name,
                          use_episodic_life_env=True):
  """Make the single-agent Atari environment."""

  def gen_env():
    game = pyspiel.load_game(
        "atari", {
            "gym_id": gym_id,
            "seed": seed,
            "idx": idx,
            "capture_video": capture_video,
            "run_name": run_name,
            "use_episodic_life_env": use_episodic_life_env
        })
    return Environment(
        game,
        chance_event_sampler=ChanceEventSampler(seed=seed),
        observation_type=ObservationType.OBSERVATION)

  return gen_env


def make_single_env(game_name, seed):

  def gen_env():
    game = pyspiel.load_game(game_name)
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

  return gen_env


def main(_):
  setup_logging()

  batch_size = int(FLAGS.num_envs * FLAGS.num_steps)

  if FLAGS.game_name == "atari":
    # pylint: disable=unused-import
    # pylint: disable=g-import-not-at-top
    import open_spiel.python.games.atari

  current_day = datetime.now().strftime("%d")
  current_month_text = datetime.now().strftime("%h")
  run_name = f"{FLAGS.game_name}__{FLAGS.exp_name}__"
  if FLAGS.game_name == "atari":
    run_name += f"{FLAGS.gym_id}__"
  run_name += f"{FLAGS.seed}__{current_month_text}__{current_day}__{int(time.time())}"

  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" %
      ("\n".join([f"|{key}|{value}|" for key, value in vars(FLAGS).items()])),
  )

  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)
  torch.backends.cudnn.deterministic = FLAGS.torch_deterministic

  device = torch.device(
      "cuda" if torch.cuda.is_available() and FLAGS.cuda else "cpu")
  logging.info("Using device: %s", str(device))

  if FLAGS.game_name == "atari":
    envs = SyncVectorEnv([
        make_single_atari_env(FLAGS.gym_id, FLAGS.seed + i, i, False,
                              run_name)() for i in range(FLAGS.num_envs)
    ])
    agent_fn = PPOAtariAgent
  else:
    envs = SyncVectorEnv([
        make_single_env(FLAGS.game_name, FLAGS.seed + i)()
        for i in range(FLAGS.num_envs)
    ])
    agent_fn = PPOAgent

  game = envs.envs[0]._game  # pylint: disable=protected-access
  info_state_shape = game.observation_tensor_shape()

  num_updates = FLAGS.total_timesteps // batch_size
  agent = PPO(
      input_shape=info_state_shape,
      num_actions=game.num_distinct_actions(),
      num_players=game.num_players(),
      player_id=0,
      num_envs=FLAGS.num_envs,
      steps_per_batch=FLAGS.num_steps,
      num_minibatches=FLAGS.num_minibatches,
      update_epochs=FLAGS.update_epochs,
      learning_rate=FLAGS.learning_rate,
      gae=FLAGS.gae,
      gamma=FLAGS.gamma,
      gae_lambda=FLAGS.gae_lambda,
      normalize_advantages=FLAGS.norm_adv,
      clip_coef=FLAGS.clip_coef,
      clip_vloss=FLAGS.clip_vloss,
      entropy_coef=FLAGS.ent_coef,
      value_coef=FLAGS.vf_coef,
      max_grad_norm=FLAGS.max_grad_norm,
      target_kl=FLAGS.target_kl,
      device=device,
      writer=writer,
      agent_fn=agent_fn,
  )

  n_reward_window = 50
  recent_rewards = collections.deque(maxlen=n_reward_window)
  time_step = envs.reset()
  for update in range(num_updates):
    for _ in range(FLAGS.num_steps):
      agent_output = agent.step(time_step)
      time_step, reward, done, unreset_time_steps = envs.step(
          agent_output, reset_if_done=True)

      if FLAGS.game_name == "atari":
        # Get around the fact that
        # stable_baselines3.common.atari_wrappers.EpisodicLifeEnv will modify
        # rewards at the LIFE and not GAME level by only counting
        # rewards of finished episodes
        for ts in unreset_time_steps:
          info = ts.observations.get("info")
          if info and "episode" in info:
            real_reward = info["episode"]["r"]
            writer.add_scalar("charts/player_0_training_returns", real_reward,
                              agent.total_steps_done)
            recent_rewards.append(real_reward)
      else:
        for ts in unreset_time_steps:
          if ts.last():
            real_reward = ts.rewards[0]
            writer.add_scalar("charts/player_0_training_returns", real_reward,
                              agent.total_steps_done)
            recent_rewards.append(real_reward)

      agent.post_step(reward, done)

    if FLAGS.anneal_lr:
      agent.anneal_learning_rate(update, num_updates)

    agent.learn(time_step)

    if update % FLAGS.eval_every == 0:
      logging.info("-" * 80)
      logging.info("Step %s", agent.total_steps_done)
      logging.info("Summary of past %i rewards\n %s",
                   n_reward_window,
                   pd.Series(recent_rewards).describe())

  writer.close()
  logging.info("All done. Have a pleasant day :)")


if __name__ == "__main__":
  app.run(main)
