# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Benchmark performance of games by counting the number of rollouts in a fixed
time frame.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
from absl import logging
import time
import numpy as np
import pandas as pd

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("games", "*",
                    "Benchmark only specific games (semicolon separated). "
                    "Use * to benchmark all (loadable) games.")
flags.DEFINE_float("time_limit", 10.,
                   "Time limit per game (in seconds).")
flags.DEFINE_integer("give_up_after", 100,
                     "Give up rollout when the history length is exceeded.")


def rollout_until_timeout(game, time_limit, give_up_after):
  is_time_out = lambda t: time.time() - t > time_limit
  num_rollouts = 0
  num_giveups = 0
  start = time.time()
  while not is_time_out(start):
    state = game.new_initial_state()
    while not state.is_terminal():
      if len(state.history()) > give_up_after:
        num_giveups += 1
        break
      action = random.choice(state.legal_actions(state.current_player()))
      state.apply_action(action)
    num_rollouts += 1
  time_elapsed = time.time() - start
  return num_rollouts, num_giveups, time_elapsed


def main(_):
  if FLAGS.games == "*":
    games_list = [game.short_name for game in pyspiel.registered_games()
                  if game.default_loadable
                  and game.short_name != "coop_box_pushing"]
  else:
    games_list = FLAGS.games.split(";")

  logging.info(f"Running benchmark for {len(games_list)} games.")
  logging.info(f"This will take approximately "
               f"{len(games_list) * FLAGS.time_limit} seconds.")

  rollouts = []
  giveups = []
  for game_name in games_list:
    logging.info(f"Running benchmark on {game_name}")
    game = pyspiel.load_game(game_name)
    num_rollouts, num_giveups, time_elapsed = rollout_until_timeout(
        game, FLAGS.time_limit, FLAGS.give_up_after)

    rollouts.append(num_rollouts / time_elapsed)
    giveups.append(num_giveups / time_elapsed)

  with pd.option_context('display.max_rows', None,
                         'display.max_columns', None,
                         'display.width', 200):
    df = pd.DataFrame(
        {"Game": games_list,
         "Rollouts/sec": rollouts,
         "Give ups/sec": giveups})

    print("---")
    print("Results for following benchmark configuration:")
    print("time_limit =", FLAGS.time_limit)
    print("give_up_after =", FLAGS.give_up_after)
    print("---")
    print(df)


if __name__ == "__main__":
  app.run(main)
