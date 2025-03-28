# Copyright 2023 DeepMind Technologies Limited
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

"""Run some analyses on some Atari data sets."""

# pylint: disable=unused-import

import sys
import time
from absl import app
from absl import flags
import numpy as np

import pyspiel
from open_spiel.python.voting import approval
from open_spiel.python.voting import base
from open_spiel.python.voting import borda
from open_spiel.python.voting import copeland
from open_spiel.python.voting import kemeny_young
from open_spiel.python.voting import maximal_lotteries
from open_spiel.python.voting import plurality
from open_spiel.python.voting import ranked_pairs
from open_spiel.python.voting import schulze
from open_spiel.python.voting import soft_condorcet_optimization as sco
from open_spiel.python.voting import stv
from open_spiel.python.voting.examples import atari_datasets

_DATASET_PATH_PREFIX = flags.DEFINE_string(
    "dataset_path_prefix", default=".", help="Where to find the dataset files")


def main(_):
  print("Loading dataset(s)...")
  dataset_filename = (_DATASET_PATH_PREFIX.value + "/" +
                      atari_datasets.RAINBOW_TABLE5)
  dataset = atari_datasets.parse_atari_table(dataset_filename)

  # If you load others, you can merge some columns from them like this:
  # dataset.add_column(dataset_ag57.get_column("random"), "random")
  # dataset.add_column(dataset_ag57.get_column("human"), "human")

  print(dataset.agent_names)
  print(dataset.game_names)
  print(f"Num agents: {len(dataset.agent_names)}")
  print(f"Num games: {len(dataset.game_names)}")

  # Alts for rainbow table 5:
  # dqn a3c ddqn prior-ddqn dueling-ddqn distrib-dqn noisy-dqn rainbow

  game_names = []
  profile = base.PreferenceProfile(alternatives=dataset.agent_names)
  for game_name, scores in dataset.table_data.items():
    profile.add_vote_from_values(scores)
    game_names.append(game_name)

  # Group up the profile and then print it to show that every vote is unique.
  profile.group()
  print(profile)

  print("Margin matrix:")
  margin_matrix = profile.margin_matrix()
  print(margin_matrix)
  print(
      "Weak Condorcet winners? "
      + f"{profile.condorcet_winner(False, margin_matrix)}"
  )
  print(
      "Strong Condorcet winner? "
      + f"{profile.condorcet_winner(True, margin_matrix)}"
  )

  voting_methods = [
      approval.ApprovalVoting(k=3),
      borda.BordaVoting(),
      copeland.CopelandVoting(),
      kemeny_young.KemenyYoungVoting(),
      maximal_lotteries.MaximalLotteriesVoting(iterative=True),
      plurality.PluralityVoting(),
      ranked_pairs.RankedPairsVoting(),
      schulze.SchulzeVoting(),
      stv.STVVoting(num_winners=3),
  ]
  for method in voting_methods:
    print("")
    print(method.name())
    outcome = method.run_election(profile)
    print(outcome.pretty_table_string())

  print("Soft Condorcet Optimization (Python):")
  py_sco_solver = sco.SoftCondorcetOptimizer(
      profile,
      batch_size=4,
      rating_lower_bound=-100.0,
      rating_upper_bound=100.0,
      temperature=1,
  )
  start_time = time.time()
  ratings, ranking = py_sco_solver.run_solver(10000, learning_rate=0.01)
  end_time = time.time()
  print(f"Time taken: {end_time - start_time}")
  alt_idx = profile.alternatives_dict
  for alt in ranking:
    print(f"  {alt}: {ratings[alt_idx[alt]]}")

  print("Soft Condorcet Optimization Sigmoid (C++):")
  cpp_sco_solver = pyspiel.sco.SoftCondorcetOptimizer(
      profile.to_list_of_tuples(),
      rating_lower_bound=-100.0,
      rating_upper_bound=100.0,
      batch_size=4,
      temperature=1,
      rng_seed=0,
  )
  start_time = time.time()
  cpp_sco_solver.run_solver(10000, learning_rate=0.01)
  end_time = time.time()
  print(f"Time taken: {end_time - start_time}")
  ratings_dict = cpp_sco_solver.ratings()
  for alt in ranking:
    print(f"  {alt}: {ratings_dict[alt]}")

  print("Soft Condorcet Optimization Fenchel-Young (C++):")
  cpp_fy_solver = pyspiel.sco.FenchelYoungOptimizer(
      profile.to_list_of_tuples(),
      rating_lower_bound=-100.0,
      rating_upper_bound=100.0,
      batch_size=4,
      temperature=1,
      rng_seed=0,
  )
  start_time = time.time()
  cpp_fy_solver.run_solver(10000, learning_rate=0.01)
  end_time = time.time()
  print(f"Time taken: {end_time - start_time}")
  ratings_dict = cpp_fy_solver.ratings()
  for alt in ranking:
    print(f"  {alt}: {ratings_dict[alt]}")


if __name__ == "__main__":
  app.run(main)
