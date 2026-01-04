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

"""Chat bot Arena dataset."""

# pylint: disable=unused-import

import collections
import sys
from absl import app
from absl import flags
import numpy as np
import pandas as pd
import pygraphviz as pgv

from open_spiel.python.utils import gfile

from open_spiel.python.algorithms import nash_averaging
from open_spiel.python.voting import approval
from open_spiel.python.voting import base
from open_spiel.python.voting import borda
from open_spiel.python.voting import copeland
from open_spiel.python.voting import kemeny_young
from open_spiel.python.voting import maximal_lotteries
from open_spiel.python.voting import plurality
from open_spiel.python.voting import ranked_pairs
from open_spiel.python.voting import schulze
from open_spiel.python.voting import stv


SEED = 23875711

# Downloaded from: https://lmsys.org/blog/2023-07-20-dataset/
DATASET_FILE = "/tmp/chatbot_arena_battles.csv"


def parse_battles_dataset(filter_ties=False):
  """Parse the data set from the raw CSV."""
  dataset = []
  model_names = {}
  with gfile.Open(DATASET_FILE, "r") as f:
    lines = f.readlines()
  for line in lines:
    if line.startswith("#"):
      continue
    # ,question_id,model_a,model_b,winner,judge,conversation_a,conversation_b,turn,anony,language,tstamp,openai_moderation,toxic_chat_tag
    parts = line.split(",")
    model_a, model_b, winner = (
        parts[2].strip(),
        parts[3].strip(),
        parts[4].strip(),
    )
    if filter_ties and winner.startswith("tie"):
      continue
    else:
      model_names[model_a] = True
      model_names[model_b] = True
      if winner == "model_a":
        dataset.append((model_a, model_b, -1))
      elif winner == "model_b":
        dataset.append((model_a, model_b, 1))
      else:
        assert winner.startswith("tie")
        dataset.append((model_a, model_b, 0))
  return list(model_names.keys()), dataset


def chatbot_arena_vase(model_names, dataset):
  """Run VasE over Chatbot Arena data set."""

  alternatives = model_names[:]
  profile = base.PreferenceProfile(alternatives=alternatives)
  for datapoint in dataset:
    alt_a, alt_b, outcome = datapoint
    if outcome == 0:
      pass
    elif outcome == -1:
      profile.add_vote([alt_a, alt_b])
    elif outcome == 1:
      profile.add_vote([alt_b, alt_a])

  margin_matrix = profile.margin_matrix()
  strong_cond_winners = profile.condorcet_winner(True, margin_matrix)
  weak_cond_winners = profile.condorcet_winner(False, margin_matrix)
  print(f"Strong Condorcet winner? {strong_cond_winners}")
  print(f"Weak Condorcet winner(s)? {weak_cond_winners}")

  voting_methods = [
      # approval.ApprovalVoting(k=8),
      # borda.BordaVoting(),
      copeland.CopelandVoting(),
      # kemeny_young.KemenyYoungVoting(),
      # Use verbose=True to get more information about the levels
      maximal_lotteries.MaximalLotteriesVoting(iterative=True),
      # maximal_lotteries.MaximalLotteriesVoting(iterative=True, verbose=True),
      # plurality.PluralityVoting(),
      ranked_pairs.RankedPairsVoting(),
      # stv.STVVoting(num_winners=8)
      schulze.SchulzeVoting(),
  ]
  for method in voting_methods:
    print("")
    print(method.name())
    outcome = method.run_election(profile)
    print(outcome.pretty_table_string())
    # print(outcome.pretty_latex_table(header=method.name()))


def ranked_pairs_viz(model_names, dataset):
  """Produce the ranked pairs visualization."""

  alternatives = model_names[:]
  profile = base.PreferenceProfile(alternatives=alternatives)
  num_alternatives = len(alternatives)
  alt_dict = profile.alternatives_dict
  for datapoint in dataset:
    alt_a, alt_b, outcome = datapoint
    if outcome == 0:
      pass
    elif outcome == -1:
      profile.add_vote([alt_a, alt_b])
    elif outcome == 1:
      profile.add_vote([alt_b, alt_a])
  margin_matrix = profile.margin_matrix()
  method = ranked_pairs.RankedPairsVoting()
  outcome = method.run_election(profile)
  graph_mat = outcome.graph
  # Visualize only over the top 8:
  keep_alternatives = [
      "gpt-4",
      "claude-v1",
      "claude-instant-v1",
      "guanaco-33b",
      "gpt-3.5-turbo",
      "wizardlm-13b",
      "palm-2",
      "vicuna-13b",
  ]
  keep_alternatives.sort()
  for j in range(num_alternatives):
    idx = num_alternatives - j - 1
    alt = alternatives[idx]
    if alt not in keep_alternatives:
      graph_mat = np.delete(graph_mat, (idx), axis=0)
      graph_mat = np.delete(graph_mat, (idx), axis=1)
  orig_alternatives = model_names[:]
  alternatives = keep_alternatives
  m = len(alternatives)
  graph = pgv.AGraph(directed=True, strict=True)
  for alternative in alternatives:
    graph.add_node(alternative)
  for i in range(m):
    for j in range(m):
      if graph_mat[i, j] == 1:
        graph.add_edge(alternatives[i], alternatives[j])
        idx_i = alt_dict[alternatives[i]]
        idx_j = alt_dict[alternatives[j]]
        edge = graph.get_edge(
            orig_alternatives[idx_i], orig_alternatives[idx_j]
        )
        edge.attr["label"] = margin_matrix[idx_i, idx_j]
  graph.write("/tmp/chatbot_arena_rps.dot")  # write to simple.dot
  graph.draw(
      "/tmp/chatbot_arena_rps.png",
      # args='-Gdpi=100',
      prog="dot",
  )  # , args="-n2")  # draw
  print("Wrote to /tmp/chatbot_arena_rps.png")


def main(_):
  model_names, dataset = parse_battles_dataset()
  model_names.sort()
  print(f"{len(model_names)} models.")
  print(f"{len(dataset)} datapoints.")
  chatbot_arena_vase(model_names, dataset)
  ranked_pairs_viz(model_names, dataset)


if __name__ == "__main__":
  np.random.seed(SEED)
  app.run(main)
