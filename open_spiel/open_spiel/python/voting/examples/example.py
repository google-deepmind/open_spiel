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

"""Simple basic example."""

# pylint: disable=unused-import

import sys
from absl import app
from absl import flags
import numpy as np

from open_spiel.python.voting import base
from open_spiel.python.voting import copeland


def main(_):
  # Create a preference profile that represents the following votes:
  #   A > B > C
  #   A > C > B
  #   C > A > B
  #   C > A > B
  #   B > C > A
  # This profile has three alternatives: A, B, and C. The strings here "A", "B",
  # "C" represent the alternative's ID and is of type base.AlternativeId.
  # (They can be strings or integers.)
  alternatives = ["A", "B", "C"]

  # Easiest way to make this profile:
  _ = base.PreferenceProfile(alternatives=alternatives, votes=[
      ["A", "B", "C"], ["A", "C", "B"], ["C", "A", "B"], ["C", "A", "B"],
      ["B", "C", "A"]
  ])

  # Note that the C > A > B vote is there twice, so another common way to show
  # this is:
  #   1: A > B > C
  #   1: A > C > B
  #   2: C > A > B
  #   1: B > C > A
  # and can be created with the WeightedVote type directly.
  profile = base.PreferenceProfile(alternatives=alternatives, votes=[
      base.WeightedVote(1, ["A", "B", "C"]),
      base.WeightedVote(1, ["A", "C", "B"]),
      base.WeightedVote(2, ["C", "A", "B"]),
      base.WeightedVote(1, ["B", "C", "A"])
  ])

  # Print some information about the profile
  print(f"Number of alternatives: {profile.num_alternatives()}")
  print(f"Number of votes: {profile.num_votes()}")
  print(f"Alternatives: {profile.alternatives}")
  print("Profile:")
  print(profile)

  # Print a reverse mapping of AlternativeId -> index
  # indices will always be numbered 0 to num_alternatives - 1.
  # Some methods work directly with the indices.
  alt_idx = profile.alternatives_dict
  print("Alternative ids -> index map:")
  print(alt_idx)

  # Iterating through a profile
  print("Iterating through profile:")
  for vote in profile.votes:
    # Each item is a weighted vote:
    print(f"  {vote.weight}: {vote.vote}")

  # Margin matrix and Condorcet winner check
  margin_matrix = profile.margin_matrix()
  cond_winners = profile.condorcet_winner(strong=True,
                                          margin_matrix=margin_matrix)
  print("Margin matrix:")
  print(margin_matrix)
  print(f"Condorcet winners: {cond_winners}")

  # Run Copeland on this profile and print the results
  method = copeland.CopelandVoting()
  outcome = method.run_election(profile)
  print("Copeland outcome:")
  print(outcome.pretty_table_string())


if __name__ == "__main__":
  app.run(main)
