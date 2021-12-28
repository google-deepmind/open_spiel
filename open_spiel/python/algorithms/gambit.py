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

# Lint as python3
"""Export game trees in gambit format.

An exporter for the .efg format used by Gambit:
http://www.gambit-project.org/gambit14/formats.html

See `examples/gambit_example.py` for an example of usage.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools


def quote(x):
  return f"\"{x}\""


def export_gambit(game):
  """Builds gambit representation of the game tree.

  Args:
    game: A `pyspiel.Game` object.

  Returns:
    string: Gambit tree
  """
  players = " ".join([f"\"Pl{i}\"" for i in range(game.num_players())])
  ret = f"EFG 2 R {quote(game)} {{ {players} }} \n"

  terminal_idx = 1
  chance_idx = 1

  # We will keep separate infoset idx per each player.
  # Note that gambit infosets start at 1, but we start them here at 0 because
  # they get incremented when accessed from infoset_tables below.
  infoset_idx = [0] * game.num_players()

  def infoset_next_id(player):
    nonlocal infoset_idx
    infoset_idx[player] += 1
    return infoset_idx[player]

  infoset_tables = [
      collections.defaultdict(functools.partial(infoset_next_id, player))
      for player in range(game.num_players())
  ]

  def build_tree(state, depth):
    nonlocal ret, terminal_idx, chance_idx, infoset_tables

    ret += " " * depth  # add nice spacing
    state_str = str(state)
    if len(state_str) > 10:
      state_str = ""

    if state.is_terminal():
      utils = " ".join(map(str, state.returns()))
      ret += f"t {quote(state_str)} {terminal_idx} \"\" {{ {utils} }}\n"
      terminal_idx += 1
      return

    if state.is_chance_node():
      ret += f"c {quote(state_str)} {chance_idx} \"\" {{ "
      for action, prob in state.chance_outcomes():
        action_str = state.action_to_string(state.current_player(), action)
        ret += f"{quote(action_str)} {prob:.16f} "
      ret += " } 0\n"
      chance_idx += 1

    else:  # player node
      player = state.current_player()
      gambit_player = player + 1  # cannot be indexed from 0
      infoset = state.information_state_string()
      infoset_idx = infoset_tables[player][infoset]

      ret += f"p {quote(state_str)} {gambit_player} {infoset_idx} \"\" {{ "
      for action in state.legal_actions():
        action_str = state.action_to_string(state.current_player(), action)
        ret += f"{quote(action_str)} "
      ret += " } 0\n"

    for action in state.legal_actions():
      child = state.child(action)
      build_tree(child, depth + 1)

  build_tree(game.new_initial_state(), 0)
  return ret
