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

"""OpenSpiel API."""

import random
from typing import Any, List, Text, Tuple, Dict

from open_spiel.python.examples.meta_cfr.sequential_games import world_representation
import pyspiel


class WorldState(world_representation.WorldState):
  """World state representation for openspiel games.

  This class implements world_representation class for openspiel games.

  Attributes:
    game_name: Name of openspiel game we want to initialize.
    config: Config containing game parameters to initialize the game.
    state: Initial state of an openspeil game.
    chance_policy: The policy of the chance node in the game tree.
  """

  def __init__(self, game_name: str, config: Dict[str, Any],
               perturbation: bool, random_seed: int = 100):
    self._perturbation = perturbation
    self._history = []
    self._random_seed = random_seed
    self.game_name = game_name
    self.config = config
    self._game = pyspiel.load_game(self.game_name, self.config)
    if str(self._game.get_type().dynamics) == "Dynamics.SIMULTANEOUS":
      self._game = pyspiel.convert_to_turn_based(self._game)
    # initial_state
    self.state = self._game.new_initial_state()
    self.chance_policy = self.get_chance_policy()
    random.seed(self._random_seed)

  def get_distinct_actions(self) -> List[int]:
    """See base class."""
    return list(range(self._game.num_distinct_actions()))

  def is_terminal(self) -> bool:
    """See base class."""
    return self.state.is_terminal()

  def get_actions(self) -> List[Any]:
    """See base class."""
    if self.is_terminal():
      return [[], [], []]
    actions = [[0], [0], [0]]
    if self.state.is_chance_node():
      legal_actions = [
          action for (action, prob) in self.state.chance_outcomes()
      ]
    else:
      legal_actions = self.state.legal_actions()
    actions[self.state.current_player() + 1] = legal_actions
    return actions

  def get_infostate_string(self, player: int) -> Text:
    """See base class."""
    infostate = self.state.information_state_string(player - 1)
    return str(len(self._history)) + "|" + str(infostate)

  def apply_actions(self, actions: Tuple[int, int, int]) -> None:
    """See base class."""
    self.state.apply_action(actions[self.state.current_player() + 1])
    self.chance_policy = self.get_chance_policy()
    self._history.append(actions)

  def get_utility(self, player: int) -> float:
    """See base class."""
    assert self.is_terminal()
    return float(self.state.returns()[player - 1])

  def get_chance_policy(self) -> Dict[int, float]:
    """See base class."""
    if self.is_terminal():
      return {}

    if not self.state.is_chance_node():
      return {0: 1}

    chance_policy = {
        action: prob for (action, prob) in self.state.chance_outcomes()
    }

    if self._perturbation:
      probs = [random.random() for _ in self.state.chance_outcomes()]
      chance_policy = {
          action: probs[i] / sum(probs)
          for i, (action, prob) in enumerate(self.state.chance_outcomes())
      }

    return chance_policy
