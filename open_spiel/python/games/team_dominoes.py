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
"""Dominoes (4 players) implemented in Python.

https://en.wikipedia.org/wiki/Dominoes#Latin_American_Version
"""

import collections
import copy
import itertools

import numpy as np

import pyspiel

_NUM_PLAYERS = 4
_PIPS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
_DECK = list(itertools.combinations_with_replacement(_PIPS, 2))
_EDGES = [None, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class Action:
  """Represent player possible action."""

  def __init__(self, player, tile, edge):
    self.player = player
    self.tile = tile
    self.edge = edge

  def __str__(self):
    return f"p{self.player} tile:{self.tile} pip:{self.edge}"

  def __repr__(self):
    return self.__str__()


def create_possible_actions():
  actions = []
  for player in range(_NUM_PLAYERS):
    for tile in _DECK:
      for edge in _EDGES:
        if edge in tile or edge is None:
          actions.append(Action(player, tile, edge))
  return actions


_ACTIONS = create_possible_actions()
_ACTIONS_STR = [str(action) for action in _ACTIONS]

_HAND_SIZE = 7

_MAX_GAME_LENGTH = 28

_GAME_TYPE = pyspiel.GameType(
    short_name="python_team_dominoes",
    long_name="Python Team Dominoes (4 players)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(_ACTIONS),
    max_chance_outcomes=len(_DECK),
    min_utility=-100,
    max_utility=100,
    num_players=_NUM_PLAYERS,
    # deal: 28 chance nodes + play: 28 player nodes
    max_game_length=_MAX_GAME_LENGTH,
    utility_sum=0.0,
)


class DominoesGame(pyspiel.Game):
  """A Python version of Block Dominoes."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return DominoesState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return DominoesObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params
    )


class DominoesState(pyspiel.State):
  """A python version of the Block Dominoes state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.actions_history = []
    self.open_edges = []
    self.hands = [[] for _ in range(_NUM_PLAYERS)]
    self.deck = copy.deepcopy(_DECK)
    self._game_over = False
    self._next_player = pyspiel.PlayerId.CHANCE
    self._current_deal_player = 0  # NEW ATTRIBUTE

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    if self.deck:  # deal phase
      return pyspiel.PlayerId.CHANCE
    return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    assert player == self._next_player
    return self.get_legal_actions(player)

  def get_legal_actions(self, player):
    """Returns a list of legal actions."""
    assert player >= 0

    actions = []
    hand = self.hands[player]

    # first move, no open edges
    if not self.open_edges:
      for tile in hand:
        actions.append(Action(player, tile, None))
    else:
      for tile in hand:
        if tile[0] in self.open_edges:
          actions.append(Action(player, tile, tile[0]))
        if tile[0] != tile[1] and tile[1] in self.open_edges:
          actions.append(Action(player, tile, tile[1]))

    actions_idx = [_ACTIONS_STR.index(str(action)) for action in actions]
    actions_idx.sort()
    return actions_idx

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    p = 1.0 / len(self.deck)
    return [(_DECK.index(i), p) for i in self.deck]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      # Deal tiles to players in order (0, 1, 2, 3)
      hand_to_add_tile = self.hands[self._current_deal_player]
      tile = _DECK[action]
      self.deck.remove(tile)
      hand_to_add_tile.append(tile)
      self._current_deal_player = (self._current_deal_player + 1) % 4

      # Check if all hands are of _HAND_SIZE
      if not all(len(hand) == _HAND_SIZE for hand in self.hands):
        return  # more tiles to deal

      for hand in self.hands:
        hand.sort()

      self._next_player = 0
    else:
      action = _ACTIONS[action]
      self.actions_history.append(action)
      my_idx = self.current_player()
      my_hand = self.hands[my_idx]
      my_hand.remove(action.tile)
      self.update_open_edges(action)

      if not my_hand:
        self._game_over = True  # player played his last tile
        return

      for i in range(1, 5):
        next_idx = (my_idx + i) % 4
        next_legal_actions = self.get_legal_actions(next_idx)

        if next_legal_actions:
          self._next_player = next_idx
          return

        # Check if a team has played all their tiles.
        if not (self.hands[0] or self.hands[2]) or not (
            self.hands[1] or self.hands[3]
        ):
          self._game_over = True
          return

      # all players are blocked. Game is stuck.
      self._game_over = True

  def update_open_edges(self, action):
    if not self.open_edges:
      self.open_edges = list(action.tile)
    else:
      self.open_edges.remove(action.edge)
      new_edge = (
          action.tile[0] if action.tile[0] != action.edge else action.tile[1]
      )
      self.open_edges.append(new_edge)

    self.open_edges.sort()

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal {_DECK[action]}"
    return _ACTIONS_STR[action]

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self.is_terminal():
      return [0 for _ in range(_NUM_PLAYERS)]

    sum_of_pips0 = sum(t[0] + t[1] for t in (self.hands[0] + self.hands[2]))
    sum_of_pips1 = sum(t[0] + t[1] for t in (self.hands[1] + self.hands[3]))

    if sum_of_pips1 == sum_of_pips0:
      return [0 for _ in range(_NUM_PLAYERS)]

    if sum_of_pips1 > sum_of_pips0:
      return [sum_of_pips1, -sum_of_pips1, sum_of_pips1, -sum_of_pips1]
    return [-sum_of_pips0, sum_of_pips0, -sum_of_pips0, sum_of_pips0]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    hand0 = [str(c) for c in self.hands[0]]
    hand1 = [str(c) for c in self.hands[1]]
    hand2 = [str(c) for c in self.hands[2]]
    hand3 = [str(c) for c in self.hands[3]]
    board = self.draw_board()
    return (
        f"hand0:{hand0}\n"
        f"hand1:{hand1}\n"
        f"hand2:{hand2}\n"
        f"hand3:{hand3}\n\n"
        f"board: {board}"
    )

  def draw_board(self):
    """Draw the board' in a human readable format."""
    board = collections.deque()
    current_open_edges = None
    for action in self.actions_history:
      # check if action is played on an empty board
      if action.edge is None:
        board.append(action.tile)
        # pylint: disable=unused-variable
        current_open_edges = list(action.tile)
      # check if action edge matches last played edge in the left or right
      elif action.edge == current_open_edges[0]:
        # invert the tile if the edge is on the right:
        tile = (
            (action.tile[1], action.tile[0])
            if action.tile[0] == current_open_edges[0]
            else action.tile
        )
        board.appendleft(tile)

      elif action.edge == current_open_edges[1]:
        # invert the tile if the edge is on the left:
        tile = (
            (action.tile[1], action.tile[0])
            if action.tile[1] == current_open_edges[1]
            else action.tile
        )
        board.append(tile)

      current_open_edges = board[0][0], board[-1][1]

    # TODO(someone): move this to a test
    assert len(board) == len(self.actions_history)
    return list(board)


class DominoesObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", 4, (4,))]

    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      # each tile is represented using 3 integers:
      # 2 for the pips, and 1 to distinguish between (0,0) to empty slot for
      # a tile.
      pieces.append(("hand", 21, (7, 3)))  # 7 tiles per hand
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:
        # list of all played actions, each action is represented using 5
        # integers:
        # 2 for the played tile (0-6),
        # 1 for the covered edge (0-6),
        # 1 for which player (0,1,3,4),
        # 1 to distinguish between actual move and empty slot for a move (0/1).
        # the None (play on an empty board) edge represented using 0.
        pieces.append(("actions_history", 125, (25, 5)))
      else:
        # last action, represented in the same way as in "actions_history"
        # but without the last integer.
        pieces.append(("last_action", 4, (4,)))
        pieces.append(("hand_sizes", 4, (4,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index : index + size].reshape(shape)
      index += size

  def copy_indices(self, dest, source, index_list):
    for idx in index_list:
      dest[idx] = source[idx]

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

    self.tensor.fill(0)

    if "player" in self.dict:
      self.dict["player"][player] = 1
      self.dict["player"][1 - player] = 0

    if "hand_sizes" in self.dict:
      my_hand_size = len(state.hands[player])
      opp_hand_size = len(state.hands[1 - player])
      self.dict["hand_sizes"][0] = my_hand_size
      self.dict["hand_sizes"][1] = opp_hand_size

    if "edges" in self.dict:
      if state.open_edges:
        self.copy_indices(self.dict["edges"], state.open_edges, [0, 1])
      else:
        self.dict["edges"][0] = 0.0
        self.dict["edges"][1] = 0.0

    if "hand" in self.dict:
      for i, tile in enumerate(state.hands[player]):
        self.copy_indices(self.dict["hand"][i], tile, [0, 1])
        self.dict["hand"][i][2] = 1.0

    if "actions_history" in self.dict:
      for i, action in enumerate(state.actions_history):
        self.copy_indices(self.dict["actions_history"][i], action.tile, [0, 1])
        self.dict["actions_history"][i][2] = (
            action.edge if action.edge is not None else 0.0
        )
        self.dict["actions_history"][i][3] = action.player
        self.dict["actions_history"][i][4] = 1.0

    if "last_action" in self.dict:
      if state.actions_history:
        action = state.actions_history[-1]
        self.copy_indices(self.dict["last_action"], action.tile, [0, 1])
        self.dict["last_action"][2] = (
            action.edge if action.edge is not None else 0.0
        )
        self.dict["last_action"][3] = action.player

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "hand" in self.dict:
      pieces.append(f"hand:{state.hands[player]}")
    if "actions_history" in self.dict:
      pieces.append(f"history:{str(state.actions_history)}")
    if "last_action" in self.dict and state.actions_history:
      pieces.append(f"last_action:{str(state.actions_history[-1])}")
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, DominoesGame)
