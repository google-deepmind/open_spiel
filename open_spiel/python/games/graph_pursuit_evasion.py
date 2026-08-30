# Copyright 2026 DeepMind Technologies Limited
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

"""Two-player pursuit-evasion on an arbitrary typed graph.

The pursuer (player 0) and evader (player 1) move alternately along an
undirected graph. Edges have transport types and each player can be restricted
to a subset of those types. The evader's location is hidden from the pursuer
between scheduled reveal rounds, while the type of each evader move is public.
The pursuer wins by occupying the evader's node; otherwise the evader wins when
the configured number of rounds elapses.
"""

import dataclasses

import numpy as np

import pyspiel


_NUM_PLAYERS = 2
_PURSUER = 0
_EVADER = 1
_DEFAULT_PARAMS = {
    "graph": (
        "0-1:taxi,1-2:bus,2-3:rail,3-4:taxi,4-5:bus,5-0:rail,"
        "1-4:taxi,2-5:bus"
    ),
    "edge_types": "taxi,bus,rail",
    "pursuer_edge_types": "taxi,bus,rail",
    "evader_edge_types": "taxi,bus,rail",
    "pursuer_start": 0,
    "evader_start": 3,
    "max_rounds": 12,
    "reveal_interval": 3,
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_graph_pursuit_evasion",
    long_name="Python Graph Pursuit-Evasion",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification=_DEFAULT_PARAMS,
)


@dataclasses.dataclass(frozen=True, order=True)
class Move:
  """A directed move derived from an undirected typed edge."""

  source: int
  target: int
  edge_type: int


def _parse_names(value, parameter):
  names = tuple(name.strip() for name in str(value).split(",") if name.strip())
  if not names:
    raise ValueError(f"{parameter} must contain at least one name")
  if len(set(names)) != len(names):
    raise ValueError(f"{parameter} contains duplicate names: {value!r}")
  return names


def _parse_graph(value, edge_type_to_id):
  moves = set()
  nodes = set()
  for encoded_edge in str(value).split(","):
    encoded_edge = encoded_edge.strip()
    if not encoded_edge:
      continue
    try:
      endpoints, edge_type_name = encoded_edge.split(":", maxsplit=1)
      source_name, target_name = endpoints.split("-", maxsplit=1)
      source = int(source_name)
      target = int(target_name)
    except ValueError as error:
      raise ValueError(
          "graph edges must use the format 'source-target:type'; got "
          f"{encoded_edge!r}"
      ) from error
    edge_type_name = edge_type_name.strip()
    if source < 0 or target < 0 or source == target:
      raise ValueError(f"invalid graph edge: {encoded_edge!r}")
    if edge_type_name not in edge_type_to_id:
      raise ValueError(
          f"unknown edge type {edge_type_name!r} in {encoded_edge!r}"
      )
    edge_type = edge_type_to_id[edge_type_name]
    nodes.update((source, target))
    moves.add(Move(source, target, edge_type))
    moves.add(Move(target, source, edge_type))
  if not moves:
    raise ValueError("graph must contain at least one edge")
  expected_nodes = set(range(max(nodes) + 1))
  if nodes != expected_nodes:
    raise ValueError("graph node ids must be consecutive and start at 0")
  return tuple(sorted(moves)), len(nodes)


class GraphPursuitEvasionGame(pyspiel.Game):
  """A configurable pursuit-evasion game on a typed graph."""

  def __init__(self, params=None):
    params = dict(params or {})
    resolved = dict(_DEFAULT_PARAMS)
    resolved.update(params)

    self.edge_type_names = _parse_names(resolved["edge_types"], "edge_types")
    edge_type_to_id = {
        edge_type: index
        for index, edge_type in enumerate(self.edge_type_names)
    }
    self.moves, self.num_nodes = _parse_graph(
        resolved["graph"], edge_type_to_id
    )
    self.wait_action = len(self.moves)
    self._action_by_move = {
        move: action for action, move in enumerate(self.moves)
    }
    self._moves_by_source = [[] for _ in range(self.num_nodes)]
    for action, move in enumerate(self.moves):
      self._moves_by_source[move.source].append(action)

    self.allowed_edge_types = (
        self._parse_allowed_types(
            resolved["pursuer_edge_types"],
            "pursuer_edge_types",
            edge_type_to_id,
        ),
        self._parse_allowed_types(
            resolved["evader_edge_types"], "evader_edge_types", edge_type_to_id
        ),
    )
    self.pursuer_start = int(resolved["pursuer_start"])
    self.evader_start = int(resolved["evader_start"])
    self.max_rounds = int(resolved["max_rounds"])
    self.reveal_interval = int(resolved["reveal_interval"])
    for start in (self.pursuer_start, self.evader_start):
      if not 0 <= start < self.num_nodes:
        raise ValueError(f"start node {start} is not present in the graph")
    if self.pursuer_start == self.evader_start:
      raise ValueError("pursuer_start and evader_start must differ")
    if self.max_rounds <= 0:
      raise ValueError("max_rounds must be positive")
    if self.reveal_interval <= 0:
      raise ValueError("reveal_interval must be positive")

    game_info = pyspiel.GameInfo(
        num_distinct_actions=len(self.moves) + 1,
        max_chance_outcomes=0,
        num_players=_NUM_PLAYERS,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=self.max_rounds * 2,
    )
    super().__init__(_GAME_TYPE, game_info, params)

  def _parse_allowed_types(self, value, parameter, edge_type_to_id):
    names = _parse_names(value, parameter)
    unknown = set(names) - set(edge_type_to_id)
    if unknown:
      raise ValueError(f"{parameter} contains unknown types: {sorted(unknown)}")
    return frozenset(edge_type_to_id[name] for name in names)

  def new_initial_state(self):
    return GraphPursuitEvasionState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return GraphPursuitEvasionObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params,
        self,
    )

  def move_action(self, source, target, edge_type):
    """Returns the action id for a directed typed move."""
    if isinstance(edge_type, str):
      try:
        edge_type = self.edge_type_names.index(edge_type)
      except ValueError as error:
        raise ValueError(f"unknown edge type: {edge_type!r}") from error
    try:
      return self._action_by_move[Move(source, target, edge_type)]
    except KeyError as error:
      raise ValueError(
          f"no {self.edge_type_names[edge_type]} edge from {source} to {target}"
      ) from error


class GraphPursuitEvasionState(pyspiel.State):
  """State for graph pursuit-evasion."""

  def __init__(self, game):
    super().__init__(game)
    self.positions = [game.pursuer_start, game.evader_start]
    self.round = 0
    self.last_revealed_evader = game.evader_start
    self.last_evader_edge_type = None
    self._next_player = _PURSUER
    self._is_terminal = False
    self._pursuer_return = 0.0
    start = (
        f"start:pursuer={game.pursuer_start},evader={game.evader_start}"
    )
    self._information_history = [[start], [start]]

  def current_player(self):
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    return self._next_player

  def _legal_actions(self, player):
    if self._is_terminal or player != self._next_player:
      return []
    game = self.get_game()
    actions = [
        action
        for action in game._moves_by_source[self.positions[player]]
        if game.moves[action].edge_type in game.allowed_edge_types[player]
    ]
    actions.append(game.wait_action)
    return actions

  def _apply_action(self, action):
    game = self.get_game()
    player = self._next_player
    source = self.positions[player]
    if action == game.wait_action:
      target = source
      edge_type = None
    else:
      move = game.moves[action]
      target = move.target
      edge_type = move.edge_type
      self.positions[player] = target

    if player == _PURSUER:
      self._record_pursuer_move(source, target, edge_type)
      if self.positions[_PURSUER] == self.positions[_EVADER]:
        self._finish(pursuer_wins=True)
      else:
        self._next_player = _EVADER
      return

    self.round += 1
    self.last_evader_edge_type = edge_type
    reveal = self.round % game.reveal_interval == 0
    if reveal:
      self.last_revealed_evader = target
    self._record_evader_move(source, target, edge_type, reveal)
    if self.positions[_PURSUER] == self.positions[_EVADER]:
      self._finish(pursuer_wins=True)
    elif self.round >= game.max_rounds:
      self._finish(pursuer_wins=False)
    else:
      self._next_player = _PURSUER

  def _record_pursuer_move(self, source, target, edge_type):
    if edge_type is None:
      event = f"P:wait@{source}"
    else:
      name = self.get_game().edge_type_names[edge_type]
      event = f"P:{name}:{source}->{target}"
    for history in self._information_history:
      history.append(event)

  def _record_evader_move(self, source, target, edge_type, reveal):
    edge_name = (
        "wait"
        if edge_type is None
        else self.get_game().edge_type_names[edge_type]
    )
    pursuer_event = f"E:{edge_name}"
    if reveal:
      pursuer_event += f":reveal={target}"
    self._information_history[_PURSUER].append(pursuer_event)
    self._information_history[_EVADER].append(
        f"E:{edge_name}:{source}->{target}"
    )

  def _finish(self, pursuer_wins):
    self._is_terminal = True
    self._pursuer_return = 1.0 if pursuer_wins else -1.0

  def _action_to_string(self, player, action):
    game = self.get_game()
    actor = "Pursuer" if player == _PURSUER else "Evader"
    if action == game.wait_action:
      return f"{actor} waits"
    move = game.moves[action]
    edge_name = game.edge_type_names[move.edge_type]
    return f"{actor} moves {move.source}->{move.target} by {edge_name}"

  def is_terminal(self):
    return self._is_terminal

  def returns(self):
    if not self._is_terminal:
      return [0.0, 0.0]
    return [self._pursuer_return, -self._pursuer_return]

  def __str__(self):
    return (
        f"Pursuer: {self.positions[_PURSUER]}, "
        f"Evader: {self.positions[_EVADER]}, "
        f"Last reveal: {self.last_revealed_evader}, "
        f"Round: {self.round}/{self.get_game().max_rounds}"
    )


class GraphPursuitEvasionObserver:
  """Player-relative current observation or perfect-recall history."""

  def __init__(self, iig_obs_type, params, game):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self._iig_obs_type = iig_obs_type
    pieces = [
        ("player", _NUM_PLAYERS, (_NUM_PLAYERS,)),
        ("own_position", game.num_nodes, (game.num_nodes,)),
        ("known_opponent_position", game.num_nodes, (game.num_nodes,)),
        (
            "last_evader_edge_type",
            len(game.edge_type_names) + 1,
            (len(game.edge_type_names) + 1,),
        ),
        ("round_progress", 1, (1,)),
    ]
    self.tensor = np.zeros(sum(size for _, size, _ in pieces), np.float32)
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index : index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    self.tensor.fill(0)
    self.dict["player"][player] = 1
    self.dict["own_position"][state.positions[player]] = 1
    if player == _PURSUER:
      known_opponent = state.last_revealed_evader
    else:
      known_opponent = state.positions[_PURSUER]
    self.dict["known_opponent_position"][known_opponent] = 1
    edge_index = (
        0
        if state.last_evader_edge_type is None
        else state.last_evader_edge_type + 1
    )
    self.dict["last_evader_edge_type"][edge_index] = 1
    self.dict["round_progress"][0] = (
        state.round / state.get_game().max_rounds
    )

  def string_from(self, state, player):
    if self._iig_obs_type.perfect_recall:
      return " | ".join(state._information_history[player])
    own = state.positions[player]
    opponent = (
        state.last_revealed_evader
        if player == _PURSUER
        else state.positions[_PURSUER]
    )
    edge = (
        "none"
        if state.last_evader_edge_type is None
        else state.get_game().edge_type_names[state.last_evader_edge_type]
    )
    return (
        f"player={player} own={own} known_opponent={opponent} "
        f"last_evader_edge={edge} round={state.round}"
    )


pyspiel.register_game(_GAME_TYPE, GraphPursuitEvasionGame)
