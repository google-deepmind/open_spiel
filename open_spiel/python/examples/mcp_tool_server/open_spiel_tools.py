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

"""OpenSpiel MCP tools."""

import dataclasses
import json
import math
import random
from typing import Any
import uuid
from fastmcp.tools import tool

import pyspiel

from open_spiel.python.algorithms import ismcts
from open_spiel.python.algorithms import mcts


@dataclasses.dataclass
class GameSession:
  """A game session."""

  player_id: int
  state: pyspiel.State

  def maybe_step(self):
    while self.state.current_player() != self.player_id:
      if self.state.is_terminal():
        return
      if self.state.is_chance_node():
        outcomes, probs = zip(*self.state.chance_outcomes())
        action = random.choices(outcomes, weights=probs, k=1)[0]
        self.state.apply_action(action)
      else:
        self.state.apply_action(random.choice(self.state.legal_actions()))


  def reset(self):
    self.state = self.state.get_game().new_initial_state()
    self.maybe_step()


class OpenSpielTools:
  """Exposes OpenSpiel API as tools."""

  def __init__(self):
    self._sessions: dict[str, GameSession] = {}

  @tool()
  def list_games(self) -> list[str]:
    """List all registered games in OpenSpiel.

    Returns:
      A list of short names for all games registered in the OpenSpiel library.
      Each name can be used with `game_info` or `pyspiel.load_game` to get more
      details or instantiate the game.
    """
    return pyspiel.registered_names()

  @tool()
  def game_info(self, game_name: str) -> dict[str, Any]:
    """Get detailed information about a specific OpenSpiel game.

    Args:
      game_name: The short name of the game (e.g., 'tic_tac_toe', 'kuhn_poker').
        Must be a valid name from the list returned by `list_games`.

    Returns:
      A dictionary containing game metadata with the following keys:
        - short_name: The game's short identifier.
        - long_name: The full descriptive name of the game.
        - chance_mode: How chance is handled (e.g., 'DETERMINISTIC',
        'EXPLICIT').
        - information: Information structure (e.g., 'PERFECT', 'IMPERFECT').
        - dynamics: Game dynamics (e.g., 'SEQUENTIAL', 'SIMULTANEOUS').
        - utility: Utility type (e.g., 'ZERO_SUM', 'GENERAL_SUM').
        - max_num_players: Maximum number of players supported.
        - min_num_players: Minimum number of players required.
        - min_utility: The minimum possible utility value for any player.
        - max_utility: The maximum possible utility value for any player.
        - parameters: The parameters of the game.
        - action_structs_only: a boolean indicating whether the game uses
        action structs only.
        - action_struct_spec: the fields and types required in the JSON string
        representation of an action struct
        - action_struct_example: an example action struct as a JSON string
    """

    game = pyspiel.load_game(game_name)
    game_type = game.get_type()
    action_struct_spec, action_struct_example = game.action_struct_spec()
    return {
        'short_name': game_type.short_name,
        'long_name': game_type.long_name,
        'chance_mode': game_type.chance_mode.name,
        'information': game_type.information.name,
        'dynamics': game_type.dynamics.name,
        'utility': game_type.utility.name,
        'max_num_players': game_type.max_num_players,
        'min_num_players': game_type.min_num_players,
        'min_utility': game.min_utility(),
        'max_utility': game.max_utility(),
        'parameters': game.get_parameters(),
        'action_structs_only': game_type.action_structs_only,
        'action_struct_spec': action_struct_spec,
        'action_struct_example': action_struct_example,
    }

  @tool()
  def start_game(self, game_name: str, player_id: int) -> str:
    """Start a new game.

    Args:
      game_name: The short name of the game.
      player_id: The ID of the player.

    Returns:
      The ID of the game session.
    """
    game = pyspiel.load_game(game_name)
    session_id = str(uuid.uuid4())
    session = GameSession(player_id, game.new_initial_state())
    session.maybe_step()
    self._sessions[session_id] = session
    return session_id

  @tool()
  def reset_game(self, session_id: str):
    """Reset a game.

    Args:
      session_id: The ID of the game session.
    """
    self._sessions[session_id].reset()

  @tool()
  def current_player(self, session_id: str) -> int:
    """Get the current player of a game.

    Args:
      session_id: The ID of the game session.

    Returns:
      The ID of the current player. Returns -4 if the game is over.
    """
    state = self._sessions[session_id].state
    return state.current_player()

  @tool()
  def legal_actions(self, session_id: str) -> list[str]:
    """Get the legal actions of a game.

    Args:
      session_id: The ID of the game session.

    Returns:
      A list of legal actions.
    """
    state = self._sessions[session_id].state
    legal_actions = state.legal_actions()
    return [state.action_to_string(a) for a in legal_actions]

  @tool()
  def play_action(self, session_id: str, action: str):
    """Play an action in a game.

    Args:
      session_id: The ID of the game session.
      action: The action to play.
    """
    session = self._sessions[session_id]
    state = session.state
    if state.is_terminal():
      raise ValueError('Game is already over.')
    if state.current_player() != session.player_id:
      raise ValueError('It is not your turn')
    try:
      # try parsing the action as a JSON string. If it succeeds, assume it is
      # an action struct represented as a JSON string.
      _ = json.loads(action)
      try:
        action_struct = state.get_game().ActionStruct(action)
      except RuntimeError as e:
        raise ValueError(
            f'Detected valid JSON but failed to parse action struct: {e}'
        ) from e
      status = state.validate_action_struct(action_struct)
      if not status.ok():
        raise ValueError(f'Invalid action: {status.message()}')
      state.apply_action_struct(action_struct)
    except json.JSONDecodeError:
      # not valid JSON, us string_to_action
      action = state.string_to_action(action)
      state.apply_action(action)
    session.maybe_step()

  @tool()
  def get_return(self, session_id: str) -> float:
    """Get the return of a game.

    Args:
      session_id: The ID of the game session.

    Returns:
      The return for the player.
    """
    session = self._sessions[session_id]
    return session.state.returns()[session.player_id]

  @tool()
  def get_observation(self, session_id: str) -> str:
    """Get the player's observation of a game.

    Args:
      session_id: The ID of the game session.

    Returns:
      The player's observation of the game.
    """
    session = self._sessions[session_id]
    return session.state.observation_string(session.player_id)

  @tool()
  def mcts_action_chooser(self, session_id: str, num_simulations: int):
    """Get an action recommendation using MCTS.

    Args:
      session_id: The ID of the game session.
      num_simulations: The number of MCTS simulations to use.

    Returns:
      The recommended action to play.
    """
    session = self._sessions[session_id]
    state = session.state
    if state.is_terminal():
      raise ValueError('Game is already over.')
    util_range = state.get_game().max_utility() - state.get_game().min_utility()
    uct_c = math.sqrt(2) * util_range
    evaluator = mcts.RandomRolloutEvaluator()
    if (state.get_game().get_type().information ==
        pyspiel.GameType.Information.PERFECT_INFORMATION):
      bot = mcts.MCTSBot(state.get_game(), uct_c=uct_c,
                         max_simulations=num_simulations, evaluator=evaluator)
    else:
      bot = ismcts.ISMCTSBot(state.get_game(), evaluator=evaluator, uct_c=uct_c,
                             max_simulations=num_simulations)
    action = bot.step(state)
    return state.action_to_string(action)
