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

"""Plays a round of Tarok with actions from user input."""

import pyspiel


def play_tarok_game():
  game = pyspiel.load_game("tarok(players=3)")
  state = game.new_initial_state()
  while not state.is_terminal():
    print_info(game, state)
    state.apply_action(int(input("Enter action: ")))
    print("-" * 70, "\n")
  print(state.current_game_phase())
  print("Players' scores: {}".format(state.rewards()))


def print_info(unused_game, state):
  """Print information about the game state."""
  print("Game phase: {}".format(state.current_game_phase()))
  print("Selected contract: {}".format(state.selected_contract()))
  print("Current player: {}".format(state.current_player()))
  player_cards = state.player_cards(state.current_player())
  action_names = [state.card_action_to_string(a) for a in player_cards]
  print("\nPlayer cards: {}".format(
      list(zip(action_names, player_cards))))

  if state.current_game_phase() == pyspiel.TarokGamePhase.TALON_EXCHANGE:
    print_talon_exchange_info(state)
  elif state.current_game_phase() == pyspiel.TarokGamePhase.TRICKS_PLAYING:
    print_tricks_playing_info(state)
  else:
    print()

  legal_actions = state.legal_actions()
  action_names = [state.action_to_string(a) for a in state.legal_actions()]
  print("Legal actions: {}\n".format(
      list(zip(action_names, legal_actions))))


def print_talon_exchange_info(state):
  talon = [[state.card_action_to_string(x) for x in talon_set]
           for talon_set in state.talon_sets()]
  print("\nTalon: {}\n".format(talon))


def print_tricks_playing_info(state):
  trick_cards = state.trick_cards()
  action_names = [state.card_action_to_string(a) for a in trick_cards]
  print("\nTrick cards: {}\n".format(
      list(zip(action_names, trick_cards))))


if __name__ == "__main__":
  play_tarok_game()
