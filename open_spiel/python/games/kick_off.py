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
"""Kick Off Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel


class Action(enum.IntEnum):
    FOLD = 0
    CALL = 1
    BET_1_5 = 2
    BET_3 = 3
    BET_5 = 4
    RAISE_2_5 = 5
    RAISE_5 = 6
    RAISE_8 = 7
    ALL_IN = 8


# Define a dictionary mapping each action to its corresponding amount
ACTION_AMOUNTS = {
    Action.FOLD: 0,
    Action.BET_1_5: 1.5,
    Action.BET_3: 3,
    Action.BET_5: 5,
    Action.RAISE_2_5: 2.5,
    Action.RAISE_5: 5,
    Action.RAISE_8: 8,
}


# Add a property to fetch the amount
def amount(action):
    return ACTION_AMOUNTS[action]


_INITIAL_STACK = 20

_NUM_PLAYERS = 4
_DECK = frozenset(
    (rank, suit) for rank in
    ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    for suit in ['h', 'd', 'c', 's'
                 ]  #h : hearts, d: diamonds, c : clubs, s : spades
)

_GAME_TYPE = pyspiel.GameType(
    short_name="python_kuhn_poker",
    long_name="Python Kuhn Poker",
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
    provides_factored_observation_string=True)

_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=len(Action),
                              max_chance_outcomes=len(_DECK),
                              num_players=_NUM_PLAYERS,
                              min_utility=-_INITIAL_STACK,
                              max_utility=_INITIAL_STACK,
                              utility_sum=0.0,
                              max_game_length=64)


class KickOffGame(pyspiel.Game):
    """A Python version of Kuhn poker."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return KickOffState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return KickOffObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
            params)


class KickOffState(pyspiel.State):
    """A python version of the Kuhn poker state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self.cards = []  # List of cards dealt to players and community cards
        self.bets = [0] * _NUM_PLAYERS  # List of bets for each player
        self.pot = 0.0  # Total pot
        self.players_stack = [
            _INITIAL_STACK
        ] * _NUM_PLAYERS  # Starting stack for each player (20 BB)
        self._game_over = False
        self._next_player = 0
        self._round = "preflop"  # Current round of betting (preflop, flop, turn, river)
        self._active_players = set(
            range(_NUM_PLAYERS))  # Set of active players in the current hand
        self._community_cards = []  # Community cards (flop, turn, river)

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every sequential-move game with chance.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self._round == "preflop" and len(self.cards) < _NUM_PLAYERS * 2:
            # Dealing hole cards to players
            return pyspiel.PlayerId.CHANCE
        else:
            return self._next_player

    def _advance_to_next_player(self):
        """Move to the next player, cycling back to 0 if at the end."""
        self._next_player = (self._next_player + 1) % _NUM_PLAYERS

    def _deal_cards(self):
        """Deals the hole cards to players and the community cards."""
        if self._round == "preflop" and len(self.cards) < _NUM_PLAYERS * 2:
            # Deal 2 hole cards to each player (total 8 cards for 4 players)
            self.cards = [
                f"{rank}{suit}" for rank in [
                    '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q',
                    'K', 'A'
                ] for suit in ['h', 'd', 'c', 's']
            ]  # Ensure this is shuffled
            # Shuffle and assign hole cards for each player
            pass

        elif self._round == "flop" and len(self._community_cards) == 0:
            # Deal Flop (3 community cards)
            self._community_cards.extend(
                ["card1", "card2",
                 "card3"])  # Placeholder for actual deal logic

        elif self._round == "turn" and len(self._community_cards) == 3:
            # Deal Turn (1 community card)
            self._community_cards.append("card4")  # Placeholder

        elif self._round == "river" and len(self._community_cards) == 4:
            # Deal River (1 community card)
            self._community_cards.append("card5")  # Placeholder

    def _update_pot(self):
        """Update the pot with the current player's bet."""
        self.pot += self.bets[self._next_player]

    def _legal_actions(self, player):
        """
    Returns a list of legal actions for the current player, considering the current bet.
    """
        assert 0 <= player < _NUM_PLAYERS, "Invalid player index."

        # Common legal actions
        legal_actions = [Action.FOLD]

        # Check if the player can call the current bet
        if self._current_bet > 0 and self._player_stacks[
                player] >= self._current_bet:
            legal_actions.append(Action.CALL)

        # Add betting actions if the player can afford them and if they exceed the current bet
        if self._player_stacks[player] >= max(1.5, self._current_bet + 1.5):
            legal_actions.append(Action.BET_1_5)
        if self._player_stacks[player] >= max(3, self._current_bet + 3):
            legal_actions.append(Action.BET_3)
        if self._player_stacks[player] >= max(5, self._current_bet + 5):
            legal_actions.append(Action.BET_5)

        # Add raising actions if the current bet is not all-in and the player can afford the raise
        if self._current_bet > 0:
            if self._player_stacks[player] >= self._current_bet + 2.5:
                legal_actions.append(Action.RAISE_2_5)
            if self._player_stacks[player] >= self._current_bet + 5:
                legal_actions.append(Action.RAISE_5)
            if self._player_stacks[player] >= self._current_bet + 8:
                legal_actions.append(Action.RAISE_8)

        # Add the all-in option if the player has chips
        if self._player_stacks[player] > 0:
            legal_actions.append(Action.ALL_IN)

        return legal_actions

    def chance_outcomes(self):
        """
    Returns the possible chance outcomes and their probabilities.
    This determines the next card to be dealt from the remaining deck.
    """
        assert self.is_chance_node(
        ), "This method should only be called at chance nodes."

        # Remaining cards in the deck
        remaining_deck = sorted(_DECK - set(self.cards))

        # Compute the probability for each outcome
        num_outcomes = len(remaining_deck)
        if num_outcomes == 0:
            raise ValueError("No cards left in the deck for chance outcomes.")

        probability = 1.0 / num_outcomes
        return [(card, probability) for card in remaining_deck]

    def _apply_action(self, action):
        """
    Applies the specified action to the game state.
    Handles both chance nodes (dealing cards) and player actions.
    """
        if self.is_chance_node():
            # Dealing a card at a chance node
            self.cards.append(action)
        else:
            # Player actions
            self.bets.append(action)

            if action == Action.CALL:
                # Add the call amount to the pot
                self.pot[self._next_player] += self._current_bet
            elif action in {Action.BET_1_5, Action.BET_3, Action.BET_5}:
                # Handle betting actions
                bet_amount = amount(action)
                self._current_bet = bet_amount
                self.pot[self._next_player] += bet_amount
            elif action in {Action.RAISE_2_5, Action.RAISE_5, Action.RAISE_8}:
                # Handle raising actions
                raise_amount = amount(action)
                self._current_bet += raise_amount
                self.pot[self._next_player] += self._current_bet
            elif action == Action.ALL_IN:
                # All-in action
                self.pot[self._next_player] += self._player_stacks[
                    self._next_player]
                self._player_stacks[self._next_player] = 0

            # Advance to the next player
            self._advance_to_next_player()

            # Determine if the game should end
            if self._should_end_game():
                self._game_over = True

    def _should_end_game(self):
        """
    Determines if the game should end based on:
    - Minimum pot contributions.
    - Number of actions taken.
    """
        return (min(self.pot) >= 2
                or (len(self.bets) == 2 and self.bets[-1] == Action.FOLD)
                or len(self.bets) == 3)

    def _action_to_string(self, player, action):
        """Converts an action to a human-readable string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal:{action}"  # For chance actions (dealing cards)
        elif action == Action.FOLD:
            return "Fold"
        elif action == Action.CALL:
            return "Call"
        elif action == Action.BET_1_5:
            return "Bet 1.5 BB"
        elif action == Action.BET_3:
            return "Bet 3 BB"
        elif action == Action.BET_5:
            return "Bet 5 BB"
        elif action == Action.RAISE_2_5:
            return "Raise 2.5 BB"
        elif action == Action.RAISE_5:
            return "Raise 5 BB"
        elif action == Action.RAISE_8:
            return "Raise 8 BB"
        elif action == Action.ALL_IN:
            return "All-In"
        else:
            return "Unknown Action"

    def is_terminal(self):
        """Returns True if the game is over."""
        # The game ends if `_game_over` is True or other custom conditions
        return self._game_over

    def returns(self):
        """Calculates the total reward for each player at the end of the game."""
        if not self._game_over:
            return [0.0] * _NUM_PLAYERS

        # Distribute the pot based on the outcome
        winnings = sum(self.pot)  # Total pot size

        # Simple evaluation logic for now (assuming card comparison determines the winner)
        player_cards = self.cards[:
                                  _NUM_PLAYERS]  # First N cards dealt to players
        max_card = max(player_cards)  # Best card determines the winner
        winners = [
            i for i, card in enumerate(player_cards) if card == max_card
        ]

        # Split the pot among winners (handle ties)
        reward = winnings / len(winners)
        return [
            reward if i in winners else -reward / (_NUM_PLAYERS - 1)
            for i in range(_NUM_PLAYERS)
        ]

    def __str__(self):
        """String representation of the game state for debugging purposes."""
        card_str = " ".join([f"{rank}{suit}" for rank, suit in self.cards])
        bet_str = " | ".join([
            self._action_to_string(i % _NUM_PLAYERS, action)
            for i, action in enumerate(self.bets)
        ])
        pot_str = ", ".join(
            [f"Player {i}: {amt} BB" for i, amt in enumerate(self.pot)])
        current_player = f"Current Player: {self._next_player}"
        game_status = "Game Over" if self._game_over else "In Progress"

        return (f"Cards: {card_str}\n"
                f"Bets: {bet_str}\n"
                f"Pot: {pot_str}\n"
                f"{current_player}\n"
                f"Status: {game_status}")


class KickOffObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(
                f"Observation parameters not supported; passed {params}")

        # Determine which observation pieces we want to include.
        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS, ))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("private_card", len(_DECK), (len(_DECK), )))
        if iig_obs_type.public_info:
            if iig_obs_type.perfect_recall:
                pieces.append(
                    ("betting", len(Action), (_NUM_PLAYERS, len(Action))))
            else:
                pieces.append(
                    ("pot_contribution", _NUM_PLAYERS, (_NUM_PLAYERS, )))

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        # Build the named & reshaped views of the bits of the flat tensor.
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index + size].reshape(shape)
            index += size

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if "player" in self.dict:
            self.dict["player"][player] = 1
        if "private_card" in self.dict and len(state.cards) > player:
            card = state.cards[player]
            self.dict["private_card"][_DECK.index(card)] = 1
        if "pot_contribution" in self.dict:
            self.dict["pot_contribution"][:] = state.pot
        if "betting" in self.dict:
            for turn, action in enumerate(state.bets):
                self.dict["betting"][turn % _NUM_PLAYERS, action] = 1

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")
        if "private_card" in self.dict and len(state.cards) > player:
            pieces.append(f"card:{state.cards[player]}")
        if "pot_contribution" in self.dict:
            pot_str = " ".join([f"{amt} BB" for amt in state.pot])
            pieces.append(f"pot[{pot_str}]")
        if "betting" in self.dict and state.bets:
            action_str = " | ".join([
                self._action_to_string(turn % _NUM_PLAYERS, action)
                for turn, action in enumerate(state.bets)
            ])
            pieces.append(f"betting[{action_str}]")
        return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, KickOffGame)
