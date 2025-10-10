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

"""A Python wrapper for playing repeated games of poker using Pokerkit.

This module defines the `RepeatedPokerkit` game, which allows for playing
multiple hands of poker within a single game episode. It supports features like
blind schedules for tournament play and stack management across hands.
"""

import copy
import dataclasses

from absl import logging
import numpy as np
import pokerkit

from open_spiel.python.games import pokerkit_wrapper
import pyspiel


dataclass = dataclasses.dataclass
DEFAULT_NUM_HANDS = 1
INACTIVE_PLAYER_SEAT = -1
INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE = -1


@dataclass
class BlindLevel:
  num_hands: int
  small_blind: int
  big_blind: int


@dataclass
class BetSizeLevel:
  num_hands: int
  small_bet_size: int
  big_bet_size: int


@dataclass
class BringInLevel:
  num_hands: int
  bring_in: int


# WARNING: pickle serialization/deserialization of this does not work properly
# (specifically, when handling the nested kGame for the python_pokerkit_wrapper
# inputs). In meantime we have disabled testing the GameType for this game.
# TODO: b/445192003 - Investigate + resolve the nested kGame issue.
_DEFAULT_PARAMS = {
    "pokerkit_game_params": {
        "name": "python_pokerkit_wrapper",
        # See pokerkit_wrapper_test.py for other options, e.g.
        # "variant": "NoLimitTexasHoldem",
        # "num_players": 2,
        # "blinds": "50 100",
        # "stack_sizes": "20000 20000",
        # ...
    },
    "max_num_hands": DEFAULT_NUM_HANDS,
    "reset_stacks": False,
    "rotate_dealer": True,
    "blind_schedule": "",
    "bet_size_schedule": "",
    "bring_in_schedule": "",
    "first_button_player": (
        # Will be changed to the last player if not overridden.
        -1
    ),
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_repeated_pokerkit",
    long_name="Python Repeated Pokerkit",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=10,  # Arbitrarily chosen to match universal_poker
    min_num_players=2,  # Arbitrarily chosen to match universal_poker
    # Not yet supported. Instead use the observation string.
    provides_information_state_string=False,
    provides_observation_string=True,
    # Tensors are not yet supported. Instead use the corresponding strings.
    provides_information_state_tensor=False,
    provides_observation_tensor=False,
    # WARNING: pickle serialization/deserialization of our _DEFAULT_PARAMS does
    # not work properly. In meantime we have disabled testing the GameType for
    # this game.
    # TODO: b/445192003 - Investigate + resolve the serialization issue.
    parameter_specification=_DEFAULT_PARAMS,
)


# TODO(jhtschultz): Consider extracting common parsing logic into a separate
# helper function.
def parse_blind_schedule(blind_schedule_str: str) -> list[BlindLevel]:
  """Parses a blind schedule string into a list of BlindLevel objects.

  Port of ParseBlindSchedule from repeated_poker.cc.

  Parses blind schedule string of the form
    <blind_level_1>;...;<blind_level_n>
  where each blind level is of the form
    <num_hands>:<small_blind>/<big_blind>

  Args:
    blind_schedule_str: A string specifying the blind schedule. The format is a
      semicolon-separated list of blind levels, where each level is a
      colon-separated tuple of `num_hands` and `<small_blind>/<big_blind>`.

  Returns:
    A list of BlindLevel objects parsed from the input string.
  """
  if not blind_schedule_str:
    return []

  blind_levels = []
  levels_str = blind_schedule_str.removesuffix(";").split(";")
  for level_str in levels_str:
    parts = level_str.split(":")
    if len(parts) != 2:
      raise ValueError(f"Invalid blind schedule string: {blind_schedule_str}")
    blinds = parts[1].split("/")
    if len(blinds) != 2:
      raise ValueError(f"Invalid blind schedule string: {blind_schedule_str}")
    num_hands = int(parts[0])
    small_blind = int(blinds[0])
    big_blind = int(blinds[1])
    blind_levels.append(
        BlindLevel(
            num_hands,
            small_blind,
            big_blind,
        )
    )
  return blind_levels


def parse_bet_size_schedule(bet_size_schedule_str: str) -> list[BetSizeLevel]:
  """Parses a bet-size schedule string into a list of BetSizeLevel objects.

  Parsed bet-size schedule strings are of the form
    <bet_size_level_1>;...;<bet_size_level_n>
  where each bet-size level is of the form
    <num_hands>:<small_bet_size>/<big_bet_size>

  Args:
    bet_size_schedule_str: A string specifying the bet-size schedule. The format
      is a semicolon-separated list of bet-size levels, where each level is a
      colon-separated tuple of `num_hands` & `<small_bet_size>/<big_bet_size>`.

  Returns:
    A list of BetSizeLevel objects parsed from the input string.
  """
  if not bet_size_schedule_str:
    return []

  bet_sizes = []
  levels_str = bet_size_schedule_str.removesuffix(";").split(";")
  for level_str in levels_str:
    parts = level_str.split(":")
    if len(parts) != 2:
      raise ValueError(
          f"Invalid bet-size schedule string: {bet_size_schedule_str}"
      )
    bet_size_parts = parts[1].split("/")
    if len(bet_size_parts) != 2:
      raise ValueError(
          f"Invalid bet-size schedule string: {bet_size_schedule_str}"
      )
    num_hands = int(parts[0])
    small_bet_size = int(bet_size_parts[0])
    big_bet_size = int(bet_size_parts[1])
    bet_sizes.append(BetSizeLevel(num_hands, small_bet_size, big_bet_size))
  return bet_sizes


def parse_bring_in_schedule(bring_in_schedule_str: str) -> list[BringInLevel]:
  """Parses a bring-in schedule string into a list of BringInLevel objects.

  Parsed bring-in schedule strings are of the form
    <bring_in_level_1>;...;<bring_in_level_n>
  where each bring-in level is of the form
    <num_hands>:<bring_in>

  Args:
    bring_in_schedule_str: A string specifying the bring-in schedule. The format
      is a semicolon-separated list of bring-in levels, where each level is a
      colon-separated tuple of `num_hands` and `bring_in`.

  Returns:
    A list of BringInLevel objects parsed from the input string.
  """
  if not bring_in_schedule_str:
    return []
  bring_ins = []
  levels_str = bring_in_schedule_str.removesuffix(";").split(";")
  for level_str in levels_str:
    parts = level_str.split(":")
    if len(parts) != 2:
      raise ValueError(
          f"Invalid bring-in schedule string: {bring_in_schedule_str}"
      )
    num_hands = int(parts[0])
    bring_in = int(parts[1])
    bring_ins.append(BringInLevel(num_hands, bring_in))
  return bring_ins


# TODO: b/437724266 - Refactor to remove direct usage of protected fields and
# then remove the above disable. (For now we're intentionally using them to
# ensure that this code mimics the RepeatedPoker game's behavior as closely as
# possible during the initial commit.)
#
# pylint: disable=protected-access


class RepeatedPokerkit(pyspiel.Game):
  """Wrapper around PokerkitWrapper for playing multiple hands in one episode.

  Wrapper around PokerkitWrapper (i.e.  'double-wrapped' Pokerkit) for playing
  multiple hands within the same game episode. This enables simulating both cash
  games and tournaments.

  TODO: b/444333187 - does not fully support games besides NoLimitTexasHoldem
  yet. See the TODOs below for more details.

  Parameters:
    "pokerkit_game_params":  (required)
        Specifies the underlying pokerkit game to begin play. Note that the
        params will be updated to reflect the current state of the repeated
        game (e.g., number of players, blinds, etc.).
        The start of this string must be "python_pokerkit_wrapper(" or
        "python_pokerkit_wrapper_acpc_style(" and then end must be a closing
        paren ")".
    "max_num_hands": int (required)
        The maximum number of hands to play in the episode. This is a required
        parameter because it should be set deliberately in relation to the
        blind schedule when playing a tournament.
    "reset_stacks": bool (required)
        Whether to reset the stacks at the start of each hand. Required.
    "rotate_dealer": bool (optional, default=True)
        Whether to rotate the dealer at the start of each hand (which for
        pokerkit actually means rotating *the seats* so that the Button is the
        last stack provided to pokerkit). This defaults to true as it is always
        done in practice. NOTE: We will ignore this parameter being False if the
        resulting dealer player busts out when reset_stacks is False and there
        are still additional hands to play. I.e. in that situation the next
        not-yet-busted player will become the dealer going forwards.
    "blind_schedule": string (optional)
        Specifies the blind schedule for playing a tournament. The format is:
        <blind_level_1>;<blind_level_2>;...<blind_level_n> where each blind
        level is of the form <num_hands>:<small_blind>/<big_blind>. If play
        continues beyond the number of hands specified in the last blind level,
        the last blind level will continue to be used.
    "bet_size_schedule": string (optional)
        Specifies the bet-size schedule for playing a tournament. The format is:
        <bet_size_level_1>;<bet_size_level_2>;...<bet_size_level_n> where each
        bet-size level is of the form
        <num_hands>:<small_bet_size>/<big_bet_size>.  If play continues beyond
        the number of hands specified in the last bet-size level, the last
        bet-size level will continue to be used.
    "bring_in_schedule": string (optional)
        Specifies the bring-in schedule for playing a tournament. The format is:
        <bring_in_level_1>;<bring_in_level_2>;...<bring_in_level_n> where each
        bring-in level is of the form <num_hands>:<bring_in>. If play continues
        beyond the number of hands specified in the last bring-in level, the
        last bring-in level will continue to be used.
    "first_button_player": int (optional, default=-1)
        The player ID of the first player that will be the last seat, and
        therefore the Button, when performing seat assignments for the very
        first hand. (Pokerkit always assigns positions based on player order,
        with the last player corresponding to the button.)
        If left as the default value -1, this value will be overridden with the
        last player ID, effectively applying pokerkit's default interpretation
        onto the player IDs' order for the first hand. For example:
        - 1 in heads-up games (effectively making the BB the player with ID 0
          and the SB/BTN the player with ID 1)
        - 2 in 3-player games (effectively making the SB the player ID with ID
          0, the BB the player with ID 1, and the BTN the player with ID 2)
        - 5 in 6-player games (effectively making the SB the player ID with ID
          0, the BB the player with ID 1, the UTG player with ID 2 ... the BTN
          the player with ID 2)
        and so on.

  Returns are calculated by summing the returns for each hand.

  pylint: disable=g-bad-todo (copied from RepeatedPoker)
  TODO: jhtschultz - Support payout structures for tournaments.
  pylint: enable=g-bad-todo

  Note that this implementation imposes some slightly stricter assumptions on
  the game definition:
   1. Exactly two blinds.
   2. At least 2 rounds.
  Both of these are very standard assumptions for poker as played in practice.

  NOTE: we use simplified moving button rules. See
  https://en.wikipedia.org/wiki/Betting_in_poker#When_a_player_in_the_blinds_leaves_the_game
  This is common in online poker games generally speaking, and it is used here
  because the logic for remapping each hand to a gamedef becomes quite
  complex and highly error-prone when using dead button rules.

  NOTE: This implementation differs from RepeatedPoker in that it *does* support
  normal tournament rules for players with stack sizes less than one Big Blind.
  (Specifically: we allow players to stay in and play as per normal, rather than
  automatically eliminating them in such cases. This means that the total
  rewards should always add up to the number of total chips when the tournament
  started, as would be assumed for typical poker tournaments.)
  """

  def __init__(self, params=None):
    self.params = params
    self._pokerkit_game_params = {}
    self._max_num_hands = params.get("max_num_hands")
    if not self._max_num_hands or not isinstance(self._max_num_hands, int):
      raise ValueError(
          "max_num_hands must be an int >= 1. Got: %d" % self._max_num_hands
      )
    self._reset_stacks = params.get("reset_stacks")
    self._rotate_dealer = params.get("rotate_dealer")
    self._blind_schedule = params.get("blind_schedule")
    self._bet_size_schedule = params.get("bet_size_schedule")
    self._bring_in_schedule = params.get("bring_in_schedule")

    self._base_game_params = params.get("pokerkit_game_params")
    if not self._base_game_params:
      raise ValueError("pokerkit_game_params must be provided.")
    if self._base_game_params.get("name") not in [
        "python_pokerkit_wrapper",
        "python_pokerkit_wrapper_acpc_style",
    ]:
      raise ValueError(
          "pokerkit_game_params must parse to python_pokerkit_wrapper or"
          " python_pokerkit_wrapper_acpc_style game params. Instead got game"
          f" name {self._base_game_params.get('name')} inside "
          f" parsed params of: {self._base_game_params}"
      )
    # NOTE: Not used anywhere in the actual 'State' below! Just used to figure
    # out the basic game properties at the very very start of the repeated game
    # (e.g. num players, deck size, utility calculations, etc)
    #
    # Additionally, this may not have the seat rotation specified by the
    # first_button_player param (since we use properties of this game in order
    # to actually set that value in certain cases).
    self._base_game = pyspiel.load_game(
        pyspiel.game_parameters_to_string(self._base_game_params)
    )
    self._game_info = pyspiel.GameInfo(
        num_distinct_actions=self._calculate_num_distinct_actions(),
        max_chance_outcomes=self._base_game.deck_size,
        num_players=self._base_game.num_players(),
        min_utility=self._calculate_min_utility(),
        max_utility=self._calculate_max_utility(),
        utility_sum=0.0,
        max_game_length=self._calculate_max_game_length(),
    )

    provided_first_button_player = params.get("first_button_player")
    if (
        provided_first_button_player is None
        or provided_first_button_player < -1
        or provided_first_button_player >= self._base_game.num_players()
    ):
      raise ValueError(
          "first_button_player must be a valid player ID or the value -1. Got:"
          f" {provided_first_button_player}"
      )
    self._first_button_player = (
        provided_first_button_player
        if provided_first_button_player >= 0
        else self._base_game.num_players() - 1
    )

    super().__init__(_GAME_TYPE, self._game_info, self.params)

  def _calculate_num_distinct_actions(self):
    """Returns an upper bound on the number of distinct player actions.

    Port of RepeatedPokerGame::NumDistinctActions.

    The action space must be able to support all legal bet sizes. This remains
    constant in the case of resetting stacks, but otherwise will increase as
    stack sizes increases. Players can bet more than their opponent's stack
    size, so the upper bound is set to the total number of chips. This isn't a
    tight upper bound, but it's very close and should not matter for practical
    purposes.
    """
    if self._reset_stacks:
      return self._base_game.num_distinct_actions()
    else:
      return sum(self._base_game.stack_sizes)

  def _calculate_min_utility(self):
    """Returns an upper-bound (magnitude) on minimum utility for the game.

    Port of RepeatedPokerGame::MinUtility.

    Note: Unlike universal_poker, pokerkit_wrapper doesn't have fine-grained
    utility calculations based on e.g. fixed limit bet sizes, so this may be
    significantly less tight in some cases!
    """
    if self._reset_stacks:
      return self._base_game.min_utility() * self._max_num_hands
    else:
      return max(self._base_game.stack_sizes) * -1

  def _calculate_max_utility(self):
    """Returns an upper-bound on maximum utility for the game.

    Port of RepeatedPokerGame::MaxUtility.

    Note: Unlike universal_poker, pokerkit_wrapper doesn't have fine-grained
    utility calculations based on e.g. fixed limit bet sizes, so this will be
    significantly less tight!
    """
    if self._reset_stacks:
      return self._base_game.max_utility() * self._max_num_hands
    else:
      return sum(self._base_game.stack_sizes)

  def game_info(self):
    return self._game_info

  def _calculate_max_game_length(self):
    """Provides a (very rough) upper bound on the maximum game length."""
    # Warning: this may be too small if the effective stack sizes are being used
    # to determine the max game length in the future. Since later hands may be
    # longer than the initial hand's length, e.g. 1 chip vs 1000 chips heads up.
    return self._base_game.max_game_length() * self._max_num_hands

  def new_initial_state(self):
    return RepeatedPokerkitState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return RepeatedPokerkitObserver(self, iig_obs_type, params)

  # Not yet actually supported. Providing this 'return hardcoded value'
  # placeholder implementation instead of throwing an error to avoid crashing
  # anything in OpenSpiel that doesn't properly respect the
  # provides_information_state_tensor=False setting above.
  def information_state_tensor_size(self) -> int:
    logging.warning(
        "information_state_tensor_size() is not yet supported for"
        " RepeatedPokerkit. Returning a default value of 1 for now."
    )
    return 1


class RepeatedPokerkitState(pyspiel.State):
  """Represents an _OpenSpiel_ 'state' for the RepeatedPokerkit game.

  Port of repeated_poker::RepeatedPokerState.

  As described by the name, this class indeed wraps a `pokerkit.State` object
  and provides the necessary interface for OpenSpiel's `pyspiel.State`.
  """

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    self._hand_number = 0
    self._is_terminal = False
    # Represents the stack sizes at the start of the current hand. To access
    # the stacks for the hand in progress, use the underlying
    # PokerkitWrapperState.
    self._stacks: list[int] = None
    self._dealer = pyspiel.PlayerId.INVALID
    self._seat_to_player: dict[int, int] = {}
    self._player_to_seat: dict[int, int] = {}
    self._small_blind = INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
    self._big_blind = INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
    self._small_bet_size = INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
    self._big_bet_size = INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
    self._bring_in = INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE

    self._wrapped_state_hand_histories: list[str] = []
    # 2-D list where each inner list has length _num_players
    self._hand_returns: list[list[float]] = [[0.0] * game.num_players()]

    super().__init__(game)

    # Load the underlying pokerkit wrapper game and state that we're going to
    # actually send actions / update whenever starting a new hand. Also, save
    # its parameters back out so that we can mutate it within the state without
    # modifying the original game params that were passed in to the
    # RepeatedPokerkit constructor.
    #
    # NOTE: Do not confuse with the ._base_game on the RepeatedPokerGame above!!
    # (They are the same type, but only this one is actually 'used' for real
    # gameplay purposes.)
    self.pokerkit_wrapper_game: pyspiel.Game = pyspiel.load_game(
        # NOTE: This is the one time we load the original params in
        # ._base_game_params within the RepeatedPokerkitState. Everywhere else
        # we're going to use the ._pokerkit_game_params that are set below.
        pyspiel.game_parameters_to_string(game._base_game_params)
    )
    self._pokerkit_game_params = self.pokerkit_wrapper_game.get_parameters() | {
        "name": game._base_game_params.get("name")
    }
    self.pokerkit_wrapper_state: pyspiel.State = (
        self.pokerkit_wrapper_game.new_initial_state()
    )

    # Initial setup logic
    self._num_active_players = self.pokerkit_wrapper_game.num_players()
    self._hand_number = 0
    assert game._max_num_hands >= 1

    players = range(game.num_players())
    self._stacks = [
        self.pokerkit_wrapper_state._wrapped_state.starting_stacks[p]
        for p in players
    ]
    self._seat_to_player = {i: i for i in players}
    self._player_to_seat = {i: i for i in players}

    # Pokerkit has no equivalent of firstPlayer, so no need to port most of the
    # remaining RepeatedPokerState constructor logic here. BUT, we do at least
    # need to initialize dealer position to allow for users to specify an
    # arbitrary starting seat rotation that doesn't depend on which player ID
    # happens to be first or last.
    #
    # NOTE: This is a player ID, not a seat number!
    self._dealer = game._first_button_player
    if self._dealer < 0:
      raise ValueError(
          "first_button_player must be a valid player ID. Got:"
          f" {game._first_button_player}"
      )
    if self._dealer >= self.num_players():
      raise ValueError(
          "first_button_player must be a valid player ID. Got:"
          f" {game._first_button_player}"
      )
    if self._dealer >= self._num_active_players:
      raise ValueError(
          "first_button_player was unexepctedly high; ID. Got:"
          f" {game._first_button_player} but there are only"
          f" {self._num_active_players} active players upon initialization."
      )

    self._blind_schedule_levels: list[BlindLevel] = parse_blind_schedule(
        game._blind_schedule
    )
    self._bet_size_schedule_levels: list[BetSizeLevel] = (
        parse_bet_size_schedule(game._bet_size_schedule)
    )
    self._bring_in_schedule_levels: list[BringInLevel] = (
        parse_bring_in_schedule(game._bring_in_schedule)
    )
    if game._blind_schedule and not self._blind_schedule_levels:
      raise ValueError("Failed to parse blind schedule.")
    if game._bet_size_schedule and not self._bet_size_schedule_levels:
      raise ValueError("Failed to parse bet-size schedule.")
    if game._bring_in_schedule and not self._bring_in_schedule_levels:
      raise ValueError("Failed to parse bring-in schedule.")

    variant = game._base_game_params.get("variant")
    uses_blinds = (
        # NOTE: Assumes that if not specified, the default pokerkit_wrapper game
        # variant is going to be a Texas Holdem style game with blinds.
        variant is None
        or variant
        in pokerkit_wrapper.VARIANT_PARAM_USAGE["raw_blinds_or_straddles"]
    )
    if uses_blinds:
      if self._blind_schedule_levels:
        self._small_blind = self._blind_schedule_levels[0].small_blind
        self._big_blind = self._blind_schedule_levels[0].big_blind
      elif self.pokerkit_wrapper_state._wrapped_state.blinds_or_straddles:
        # Identify the small and big blind from the underlying pokerkit game.
        # (When set, it's always a list of integers with length equal to the
        # number of players in the pokerkit game)
        blinds = self.pokerkit_wrapper_state._wrapped_state.blinds_or_straddles
        num_blinds = 0
        for blind in blinds:
          if blind > 0:
            num_blinds += 1
            if self._small_blind == INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE:
              self._small_blind = blind
            else:
              self._big_blind = max(self._small_blind, blind)
              self._small_blind = min(self._small_blind, blind)
        if (
            any([blind != 0 for blind in blinds])
            or len(blinds) != self.num_players()
        ):
          assert num_blinds == 2
          assert self._small_blind > 0
          assert self._big_blind > 0
      else:
        # Many game variants (correctly) have no blinds.
        pass

    if not uses_blinds and game._rotate_dealer:
      raise ValueError(
          "Attempting to rotate dealers in a poker variant that does not have "
          " a rotating Button position. This is not supported. Poker variant "
          f" was: {variant}"
      )

    betting_structure = (
        self.pokerkit_wrapper_state._wrapped_state.betting_structure
    )
    if self._bet_size_schedule_levels:
      if betting_structure != pokerkit.state.BettingStructure.FIXED_LIMIT:
        raise ValueError(
            "Attempting to use a bet-size schedule, but the underlying pokerkit"
            " game does not have a fixed-limit betting structure. Aborting to"
            " avoid setting bet sizes to inaccurate values."
        )
      self._small_bet_size = self._bet_size_schedule_levels[0].small_bet_size
      self._big_bet_size = self._bet_size_schedule_levels[0].big_bet_size
    elif betting_structure == pokerkit.state.BettingStructure.FIXED_LIMIT:
      self._small_bet_size = self.pokerkit_wrapper_state.get_game().params.get(
          "small_bet"
      )
      self._big_bet_size = self.pokerkit_wrapper_state.get_game().params.get(
          "big_bet"
      )
    else:
      # Other game variants (correctly) have no bet sizes.
      logging.info("Found no small_bet or big_bet.")
      pass
    if self._bring_in_schedule_levels:
      if betting_structure != pokerkit.state.BettingStructure.FIXED_LIMIT:
        raise ValueError(
            "Attempting to use a bring-in schedule, but the underlying pokerkit"
            " game does not have a fixed-limit betting structure. Aborting to"
            " avoid setting bring-in to inaccurate values."
        )
      self._bring_in = self._bring_in_schedule_levels[0].bring_in
    elif betting_structure == pokerkit.state.BettingStructure.FIXED_LIMIT:
      self._bring_in = self.pokerkit_wrapper_state._wrapped_state.bring_in
    else:
      # Other game variants (correctly) have no bring-in.
      logging.info("Found no bring_in.")
      pass

    self.update_seat_rotation()
    self.update_blinds_bet_sizes_and_or_bring_in()
    self.update_pokerkit_wrapper()
    return

  def update_stacks(self):
    """Updates player stacks based on the outcome of the last hand.

    Port of RepeatedPokerState::UpdateStacks.

    Raises:
      RuntimeError: If `update_stacks` is called when the last pokerkit
        operation was not a `ChipsPulling` operation, which indicates the hand
        is not in a state where stacks can be safely updated.
    """
    if self.get_game()._reset_stacks:
      return
    # Unlike ACPC, pokerkit doesn't include chips in the payoff if they've
    # actively been bet / are in the pot. So, we need to be careful not to start
    # 'leaking' out chips from players' stacks if this is called at points where
    # the hand isn't actually over yet (e.g. start of the hand / any time the
    # hand is in progress)
    if all([
        isinstance(op, pokerkit.BlindOrStraddlePosting)
        for op in self.pokerkit_wrapper_state._wrapped_state.operations
    ]):
      return

    if not isinstance(
        self.pokerkit_wrapper_state._wrapped_state.operations[-1],
        pokerkit.ChipsPulling,
    ):
      raise RuntimeError(
          "Attempting to update stacks, but last pokerkit operation was not a"
          " ChipsPulling operation. Aborting to avoid setting stacks to"
          " inaccurate values (as there could be chips in the pot that we would"
          " effectively 'leak'). Operations was:",
          self.pokerkit_wrapper_state._wrapped_state.operations,
      )

    for player in range(self.num_players()):
      seat = self._player_to_seat[player]
      if seat != INACTIVE_PLAYER_SEAT:
        self._stacks[
            player
        ] += self.pokerkit_wrapper_state._wrapped_state.payoffs[seat]

  def update_seat_assignments_unrotated(self):
    """Updates our seat assignments for the next hand, without rotating seats.

    Port of RepeatedPokerState::UpdateSeatAssignments.

    As per the name, this doesn't handle the seat rotations necessary for
    moving the Button in pokerkit games. See instead
    `update_dealer_and_rotate_seats` below for more details.
    """
    if self.get_game()._reset_stacks:
      return

    self._player_to_seat = {}
    self._seat_to_player = {}
    next_open_seat = 0
    for player in range(self.num_players()):
      # Worst case the player should have 0 chips left; if they have a negative
      # number then there was likely a major bug in the game logic elsewhere.
      assert(self._stacks[player] >= 0)
      if self._stacks[player] == 0:
        self._player_to_seat[player] = INACTIVE_PLAYER_SEAT
      else:
        self._player_to_seat[player] = next_open_seat
        self._seat_to_player[next_open_seat] = player
        next_open_seat += 1
    self._num_active_players = next_open_seat

  def update_dealer(self):
    """Updates the dealer value. NOTE: does NOT rotate seats based on it!

    This is partially a port of RepeatedPokerState::UpdateDealer, with the
    exception of seat rotation logic (which we handle separately due to
    pokerkit differing from ACPC in how it works).

    Raises:
      RuntimeError: If an infinite loop is detected when rotating the dealer.
    """
    if (
        not self.get_game()._rotate_dealer
        # We MUST rotate the dealer position if the current dealer has busted
        # out (even if rotate_dealer is False). Otherwise we will be unable to
        # compute a proper seat rotation later on.
        and self._player_to_seat[self._dealer] != INACTIVE_PLAYER_SEAT
    ):
      return

    # Make sure our seat <-> player mappings are consistent prior to rotating.
    for player, seat in self._player_to_seat.items():
      if seat != INACTIVE_PLAYER_SEAT:
        assert self._seat_to_player[seat] == player
    for seat, player in self._seat_to_player.items():
      assert self._player_to_seat[player] == seat

    # NOTE: Dealer in terms of *player ID*, not seat number.
    self._dealer = (int(self._dealer) + 1) % self.num_players()

    # Arbitrary choice that we know DEFINITELY represents hitting an infinite
    # loop below if reached.
    definitely_infinite_loop_count = 10000
    # Make sure that we remember to update the above value if someone later
    # drastically increase the max number of players in the game.
    assert definitely_infinite_loop_count > self.num_players() * 10
    loop_counter = 0
    while self._player_to_seat[self._dealer] == INACTIVE_PLAYER_SEAT:
      self._dealer = (int(self._dealer) + 1) % self.num_players()

      loop_counter += 1
      if loop_counter >= definitely_infinite_loop_count:
        raise RuntimeError(
            "Detected infinite loop when attempting to rotate the dealer."
        )

  def update_seat_rotation(self):
    """Rotates seats based on self._dealer.

    Note: deliberately does NOT depend on self._rotate_dealer being True. This
    is because it is entirely valid for a game to not rotate dealers after each
    hand, while still _starting_ with a specific non-default rotation / button
    player that is not the first or last player ID. Similarly, even if we don't
    _normally_ rotate dealer with that setting set to False, in practice we may
    still have to rotate dealer in the event that the dealer busts out.

    Typically called during 1. initialization and 2. immediately after the
    dealer has been updated, following the end of a hand.

    Following this method call, if the game rotates dealers then the seats
    should be updated (to work with pokerkit's seat rotation logic) such that:

    For N > 2 seats: the Button is now the last seat and that the SB is the
    first seat, with players in between proceeding clockwise in standard
    post-flop game order. I.e.
    - SB is seat 0
    - BB (if any) is seat 1
    - UTG (if any) is seat 2
    - ...
    - Button is seat N-1

    For N=2 seats (headsup):
    - the SB / BTN is seat 0
    - the BB is seat 1

    For more details, see the pokerkit docs:
    https://pokerkit.readthedocs.io/en/stable/simulation.html#position
    https://pokerkit.readthedocs.io/en/stable/examples.html
    """
    # At each call of update_seat_assignments_unrotated it effetively 'resets'
    # the rotation entirely, so we cannot simply rotate each seat by 1 forwards.
    # Instead we must compute the exact amount to rotate the player with ID
    # self._dealer such that their seat will end up as the last seat in the
    # list, i.e. _num_active_players - 1.
    #
    # This is 'safe' / requires no modulo because _num_active_players will be
    # strictly greater than the seat number. Since - again - before ever calling
    # this method we would have already reset all the seats / removed any gaps
    # due to players busting out.
    assert self._player_to_seat[self._dealer] >= 0
    assert self.num_players() > self._player_to_seat[self._dealer]
    seat_rotation = (
        self._num_active_players - 1 - self._player_to_seat[self._dealer]
    )
    assert seat_rotation >= 0
    next_player_to_seat = {
        player: (
            (unrotated_seat + seat_rotation) % self._num_active_players
            if unrotated_seat != INACTIVE_PLAYER_SEAT
            else INACTIVE_PLAYER_SEAT
        )
        for player, unrotated_seat in self._player_to_seat.items()
    }
    assert next_player_to_seat[self._dealer] == self._num_active_players - 1
    assert (
        len([
            seat
            for seat in next_player_to_seat.values()
            if seat != INACTIVE_PLAYER_SEAT
        ])
        == self._num_active_players
    )
    self._player_to_seat = next_player_to_seat
    self._seat_to_player = {
        seat: player
        for player, seat in next_player_to_seat.items()
        if seat != INACTIVE_PLAYER_SEAT
    }
    assert all([player >= 0 for player in self._seat_to_player])
    assert len(self._seat_to_player) == self._num_active_players

  def update_blinds_bet_sizes_and_or_bring_in(self):
    """Updates the blinds for the next hand.

    Port of RepeatedPokerState::UpdateBlinds, with a couple of changes to
    accomodate pokerkit's differences from ACPC + also support small_bet,
    big_bet, and bring_in for other poker variants.
    """
    # (No need to update seat assignments for blinds like in
    # RepeatedPoker.UpdateBlinds. Since pokerkit instead directly maps each
    # stack to table position based on index, which we're handling elsewhere
    # via seat rotations.)

    def update_blind_bring_in_or_bet_size(level):
      if isinstance(level, BlindLevel):
        self._small_blind = level.small_blind
        self._big_blind = level.big_blind
      elif isinstance(level, BetSizeLevel):
        self._small_bet_size = level.small_bet_size
        self._big_bet_size = level.big_bet_size
      elif isinstance(level, BringInLevel):
        self._bring_in = level.bring_in
      else:
        raise ValueError(f"Unsupported schedule level type: {type(level)}")

    for schedule in [
        self._blind_schedule_levels,
        self._bring_in_schedule_levels,
        self._bet_size_schedule_levels,
    ]:
      if schedule:
        num_hands = 0
        for level in schedule:
          if self._hand_number < num_hands + level.num_hands:
            update_blind_bring_in_or_bet_size(level)
            break
          num_hands += level.num_hands
        else:
          # If we've exceeded the schedule (and never break-d above), use the
          # last level.
          update_blind_bring_in_or_bet_size(schedule[-1])

  def update_pokerkit_wrapper(self):
    """Updates the underlying PokerkitWrapper and State.

    Port of RepeatedPokerState::UpdateUniversalPoker, with a couple of changes
    to accommodate the differences between pokerkit and ACPC.
    """
    # NOTE: Updating the ._pokerkit_game_params here, NOT the original params
    # stored in the RepeatedPokerkitGame above!
    self._pokerkit_game_params["num_players"] = self._num_active_players
    stacks = [None] * self._num_active_players
    for p in range(self.num_players()):
      if self._player_to_seat[p] != INACTIVE_PLAYER_SEAT:
        # NOTE: Unlike RepeatedPoker we need to ensure that these are actually
        # in the specific 'rotated' seat order (due to how pokerkit works).
        stacks[self._player_to_seat[p]] = self._stacks[p]
    self._pokerkit_game_params["stack_sizes"] = " ".join(str(s) for s in stacks)

    if (
        self._small_blind != INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
        or self._big_blind != INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE
    ):
      self._pokerkit_game_params["blinds"] = (
          f"{self._small_blind} {self._big_blind}"
      )
    else:
      if "blinds" in self._pokerkit_game_params:
        del self._pokerkit_game_params["blinds"]

    # Unlike blinds or stack_sizes, the remaining params are all individually
    # passed in. Allowing us to easily handle them all in the exact same way.
    def set_or_remove_from_params(param, value):
      if value != INVALID_BLIND_BET_SIZE_OR_BRING_IN_VALUE:
        self._pokerkit_game_params[param] = value
      else:
        if param in self._pokerkit_game_params:
          del self._pokerkit_game_params[param]

    set_or_remove_from_params("bring_in", self._bring_in)
    set_or_remove_from_params("small_bet", self._small_bet_size)
    set_or_remove_from_params("big_bet", self._big_bet_size)

    # No need to determine a 'first player' like in
    # RepeatedPoker::UpdateUniversalPoker seat we handle this all via seat
    # rotations.

    # Finally, load a new pokerkit_wrapper game and state with the updated
    # params.
    self.pokerkit_wrapper_game = pyspiel.load_game(
        # NOTE: Using the just-updated ._pokerkit_game_params here, NOT the
        # original params passed into the RepeatedPokerkitGame above!
        pyspiel.game_parameters_to_string(self._pokerkit_game_params)
    )
    self.pokerkit_wrapper_state = self.pokerkit_wrapper_game.new_initial_state()

  def _apply_action(self, action):
    """Port of RepeatedPokerState::DoApplyAction for PokerkitWrapper."""
    self.pokerkit_wrapper_state.apply_action(action)
    if not self.pokerkit_wrapper_state.is_terminal():
      return

    assert not self.pokerkit_wrapper_state._wrapped_state.status

    # Record hand-level information.
    for i, per_seat_returns in enumerate(self.pokerkit_wrapper_state.returns()):
      player: int = self._seat_to_player[i]
      self._hand_returns[-1][player] = per_seat_returns
    wrapped_state = self.pokerkit_wrapper_state._wrapped_state
    self._wrapped_state_hand_histories.append(str(wrapped_state))
    # Terminate or start a new hand
    if self._hand_number + 1 == self.get_game()._max_num_hands:
      self._is_terminal = True
      # Necessary to ensure that if returning stacks inside to_struct() that it
      # gets the correct final stacks when in a terminal state.
      self.update_stacks()
      return
    self._hand_number += 1
    self._hand_returns.append([0.0] * self.num_players())
    self.update_stacks()
    self.update_seat_assignments_unrotated()
    if self._num_active_players == 1:
      # We're playing a tournament and we have a winner
      self._is_terminal = True
      return
    self.update_dealer()
    self.update_seat_rotation()
    self.update_blinds_bet_sizes_and_or_bring_in()
    self.update_pokerkit_wrapper()

  def current_player(self):
    """Returns the current player."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    elif self.is_chance_node():
      return pyspiel.PlayerId.CHANCE
    else:
      assert self.pokerkit_wrapper_state.current_player() not in [
          pyspiel.PlayerId.TERMINAL,
          pyspiel.PlayerId.CHANCE,
      ]
      assert self.pokerkit_wrapper_state.current_player() >= 0
      return self._seat_to_player[self.pokerkit_wrapper_state.current_player()]

  # pylint: disable=g-bad-todo (copied from RepeatedPoker)
  # TODO: jhtschultz - Switch to rewards?
  # pylint: enable=g-bad-todo
  def returns(self):
    """Returns the total returns for each player.

    Port of RepeatedPokerState::Returns.
    """
    assert self._hand_number + 1 == len(self._hand_returns)
    returns = [0.0] * self.num_players()
    if not self.is_terminal():
      return returns
    for hand_returns in self._hand_returns:
      for i in range(self.num_players()):
        returns[i] += hand_returns[i]
    return returns

  def to_string(self):
    """Returns a string representation of the state.

    Port of RepeatedPokerState::ToString.
    """
    return (
        f"Hand {self._hand_number}\n{self.pokerkit_wrapper_state.to_string()}"
    )

  # Deliberately not porting RepeatedPokerState::ObservationString. See
  # make_py_observer + RepeatedPokerkitObserver for the observer implementation.

  def clone(self):
    """Creates a copy of the state."""
    return copy.deepcopy(self)

  def _legal_actions(self, player):
    if self.is_terminal():
      raise ValueError(
          "Attempted to get legal actions for player {player} when the game is"
          " terminal."
      )
    if self.is_chance_node():
      raise ValueError(
          "Attempted to get legal actions for player {player} when the game is"
          " a chance node."
      )
    return self.pokerkit_wrapper_state.legal_actions(
        self._player_to_seat.get(int(player))
    )

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    assert self.pokerkit_wrapper_state.chance_outcomes()
    return self.pokerkit_wrapper_state.chance_outcomes()

  def _action_to_string(self, player, action):
    if self.is_chance_node():
      if player != pyspiel.PlayerId.CHANCE:
        raise ValueError("Chance actions must be played by the chance player.")
      # Deliberately NOT mapping to seats. Chance nodes don't have a seat and
      # will use the same ID value here + in the pokerkit_wrapper_state at all
      # times (unlike our 'real' players).
      return self.pokerkit_wrapper_state.action_to_string(player, action)
    else:
      if not player >= 0 or player >= self.num_players():
        raise ValueError(
            f"Player {player} is out of range [0, {self.num_players()})"
        )
      return self.pokerkit_wrapper_state.action_to_string(
          self._player_to_seat.get(int(player)), action
      )

  def is_terminal(self):
    return self._is_terminal

  def is_chance_node(self):
    return self.pokerkit_wrapper_state.is_chance_node()

  def __str__(self):
    return f"Hand number: {self._hand_number}\n{self.pokerkit_wrapper_state}"

  def to_struct(self) -> pyspiel.repeated_pokerkit.RepeatedPokerkitStateStruct:
    """Returns the current repeated_pokerkit state struct."""
    struct = pyspiel.repeated_pokerkit.RepeatedPokerkitStateStruct()
    struct.pokerkit_state_struct = self.pokerkit_wrapper_state.to_struct()
    struct.hand_number = self._hand_number
    struct.is_terminal = self._is_terminal
    struct.stacks = self._stacks
    struct.dealer = self._dealer
    struct.seat_to_player = self._seat_to_player
    struct.player_to_seat = self._player_to_seat
    struct.small_blind = self._small_blind
    struct.big_blind = self._big_blind
    struct.small_bet_size = self._small_bet_size
    struct.big_bet_size = self._big_bet_size
    struct.bring_in = self._bring_in
    struct.hand_returns = self._hand_returns
    return struct

  def to_json(self) -> str:
    return self.to_struct().to_json()


class RepeatedPokerkitObserver:
  """RepeatedPokerkit "Observer" for creating per-player state tensors and strings."""

  def __init__(self, game, iig_obs_types, params):
    """Initializes the RepeatedPokerkitObserver."""
    if params:
      raise ValueError(
          f"Observation parameters not supported; passed {params}."
      )
    self.game = game
    self.iig_obs_types = iig_obs_types
    if self.iig_obs_types is None:
      # Effectively: default to returning observation_* instead of
      # information_state_* if not specified. Since `perfect_recall` in practice
      # chooses between information_state_* vs observation_* from the
      # perspective of OpenSpiel. See e.g.
      # python/algorithms/generate_playthrough.py
      self.iig_obs_types = pyspiel.IIGObservationType(perfect_recall=False)
    self.params = params

    # Not actually used since `provides_observation_tensor=False`, but necessary
    # since some tests (incorrectly?) assume these always have been created /
    # test for the attribute's existence.
    self.tensor = np.array([])
    self.dict = {}

  def set_from(self, state, player) -> None:
    """No-op; see `provides_observation_tensor=False,` above.

    Defined only because some tests (incorrectly?) assume this method will
    always be defined.

    Args:
      state: The state to extract the observation from.
      player: The player to extract the observation for.
    """
    if player < 0 or player >= state.num_players():
      raise ValueError(
          f"Player {player} is out of range [0, {state.num_players()})"
      )
    pass

  def string_from(self, state, player) -> str:
    """An observation of `state` from the PoV of `player`, as a string.

    If the Observer's iig_obs_types `perfect_recall` is True, then this is
    equivalent to returning an `information_state_string`. If `perfect_recall`
    is False, then this is equivalent to returning an `observation_string`.

    Args:
      state: The state to extract the observation from.
      player: The player to extract the observation for.

    Returns:
      A string representation of the observation.
    """
    if self.iig_obs_types.perfect_recall:
      # Playthrough tests unfortunately do not properly respect the
      # `provides_information_state_string=False` setting in the game
      # definition, so we need to avoid raising an error here.

      # raise ValueError( "Information state string not supported.")
      return "Information state string not supported."

    assert player >= 0
    # pytype: disable=protected-access
    assert player < state.num_players()
    seat_id = state._player_to_seat.get(player)
    # pytype: enable=protected-access
    if seat_id == INACTIVE_PLAYER_SEAT:
      # pylint: disable=g-bad-todo (copied from RepeatedPoker)
      # TODO: jhtschultz - consider adding an observer to pokerkit and returning
      # the public information here. This would allow players who have been
      # eliminated to continue watching the game.
      # pylint: enable=g-bad-todo
      return "Game over.\n"
    pokerkit_wrapper_observer = state.pokerkit_wrapper_game.make_py_observer(
        iig_obs_type=pyspiel.IIGObservationType(perfect_recall=False)
    )
    return pokerkit_wrapper_observer.string_from(
        state.pokerkit_wrapper_state, seat_id
    )

# TODO: b/437724266 - remove once no longer disabling at the top of this file.
# pylint: enable=protected-access

pyspiel.register_game(_GAME_TYPE, RepeatedPokerkit)
