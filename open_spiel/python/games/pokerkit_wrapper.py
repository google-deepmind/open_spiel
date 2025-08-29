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
"""Pokerkit wrapper to provide support for multiple poker game variants."""

import copy
import dataclasses
import decimal
import inspect
import math

import numpy as np
import pokerkit

import pyspiel


# For documentation on using pokerkit see
# https://pokerkit.readthedocs.io/en/0.4/reference.html, and most notably
# https://pokerkit.readthedocs.io/en/0.4/reference.html#module-pokerkit.state

# Automatic steps we want performed by pokerkit without explicit function calls.
# The selection here is based of the example comments in pokerkit/pyspiel.py.
_DEFAULT_AUTOMATIONS = (
    pokerkit.Automation.ANTE_POSTING,
    pokerkit.Automation.BET_COLLECTION,
    pokerkit.Automation.BLIND_OR_STRADDLE_POSTING,
    pokerkit.Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    pokerkit.Automation.HAND_KILLING,
    pokerkit.Automation.CHIPS_PUSHING,
    pokerkit.Automation.CHIPS_PULLING,
    # WARNING: do NOT automate CARD_BURNING, if for whatever reason you decide
    # to go re-enabling it. (If you want to do card burning, to be faithful to
    # OpenSpiel you'll need to handle it via chance nodes / explicit actions!)
)


# To add support for additional pokerkit game variants add them to this map.
_SUPPORTED_VARIANT_CLASSES = [
    pokerkit.NoLimitTexasHoldem,
    pokerkit.NoLimitShortDeckHoldem,
    # Conversationally: "Limit Texas Hold'em"
    pokerkit.FixedLimitTexasHoldem,
    # Conversationally: "Seven Card Stud"
    pokerkit.FixedLimitSevenCardStud,
    # Conversationally: "Razz"
    pokerkit.FixedLimitRazz,
    # Conversationally: "Pot Limit Omaha (PLO)"
    pokerkit.PotLimitOmahaHoldem,
]
_SUPPORTED_VARIANT_MAP = {
    # e.g. { "NoLimitTexasHoldem": pokerkit.NoLimitTexasHoldem, ... }
    variant.__name__: variant
    for variant in _SUPPORTED_VARIANT_CLASSES
}

# pylint: disable=unused-private-name
_VARIANTS_SUPPORTING_ACPC_STYLE = [
    pokerkit.NoLimitTexasHoldem.__name__,
    pokerkit.FixedLimitTexasHoldem.__name__,
]
# pylint: enable=unused-private-name


def _get_variants_supporting_param(param_name: str) -> list[str]:
  return [
      name
      for name, cls in _SUPPORTED_VARIANT_MAP.items()
      if param_name in inspect.signature(cls).parameters.keys()
  ]


_VARIANT_PARAM_USAGE = {
    param: _get_variants_supporting_param(param)
    for param in ["min_bet", "bring_in", "small_bet", "big_bet"]
}

# WARNING: do not use comma-containing string param values here. Also, do not
# use strings for single-value integers. Instead respectively use space (" ")
# separation inside your strings values or plain numeric values (e.g. ints).
#
# If you don't, you may run into problems with the default pickle
# serialization/deserialization - e.g. errors or crashes on a round-trip.
# (This would potentially be fixable via custom __getstate__ and __setstate__
# methods, but we would have to define and maintain them ourselves if so.)
_DEFAULT_PARAMS = {
    # Matches pokerkit.NoLimitTexasHoldem.__name__
    # but kept as a hardcoded string to provide clarity to readers.
    # For more details, see the pokerkit docs.
    "variant": "NoLimitTexasHoldem",
    "num_players": 2,
    # Assumed to be "<SB> <BB>", i.e. space-separated numbers (within the
    # string). Leave as empty string for any games where blinds are irrelevant,
    # such as games using a "bring_in" (e.g. limit_razz and
    # limit_seven_card_stud).
    "blinds": "50 100",
    # Minimum bet size for no-limit games. Even if set, this will be ignored for
    # all other games (e.g. 'fixed limit' games).
    # As a convienence, if unset for no-limit games where `blinds` param was
    # set, we will attempt to default to the BigBlind instead of this default 2.
    "min_bet": 2,
    "num_streets": 4,
    # Relevant for games with public hole cards. Will be ignored by all
    # other games (including the default NoLimitTexasHoldem).
    "bring_in": 50,
    # Relevant for various FixedLimit ("limit") games. Will be ignored by all
    # other games (including the default NoLimitTexasHoldem).
    "small_bet": 25,
    "big_bet": 100,
    # NOTE: not yet officially supported. Added primarily for future-proofing.
    "antes": "0 0",
    # Stack sizes for each player. Should be a string containing num_players #
    # of space-separated integers.
    #
    # NOTE: Pokerkit considers "positions", not "seat numbers". As per the docs
    # https://pokerkit.readthedocs.io/en/0.4/simulation.html#position,
    # """
    # For non-heads-up button games, this means that the zeroth player will
    # always be the small blind, the first player will always be the big blind,
    # the second player will always be UTG, and so on until the last player who
    # will always be in the button and in position. The notion of position does
    # not really exist in stud games. In this case, the player to the immediate
    # left of the dealer should be in the zeroth position while the player to
    # the immediate right of the dealer should be in the last position.
    # """
    "stack_sizes": "20000 20000",
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_pokerkit_wrapper",
    long_name="Python Pokerkit",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=10,  # Arbitrarily chosen to match universal_poker
    min_num_players=2,  # Arbitrarily chosen to match universal_poker
    # See discussion in cl/798600930 for justification on not implementing them.
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_tensor=False,
    provides_observation_string=True,
    # TODO: b/437724266 - determine what this value does and whether it should
    # be True or False / what functionality it's actually controlling.
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS,
)

# Used in the PokerkitWrapperAcpcStyle subclass; see below for more details.
_GAME_TYPE_ACPC_STYLE = pyspiel.GameType(
    short_name="python_pokerkit_wrapper_acpc_style",
    long_name="Python Pokerkit ACPC Style (mimicking UniversalPoker / ACPC)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=10,  # Arbitrarily chosen to match universal_poker
    min_num_players=2,  # Arbitrarily chosen to match universal_poker
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_tensor=True,
    provides_observation_string=True,
    # It might be possible this should be True, unfortunately we're not actually
    # sure. (The original universal_poker.cc file we based most of this on
    # doesn't appear to set this at all!)
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS,
)


# NOTE: FOLD and CHECK_OR_CALL deliberately chosen to match kFold and kCall in
# universal_poker so that it's easier to port code over from it.
ACTION_FOLD = 0
# TODO: b/437724266 - this inherently limits our ability to ever raise / perform
# a completion bet to a size of 1 chip.
ACTION_CHECK_OR_CALL = 1

# NOTE: any other play actions >1 will be interpreted as part of a raise,
# reraise, or completion bet. Generally this follows typical poker assumptions
# i.e. "raise to X" = asserting a total contribution of X chips on the current
# street. However, we do provide an alternative interpretations of these actions
# via the PokerkitWrapperAcpcStyle subclass below in order to mimic the actions
# used in UniversalPoker / ACPC, which instead treats it as a "total
# contribution of X chips across **all** streets" (i.e. across the entire hand).


def _parse_values(param_str) -> list[int | decimal.Decimal]:
  """Convert " "-separated values in a ``str`` to numerical values via pokerkit.

  Args:
    param_str: A space-separated string of numerical values.

  Returns:
    A list of numerical values, which can be either `int` or `decimal.Decimal`.
  """
  if not param_str:
    return None
  return [pokerkit.parse_value(s) for s in param_str.split(" ")]


# ------------------------------------------------------------------------------
# --- "Base-class" PokerkitWrapper implementation. ---
#


# (Note: Subclassed below by PokerkitWrapperAcpcStyle)
class PokerkitWrapper(pyspiel.Game):
  """An OpenSpiel game that wraps pokerkit."""

  def __init__(self, params=None):
    game_type = self._game_type()

    if not params:
      params = {}
    # In addition to simply filling in for None or {} params, this also makes it
    # easier to set only a subset of the arguments for testing purposes.
    # NOTE: if this ends up too bug-prone, we may later remove this behavior in
    # the future in favor of some helper or factory methods
    self.params = _DEFAULT_PARAMS | params

    variant = self.params.get("variant")
    if variant not in _SUPPORTED_VARIANT_MAP:
      raise ValueError(f"Unknown poker variant: {variant}")

    # **** **** **** **** **** **** **** **** **** **** **** **** **** ****
    # WARNING: DO NOT CALL self.num_players() BELOW UNTIL AFTER COMPLETELY
    # FINISHING INITIALIZATION OF THIS OBJECT.
    #
    # Instead, for now only use this `num_players` local variable.
    # **** **** **** **** **** **** **** **** **** **** **** **** **** ****
    #
    # Context: Calling self.num_players() too early may result in segfaults.
    # Which have historically been very challenging + time-consuming to debug
    # due to lacking clear error messages.
    #
    # (Please also do not remove this comment; LLMs love quietly slipping in
    # usage of self.num_players or self.num_players(). This giant warning is
    # helpful for reminding them that doing so is a bad idea.)
    num_players = self.params["num_players"]
    if num_players > game_type.max_num_players:
      raise ValueError(
          f"Players in game ({num_players}) exceeds"
          f" max_players ({game_type.max_num_players})."
      )
    if num_players < game_type.min_num_players:
      raise ValueError(
          f"Players in game ({num_players}) is less than"
          f" min_players ({game_type.min_num_players})."
      )

    if variant in _VARIANT_PARAM_USAGE["min_bet"]:
      # As described in the docstring for _DEFAULT_PARAMS, if the user didn't
      # specify a min_bet but *did* specify blinds, we will attempt to set
      # min_bet to the Big Blind instead of the default like most other params.
      # TODO: b/437724266 - Clean up or simplify some of this logic.
      if "min_bet" in params:  # NOTE: Checking the *original* params.
        self.params["min_bet"] = params["min_bet"]
      # In most nolimit games this will just be the big blind, so it's
      # annoying and counterintuitive for users to have to specify it just to
      # avoid getting the default value we defined above.
      else:
        # NOTE: As above, checking the ORIGINAL params - not the self.params we
        # created inside this __init__().
        if "blinds" in params:
          parsed_blinds = _parse_values(params.get("blinds"))
          if parsed_blinds is None or len(parsed_blinds) != 2:
            raise ValueError(
                "blinds must be a space-separated list [Small Blind, Big"
                f" Blind]): ({params['blinds']})"
            )
          # NOTE: setting it on the self.params we created, meaning it will
          # (correctly) override the default values we pulled in iniiially.
          self.params["min_bet"] = parsed_blinds[1]
        # else: we will just use the default's param like everything else.

    else:  # variant not in _VARIANT_PARAM_USAGE["min_bet"]
      del self.params["min_bet"]

    self._is_bring_in_variant = variant in _VARIANT_PARAM_USAGE["bring_in"]

    for param in ["bring_in", "small_bet", "big_bet"]:
      if variant in _VARIANT_PARAM_USAGE[param]:
        # TODO: b/437724266 - Consider whether we should allow this to use
        # default values brought into self.params, rather than requiring the
        # user to _explicitly_ set these in all circumstances. (See e.g. how we
        # handle min_bet above by checking against the original unmodified input
        # params.)
        if param not in self.params:
          raise ValueError(
              f"Parameter {param} is required for variant {variant}"
          )
        if self.params.get(param) <= 1:
          raise ValueError(
              f"Parameter {param} must be >= 2, but is {self.params.get(param)}"
          )
      else:
        del self.params[param]

    self.stack_sizes = _parse_values(self.params["stack_sizes"])
    if self.stack_sizes is None or len(self.stack_sizes) != num_players:
      raise ValueError(
          "stack_sizes must be a space-separated list of "
          f" {num_players} values: ({self.params['stack_sizes']})"
      )

    self.blinds = _parse_values(self.params["blinds"])
    if self.blinds is None or len(self.blinds) != 2:
      raise ValueError(
          "blinds must be a space-separated list [Small Blind, Big Blind]): "
          f"({self.params['blinds']})"
      )

    # --- Card, and Deck Information ---
    # TODO: b/437724266 - depending on game variant we may want to use a
    # different deck in the future (e.g. Kuhn, shortdeck, etc).
    self._deck = pokerkit.Deck.STANDARD
    # TODO: b/437724266 - probably should make accessors for all of these to
    # avoid people mutating them out from underneath us.
    self.deck_size = len(self._deck)
    self.card_to_int = {card: i for i, card in enumerate(self._deck)}
    self.int_to_card = {i: card for i, card in enumerate(self._deck)}
    self.num_streets = self.params.get("num_streets")

    # --- GameInfo and GameType Setup ---
    # TODO: b/437724266 - Add support for calculating a tighter max_action
    # bound, e.g. in limit games where the player cannot bet their entire stack
    # at once.
    #
    # See FOLD / CHECK actions defined above for more details.
    num_distinct_actions = max(self.stack_sizes) + 1
    total_chips = sum(self.stack_sizes)

    # Initialize to None since the function is going to attempt to check this
    # to see if we cached it already.
    self._max_game_length_estimate = None
    self._max_game_length_estimate = self._calculate_max_game_length()

    self._game_info = pyspiel.GameInfo(
        num_distinct_actions=num_distinct_actions,
        max_chance_outcomes=self.deck_size,
        num_players=num_players,
        # TODO: b/437724266 - Add support for calculating tighter utility
        # bounds rather than this very rough estimate. This estimate currently
        # assumes that all games will always be able to get all the chips in,
        # even in e.g. limit games (where the stack sizes can be too large
        # relative to the bet limits for doing so to actually be possible).
        min_utility=float(-max(self.stack_sizes)),
        max_utility=float(total_chips - min(self.stack_sizes)),
        utility_sum=0.0,
        max_game_length=self._calculate_max_game_length(),
    )
    super().__init__(game_type, self._game_info, self.params)

  def game_info(self):
    return self._game_info

  def _calculate_max_game_length(self):
    """Estimate the maximum game length based on the input parameters."""
    if self._max_game_length_estimate is not None:
      return self._max_game_length_estimate
    # TODO: b/437724266 - Add support for calculating a tighter max_game_length
    # bound.
    return 1000

  def new_initial_state(self):
    return PokerkitWrapperState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return PokerkitWrapperObserver(self, params)

  def raise_error_if_player_out_of_range(self, player):
    if player < 0 or player >= self.num_players():
      raise ValueError(
          f"Player {player} is out of range [0, {self.num_players()})"
      )

  def wrapped_state_factory(self):
    """Creates the "wrapped" (not "PokerkitWrapper...") pokerkit.state object.

    This is used for both the 'real' wrapped pokerkit state object, AND also as
    needed to create 'dummy' Pokerkit objects for other purposes. (Hence why
    this is inside the main 'game' class and not the 'State' class below it.)

    Returns:
      A `pokerkit.State` object configured with the game parameters.
    """
    # TODO: b/437724266 - Stop directly accessing values from the params and
    # instead grab the values we need via getters.
    params = self.params
    variant = params.get("variant")
    poker_variant_class = _SUPPORTED_VARIANT_MAP.get(variant)
    if not poker_variant_class:
      raise ValueError(f"Unknown / unsupported poker variant: {variant}")

    args = {
        "automations": _DEFAULT_AUTOMATIONS,
        "ante_trimming_status": False,
        # Deliberately skipping the following since they vary by game variant:
        # - raw_blinds_or_straddles, min_bet
        # - bring_in, small_bet, big_bet
        # As per pokerkit docs:
        # """
        # In Pokerkit, the term ``raw`` is used to denote the fact that
        # they can be supplied in many forms and will be "parsed" or
        # "evaluated" further to convert them into a more ideal form.
        # For instance, ``0`` will be interpreted as no ante for all
        # players. Another value will be interpreted as that value as the
        # antes for all. ``[0, 2]`` and ``{1: 2} will be considered as the
        # big blind ante whereas ``{-1: 2}`` will be considered as the
        # button ante.
        # """
        "raw_antes": _parse_values(params.get("antes")),
        "raw_starting_stacks": _parse_values(params.get("stack_sizes")),
        "player_count": params.get("num_players"),
    }
    if not self.is_bring_in_variant():
      blinds_or_straddles = _parse_values(params.get("blinds"))
      args["raw_blinds_or_straddles"] = blinds_or_straddles

    for param in ["min_bet", "bring_in", "small_bet", "big_bet"]:
      assert not (
          param in params and variant not in _VARIANT_PARAM_USAGE[param]
      )
      assert not (
          param not in params and variant in _VARIANT_PARAM_USAGE[param]
      )
      if param in params:
        args[param] = params.get(param)

    # Disable card burning everywhere. Card burning has no purpose here since
    # our cards's backs cannot be marked, unlike with e.g. real physical poker
    # cards).
    #
    # We use this funny-looking 'call create_state() then pass streets back into
    # State() constructor' rather than setting anything directly / maually using
    # 'replace' on the top-level state object as per guidance in
    # https://pokerkit.readthedocs.io/en/latest/tips.html#read-only-and-read-write-values
    helper_state = poker_variant_class.create_state(**args)
    streets_override = list(helper_state.streets)
    for i in range(len(streets_override)):
      streets_override[i] = dataclasses.replace(
          streets_override[i], card_burning_status=False
      )
    # (The purpose of this is creating a fresh identical State object to the
    # above line `helper_state = poker_variant_class.create_state(**args)`,
    # except with different street inputs controlling card burning as discussed
    # above, and WITHOUT directly mutating any of the internal pokerkit.State
    # fields - as requested by the pokerkit docs.)
    #
    # NOTE: This is NOT a PokerkitWrapperState object, but a pokerkit.State
    # object (ie the 'wrapped' object).
    returned_state = pokerkit.State(
        automations=helper_state.automations,
        deck=helper_state.deck,
        hand_types=helper_state.hand_types,
        streets=streets_override,  # Note that this uses the 'override' version!
        betting_structure=helper_state.betting_structure,
        ante_trimming_status=helper_state.ante_trimming_status,
        raw_antes=helper_state.antes,
        raw_blinds_or_straddles=helper_state.blinds_or_straddles,
        bring_in=helper_state.bring_in,
        raw_starting_stacks=helper_state.starting_stacks,
        player_count=helper_state.player_count,
        mode=helper_state.mode,
        starting_board_count=helper_state.starting_board_count,
        divmod=helper_state.divmod,
        rake=helper_state.rake,
    )

    # Double check that we didn't make mistakes when disabing card burning just
    # now. (This is a bit overkill, but we REALLY don't want there to be any
    # chance of accidentally leaving card burning enabled anywhere).
    for street in returned_state.streets:
      assert not street.card_burning_status
    return returned_state

  def _game_type(self):
    return _GAME_TYPE

  def is_bring_in_variant(self):
    return self._is_bring_in_variant


class PokerkitWrapperState(pyspiel.State):
  """Represents an _OpenSpiel_ 'state' for the PokerkitWrapper game.

  As described by the name, this class indeed wraps a `pokerkit.State` object
  and provides the necessary interface for OpenSpiel's `pyspiel.State`.
  """

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._wrapped_state: pokerkit.State = game.wrapped_state_factory()

  def current_player(self):
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    elif self.is_chance_node():
      return pyspiel.PlayerId.CHANCE
    else:
      return self._wrapped_state.actor_index

  def _legal_actions(self, player):
    self.get_game().raise_error_if_player_out_of_range(player)

    if self.is_terminal():
      return []
    actions = []

    # Handling player nodes first since they're the most straightforward.
    if self._wrapped_state.can_fold():
      actions.append(ACTION_FOLD)
    if self._wrapped_state.can_check_or_call():
      actions.append(ACTION_CHECK_OR_CALL)
    if self._wrapped_state.can_complete_bet_or_raise_to():
      actions = actions + list(
          range(
              max(
                  self._wrapped_state.min_completion_betting_or_raising_to_amount,
                  # bet size 0 is equivalent to check/call, and bet size 1 is
                  # disallowed due to us needing that number for FOLD
                  2,
              ),
              self._wrapped_state.max_completion_betting_or_raising_to_amount
              + 1,
          )
      )
    if (
        self._wrapped_state.can_check_or_call()
        or self._wrapped_state.can_fold()
        or self._wrapped_state.can_complete_bet_or_raise_to()
    ):
      return sorted(actions)

    # Otherwise we're in a chance node and the actions must be dealable cards!
    # TODO: b/437724266 - extract out this to helper function for here +
    # chance_outcomes to share
    return sorted(
        self.get_game().card_to_int[c]
        for c in self._wrapped_state.get_dealable_cards()
    )

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted([
        self.get_game().card_to_int[c]
        for c in self._wrapped_state.get_dealable_cards()
    ])
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    # Handling player actions first since it's the more striaghtforward case.
    if not self.is_chance_node():
      if action == ACTION_FOLD:
        self._wrapped_state.fold()
      elif action == ACTION_CHECK_OR_CALL:
        self._wrapped_state.check_or_call()
      else:
        # Warning: this means that it's impossible to compelete bets or raise to
        # sizes that equal one of the actions above!
        self._wrapped_state.complete_bet_or_raise_to(action)
      return

    # else it's a chance node (which is a bit more complicated):

    card_from_action = self.get_game().int_to_card[action]
    if card_from_action not in self._wrapped_state.get_dealable_cards():
      raise ValueError(
          f"Chance action maps to non-dealable card {card_from_action}: "
          f"{self._wrapped_state.can_deal_hole()} "
          f"{self._wrapped_state.can_deal_board()}"
      )
    if self._wrapped_state.can_deal_hole():
      if self._wrapped_state.can_deal_board():
        raise ValueError(
            "Chance node with both dealable hole and board cards: "
            f"{self._wrapped_state.can_deal_hole()} "
            f"{self._wrapped_state.can_deal_board()}"
        )
      self._wrapped_state.deal_hole(card_from_action)
    elif self._wrapped_state.can_deal_board():
      self._wrapped_state.deal_board(card_from_action)
    else:
      raise ValueError(
          "Chance node with no dealable cards: "
          f"{self._wrapped_state.can_deal_hole()} "
          f"{self._wrapped_state.can_deal_board()}"
      )

  def _action_to_string(self, player, action):
    if action == ACTION_FOLD:
      return "Fold"
    if action == ACTION_CHECK_OR_CALL:
      amount = self._wrapped_state.checking_or_calling_amount
      return "Check" if amount == 0 else f"Call({amount})"
    return f"Bet/Raise to {action}"

  def is_terminal(self):
    # .status will be False IFF the game has not started or if it's been
    # terminated. As such, by ruling out game not being started we can verify
    # if we are in a terminal state.
    #
    # For more details, see the comments in pokerkit/state.py.
    game_never_started = (
        self._wrapped_state.can_deal_hole() and not self._wrapped_state.status
    )
    if game_never_started:
      return False

    return not self._wrapped_state.status

  def is_chance_node(self):
    # Chance nodes should only refer to situations where a card is being dealt
    # from the deck. (Technically we could do the same for burned cards, but
    # we're automating that phase entirely, so it would regardless always be
    # false).
    return (
        self._wrapped_state.can_deal_hole()
        or self._wrapped_state.can_deal_board()
    )

  def returns(self):
    if not self.is_terminal():
      return [0.0 for _ in range(self.get_game().num_players())]
    payoffs = [float(p) for p in self._wrapped_state.payoffs]

    # Double check that this is consistent with the rest of pokerkit.State's
    # internal stack and starting_stack values to detect any bugs.
    for player, payoff in enumerate(payoffs):
      assert payoff == (
          self._wrapped_state.stacks[player]
          - self._wrapped_state.starting_stacks[player]
      )

    return payoffs

  def clone(self):
    raise NotImplementedError(
        "Cloning is not yet supported for PokerkitWrapperState."
    )

  def deepcopy_wrapped_state(self) -> pokerkit.State:
    """Create and return a deepcopy of the 'wrapped' pokerkit.state."""
    return copy.deepcopy(self._wrapped_state)

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    state: pokerkit.State = self._wrapped_state
    parts = []
    parts.append(f"Stacks: {state.stacks}")
    parts.append(f"Bets: {state.bets}")
    parts.append(f"Board: {state.board_cards}")
    parts.append(f"Hole Cards: {state.hole_cards}")
    parts.append(f"Pots: {list(state.pots)}")
    parts.append(f"Operations: {state.operations}")
    return " | ".join(parts)


class PokerkitWrapperObserver:
  """Pokerkit "Observer" for creating per-player state tensors and strings."""

  def __init__(self, game, params):
    """Initializes the PokerkitWrapperObserver."""
    if params:
      raise ValueError(
          f"Observation parameters not supported; passed {params}."
      )
    self.game = game

  def set_from(self, state, player) -> None:
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    state.get_game().raise_error_if_player_out_of_range(player)
    # TODO: b/437724266 - Add full support for othe game variants.
    raise NotImplementedError("Not implemented yet.")

  def string_from(self, state, player) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    state.get_game().raise_error_if_player_out_of_range(player)
    # TODO: b/437724266 - Add full support for othe game variants.
    raise NotImplementedError("Not implemented yet.")


# ------------------------------------------------------------------------------
# --- "ACPC-style" PokerkitWrapper subclass to mimick UniversalPoker ---
class PokerkitWrapperAcpcStyle(PokerkitWrapper):
  """A subclass of PokerkitWrapper that mimicks ACPC-wrapping UniversalPoker."""

  def __init__(self, params=None):
    if not params:
      params = _DEFAULT_PARAMS

    variant = params.get("variant")
    if not variant:
      variant = _DEFAULT_PARAMS.get("variant")
    if variant not in _VARIANTS_SUPPORTING_ACPC_STYLE:
      raise ValueError(
          "PokerkitWrapperAcpcStyle only supports 'limit_holdem' and "
          "'nolimit_holdem' variants."
      )
    # Note: on _game_type() to ensure that we use the correct game type in the
    # base class's init().
    super().__init__(params)

  def _game_type(self):
    """Returns the game type of the underlying game.

    Needed to ensure that inside the super().__init()__ we use the correct
    one (not the base class's).
    """
    return _GAME_TYPE_ACPC_STYLE

  def _calculate_max_game_length(self):
    """Estimate maximum game like UniversalPokerGame::MaxGameLength()."""
    if self._max_game_length_estimate is not None:
      return self._max_game_length_estimate

    dummy_wrapped_state = self.wrapped_state_factory()
    num_players = dummy_wrapped_state.player_count
    length = 1
    streets = dummy_wrapped_state.streets
    for street in streets:
      length += street.board_dealing_count
      length += len(street.hole_dealing_statuses) * num_players
    length += len(streets) * (num_players)
    max_stack = max(self.stack_sizes)
    max_blind = max(self.blinds)
    max_num_raises = math.ceil((max_stack + max_blind - 1) / max_blind)
    length += max_num_raises * (num_players - 1)
    return length

  def observation_tensor_shape(self) -> list[int]:
    return [2 * (self.num_players() + self.deck_size)]

  def new_initial_state(self):
    return PokerkitWrapperAcpcStyleState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return PokerkitWrapperAcpcStyleObserver(self, params)

  def game_type(self):
    return _GAME_TYPE_ACPC_STYLE


class PokerkitWrapperAcpcStyleState(PokerkitWrapperState):
  """State class for PokerkitWrapperAcpcStyle."""

  def information_state_tensor(self, player):
    """See google3/third_party/open_spiel/spiel.h.

    Deliberately matches
    https://source.corp.google.com/piper///depot/google3/third_party/open_spiel/games/universal_poker/universal_poker.cc;l=400
    as much as realistically possible.

    Layout is as follows:

    my player number: num_players bits
    my cards: Initial deck size bits (1 means you have the card), i.e.
              MaxChanceOutcomes() = NumSuits * NumRanks
    public cards: Same as above, but for the public cards.
     action sequence: (max game length)*2 bits (fold/raise/call/all-in)
     action sequence sizings: (max game length) integers with value >= 0,
                               0 when corresponding to 'deal' or 'check'.

    (+ another initial deck size * num_players bits for 'up-cards' in bring-in
      variants, ie games that also have public hole cards)

    Args:
      player: The player ID for whom to generate the information state tensor.

    Returns:
      A numpy array representing the information state for the given player.
    """
    game = self.get_game()
    game.raise_error_if_player_out_of_range(player)

    wrapped_state = self._wrapped_state
    num_players = game.num_players()
    deck_size = game.deck_size
    max_game_length = game.max_game_length()
    max_chance_outcomes = game.max_chance_outcomes()

    # --- Initialize the tensor (with length matching universal_poker's) ---
    #
    # Specifically: shape should match InformationStateTensorShape() in
    # http://google3/third_party/open_spiel/games/universal_poker/universal_poker.cc?l=1131
    tensor_length = (
        num_players + 2 * max_chance_outcomes + (2 + 1) * max_game_length
    )
    # TODO: b/437724266 - consider adding this back in later if supporting
    # other variants in this fashion
    # if game.is_bring_in_variant():
    # tensor_length += num_players * max_chance_outcomes
    self.tensor = np.zeros(tensor_length, dtype=np.float32)

    # --- Fill in the tensor ---
    offset = 0
    # Mark who I am.
    self.tensor[player] = 1.0
    offset += num_players

    # Mark my private cards
    # (Note: it may be much more efficient to iterate over the cards of the
    # player, rather than iterating over all the cards. Consider updasting if
    # this causes a performance bottleneck in the future.)
    #
    # Technically we could probably use 'hole_cards' since we're never going to
    # be a poker variant with public hole cards (like e.g. 'bring-in' variants),
    # but in case this gets copy/pasted elsewhere it's probably good to be
    # defensive and explicitly use 'down_cards' instad.
    my_hole_cards = list(wrapped_state.get_down_cards(player))
    for card, i in game.card_to_int.items():
      if card in my_hole_cards:
        self.tensor[offset + i] = 1.0
    offset += deck_size

    # Mark the public cards
    board_cards = wrapped_state.board_cards
    # Board cards are nested inside a list, so we need to flatten them first.
    board_cards = [card[0] for card in board_cards]
    for card, i in game.card_to_int.items():
      if card in board_cards:
        self.tensor[offset + i] = 1.0
    offset += deck_size

    i = 0
    # Must be separate since universal_poker action_sequence doesn't actually
    # include entries for chance nodes (e.g. dealing cards).
    for operation in wrapped_state.operations:
      # Goal: mark action sequences to match Pokerkit as follows:
      # - Call = 1 0
      # - Raise = 0 1
      # - All-in = 1 1
      # - Fold = 0 0
      # - Deal = 0 0
      #
      # plus, (for legacy reasons) the corresponding *unabstracted* sizings
      # another max_game_length down in the vector (ie no longer as 0s or 1s).
      #
      # In total this will take up (2 + 1) * max_game_length entries.
      #
      # Relevant player operations in Pokerkit include:
      # - CompletionBettingOrRaisingTo
      # - CheckingOrCalling
      # - Folding
      #
      # Relevant card operations in Pokerkit include:
      # - HoleDealing
      # - BoardDealing
      #
      # Note: we have to manually increment i only during matching operations
      # since the Pokerkit operations don't necessarily line up with the
      # universal_poker action_sequence. (There will be other irrelevant things
      # like e.g. BlindOrStradlePostiong, BetCollection, etc).
      if isinstance(operation, pokerkit.CheckingOrCalling):
        self.tensor[offset + i * 2] = 1.0
        self.tensor[offset + (i * 2) + 1] = 0.0
        # Judging by
        # https://source.corp.google.com/piper///depot/google3/third_party/open_spiel/games/universal_poker/universal_poker.h;l=160;bpv=1;bpt=1;rcl=796286203
        # it sounds like universal_poker only records '0' for call sizings??
        # In which case, we should actaully *NOT* record the size here...
        #
        # self.tensor[
        #     offset + choice_index + max_game_length*2] = operation.amount
        #
        # In the future we may want to record it anyways though. And, regardless
        # we do need to incrememnt choice_index since it DOES show up in the
        # sequence / takes a slot in the tensor (just, left to 0).
        i += 1
      if isinstance(operation, pokerkit.CompletionBettingOrRaisingTo):
        # WARNING: we don't distinguish between 'all in' and normal raise
        # here, which means it won't exactly match universal_poker's tensor.
        self.tensor[offset + i * 2] = 0.0
        self.tensor[offset + (i * 2) + 1] = 1.0
        # Also mark the unabstracted action sequence sizings
        # WARNING: this is the amount *being raised at this moment*, not the
        # total contribution by the player across the course of the game up to
        # this point (Which is what universal_poker records here).
        #
        # TODO: b/437611559 - Consider updating to match universal_poker's
        # tensor.
        self.tensor[offset + i + max_game_length * 2] = operation.amount
        i += 1
      # WARNING: THIS MAKES IT VERY DIFFICULT TO TELL FOLDING FROM DEALING IN
      # THE TENSOR HERE.
      #
      # See also this TODO from universal_poker.cc:
      # """"
      # TODO(author2): Should this be 11?
      # """"
      if isinstance(operation, pokerkit.Folding):
        self.tensor[offset + i * 2] = 0.0
        self.tensor[offset + (i * 2) + 1] = 0.0
        # Folds get 0 in the tensor, so no need to mutate anything here...
        # ... but they do TAKE UP A SLOT so we need to increment i still
        i += 1
      # max_game_length * 2 for one-hot encoding of fold/raise/call.
      if isinstance(operation, pokerkit.HoleDealing) or isinstance(
          operation, pokerkit.BoardDealing
      ):
        self.tensor[offset + i * 2] = 0.0
        self.tensor[offset + (i * 2) + 1] = 0.0
        i += 1
    # *2 for the hone-hot encoding fold/raise/call, plus *1 for the sizings.
    offset += max_game_length * 3

    # TODO: b/437724266 - consider adding this back in later if supporting
    # other variants in this fashion
    # if game.is_bring_in_variant():
    #   for p in range(num_players):
    #     up_cards = wrapped_state.get_up_cards(p)
    #     for card, i in game.card_to_int.items():
    #       if card in up_cards:
    #         self.tensor[offset + i] = 1.0
    #   offset += num_players * max_chance_outcomes

    assert tensor_length == offset
    return self.tensor

  def information_state_string(self, player):
    if player < 0 or player >= self.get_game().num_players():
      raise ValueError(
          f"Player {player} is out of range [0,"
          f" {self.get_game().num_players()})."
      )

    pot = 0
    for p in self._wrapped_state.pots:
      if player in p.player_indices:
        # NOTE: Might need to be adjusted in the future if we add support for
        # ranked games, as this technically includes BOTH ranked AND unraked
        # amounts. For more details see
        # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.Pot
        # (e.g. you might need to change it to p.unraked_amount.)
        pot += p.amount
    for pl in range(self.get_game().num_players()):
      if pl != player:
        pot += self._wrapped_state.bets[pl]
    remaining_money: str = " ".join(
        [str(s) for s in self._wrapped_state.stacks]
    )
    betting_sequence = []
    for operation in self._wrapped_state.operations:
      if isinstance(operation, pokerkit.CheckingOrCalling):
        betting_sequence.append("c")
      elif isinstance(operation, pokerkit.Folding):
        betting_sequence.append("f")
      elif isinstance(operation, pokerkit.CompletionBettingOrRaisingTo):
        betting_sequence.append(f"r{operation.amount}")
    betting_sequence = "".join(betting_sequence)
    street_index = self._wrapped_state.street_index
    private = " ".join([
        str(c.rank) + str(c.suit)
        for c in self._wrapped_state.get_censored_hole_cards(player)
    ])
    board = " ".join([
        str(c[0].rank) + str(c[0].suit) for c in self._wrapped_state.board_cards
    ])
    return (
        f"[Round {street_index}]"
        f"[Player: {player}]"
        f"[Pot: {pot}]"
        f"[Money: {remaining_money}]"
        f"[Private:{private}]"
        f"[Public: {board}]"
        f"[Sequences: {betting_sequence}]"
    )


class PokerkitWrapperAcpcStyleObserver(PokerkitWrapperObserver):
  """Observer class for PokerkitWrapperAcpcStyle."""

  def __init__(self, game, params):
    super().__init__(game, params)
    # Reinitialize tensor based on ACPC game shape
    self.tensor_size = game.observation_tensor_shape()
    self.tensor = np.zeros(self.tensor_size, dtype=np.float32)
    self.dict = {}

  def set_from(self, state, player) -> None:
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`.

    WARNING: THIS OCCURS VIA MUTATION! Note how unlike with the C++ functions,
    this function doesn't return anything (here, and generally all the other
    python Observer stuff too).

    Intentionally designed to mimick the logic in
    UniversalPokerState::ObservationTensor (which is per-player).

    Args:
      state: The `PokerkitWrapperState` to observe.
      player: The player ID for whom to generate the observation.

    Returns:
      None. This method updates the internal `tensor` and `dict` in place.
    """
    game = state.get_game()
    game.raise_error_if_player_out_of_range(player)

    # TODO: b/437724266 - add mutation of the 'dict' thing as well (in addition
    # to just the tensor like we're doing currently).

    self.tensor.fill(0)
    game = state.get_game()
    wrapped_state_clone: pokerkit.State = state.deepcopy_wrapped_state()
    num_players = game.num_players()
    num_cards = len(wrapped_state_clone.deck)

    offset = 0
    # Mark who I am.
    self.tensor[player] = 1.0
    offset += num_players

    # Mark private hole cards
    for card in wrapped_state_clone.get_down_cards(player):
      self.tensor[offset + game.card_to_int[card]] = 1.0
    offset += num_cards

    # Mark the public cards
    for card in wrapped_state_clone.board_cards:
      self.tensor[offset + game.card_to_int[card[0]]] = 1.0
    offset += num_cards

    # Mark the contribution of each player to the pot.
    for p in range(num_players):
      contribution = (
          wrapped_state_clone.starting_stacks[p] - wrapped_state_clone.stacks[p]
      )
      self.tensor[offset + p] = float(contribution)
    offset += num_players
    assert offset == self.tensor_size[0]
    return

  def string_from(self, state, player) -> str:
    """Observation of `state` from the PoV of `player`, as a string."""
    game = state.get_game()
    game.raise_error_if_player_out_of_range(player)

    # Note: we're mainly just using this for read access. So, if this is too
    # inefficient, as an optimization we could instead just directly access the
    # protected variable (i.e. deliberately disregarding its protected-ness) -
    # i.e. so long as we're careful to avoid ever mutating it from inside this
    # function.
    wrapped_state_clone: pokerkit.State = state.deepcopy_wrapped_state()
    pot = 0
    for p in wrapped_state_clone.pots:
      if player in p.player_indices:
        # NOTE: Might need to be adjusted in the future if we add support for
        # ranked games, as this technically includes BOTH ranked AND unraked
        # amounts. For more details see
        # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.Pot
        # (e.g. you might need to change it to p.unraked_amount.)
        pot += p.amount
    for pl in range(state.get_game().num_players()):
      if pl != player:
        pot += wrapped_state_clone.bets[pl]
    remaining_money: str = " ".join(
        [str(s) for s in wrapped_state_clone.stacks]
    )
    street_index = wrapped_state_clone.street_index
    private = " ".join([
        str(c.rank) + str(c.suit)
        for c in wrapped_state_clone.get_censored_hole_cards(player)
    ])
    # Total contribution to the pot by the player.
    contribution = (
        wrapped_state_clone.starting_stacks[player]
        - wrapped_state_clone.stacks[player]
    )
    return (
        f"[Round {street_index}]"
        f"[Player: {player}]"
        f"[Pot: {pot}]"
        f"[Money: {remaining_money}]"
        f"[Private:{private}]"
        # Called "ante" to match universal_poker, but in reality this has
        # nothing to do with _actual_ pokerkit antes.
        f"[Ante: {contribution}]"
    )


# ------------------------------------------------------------------------------

# TODO: b/437724266 - Re-enable registration of 'base' PokerkitWrapper once
# we have support for at least one game (i.e. _not_ attempting to mimick the
# ACPC-wrapping UniversalPoker game)
#
# pyspiel.register_game(_GAME_TYPE, PokerkitWrapper)

pyspiel.register_game(_GAME_TYPE_ACPC_STYLE, PokerkitWrapperAcpcStyle)
