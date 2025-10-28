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
r"""Pokerkit wrapper to provide support for multiple poker game variants.

The base class supports:
- Texas Hold'em
- Limit Texas Hold'em
- Seven Card Stud
- Razz
- Pot Limit Omaha (PLO)

We also provide a subclass that more closely mimics the UniversalPoker / ACPC
implementation in the C++ codebase, but only supports:
- Texas Hold'em
- Limit Texas Hold'em
"""

import copy
import dataclasses
import decimal
import inspect
import math

from absl import logging
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
    # "Limit Texas Hold'em"
    pokerkit.FixedLimitTexasHoldem,
    # "Seven Card Stud"
    pokerkit.FixedLimitSevenCardStud,
    # "Razz"
    pokerkit.FixedLimitRazz,
    # "Pot Limit Omaha (PLO)"
    pokerkit.PotLimitOmahaHoldem,
]
SUPPORTED_VARIANT_MAP = {
    # e.g. { "NoLimitTexasHoldem": pokerkit.NoLimitTexasHoldem, ... }
    variant.__name__: variant
    for variant in _SUPPORTED_VARIANT_CLASSES
}

_SHORT_DECK_VARIANTS = [
    pokerkit.NoLimitShortDeckHoldem.__name__,
]

_STANDARD_DECK_VARIANTS = [
    v.__name__
    for v in _SUPPORTED_VARIANT_CLASSES
    if v.__name__ not in _SHORT_DECK_VARIANTS
]

_VARIANTS_SUPPORTING_ACPC_STYLE = [
    pokerkit.NoLimitTexasHoldem.__name__,
    pokerkit.FixedLimitTexasHoldem.__name__,
]


def _get_variants_supporting_param(param_name: str) -> list[str]:
  return [
      name
      for name, cls in SUPPORTED_VARIANT_MAP.items()
      if param_name in inspect.signature(cls).parameters.keys()
  ]

VARIANT_PARAM_USAGE = {
    param: _get_variants_supporting_param(param)
    for param in [
        "bring_in",
        "small_bet",
        "big_bet",
        "raw_blinds_or_straddles",
        "max_completion_betting_or_raising_count",
    ]
}

_ALL_IN_FOR_ONE_CHIP_EDGECASE_STRING = "Bet/Raise to 1 [ALL-IN EDGECASE]"

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
    # string).
    # NOTE: Small Blind must be >= 1, Big Blind must be >= 2, and SB <= BB.
    # TODO: b/437724266 - Add support for any N blinds where 0 < N < num_players
    # instead of requiring exactly "<SB> <BB>". (Pokerkit already supports this,
    # we just need to ensure we're handling things properly on our end.)
    "blinds": "5 10",
    "num_streets": 4,
    # Relevant for games with public hole cards. Will be ignored by all
    # other games (including the default NoLimitTexasHoldem).
    # NOTE: Must be strictly < small_bet.
    "bring_in": 5,
    # Relevant for various FixedLimit ("limit") games. Will be ignored by all
    # other games (including the default NoLimitTexasHoldem). Must be provided
    # if bring_in is provided.
    # NOTE: Must be strictly < big_bet.
    "small_bet": 10,
    "big_bet": 20,
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
    "stack_sizes": "2000 2000",
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
    provides_information_state_string=True,
    provides_observation_string=True,
    # Tensors are not yet supported. Instead use the corresponding strings.
    provides_information_state_tensor=False,
    provides_observation_tensor=False,
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
    # TODO: b/434776281 - Toggle True again once *properly* supported via the
    # Observer class when its iig_obs_type.perfect_recall is True.
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_tensor=True,
    provides_observation_string=True,
    # TODO: b/437724266 - determine what this value does and whether it should
    # be True or False / what functionality it's actually controlling.
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS,
)


# NOTE: FOLD and CHECK_OR_CALL deliberately chosen to match kFold and kCall in
# universal_poker so that it's easier to port code over from it.
ACTION_FOLD = 0
ACTION_CHECK_OR_CALL = 1
FOLD_AND_CHECK_OR_CALL_ACTIONS = [ACTION_FOLD, ACTION_CHECK_OR_CALL]

# NOTE: with one exception (see below), all player actions with value >= 2 will
# be interpreted as part of a raise, reraise, or completion bet. Generally this
# follows typical poker assumptions, where "raise to X" declares a total
# contribution on the *current street* of X chips. (This differs from an
# ACPC-"style" implementation / universal_poker's approach, which isntead treats
# it as declaring a total contribution of X chips across **all** streets", i.e.
# the entire hand).

# NOTE: As a result of using action `1` to handle ACTION_CHECK_OR_CALL, we have
# to handle bet sizes of 1 chip via special-case 'mapping' them to another
# action value. This can can happen in exactly one case: when a player has a
# single chip left in their stack and wants to shove all-in. (We prevent bet
# sizes < 2 in all other situations by ensuring that we only ever provide
# pokerkit min_bet and small_bet parameters of size >= 2.)


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
    # Replicates pyspiel.load_game()'s behavior of using the default's value for
    # each param missing from the provided params.
    self.params = _DEFAULT_PARAMS | params
    # Prevents accidental usage params instead of self.params below.
    # NOTE: Only removes the reference from *local scope* here; for details see:
    # https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces
    del params

    variant = self.params.get("variant")
    if variant not in SUPPORTED_VARIANT_MAP:
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

    # NOTE: Will not include any params set by a user which happen to match the
    # default value. Meaning even params not present in this set might have been
    # set explicitly by a user; we cannot assume it came from a default value.
    definitely_overriden = set([
        param
        for param, v in _DEFAULT_PARAMS.items()
        if v != self.params.get(param)
    ])

    # --- Blinds, Bet Sizes, and Antes or Bring-ins ---
    self.blinds = _parse_values(self.params["blinds"])
    if (
        self.blinds is None
        or len(self.blinds) != 2
        or self.blinds[0] > self.blinds[1]
    ):
      raise ValueError(
          "blinds must parse to a list of 2 values [Small Blind, Big Blind],"
          " where SB >= 1, BB >= 2, and SB <= BB. But instead got: ",
          self.params["blinds"],
          " which parsed to: ",
          self.blinds,
      )
    if any(x <= 0 for x in self.blinds):
      raise ValueError("Blinds must all be >= 1: ", self.blinds)
    # Ensure that users do not configure games with blinds low enough that it
    # will matter that we can't support bet sizes <= 1.
    if not max(self.blinds) >= 2:
      raise ValueError("Big Blind must be at least 2: ", self.blinds)

    assert self.params["small_bet"] < self.params["big_bet"]
    if variant in VARIANT_PARAM_USAGE["bring_in"]:
      assert self.params["bring_in"] < self.params["small_bet"]

    # This is such a common + confusing mistake that it's helpful to at least
    # warn about it, even if we can't actually throw an error (without losing
    # the ability to manually choose these default values in certain cases).
    for param1, param2 in [
        ("bring_in", "small_bet"),
        ("bring_in", "big_bet"),
        ("small_bet", "big_bet"),
        ("big_bet", "small_bet"),
    ]:
      if (
          param1 in definitely_overriden
          and param2 not in definitely_overriden
          and (
              variant in VARIANT_PARAM_USAGE[param1]
              or variant in VARIANT_PARAM_USAGE[param2]
          )
      ):
        logging.warning(
            "In the following pair of tightly-coupled params, the first param"
            " differs from its default value but the second does not: %s vs"
            " %s. (NOTE: This may be intended, and is technically allowed, so"
            " we are not raising an error. But in practice, it's very likely"
            " that you set the params here incorrectly!).",
            param1,
            param2,
        )

    stack_sizes = _parse_values(self.params["stack_sizes"])
    if stack_sizes is None or len(stack_sizes) != num_players:
      raise ValueError(
          "stack_sizes must be a space-separated list of "
          f" {num_players} values. Was provided: {self.params['stack_sizes']},"
          f" which parsed to: {stack_sizes}"
      )
    self.stack_sizes = stack_sizes

    # --- Card and Deck Information ---
    self._deck = pokerkit.Deck.STANDARD
    if variant in _SHORT_DECK_VARIANTS:
      self._deck = pokerkit.Deck.SHORT_DECK_HOLDEM
    else:
      assert variant in _STANDARD_DECK_VARIANTS

    self.deck_size = len(self._deck)
    self.card_to_int = {card: i for i, card in enumerate(self._deck)}
    self.int_to_card = {i: card for i, card in enumerate(self._deck)}
    self.num_streets = self.params.get("num_streets")

    # --- GameInfo and GameType Setup ---

    # Initialize to None since the function is going to attempt to check this
    # to see if we cached it already.
    self._max_game_length_estimate = None
    self._max_game_length_estimate = self._calculate_max_game_length()

    self._game_info = pyspiel.GameInfo(
        num_distinct_actions=self._calculate_num_distinct_actions(),
        max_chance_outcomes=self.deck_size,
        num_players=num_players,
        min_utility=self._calculate_min_utility(),
        max_utility=self._calculate_max_utility(),
        utility_sum=0.0,
        max_game_length=self._calculate_max_game_length(),
    )
    super().__init__(game_type, self._game_info, self.params)

  def _calculate_num_distinct_actions(self):
    """Returns an upper bound on the number of distinct player actions.

    NOTE: This could be a bit tighter, e.g. depending on the value of min_bet
    or specifics of the game variant. That said, this just needs to be an upper
    bound, so it should be entirely fine.
    """
    # +1 is to account for ACTION_FOLD being 0
    return max(self.stack_sizes) + 1

  def _calculate_min_utility(self):
    """Returns an upper-bound (magnitude) on minimum utility for the game.

    NOTE: for sake of simplicity, this base class utility estimate assumes that
    in all games the deepest-stack will always be able to get all the chips in
    AND actually lose them. This is not as tight as it could be in many cases,
    e.g. where effective stack sizes or bet limits prevent the deepest-stack
    from actually doing so. If you need tighter bounds, consider subclassing and
    overriding.
    """
    # TODO: b/437724266 - use effective stack size instead of raw stack size.
    return float(-max(self.stack_sizes))

  def _calculate_max_utility(self):
    """Returns an upper-bound on maximum utility for the game.

    NOTE: for sake of simplicity, this base class utility estimate assumes that
    in all games the deepest-stack will always be able to win all other players'
    chips. This is not as tight as it could be in many cases, e.g. where
    bet limits prevent the deepest-stack from actually doing so. If you need
    tighter bounds, consider subclassing and overriding.
    """
    return float(sum(self.stack_sizes) - max(self.stack_sizes))

  def game_info(self):
    return self._game_info

  def _calculate_max_game_length(self):
    """Provides a (very rough) upper bound on the maximum game length."""
    if self._max_game_length_estimate is not None:
      return self._max_game_length_estimate

    # Both theoretically and in practice, games will be MUCH shorter than this.
    # However, this does set an ironclad upper bounds on the game length,
    # which is 'good enough' for our purposes.
    # NOTE: Assumes that in nolimit games, min_bet is exactly one big blind.
    min_reraise_amount = max(self.blinds)
    return (
        # Upper bound on check actions from players not betting, on all streets.
        self.params.get("num_players") * max(self.params.get("num_streets"), 1)
        # Upper bound on reraises + calls by the other players. (This is a
        # factor of total chips, and so doesn't need to consider streets.)
        + (max(self.stack_sizes) // min_reraise_amount)
        * max(1, self.params.get("num_players") - 1)
        # Upper bound on performing all deal actions
        + self.deck_size
    )

  def new_initial_state(self):
    return PokerkitWrapperState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    return PokerkitWrapperObserver(self, iig_obs_type, params)

  def raise_error_if_player_out_of_range(self, player):
    if player < 0 or player >= self.num_players():
      raise ValueError(
          f"Player {player} is out of range [0, {self.num_players()})"
      )

  # Not yet actually supported. Providing this 'return hardcoded value'
  # placeholder implementation instead of throwing an error to avoid crashing
  # anything in OpenSpiel that doesn't properly respect the
  # provides_information_state_tensor=False setting above.
  def information_state_tensor_size(self) -> int:
    logging.warning(
        "information_state_tensor_size() is not yet supported for"
        " PokerkitWrapper. Returning a default value of 1 for now."
    )
    return 1

  def _wrapped_game_template_factory(self):
    """Creates a default pokerkit 'template' that can easily construct states.

    For more details, see:
    https://pokerkit.readthedocs.io/en/stable/simulation.html#pre-defined-games
    ```
    In certain use cases, one may want to create a template from which just the
    starting stacks and the number of players can be specified. In PokerKit,
    this can be done by creating an instance of the game that acts as a state
    factory from which states are initialized.
    ```

    NOTE: By definition, this CANNOT disable card burning, as it's using the
    default pokerkit game variants (all of which have card burning enabled /
    do not allow custom Street inputs). If using this to create a
    wrapped_state you'll want to pass in a custom Streets to disable it!

    WARNING: Don't attempt to merge this into wrapped_state_factory(). You'll
    need this for multiple reasons, including those other than creating a game
    object to track game state. Notably, we have to pass the Pokerkit.Poker into
    calls to HandHistory.from_game_state(...) to construct PHHs.

    Returns:
      A "variant" sub-class pokerkit.Poker object.
    """
    # TODO: b/437724266 - Stop directly accessing values from the params and
    # instead grab the values we need via getters.
    params = self.params
    variant = params.get("variant")
    poker_variant_class = SUPPORTED_VARIANT_MAP.get(variant)
    if not poker_variant_class:
      raise ValueError(f"Unknown / unsupported poker variant: {variant}")

    def in_scope_for_variant(k):
      return k in inspect.signature(poker_variant_class).parameters.keys()

    assert _parse_values(params.get("stack_sizes")) == self.stack_sizes
    game_args = {
        k: v
        for k, v in {
            "automations": _DEFAULT_AUTOMATIONS,
            "ante_trimming_status": False,
            "player_count": params.get("num_players"),
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
            "raw_blinds_or_straddles": self.blinds,
            # Unlike most other args, we don't support custom user inputs for
            # this. Despite the name, it's only used in nolimit games (NOT any
            # of the fixed limit variants, which compute their minimum bet size
            # in other ways). And in practice, almost everyone just uses the big
            # blind as the minimum bet size in nolimit games.
            "min_bet": max(self.blinds),  # Always set to the big blind
            # All remaining args we need line up exactly with the params
            # key/values.
        }.items()
        if in_scope_for_variant(k)
    } | {k: v for k, v in params.items() if in_scope_for_variant(k)}

    # Double check that we didn't somehow end up in a situation where we are
    # allowing bet sizes that could conflict with the reserved values for our
    # fold / check actions.
    min_bet = game_args.get("min_bet")
    if min_bet is not None:
      assert min_bet > 0
      assert min_bet not in FOLD_AND_CHECK_OR_CALL_ACTIONS
    # TODO: b/434776281 - consider moving additional validation checks out of
    # the constructor and into this spot here right before we actually use them.

    return poker_variant_class(**game_args)

  def wrapped_state_factory(self) -> pokerkit.State:
    """Creates the "wrapped" (not "PokerkitWrapper...") pokerkit.state object.

    As a convienence, this method additionally disables card burning on all
    streets. (Card burning would have no purpose here since our cards' backs
    cannot be marked, unlike with e.g. real physical poker cards).

    Returns:
      A `pokerkit.State` object configured with the game parameters.
    """
    # For more details on why this works (i.e. only two arguments), see
    # the docstring in the _wrapped_game_template_factory() method.
    #
    # NOTE: All default pokerkit games have card burning enabled. **This
    # cannot be prevented, whether via create_state OR via the 'template'
    # approach we use here.** We'll need to create our own state object
    # directly using the pokerkit.State constructor to (properly) disable it.
    game_template = self._wrapped_game_template_factory()
    stacks = _parse_values(self.params.get("stack_sizes"))
    number_players = self.params.get("num_players")
    helper_state = game_template(stacks, number_players)

    # Doing this instead of just directly mutating state.streets as per
    # https://pokerkit.readthedocs.io/en/stable/tips.html#read-only-and-read-write-values
    # """
    #  [The attributes of pokerkit.state.State... should never be modified...
    #  [I]nstead let PokerKit modify them through public method calls. In other
    #  words, the user must only read from the state’s attributes or call public
    #  methods (which may modify them).
    # """
    streets_override = [
        pokerkit.state.Street(
            **dataclasses.asdict(street)
            | {
                "card_burning_status": False,
            }
        )
        for street in helper_state.streets
    ]
    args = copy.deepcopy(
        {
            # A select few args have "raw_" prefixes in the State constructor.
            # Unfortunately, although the constructed State objects have both
            # versions, in practice the non-"raw_" versions are the only ones
            # that are actually (reliably) set to non-None values.
            # As such we have to source these "raw_" args from the equivalent
            # non-"raw_" versions.
            #
            # For example:
            # - giving `raw_starting_stacks` arg `state.starting_stacks`'s value
            # - giving `raw_antes` arg `state.antes`'s value
            # - giving `raw_blinds_or_straddles` arg `state.blinds`'s value
            arg: helper_state.__dict__.get(arg.replace("raw_", ""))
            for arg in inspect.signature(pokerkit.State).parameters.keys()
        }
        | {"streets": streets_override}
    )
    assert len(args) == len(inspect.signature(pokerkit.State).parameters.keys())
    assert None not in args.values()
    # NOTE: CANNOT USE THE "template" APPROACH, NOR .create_state() HERE
    # (without allowing card burning - since neither of these options accept an
    # input 'streets').
    returned_state: pokerkit.State = pokerkit.State(**args)
    # Double checks that we didn't make mistakes when disabing card burning just
    # now. (This is a bit overkill, but we REALLY don't want there to be any
    # chance of accidentally leaving card burning enabled anywhere).
    for street in returned_state.streets:
      assert not street.card_burning_status
    return returned_state

  def _game_type(self):
    return _GAME_TYPE


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
    return self._legal_actions_base(player)

  def _legal_actions_base(self, player):
    wrapped_state: pokerkit.State = self._wrapped_state
    game: PokerkitWrapper = self.get_game()
    game.raise_error_if_player_out_of_range(player)

    if self.is_terminal():
      raise ValueError(
          f"Attempted to get legal actions for player {player} when the"
          " game is terminal."
      )

    # player is not the active player, and so can't take any actions.
    if player != wrapped_state.actor_index:
      raise ValueError(
          f"Attempted to get legal actions for player {player} when the current"
          f" player is {wrapped_state.actor_index}."
      )

    if not (
        wrapped_state.can_check_or_call()
        or wrapped_state.can_fold()
        or wrapped_state.can_complete_bet_or_raise_to()
    ):
      raise ValueError(
          f"Attempted to get legal actions for player {player} when the"
          " player is not in a position to check, fold, or bet/raise."
      )

    actions = []

    # Handling player nodes first since they're the most straightforward.
    if wrapped_state.can_fold():
      actions.append(ACTION_FOLD)
    if wrapped_state.can_check_or_call():
      actions.append(ACTION_CHECK_OR_CALL)
    if wrapped_state.can_complete_bet_or_raise_to():
      valid_bet_sizes = []
      betting_structure = wrapped_state.betting_structure
      min_bet = wrapped_state.min_completion_betting_or_raising_to_amount
      max_bet = wrapped_state.max_completion_betting_or_raising_to_amount
      assert min_bet is not None and max_bet is not None

      if betting_structure == pokerkit.state.BettingStructure.FIXED_LIMIT:
        # min and max completion, bet, and raise amounts are identical as per:
        # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.BettingStructure.FIXED_LIMIT
        assert min_bet == max_bet
        fixed_size: int | None = min_bet
        if (
            fixed_size is not None
            and fixed_size not in FOLD_AND_CHECK_OR_CALL_ACTIONS
        ):
          valid_bet_sizes.append(fixed_size)
      elif (
          betting_structure == pokerkit.state.BettingStructure.NO_LIMIT
          or betting_structure == pokerkit.state.BettingStructure.POT_LIMIT
      ):
        assert min_bet <= max_bet
        valid_bet_sizes.extend([
            amount
            for amount in range(
                min_bet,
                max_bet + 1,
            )
            if amount not in FOLD_AND_CHECK_OR_CALL_ACTIONS
        ])
      else:
        raise RuntimeError(
            f"Unsupported pokerkit betting structure: {betting_structure}"
        )
      assert all(
          amount not in FOLD_AND_CHECK_OR_CALL_ACTIONS
          for amount in valid_bet_sizes
      )

      if len(valid_bet_sizes) == 1:
        only_valid_bet = valid_bet_sizes[0]
        assert only_valid_bet == min_bet and only_valid_bet == max_bet
        current_street: pokerkit.Street = wrapped_state.streets[
            wrapped_state.street_index
        ]
        # "Global minimum" size for the current street, regardless of player.
        # TODO: b/437724266 - double check this actually is independent from
        # player!
        street_minimum_size = (
            current_street.min_completion_betting_or_raising_amount
        )
        if only_valid_bet < street_minimum_size:
          # In 2-person games, the only way for this to happen is if the
          # effective stack size is less than the bet size and the player is
          # (effectively) shoving. So we can trivially verify that this isn't
          # happening incorrectly.
          #
          # NOTE: This cares about _effective_ stack size, not the player's
          # actual stack size! Since this behavior also happens when the *other*
          # player's stack size is less than the minimum for the street.
          if game.num_players() == 2:
            if only_valid_bet != wrapped_state.get_effective_stack(player):
              raise RuntimeError(
                  f"Player {player} has effective stack size of"
                  f" {wrapped_state.get_effective_stack(player)} (and has"
                  f" 'real' stack size of {wrapped_state.stacks[player]} chips)"
                  f"  but only valid bet is {only_valid_bet}, which is"
                  " unexpectedly less than the minimum value for the current"
                  " street AND also doesn't equal their effective stack size."
                  f" {street_minimum_size}."
              )
          # In 3+ person games, this is significantly more complicated to verify
          # since side-pots can significantly complicate the math, especially
          # in fixed-limit games. Any asserts we could write here would be
          # overly bug-prone + require significant testing anyways.
          pass

      # Handle edge case in both no limit and fixed limit where the player has
      # so few chips that a legal bet would have conflicted with one of our
      # reserved actions, and so got filtered out above.
      if not valid_bet_sizes and wrapped_state.stacks[player] > 0:
        # If the stack size is non-zero, then we *should* be in a situation
        # where we were going to try to bet 1 chip, which got filtered out due
        # to it being in the reserved actions set. (Anything else would mean
        # there's a bug in our code above.)
        assert wrapped_state.stacks[player] == 1
        assert min_bet == 1 and max_bet == 1
        # This decision to map to 2 is arbitrary! We could alternatively have
        # mapped to any other (reasonably small) positive integer not in the
        # reserved actions set.
        logging.warning(
            "Mapping shove size 1 chip to action 2 in legal_actions to avoid"
            " conflict with 'reserved' actions.",
        )
        valid_bet_sizes.append(2)

      # TODO: b/437724266 - Extract this entire giant if-statement block out
      # to a helper function to make this overall more readable. (In particular,
      # having the .extend() for the entire can_complete_bet_or_raise_to()
      # handling at the very bottom down here is difficult to read + bug
      # prone).
      assert set(valid_bet_sizes).isdisjoint(FOLD_AND_CHECK_OR_CALL_ACTIONS)
      actions.extend(valid_bet_sizes)

    assert actions
    return sorted(actions)

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities.

    NOTE: Pokerkit has a known-issue of returning certain mucked cards in N>2
    player games. As such, we additionally ourselves 'double check' to ensure
    that there are no cards being incorrectly returned by pokerkit itself. And
    in such cases either A. remove such offending cards, or B. raise an error
    (respectively depending on whether it's this exact 'mucked card' known issue
    vs something else).)

    Raises:
      RuntimeError: If `pokerkit.State.get_dealable_cards()` returns cards that
        are determined to be non-dealable, even after resolving a known issue
        with mucked cards.
    """
    assert self.is_chance_node()
    wrapped_state: pokerkit.State = self._wrapped_state
    # pokerkit has a bug in get_dealable_cards() where it sometimes returns
    # cards in N>2 player games that were already dealt to a different player if
    # that player mucked. We want to filter those out here to avoid bugs
    # in our code later when applying the action (at which point pokerkit will
    # correctly raise an error sayign that the card is not recommened to be
    # dealt).
    # Flattened list of hole cards
    flattened_hole_cards = []
    for sublist in wrapped_state.hole_cards:
      flattened_hole_cards.extend(sublist)
    flattened_board_cards = []
    for sublist in wrapped_state.board_cards:
      flattened_board_cards.extend(sublist)
    double_checked_used_cards = set(wrapped_state.deck) - set(
        flattened_hole_cards
        + flattened_board_cards
        + wrapped_state.burn_cards
        + wrapped_state.mucked_cards
    )
    pokerkit_computed_dealable_cards = set(wrapped_state.get_dealable_cards())
    if pokerkit_computed_dealable_cards != double_checked_used_cards:
      logging.warning(
          "pokerkit.State.get_dealable_cards() did not return the same set of"
          " cards as the set of used cards. %s != %s",
          pokerkit_computed_dealable_cards,
          double_checked_used_cards,
      )
    # Make sure that there's no bugs where *our* logic forgets to filter out
    # cards vs pokerkit's logic
    assert double_checked_used_cards.issubset(pokerkit_computed_dealable_cards)

    for card in pokerkit_computed_dealable_cards - double_checked_used_cards:
      if card in wrapped_state.mucked_cards:
        logging.warning(
            "Confirmed incorrectly-dealable card is an instance of known issue"
            " in pokerkit (card %s is one of the mucked_card). ***REMOVING "
            "CARD %s FROM THE RETURNED CARDS AS A WORKAROUD.****",
            card,
            card,
        )
        pokerkit_computed_dealable_cards.remove(card)
    if pokerkit_computed_dealable_cards != double_checked_used_cards:
      raise RuntimeError(
          "pokerkit.State.get_dealable_cards() still contains cards that we "
          " have determined should not be dealable - even after removing some "
          " cards that were included due to known issues with pokerkit. This"
          " means there are likely MORE problems in pokerkit that we have no"
          " identified yet! Post-'surgery' get_dealable_cards() was"
          f" {pokerkit_computed_dealable_cards}, but when double-checking"
          " ourselves we determined the list should instead be"
          f" {double_checked_used_cards}"
      )

    outcomes = sorted([
        self.get_game().card_to_int[c] for c in pokerkit_computed_dealable_cards
    ])
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    wrapped_state: pokerkit.State = self._wrapped_state
    # Handling player actions first since it's the more striaghtforward case.
    if not self.is_chance_node():
      # NOTE: using the _base version of the method directly. Since even in
      # situations where legal_actions is overridden, we still want to avoid
      # using the overriding class's actions (which could be different in
      # situations where we're 'mapping' actions in some way).
      if action not in self._legal_actions_base(self.current_player()):
        raise ValueError(
            f"Attempted to apply illegal action {action} to player"
            f" {self.current_player()}. Legal actions were"
            f" {self.legal_actions()}"
        )

      if action == ACTION_FOLD:
        wrapped_state.fold()
      elif action == ACTION_CHECK_OR_CALL:
        wrapped_state.check_or_call()
      else:
        if wrapped_state.can_complete_bet_or_raise_to(action):
          wrapped_state.complete_bet_or_raise_to(action)
        else:
          # Should be an edge case when trying to shove with exactly 1 chip,
          # where we remapped the shove to action 2 to avoid conflicting with
          # reserved actions.
          assert action == 2
          assert wrapped_state.stacks[wrapped_state.actor_index] == 1
          assert wrapped_state.can_complete_bet_or_raise_to(1)
          assert wrapped_state.stacks[wrapped_state.actor_index] == 1
          action_string = self._action_to_string_base(
              wrapped_state.actor_index, action
          )
          if _ALL_IN_FOR_ONE_CHIP_EDGECASE_STRING not in action_string:
            raise ValueError(
                f"Expected to find {_ALL_IN_FOR_ONE_CHIP_EDGECASE_STRING} for"
                f" action {action} but got action string {action_string}"
            )
          wrapped_state.complete_bet_or_raise_to(1)
      return

    # else it's a chance node (which is a bit more complicated):

    card_from_action = self.get_game().int_to_card[action]
    if card_from_action not in wrapped_state.get_dealable_cards():
      raise ValueError(
          f"Chance action maps to non-dealable card {card_from_action}: "
          f"{wrapped_state.can_deal_hole()} "
          f"{wrapped_state.can_deal_board()}"
      )
    # If these ever fail it means that wrapped_state.get_dealable_cards()
    # gave us a card that it shouldn't have!
    # assert card_from_action not in wrapped_state.mucked_cards
    assert card_from_action not in wrapped_state.burn_cards
    assert card_from_action not in wrapped_state.board_cards
    for player_hole_cards in wrapped_state.hole_cards:
      assert card_from_action not in player_hole_cards

    if wrapped_state.can_deal_hole() and wrapped_state.can_deal_board():
      raise ValueError(
          "Chance node with both dealable hole and board cards: "
          f"{wrapped_state.can_deal_hole()} "
          f"{wrapped_state.can_deal_board()}"
      )
    elif wrapped_state.can_deal_hole():
      wrapped_state.deal_hole(card_from_action)
    elif wrapped_state.can_deal_board():
      wrapped_state.deal_board(card_from_action)
    else:
      raise ValueError(
          "Chance node has dealable card, but cannot deal hole or board cards :"
          f"{wrapped_state.can_deal_hole()} "
          f"{wrapped_state.can_deal_board()}"
      )

  # TODO: b/437724266 - update string structure (not bet sizings!) to match
  # universal_poker/universal_poker.cc ActionToString where practical. E.g.
  # 'Deal {card}' and 'player={player} move={move}'.
  def _action_to_string(self, player, action):
    """Returns the string representation of an action.

    Assumes that input player actions corresponding to completion-bets or raises
    are in 'Pokerkit' style rather than 'ACPC' style, i.e. raising to a total
    contribution on the current street only (rather than a total contribution
    across all streets).

    Args:
      player: The player for whom the action string is generated.
      action: The action to convert to a string.
    """
    return self._action_to_string_base(player, action)

  def _action_to_string_base(self, player, action):
    """Core "base" logic from above, separated to bypass method overrides."""
    wrapped_state: pokerkit.State = self._wrapped_state
    if self.is_chance_node():
      assert player is not None
      assert player < 0
      if wrapped_state.can_deal_hole():
        return f"Deal Hole Card: {self.get_game().int_to_card[action]}"
      elif wrapped_state.can_deal_board():
        return f"Deal Board Card: {self.get_game().int_to_card[action]}"
      else:
        raise ValueError(
            f"Chance node with action {action} but cannot deal cards. "
            f"can_deal_hole: {wrapped_state.can_deal_hole()} "
            f"can_deal_board: {wrapped_state.can_deal_board()}"
        )
    if player != wrapped_state.actor_index:
      raise ValueError(
          "Attempted to call _action_to_string() for a not-currently-active"
          "player (who shouldn't have any actions available)."
      )
    assert player == wrapped_state.actor_index
    if action == ACTION_FOLD:
      return f"Player {player}: Fold"
    elif action == ACTION_CHECK_OR_CALL:
      amount = wrapped_state.checking_or_calling_amount
      return f"Player {player}: Check" if amount == 0 else f"Call({amount})"
    elif action >= 0 and action not in FOLD_AND_CHECK_OR_CALL_ACTIONS:
      if not (action == 2 and wrapped_state.stacks[player] == 1):
        return f"Player {player}: Bet/Raise to {action}"

      # EDGE CASE: If stack size is 1 and action is 2, this is likely a case
      # where the action was 'remapped' from 1 to 2 to avoid collision with
      # CHECK_OR_CALL action.
      assert wrapped_state.min_completion_betting_or_raising_to_amount == 1
      assert wrapped_state.max_completion_betting_or_raising_to_amount == 1
      # As per the docs:
      # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.payoffs
      # "Negated versions of these values can be thought of as contributions
      # to the hand by the individual players."
      #
      # Bets on the other hand has the positive amounts the player already
      # contributed on this street.
      #
      # TODO: b/437724266 - figure out whether things like blinds or bring_in
      # (which have laready been subtracted from starting_stacks by this point)
      # will or will not show up in the bets list, and how that may affect this
      # in preflop situations specifically. Since, although in practice this
      # appears to work + the tests aren't flakey, to be safe we should
      # investigate and document exactly what is happening in this spot.
      contribution_on_prior_streets_or_non_bets = -1 * (
          wrapped_state.payoffs[player] + wrapped_state.bets[player]
      )
      assert (
          wrapped_state.starting_stacks[player]
          - contribution_on_prior_streets_or_non_bets
          == 1
      )
      return f"Player {player}: Bet/Raise to 1 [ALL-IN EDGECASE]"
    else:
      raise ValueError(
          f"Invalid action {action} for player {player}. 'legal actions' was "
          f" {self.legal_actions()}"
      )

  def is_terminal(self):
    # .status is True if the state is not terminal, as per
    # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.status,
    assert self._wrapped_state.status is not None
    is_terminal = not self._wrapped_state.status
    if is_terminal:
      assert isinstance(
          self._wrapped_state.operations[-1], pokerkit.ChipsPulling
      )
    return is_terminal

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
    return copy.deepcopy(self)

  def deepcopy_wrapped_state(self) -> pokerkit.State:
    """Create and return a deepcopy of the 'wrapped' pokerkit.state."""
    return copy.deepcopy(self._wrapped_state)

  def to_struct(self) -> pyspiel.pokerkit_wrapper.PokerkitStateStruct:
    legal_actions = []
    if not self.is_terminal():
      legal_actions = self.legal_actions()
    # TODO(jhtschultz): If having the PHHs embeded in the observation string
    # would be helpful pass in pyspiel.IIGObservationType(perfect_recall=True))
    # here instead.
    # (Either way we'll already be putting the PHHs on the struct directly
    # though, so doing so probably isn't _necessary_ - see below for details.)
    observer = self.get_game().make_py_observer()
    obs_strs = [
        observer.string_from(self, p)
        for p in range(self.get_game().num_players())
    ]
    game = self.get_game()
    stacks = [int(s) for s in self._wrapped_state.stacks]
    bets = [int(b) for b in self._wrapped_state.bets]

    flattened_board_cards = []
    for sublist in self._wrapped_state.board_cards:
      flattened_board_cards.extend(sublist)
    board_cards = [game.card_to_int[c] for c in flattened_board_cards]

    hole_cards = [
        [game.card_to_int[c] for c in hc_list]
        for hc_list in self._wrapped_state.hole_cards
    ]
    pots = [int(p.amount) for p in self._wrapped_state.pots]
    burn_cards = [game.card_to_int[c] for c in self._wrapped_state.burn_cards]
    mucked_cards = [
        game.card_to_int[c] for c in self._wrapped_state.mucked_cards
    ]

    # Technically we can get these 'for free' on the observation strings above
    # by using a perfect_recall observer there. But in practice these are
    # important enough that it's nice to have them be explicitly available as a
    # first-class citizen on the actual struct.
    perfect_recall_observer = self.get_game().make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=True)
    )
    poker_hand_histories: list[list[str]] = [
        perfect_recall_observer.poker_hand_history_actions(self, p)
        for p in range(self.get_game().num_players())
    ]

    cpp_state_struct = pyspiel.pokerkit_wrapper.PokerkitStateStruct()
    cpp_state_struct.observation = obs_strs
    cpp_state_struct.legal_actions = legal_actions
    cpp_state_struct.current_player = self.current_player()
    cpp_state_struct.is_terminal = self.is_terminal()
    cpp_state_struct.stacks = stacks
    cpp_state_struct.bets = bets
    cpp_state_struct.board_cards = board_cards
    cpp_state_struct.hole_cards = hole_cards
    cpp_state_struct.pots = pots
    cpp_state_struct.burn_cards = burn_cards
    cpp_state_struct.mucked_cards = mucked_cards
    cpp_state_struct.poker_hand_histories = poker_hand_histories
    return cpp_state_struct

  def to_json(self) -> str:
    return self.to_struct().to_json()

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    state: pokerkit.State = self._wrapped_state
    parts = []
    parts.append(f"Stacks: {state.stacks}")
    parts.append(f"Bets: {state.bets}")
    parts.append(f"Board: {state.board_cards}")
    # WARNING: Will include private hole cards for all players!
    parts.append(f"Hole Cards: {state.hole_cards}")
    parts.append(f"Pots: {list(state.pots)}")
    # WARNING: Will include private hole cards for all players!
    parts.append(f"Operations: {state.operations}")
    return " | ".join(parts)


class PokerkitWrapperObserver:
  """Pokerkit "Observer" for creating per-player state tensors and strings."""

  def __init__(self, game, iig_obs_types, params):
    """Initializes the PokerkitWrapperObserver."""
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
      # open_spiel/python/algorithms/generate_playthrough.py
      self.iig_obs_types = pyspiel.IIGObservationType(perfect_recall=False)
    self.params = params

    # Not actually used since `provides_observation_tensor=False`, but necessary
    # since some tests (incorrectly?) assume these always have been created /
    # test for the attribute's existence.
    self.tensor = np.array([])
    self.dict = {}

  def poker_hand_history_actions(self, state, player) -> list[str]:
    """Returns PHH (Poker Hand History) actions from player's perspective."""
    game = state.get_game()
    game.raise_error_if_player_out_of_range(player)

    # For context see
    # https://pokerkit.readthedocs.io/en/stable/notation.html#writing-hands
    # and https://phh.readthedocs.io/en/stable/required.html
    hand_history = pokerkit.HandHistory.from_game_state(
        # pylint: disable=protected-access
        # NOTE: Technically has card burning enabled! But should be close enough
        # for our purposes here - in practice still results in the same string.
        game._wrapped_game_template_factory(),
        # Used as an alternative to .deepcopy_wrapped_state() as an efficiency
        # optimization; is safe so long as we don't modify the wrapped state.
        state._wrapped_state,
        # pylint: enable=protected-access
    )
    # Censoring other players' hole cards
    per_player_action_view: list[str] = []
    for action in hand_history.actions:
      # If not dealing a hole card no need to censor.
      if not action.startswith("d dh p"):
        per_player_action_view.append(action)
        continue
      assert action.startswith("d dh p")

      # Dealing with hole cards, but still current player, so no need to censor.
      if action.startswith(f"d dh p{player + 1}"):
        per_player_action_view.append(action)
        continue
      assert not action.startswith(f"d dh p{player + 1}")

      # else: dealing hole cards for another player. This may (and usually will)
      # contain private hole cards, so censor it.
      # (We _could_ try to censor just *private* hole cards, i.e. 'down' cards
      # but not 'up' cards in games with them, but out of an abundance of
      # caution we instead just censor all of them to be sure.)

      action_components = action.split(" ")
      # for example: d, dh, p1 => d, dh, p1
      public_info = action_components[0:3]
      # for example:
      #  9hTs => ????
      #  9hTs AhTd => ???? ????
      #  7h6c4c3d2c => ??????????
      censored = ["?" * len(s) for s in action_components[3:]]
      per_player_action_view.append(" ".join(public_info + censored))
    return per_player_action_view

  def _observation_string(self, state, player):
    """Returns a string representation of the observation for a given player."""
    # pylint: disable=protected-access
    # Directly using for readonly access as an efficiency optimization. Could
    # also use deepcopy_wrapped_state() to ensure we don't accidentally mutate
    # the underlying state here, at the cost of performing the deepcopy every
    # time this method is called.
    wrapped_state_do_not_mutate: pokerkit.State = state._wrapped_state
    # pylint: enable=protected-access

    starting_stacks: list[int] = wrapped_state_do_not_mutate.starting_stacks
    stacks: list[int] = wrapped_state_do_not_mutate.stacks

    current_street_index: int | None = wrapped_state_do_not_mutate.street_index
    # Generator returning True IFF discard/draw has been performed for a given
    # street. For more details see:
    # https://pokerkit.readthedocs.io/en/stable/reference.html#pokerkit.state.State.draw_statuses
    draw_statuses: list[bool] = [
        s for s in wrapped_state_do_not_mutate.draw_statuses
    ]
    draw_status: bool | None = (
        None
        if current_street_index is None
        else draw_statuses[current_street_index]
    )
    pots: pokerkit.Pot = [p for p in wrapped_state_do_not_mutate.pots]
    bets: list[int] = wrapped_state_do_not_mutate.bets
    player_next_to_act: int = wrapped_state_do_not_mutate.actor_index

    # --- NOTE: the following '_cards' on the state here are all generators, so
    # we need to wrap them all in list() to actually grab any values. ---

    # Unlike below, board cards are always public to everyone, so this is
    # striaghtforward.
    public_board_cards: list[pokerkit.Card] = list(
        wrapped_state_do_not_mutate.board_cards
    )

    # Hole cards for this current player that only this current player can see.
    private_down_cards: list[pokerkit.Card] = list(
        wrapped_state_do_not_mutate.get_down_cards(player)
    )

    # Creates a list containing 1. any Public (face-up) hole cards for player p,
    # plus '??'s for any of their facedown cards (**even if player p is the
    # current player or the 'observing' player**).
    # As such, this will look identical in all players' observations.
    per_player_censored_cards: list[pokerkit.Card] = [
        list(wrapped_state_do_not_mutate.get_censored_hole_cards(p))
        for p in range(wrapped_state_do_not_mutate.player_count)
    ]
    return (
        f"Player: {player}\n"
        f"||Current Street: {current_street_index}\n"
        f"||Current Street discard/draw performed: {draw_status}\n"
        f"||Next Player to act: {player_next_to_act}\n"
        f"||Pot(s): {pots} \n"
        f"||Bets: {bets}\n"
        f"||Board Cards: {public_board_cards}\n"
        f"||Player's Private Hole Cards: {private_down_cards}\n"
        # NOTE: This is intentionally the same for all observing players. I.e.
        # even for player {player}, the private hole cards will be ??-censored.
        f"||Per-player Hole Cards (public view): {per_player_censored_cards}\n"
        f"||Per-player Starting Stacks: {starting_stacks}\n"
        f"||Per-player Current Stacks: {stacks}"
    )

  def set_from(self, state, player) -> None:
    """No-op; see `provides_observation_tensor=False,` above.

    Defined only because some tests (incorrectly?) assume this method will
    always be defined.

    Args:
      state: The state to extract the observation from.
      player: The player to extract the observation for.
    """
    state.get_game().raise_error_if_player_out_of_range(player)
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
    observation_string = self._observation_string(state, player)
    if self.iig_obs_types.perfect_recall:
      # I.e. construct the 'information state string' (by simply merging the
      # observation string with the PHH actions).
      phh_action_string = ",".join(
          self.poker_hand_history_actions(state, player)
      )
      return f"{observation_string}\n||PHH Actions: {phh_action_string}"
    else:
      return self._observation_string(state, player)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# TODO: b/434776281 - extract out PokerkitWrapperAcpcStyle into a separate file.


# --- "ACPC-style" PokerkitWrapper subclass to mimick UniversalPoker ---
class PokerkitWrapperAcpcStyle(PokerkitWrapper):
  """Subclass that mimics ACPC-wrapping UniversalPoker action handling.

  NOTE: To compare with UniversalPoker, please use the PHH hand history and/or
  'returns' array as your source of truth.

  - only action handling (applying actions, listing legal actions, converting
    actions to strings) is modified. The info/observation tensors and strings
    will exactly match that of an un-subclassed PokerkitWrapper - and therefore
    will NOT match UniversalPoker.

  - action_to_string's only change is to interpet the provided
    action as an ACPC-style action; it will simply map the action back to a
    pokerkit style action before returning the string as per usual (i.e. it will
    not make any other changes to the output string, e.g. to match
    universal_poker).

  NOTE: Unlike universal_poker, this actually handles hole-card dealing
  'properly' for real-world poker games, i.e. each player gets their first hole
  card before any player gets their second hole card. You will need to shuffle
  the order with which hole cards are dealt to simulate the same exact game.
  (That said, the decks - and therefore chance actions - are the same for any
  given card between this and universal_poker, so doing so should be
  straightforward.)
  """

  def __init__(self, params=None):
    if (
        params
        and "variant" in params
        and params["variant"] not in _VARIANTS_SUPPORTING_ACPC_STYLE
    ):
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

  def game_type(self):
    return _GAME_TYPE_ACPC_STYLE

  def new_initial_state(self):
    return PokerkitWrapperAcpcStyleState(self)


class PokerkitWrapperAcpcStyleState(PokerkitWrapperState):
  """Represents an _OpenSpiel_ 'state' for the PokerkitWrapper game.

  As described by the name, this class indeed wraps a `pokerkit.State` object
  and provides the necessary interface for OpenSpiel's `pyspiel.State`.
  """

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    current_payoffs = self._wrapped_state.payoffs
    self._action_converter = ToAcpcActionConverter(
        num_players=game.num_players(),
        initial_contributions={
            # Payoffs should contain the negative initial contributions from
            # blinds and/or straddles, antes, and/or bring-ins. Hence it's
            # easier to just use it rather than checking the relevant values for
            # the given game variant.
            p: -1 * current_payoffs[p]
            for p in range(game.num_players())
        },
    )
    self._pokerkit_action_to_acpc_style_mapping = {}
    self._acpc_action_to_pokerkit_action_mapping = {}
    self._refresh_action_mappings_if_player_node()

  def _refresh_action_mappings_if_player_node(self):
    """Refreshes the action mappings if we are at a player node.

    NOTE: should be called upon initializaiton + immediately *after* applying
    any action - not at the start of e.g. _apply_action. This is because we use
    these maps in multiple mostly-pure functions such as action_to_string() and
    _legal_actions(). Otherwise we would have to call this function at the
    start of each of those functions, i.e. regenerating the dict every time they
    are called (which would be inefficient and much less elegant).
    """
    if self.is_chance_node() or self.is_terminal():
      return

    assert not self.is_chance_node() and not self.is_terminal()
    current_player = self.current_player()
    # NOTE: not calling self._legal_actions() or self._action_to_string(). Since
    # those depends on these maps *having already been updated* + are in the
    # world of ACPC-style actions we're trying to support, instead of the
    # pokerkit-style actions we have at this point.
    pokerkit_style_actions = super()._legal_actions(current_player)
    # NOTE: *also* deliberately NOT calling super()._action_to_string(). Doing
    # so has (surprisingly!) flaked inside the general-purpose OpenSpiel
    # test_game_sim (i.e. test_game_sim_python_pokerkit_wrapper_acpc_style) with
    # `TypeError: super(type, obj): obj must be an instance or subtype of type`.
    #
    # The reason appears to be due to an edge case around how python handles
    # scoping inside of list comprehensions.
    #
    # For more details on the error, see
    # https://github.com/google-deepmind/open_spiel/actions/runs/18138757383/job/51624123398#step:6:3415
    pokerkit_style_action_strings = [
        self._action_to_string_base(current_player, action)
        for action in pokerkit_style_actions
    ]
    self._pokerkit_action_to_acpc_style_mapping = (
        self._action_converter.create_player_action_map(
            current_player,
            pokerkit_style_actions,
            pokerkit_style_action_strings,
        )
    )
    self._acpc_action_to_pokerkit_action_mapping = {
        v: k for k, v in self._pokerkit_action_to_acpc_style_mapping.items()
    }

  def to_pokerkit_action(self, acpc_style_action):
    return self._acpc_action_to_pokerkit_action_mapping[acpc_style_action]

  def to_acpc_action(self, pokerkit_style_action):
    return self._pokerkit_action_to_acpc_style_mapping[pokerkit_style_action]

  def _action_to_string(self, player, action):
    """Returns the string representation of an ACPC style action.

    NOTE: Maps ACPC Style actions to a Pokerkit Style actions, then returns the
    base class's _action_to_string(). Does NOT attempt to make the output string
    exactly match that of UniversalPoker.

    Args:
      player: The player for whom the action string is generated.
      action: The ACPC-style action.
    """
    if self.is_chance_node():
      return super()._action_to_string(player, action)
    if self.is_terminal():
      raise ValueError("Cannot convert action to string for a terminal state.")

    if action not in self._acpc_action_to_pokerkit_action_mapping:
      raise ValueError(
          f"Action {action} is not a valid ACPC style action. Expected legal"
          f" actions are: {self.legal_actions(player)}"
      )
    assert not (self.is_chance_node() or self.is_terminal())
    return super()._action_to_string(
        player, self._acpc_action_to_pokerkit_action_mapping[action]
    )

  def _legal_actions(self, player: int) -> list[int]:
    """Returns the legal actions in ACPC style.

    Obtains the list of pokerkit-style actions from the base class, then
    converts them all to ACPC style.

    Args:
      player: The player for whom to get the legal actions.

    Returns:
      A list of legal actions in ACPC style.
    """
    pokerkit_style_legal_actions = super()._legal_actions(player)
    if self.is_chance_node() or self.is_terminal():
      logging.warning("called legal_actions in chance node or terminal state.")
      return pokerkit_style_legal_actions
    else:
      return [self.to_acpc_action(a) for a in pokerkit_style_legal_actions]

  def _apply_action(self, action):
    """Apply ACPC-style actions.

    Aplies ACPC actions by converting them to Pokerkit style actions and then
    passing them to the base class's _apply_action(). Also perorms various
    validation checks and bookkeeping.

    Args:
      action: The ACPC-style action to apply.

    Raises:
      RuntimeError: If the action string does not match the expected format
        after conversion.
    """
    if self.is_chance_node():
      self._action_converter.track_chance_node()
      super()._apply_action(action)

      # Refresh so that the *next* action will be handled correctly.
      self._refresh_action_mappings_if_player_node()
      return

    if self.is_terminal():
      raise ValueError("Cannot apply action to a terminal state.")

    assert not (self.is_chance_node() or self.is_terminal())
    pokerkit_style_action = self._acpc_action_to_pokerkit_action_mapping[action]

    # Avoid bugs due to accidentally using the unconverted action (which is
    # ACPC-style still at this point)
    acpc_style_action = action
    del action
    if pokerkit_style_action == ACTION_FOLD:
      super()._apply_action(pokerkit_style_action)
      self._refresh_action_mappings_if_player_node()
      return

    if pokerkit_style_action == ACTION_CHECK_OR_CALL:
      size = self._wrapped_state.checking_or_calling_amount
    elif pokerkit_style_action > ACTION_CHECK_OR_CALL:
      # Handling completion bet or raise-to. *Usually* the action is the size,
      # but there are rare exceptions.
      size = pokerkit_style_action
      # NOTE: the first call is using super() meaning it will interpret the
      # action as a *pokerkit style* action! Not an ACPC style action, like we
      # would get it using self.action_to_string().
      if (
          not super()
          ._action_to_string(self.current_player(), pokerkit_style_action)
          .endswith(f"Bet/Raise to {size}")
      ):
        # Currently there is only one very specific edge case where this is
        # allowed: betting exactly one chip.
        assert (
            super()
            ._action_to_string(self.current_player(), pokerkit_style_action)
            .endswith(_ALL_IN_FOR_ONE_CHIP_EDGECASE_STRING)
        )
        size = 1

      # NOTE: earlier we were checking the string using the **non-acpc** base
      # class handling. Pay attention to how we're doing self._action_to_string
      # now, not super()._action_to_string().
      action_string_from_acpc_action = self._action_to_string(
          self.current_player(), acpc_style_action
      )
      if not action_string_from_acpc_action.removesuffix(
          " [ALL-IN EDGECASE]"
      ).endswith(f"Bet/Raise to {size}"):
        raise RuntimeError(
            f"Resulting action string {action_string_from_acpc_action} for"
            f" ACPC-style action {acpc_style_action} does not end as expected."
            f" Size is {size} and pokerkit-style action was"
            f" {pokerkit_style_action}.",
        )

    else:
      raise ValueError(
          "Action must be a check, call, completion bet, or raise-to."
      )
    assert size is not None

    self._action_converter.track_pokerkit_style_player_action(
        self.current_player(), pokerkit_style_action, size
    )
    super()._apply_action(pokerkit_style_action)
    self._refresh_action_mappings_if_player_node()
    return


class ToAcpcActionConverter:
  """Helper class to convert PokerKit-style actions to ACPC-style actions.

  Specifically: PokerKit's betting actions generally mean "raise this player's
  contribution for *this* street to X". This differs from ACPC's betting actions
  which generally represent "raise this player's toatal contribution across all
  streets to X". This class helps track contributions to allow us to convert
  between these two styles as needed in PokerKitWrapperACPCStyle.
  """

  def __init__(
      self,
      num_players: int,
      initial_contributions: dict[int, float],
  ):
    if num_players is None:
      raise ValueError("num_players must be specified.")
    if num_players <= 1:
      raise ValueError("num_players must be at least 2.")
    if initial_contributions is None:
      raise ValueError("initial_contributions must be specified.")

    self._num_players = num_players
    self._per_street_contributions: list[dict[int, float]] = [
        {p: 0.0 for p in range(self._num_players)} | initial_contributions
    ]

    for p in range(num_players):
      assert p in self._per_street_contributions[0]
    if len(self._per_street_contributions[0]) != num_players:
      raise ValueError(
          "After merging in initial_contributions, expected"
          f" {num_players} players. Instead got"
          f" {len(initial_contributions)} players."
      )

    # NOTE: Defaults to False since we also default self.current_street to 0 and
    # don't want track_chance_node() to increment the street
    # counter if people call it prior to any player actions, e.g. dealing cards
    # in Hold'em preflop chance nodes.
    self._ready_to_increment_street = False

  def track_chance_node(self):
    """Handles bookkeeping related to chance nodes potentially changing street.

    Specifically: increments the street IFF we have not tracked any additional
    contributions since the last time we incremented the street.

    In practice, users can call this function every time they are processing
    a chance node (and trust it will increment only once to the next street, on
    the first such call).

    Raises:
      RuntimeError: If the street is incremented past the maximum number of
        streets.
    """
    if not self._ready_to_increment_street:
      # E.g. called a second or third time when dealing the 3 flop cards.
      return
    # else:
    self._ready_to_increment_street = False
    self._per_street_contributions.append(
        {p: 0.0 for p in range(self._num_players)}
    )

  def track_pokerkit_style_player_action(
      self, player, pokerkit_style_action, size
  ):
    """Tracks a completion bet or raise, or check or call, for the given player.

    Stores information for the given action **assuming said action is NOT in
    ACPC stlye**, and marks that the street is ready to be incremented upon the
    next chance node.

    (If you want to track an action that is in ACPC style, please first convert
    it to the non-ACPC 'normal' PokerkitWrapper action before calling this
    method!)

    NOTE: Size should not necessarily equal the pokerkit_style_action (even if
    it does for most bets). In particular:
    - Size should be 0 for checks
    - Size should equal the 'calling amount' for calls.
    - Size should equal the amount completion-bet or raised-to. (This is
      _usually_ equal to the action, but there are exceptions: e.g. in
      situations where betting 1 chip is allowed.)

    Args:
      player: The index of the player who performed the action.
      pokerkit_style_action: The action taken by the player, using the
        PokerkitWrapper's action encoding (e.g., ACTION_FOLD,
        ACTION_CHECK_OR_CALL, or a bet/raise amount).
      size: The numerical value associated with the action. This is 0 for folds
        and checks, the calling amount for calls, and the total amount being
        bet/raised to for completion bets or raises.
    """
    if size < 0:
      raise ValueError("Size must be non-negative.")

    # The caller may still have additional contributions on this street, but we
    # can at least guarantee that on the *next* chance node it will be a
    # different street.
    self._ready_to_increment_street = True
    if pokerkit_style_action in FOLD_AND_CHECK_OR_CALL_ACTIONS and size == 0:
      pass  # Fold and check contribute no additional chips from their stack.
      return
    elif pokerkit_style_action == ACTION_FOLD:  # and size != 0
      raise ValueError("FOLD actions should be passed as 0 size.")
    elif pokerkit_style_action == ACTION_CHECK_OR_CALL:  # and size != 0
      # This MUST be a call if we've reacehd this point, since checks
      # (i.e. size == 0) should have been handled above already above.
      assert size > 0
      # NOTE: Adding, not replacing, since call sizes in pokerkit are defined
      # as the *additional* contribution on the current street (unlike
      # completion bets or raises).
      self._per_street_contributions[-1][player] += size
    elif pokerkit_style_action > max(FOLD_AND_CHECK_OR_CALL_ACTIONS):
      if pokerkit_style_action != size and not (
          pokerkit_style_action == 2 and size == 1
      ):
        raise ValueError(
            "pokerkit_style_action must equal size for completion bets or"
            " raise-tos (except very special cases whereaction 2 can mean 'bet"
            f" 1 chip'). Got action {pokerkit_style_action} and size"
            f" {size} instead."
        )
      # Completion bet or raise-to. (Note how unlike calls, this is effectively
      # "replacing" the contribution for the current street, not just adding to
      # it.)
      self._per_street_contributions[-1][player] = size
    else:
      raise ValueError(
          "pokerkit_style_action must be a fold, check, or call. Got action"
          f" {pokerkit_style_action} instead."
      )

  def create_player_action_map(
      self,
      player: int,
      pokerkit_style_actions: list[int],
      pokerkit_style_action_strings: list[str],
  ) -> dict[int, int]:
    """Creates an dict mapping Pokerkit-style actions to Acpc-style actions."""
    if len(pokerkit_style_actions) != len(pokerkit_style_action_strings):
      raise ValueError(
          "pokerkit_style_actions and action_strings must have the same length."
      )
    pokerkit_action_and_string_pairs = list(
        zip(pokerkit_style_actions, pokerkit_style_action_strings)
    )

    # Check that the input action and action strings match as expected.
    for (
        pokerkit_style_action,
        pokerkit_style_action_string,
    ) in pokerkit_action_and_string_pairs:
      if pokerkit_style_action not in FOLD_AND_CHECK_OR_CALL_ACTIONS:
        expected_bet_raise_string = "Bet/Raise to"
        if expected_bet_raise_string not in pokerkit_style_action_string:
          raise ValueError(
              f"Action string does not contain '{expected_bet_raise_string}'."
              f" Got action string: {pokerkit_style_action_string}"
          )

    # This is the main purpose for having this separate function: it allows us
    # to calculate the total prior street contribution once, rather than having
    # to do so for every single action.
    current_street = len(self._per_street_contributions) - 1
    # NOTE: exclusive slice to avoid including the current street's
    # contributions in the sum here.
    prior_street_contributions = self._per_street_contributions[:current_street]
    total_prior_street_contribution = int(
        sum(contribution[player] for contribution in prior_street_contributions)
    )
    pokerkit_action_to_acpc_action_mapping = {
        pokerkit_style_action: self._convert_action_to_acpc_style(
            pokerkit_style_action,
            pokerkit_style_action_string,
            total_prior_street_contribution,
        )
        for pokerkit_style_action, pokerkit_style_action_string in pokerkit_action_and_string_pairs
    }
    # All our actions should be unique by definition. If not, there's likely a
    # but in our logic.
    assert len(pokerkit_action_to_acpc_action_mapping.keys()) == len(
        set(pokerkit_action_to_acpc_action_mapping.keys())
    )
    assert len(pokerkit_action_to_acpc_action_mapping.values()) == len(
        set(pokerkit_action_to_acpc_action_mapping.values())
    )
    return pokerkit_action_to_acpc_action_mapping

  def _convert_action_to_acpc_style(
      self,
      pokerkit_style_action: int,
      pokerkit_style_action_string: str,
      total_prior_street_contribution: float,
  ) -> int:
    """Helper convert indvidiual actions for convert_pokerkit_style_actions."""
    if pokerkit_style_action in FOLD_AND_CHECK_OR_CALL_ACTIONS:
      # Fold and check_or_call actions are always hardcoded to the same values
      # in ACPC style and pokerkit style, so no need to convert the action in
      # any way.
      return pokerkit_style_action

    # Completion Bet or Raise-to
    #
    # NOTE: This is the one place where we usually have to convert actions!
    # I.e. ACPC-style and pokerkit-style actions primarily differ only in this
    # specific point.
    #
    # The "all-in for one chip" edge case is the only situation where in a
    # pokerkit style action a completion-bet or raise size is different from the
    # action's value.
    is_special_edge_case_betting_one_chip = (
        pokerkit_style_action == 2
        and pokerkit_style_action_string.endswith(
            _ALL_IN_FOR_ONE_CHIP_EDGECASE_STRING
        )
    )
    pokerkit_action_size = (
        pokerkit_style_action
        if not is_special_edge_case_betting_one_chip
        else 1
    )
    assert (
        f"Bet/Raise to {pokerkit_action_size}" in pokerkit_style_action_string
    )
    return pokerkit_action_size + total_prior_street_contribution


# ------------------------------------------------------------------------------

pyspiel.register_game(_GAME_TYPE, PokerkitWrapper)
pyspiel.register_game(_GAME_TYPE_ACPC_STYLE, PokerkitWrapperAcpcStyle)
