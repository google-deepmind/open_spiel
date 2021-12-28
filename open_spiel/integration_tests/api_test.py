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

# Lint as: python3
"""Tests for open_spiel.integration_tests.api."""

import enum
import re
import unittest

from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from open_spiel.python import games  # pylint:disable=unused-import
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import sample_some_states
from open_spiel.python.mfg import games as mfg_games  # pylint:disable=unused-import
from open_spiel.python.observation import make_observation
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("test_only_games", ".*",
                    "Test only selected games (regex). Defaults to all.")

_ALL_GAMES = pyspiel.registered_games()

_GAMES_TO_TEST = set([g.short_name for g in _ALL_GAMES if g.default_loadable])

_GAMES_NOT_UNDER_TEST = [
    g.short_name for g in _ALL_GAMES if not g.default_loadable
]

_GAMES_TO_OMIT_LEGAL_ACTIONS_CHECK = set(["bridge_uncontested_bidding"])

# The list of game instances to test on the full tree as tuples
# (name to display, string to pass to load_game).
_GAMES_FULL_TREE_TRAVERSAL_TESTS = [
    ("cliff_walking", "cliff_walking(horizon=7)"),
    ("kuhn_poker", "kuhn_poker"),
    ("leduc_poker", "leduc_poker"),
    ("iigoofspiel4", ("turn_based_simultaneous_game(game=goofspiel("
                      "imp_info=True,num_cards=4,points_order=descending))")),
    ("kuhn_poker3p", "kuhn_poker(players=3)"),
    ("first_sealed_auction", "first_sealed_auction(max_value=2)"),
    ("tiny_hanabi", "tiny_hanabi"),
    ("nf_auction", ("turn_based_simultaneous_game(game="
                    "normal_form_extensive_game(game="
                    "first_sealed_auction(max_value=3)))")),
    # Disabled by default - big games, slow tests.
    # Uncomment to check the games if you modify them.
    # ("liars_dice", "liars_dice"),
    # ("tiny_bridge_2p", "tiny_bridge_2p"),
]

_GAMES_FULL_TREE_TRAVERSAL_TESTS_NAMES = [
    g[1] for g in _GAMES_FULL_TREE_TRAVERSAL_TESTS
]

TOTAL_NUM_STATES = {
    # This maps the game name to (chance, playable, terminal)
    "cliff_walking": (0, 2119, 6358),
    "kuhn_poker": (4, 24, 30),
    "leduc_poker": (157, 3780, 5520),
    "liars_dice": (7, 147456, 147420),
    "iigoofspiel4": (0, 501, 576),
    "kuhn_poker3p": (17, 288, 312),
    "first_sealed_auction": (12, 10, 14),
    "tiny_bridge_2p": (29, 53760, 53340),
    "tiny_hanabi": (3, 16, 36),
    "nf_auction": (0, 7, 36),
}

# This is kept to ensure non-regression, but we would like to remove that
# when we can interpret what are these numbers.
PERFECT_RECALL_NUM_STATES = {
    "cliff_walking": 2119,
    "kuhn_poker": 12,
    "leduc_poker": 936,
    "liars_dice": 24576,
    "iigoofspiel4": 162,
    "kuhn_poker3p": 48,
    "first_sealed_auction": 4,
    "tiny_bridge_2p": 3584,
    "tiny_hanabi": 8,
    "nf_auction": 2,
}


class EnforceAPIOnFullTreeBase(parameterized.TestCase):
  """Test properties on the full game tree, instantiating the tree only once.

  A new class, extensing this class will be dynamically created and added as
  a unittest class for the games we want to test exhaustively.
  """

  @classmethod
  def setUpClass(cls):
    super(EnforceAPIOnFullTreeBase, cls).setUpClass()

    cls.all_states = set(
        get_all_states.get_all_states(
            cls.game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=True).values())

  def test_legal_actions_empty(self):
    # We check we have some non-terminal non-random states
    self.assertTrue(
        any(not s.is_terminal() and not s.is_chance_node()
            for s in self.all_states))

    for state in self.all_states:
      if state.is_terminal():
        # Empty on terminal
        msg = ("The game %s does not return an empty list on "
               "legal_actions() for state %s" % (self.game_name, str(state)))
        self.assertEmpty(state.legal_actions(), msg=msg)
        for player in range(self.game.num_players()):
          msg = ("The game %s does not return an empty list on "
                 "legal_actions(%i) for state %s" %
                 (self.game_name, player, str(state)))
          self.assertEmpty(state.legal_actions(player), msg=msg)
      elif state.is_simultaneous_node():
        # No requirement for legal_actions to be empty, since all players act.
        pass
      elif state.is_chance_node():
        # Would be an error to request legal actions for a non-chance player.
        pass
      else:
        # Empty for players other than the current player
        current_player = state.current_player()
        for player in range(self.game.num_players()):
          if player != current_player:
            msg = ("The game {!r} does not return an empty list on "
                   "legal_actions(<not current player>) in state {}".format(
                       self.game_name, state))
            self.assertEmpty(state.legal_actions(player), msg=msg)

  def test_number_of_nodes(self):
    expected_numbers = TOTAL_NUM_STATES[self.game_name]

    num_chance_nodes = 0
    num_terminals = 0
    num_playable = 0
    for state in self.all_states:
      if state.is_terminal():
        num_terminals += 1
      elif state.is_chance_node():
        num_chance_nodes += 1
      else:
        num_playable += 1

    self.assertEqual(expected_numbers,
                     (num_chance_nodes, num_playable, num_terminals))

  def test_current_player_returns_terminal_player_on_terminal_nodes(self):
    for state in self.all_states:
      if state.is_terminal():
        self.assertEqual(pyspiel.PlayerId.TERMINAL, state.current_player())

  def test_information_state_no_argument_raises_on_terminal_nodes(self):
    for state in self.all_states:
      if state.is_terminal():
        with self.assertRaises(RuntimeError):
          state.information_state_string()

  def test_game_is_perfect_recall(self):
    # We do not count the terminal nodes here.
    expected_number_states = PERFECT_RECALL_NUM_STATES[self.game_name]
    all_states = []
    for _ in range(3):
      infostate_player_to_history = _assert_is_perfect_recall(self.game)
      all_states.append(infostate_player_to_history)
      # We compare the total number of (infostate, player) touched, to prevent
      # any regression (we skip chance nodes).
      # We use assertEqual and not assertLen to prevent the huge dict to be
      # displayed
      self.assertEqual(expected_number_states, len(infostate_player_to_history))  # pylint: disable=g-generic-assert

  def test_constant_sum(self):
    game_type = self.game.get_type()
    terminal_values = {
        tuple(state.returns())
        for state in self.all_states
        if state.is_terminal()
    }
    if game_type.utility in (pyspiel.GameType.Utility.ZERO_SUM,
                             pyspiel.GameType.Utility.CONSTANT_SUM):
      expected_sum = self.game.utility_sum()
      for returns in terminal_values:
        self.assertEqual(sum(returns), expected_sum)
    elif game_type.utility == pyspiel.GameType.Utility.GENERAL_SUM:
      all_sums = {sum(returns) for returns in terminal_values}
      self.assertNotEqual(len(all_sums), 1)
    elif game_type.utility == pyspiel.GameType.Utility.IDENTICAL:
      for returns in terminal_values:
        self.assertLen(set(returns), 1)
    else:
      raise AssertionError("Invalid utility type {}".format(game_type.utility))

  def test_information_state_functions_raises_on_chance_nodes(self):

    def _assert_information_state_on_chance_nodes_raises(state):

      if state.is_chance_node():
        with self.assertRaises(RuntimeError):
          state.information_state_string()
        with self.assertRaises(RuntimeError):
          state.information_state_tensor()

    for state in self.all_states:
      _assert_information_state_on_chance_nodes_raises(state)

  def test_current_player_infosets_no_overlap_between_players(self):
    # This is the stronger property we can currently verify. In particular,
    # we can find some state h0 where player 0 plays such that:
    # h0.information_state_string(0) == h0.information_state_string(0).

    states_for_player = [set() for _ in range(self.game.num_players())]
    for state in self.all_states:
      if not state.is_chance_node() and not state.is_terminal():
        states_for_player[state.current_player()].add(state)
      elif state.is_chance_node():
        self.assertEqual(state.get_type(), pyspiel.StateType.CHANCE)
      else:
        self.assertEqual(state.get_type(), pyspiel.StateType.TERMINAL)

    infoset_functions = [lambda s, player: s.information_state_string(player)]

    def _information_state_tensor(state, player):
      return tuple(np.asarray(state.information_state_tensor(player)).flatten())

    infoset_functions.append(_information_state_tensor)

    for infoset_function in infoset_functions:

      information_sets_per_player = []
      for player in range(self.game.num_players()):
        set_l = set(
            infoset_function(s, player) for s in states_for_player[player])
        information_sets_per_player.append(set_l)

      union = set()
      for information_set in information_sets_per_player:
        union = union.union(information_set)
      self.assertLen(union, sum([len(x) for x in information_sets_per_player]))


class Relation(enum.Enum):
  SUBSET_OR_EQUALS = 1
  EQUALS = 2


class EnforceAPIOnPartialTreeBase(parameterized.TestCase):
  """This only partially test some properties."""

  @classmethod
  def setUpClass(cls):
    super(EnforceAPIOnPartialTreeBase, cls).setUpClass()

    cls.some_states = sample_some_states.sample_some_states(
        cls.game, max_states=400)
    cls.game_type = cls.game.get_type()

  def test_sequence_lengths(self):
    try:
      max_history_len = self.game.max_history_length()
      max_move_number = self.game.max_move_number()
      max_chance_nodes_in_history = self.game.max_chance_nodes_in_history()
    except RuntimeError:
      return  # The function is likely not implemented, so skip the test.

    self.assertGreater(max_history_len, 0)
    self.assertGreater(max_move_number, 0)
    if self.game_type.chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC:
      self.assertEqual(max_chance_nodes_in_history, 0)
    else:
      self.assertGreater(max_chance_nodes_in_history, 0)

    for state in self.some_states:
      self.assertLessEqual(len(state.full_history()), max_history_len)
      self.assertLessEqual(state.move_number(), max_move_number)

      chance_nodes_in_history = 0
      for item in state.full_history():
        if item.player == pyspiel.PlayerId.CHANCE:
          chance_nodes_in_history += 1
      self.assertLessEqual(chance_nodes_in_history, max_chance_nodes_in_history)

  def test_observations_raises_error_on_invalid_player(self):
    game = self.game
    game_type = self.game_type
    game_name = self.game_name
    num_players = game.num_players()

    for state in self.some_states:
      if game_type.provides_information_state_string:
        if not state.is_chance_node():
          for p in range(num_players):
            state.information_state_string(p)
        msg = f"information_state_string did not raise an error for {game_name}"
        with self.assertRaisesRegex(RuntimeError, "player >= 0", msg=msg):
          state.information_state_string(-1)
        with self.assertRaisesRegex(RuntimeError, "player <", msg=msg):
          state.information_state_string(num_players + 1)

      if game_type.provides_information_state_tensor:
        if not state.is_chance_node():
          for p in range(num_players):
            v = state.information_state_tensor(p)
            self.assertLen(v, game.information_state_tensor_size())
        msg = f"information_state_tensor did not raise an error for {game_name}"
        with self.assertRaisesRegex(RuntimeError, "player >= 0", msg=msg):
          state.information_state_tensor(-1)
        with self.assertRaisesRegex(RuntimeError, "player <", msg=msg):
          state.information_state_tensor(num_players + 1)

      if game_type.provides_observation_tensor:
        if not state.is_chance_node():
          for p in range(num_players):
            v = state.observation_tensor(p)
            self.assertLen(v, game.observation_tensor_size())
        msg = f"observation_tensor did not raise an error for {game_name}"
        with self.assertRaisesRegex(RuntimeError, "player >= 0", msg=msg):
          state.observation_tensor(-1)
        with self.assertRaisesRegex(RuntimeError, "player <", msg=msg):
          state.observation_tensor(num_players + 1)

      if game_type.provides_observation_string:
        if not state.is_chance_node():
          for p in range(num_players):
            state.observation_string(p)
        msg = f"observation_string did not raise an error for {game_name}"
        with self.assertRaisesRegex(RuntimeError, "player >= 0", msg=msg):
          state.observation_string(-1)
        with self.assertRaisesRegex(RuntimeError, "player <", msg=msg):
          state.observation_string(num_players + 1)

  def test_legal_actions_returns_empty_list_on_opponent(self):
    if self.game_name in _GAMES_TO_OMIT_LEGAL_ACTIONS_CHECK:
      return

    for state in self.some_states:
      if state.is_terminal():
        # Empty on terminal
        msg = ("The game %s does not return an empty list on "
               "legal_actions() for state %s" % (self.game_name, state))
        self.assertEmpty(state.legal_actions(), msg=msg)
        for player in range(self.game.num_players()):
          msg = ("The game %s does not return an empty list on "
                 "legal_actions(%i) for state %s" %
                 (self.game_name, player, state))
          self.assertEmpty(state.legal_actions(player), msg=msg)
      elif state.is_simultaneous_node():
        # No requirement for legal_actions to be empty, since all players act.
        pass
      elif state.is_chance_node():
        # Would be an error to request legal actions for a non-chance player.
        pass
      else:
        # Empty for players other than the current player
        current_player = state.current_player()
        for player in range(self.game.num_players()):
          if player != current_player:
            msg = ("The game {!r} does not return an empty list on "
                   "legal_actions(<not current player>) in state {}".format(
                       self.game_name, state))
            self.assertEmpty(state.legal_actions(player), msg=msg)

  def test_private_information_contents(self):
    try:
      private_observation = make_observation(
          self.game,
          pyspiel.IIGObservationType(
              public_info=False,
              perfect_recall=False,
              private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
    except (RuntimeError, ValueError):
      return

    if private_observation.string_from(self.some_states[0], 0) is None:
      return

    player_has_private_info = [False] * self.game.num_players()

    for state in self.some_states:
      for i in range(self.game.num_players()):
        if private_observation.string_from(state, i):
          player_has_private_info[i] = True

    if (self.game_type.information ==
        pyspiel.GameType.Information.IMPERFECT_INFORMATION):
      self.assertTrue(any(player_has_private_info))
    if (self.game_type.information ==
        pyspiel.GameType.Information.PERFECT_INFORMATION):
      self.assertFalse(any(player_has_private_info))

  def test_no_invalid_public_observations(self):
    try:
      public_observation = make_observation(
          self.game,
          pyspiel.IIGObservationType(
              public_info=True,
              perfect_recall=False,
              private_info=pyspiel.PrivateInfoType.NONE))
    except (ValueError, RuntimeError):
      return

    if public_observation.string_from(self.some_states[0], 0) is None:
      return

    for state in self.some_states:
      self.assertIsNotNone(public_observation.string_from(state, 0))


def _assert_properties_recursive(state, assert_functions):

  for assert_function in assert_functions:
    assert_function(state)

  # Recursion
  # TODO(author2): We often use a `give me the next node` function and we
  # probably want a utility method for that, which works for all games.
  if state.is_terminal():
    return
  elif state.is_chance_node():
    for action, unused_prob in state.chance_outcomes():
      state_for_search = state.child(action)
      _assert_properties_recursive(state_for_search, assert_functions)
  else:
    for action in state.legal_actions():
      state_for_search = state.child(action)
      _assert_properties_recursive(state_for_search, assert_functions)


def _assert_is_perfect_recall(game):
  """Raises an AssertionError if the game is not perfect recall.

  We are willing to ensure the following property (perfect recall):
  - define X_i(h) be the sequence of information states and actions from the
    start of the game observed by player i (i.e. excluding the states and
    actions taken by the opponents unless those actions are included in player
    i's information state), along h but not including the state at h:
       X_i(h) = (s_1, a_1), (s_2, a_2), ... , (s_{t-1}, a_{t-1})
    then player i has perfect recall in this game iff: forall s in S_i,
    forall h1, h2 in s X_i(h1) == X_i(h2). Here, we check that the game has
    perfect recall if this is true for all players i (excluding chance).

    For more detail and context, see page 11 of
    http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf

  In particular, note that:
  - we want to check that holds both for
    + `std::string information_state_string(current_player)`
    + `information_state_tensor`.
  - we check that currently only from the point of view of the current
    player at the information state (i.e. we compare
    `prev_state.information_state_string(current_player)` but not
    `prev_state.information_state_string(opponent_player)`

  The strategy is the following: we traverse the full tree (of states, not
  infostates), and make sure for each node that the history we get for
  the infostate associated to that node, that is is unique with respect to
  the infostate.

  Args:
    game: A Spiel game to check.

  Returns:
    The internal cache mapping (infostate_str, player_id) to a list of one
    history leading to this infostate.
  """
  game_info = game.get_type()
  if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("The game is expected to be sequential")

  infostate_player_to_history = {}
  _assert_is_perfect_recall_recursive(
      game.new_initial_state(),
      current_history=[],
      infostate_player_to_history=infostate_player_to_history)

  return infostate_player_to_history


def _assert_is_perfect_recall_recursive(state, current_history,
                                        infostate_player_to_history):
  """Raises an AssertionError if the game is not perfect recall.

  The strategy is the following: we traverse the full tree (of states, not
  infostates), and make sure for each node that the history we get for
  the infostate associated to that node, that is is unique with respect to
  the infostate.

  Args:
    state: The current state during the recursive tree traversal.
    current_history: The current list of strictly preceding `SpielState` objects
      that lead to the current `state` (excluded).
    infostate_player_to_history: A dictionnary mapping (infostate string
      representation, current_player) to the list of one instance of actual
      predecessor nodes.
  """

  if not state.is_chance_node() and not state.is_terminal():
    current_player = state.current_player()
    infostate_str = state.information_state_string(current_player)
    key = (infostate_str, current_player)

    if key not in infostate_player_to_history:
      # First time we see the node.
      infostate_player_to_history[key] = list(current_history)
    else:
      previous_history = infostate_player_to_history[key]

      if len(previous_history) != len(current_history):
        raise AssertionError("We found 2 history leading to the same state:\n"
                             "State: {!r}\n"
                             "InfoState str: {}\n"
                             "First history ({} states): {!r}\n"
                             "Second history ({} states): {!r}\n".format(
                                 state.history(), infostate_str,
                                 len(previous_history),
                                 "|".join([str(sa) for sa in previous_history]),
                                 len(current_history),
                                 "|".join([str(sa) for sa in current_history])))

      # Check for `information_state`
      # pylint: disable=g-complex-comprehension
      expected_infosets_history = [(s.information_state_string(current_player),
                                    a)
                                   for s, a in previous_history
                                   if s.current_player() == current_player]
      # pylint: disable=g-complex-comprehension
      infosets_history = [(s.information_state_string(current_player), a)
                          for s, a in current_history
                          if s.current_player() == current_player]

      if expected_infosets_history != infosets_history:
        # pyformat: disable
        raise AssertionError("We found 2 history leading to the same state:\n"
                             "history: {!r}\n"
                             "info_state str: {}\n"
                             "**First history ({} states)**\n"
                             "states: {!r}\n"
                             "info_sets: {!r}\n"
                             "**Second history ({} states)**\n"
                             "Second info_state history: {!r}\n"
                             "Second history: {!r}\n".format(
                                 state.history(),
                                 infostate_str,
                                 len(previous_history),
                                 "|".join([str(sa) for sa in previous_history]),
                                 expected_infosets_history,
                                 len(current_history), infosets_history,
                                 "|".join([str(sa) for sa in current_history])))
        # pyformat: enable

      # Check for `information_state_tensor`
      expected_infosets_history = [
          (s.information_state_tensor(s.current_player()), a)
          for s, a in previous_history
          if s.current_player() == current_player
      ]
      infosets_history = [(s.information_state_tensor(s.current_player()), a)
                          for s, a in current_history
                          if s.current_player() == current_player]

      if not all([
          np.array_equal(x, y)
          for x, y in zip(expected_infosets_history, infosets_history)
      ]):
        raise ValueError("The history as tensor in the same infoset "
                         "are different:\n"
                         "History: {!r}\n".format(state.history()))

  # Recursion

  # TODO(author2): We often use a `give me the next node` function and we
  # probably want a utility method for that, which works for all games.
  if state.is_terminal():
    return
  else:
    for action in state.legal_actions():
      state_for_search = state.child(action)
      _assert_is_perfect_recall_recursive(
          state_for_search,
          current_history=current_history + [(state, action)],
          infostate_player_to_history=infostate_player_to_history)


def _create_test_case_classes():
  """Yields one Testing class per game to test."""
  for game_name, game_string in _GAMES_FULL_TREE_TRAVERSAL_TESTS:
    if not re.match(FLAGS.test_only_games, game_string):
      continue
    game = pyspiel.load_game(game_string)
    new_class = type("EnforceAPIFullTree_{}_Test".format(game_name),
                     (EnforceAPIOnFullTreeBase,), {})
    new_class.game_name = game_name
    new_class.game = game
    yield new_class

  for game_name in _GAMES_TO_TEST:
    if not re.match(FLAGS.test_only_games, game_name):
      continue
    game = pyspiel.load_game(game_name)
    new_class = type("EnforceAPIPartialTree_{}_Test".format(game_name),
                     (EnforceAPIOnPartialTreeBase,), {})
    new_class.game_name = game_name
    new_class.game = game
    yield new_class


def load_tests(loader, tests, pattern):  # pylint: disable=invalid-name,g-doc-args
  """Returns Dynamically created TestSuite.

  This creates one TestCase per game to test.

  See https://docs.python.org/2/library/unittest.html#load-tests-protocol.
  """
  del pattern
  tests = tuple(
      loader.loadTestsFromTestCase(test_case_class)
      for test_case_class in _create_test_case_classes())
  return unittest.TestSuite(tests=tests)


def main(_):
  absltest.main()


if __name__ == "__main__":
  # Necessary to run main via app.run for internal tests.
  app.run(main)
