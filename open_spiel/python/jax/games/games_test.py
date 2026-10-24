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

"""The game registry and the Game contract, checked for every registered game.

The registry parses OpenSpiel-style game strings and refuses duplicate names.
Every game honours the `Game` contract: its methods are jittable,
tensors have their declared shapes, a playout reaches a terminal state, and
`chance_outcomes` returns one dense probability vector indexed by chance
action id.
"""

import random
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import connect_four
from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import kuhn_poker
from open_spiel.python.jax.games import liars_dice
from open_spiel.python.jax.games import registry
from open_spiel.python.jax.games import tic_tac_toe


class RegistryTest(parameterized.TestCase):
  """The game registry and its game strings."""

  def test_parse_game_string(self):
    """Game strings parse into a name and parameters."""
    self.assertEqual(registry.parse_game_string("hex"), ("hex", {}))
    self.assertEqual(registry.parse_game_string("hex()"), ("hex", {}))
    self.assertEqual(
        registry.parse_game_string("x(num_rows=6,val=0.99,name=hello)"),
        ("x", {"num_rows": 6, "val": 0.99, "name": "hello"}),
    )
    self.assertEqual(
        registry.parse_game_string("x(a=True,b=FALSE)"),
        ("x", {"a": True, "b": False}),
    )

  @parameterized.parameters("hex(board_size=3", "hex(board_size)", "hex(=3)")
  def test_parse_game_string_rejects_malformed_strings(self, game_str):
    """Malformed game strings raise.

    Args:
      game_str: The game string.
    """
    with self.assertRaises(ValueError):
      registry.parse_game_string(game_str)

  def test_load_game_with_defaults_string_and_kwargs(self):
    """load_game takes defaults, a string or kwargs."""
    for game in (
        registry.load_game("connect_four"),
        registry.load_game("connect_four()"),
    ):
      self.assertIsInstance(game, connect_four.ConnectFour)
      self.assertEqual((game.num_rows, game.num_cols), (6, 7))
    for game in (
        registry.load_game("connect_four(rows=7,columns=8)"),
        registry.load_game("connect_four", rows=7, columns=8),
    ):
      self.assertEqual((game.num_rows, game.num_cols), (7, 8))

  def test_load_game_refuses_parameters_given_twice(self):
    """A parameter given in the string and as a kwarg raises."""
    with self.assertRaises(ValueError):
      registry.load_game("connect_four(rows=7)", columns=8)

  def test_load_game_refuses_unknown_names(self):
    """An unknown game name raises."""
    with self.assertRaisesRegex(ValueError, "unknown game 'no_such_game'"):
      registry.load_game("no_such_game")

  def test_register_game_adds_and_refuses_duplicates(self):
    """register_game adds a game and refuses a duplicate."""
    # Reaches private _GAMES of the code under test.
    # pylint: disable=protected-access
    with mock.patch.dict(registry._GAMES):
      registry.register_game("my_ttt", tic_tac_toe.TicTacToe)
      self.assertIn("my_ttt", registry.registered_names())
      self.assertIsInstance(registry.load_game("my_ttt"), tic_tac_toe.TicTacToe)
      with self.assertRaisesRegex(ValueError, "already registered"):
        registry.register_game("my_ttt", tic_tac_toe.TicTacToe)
      registry.register_game("my_ttt", kuhn_poker.KuhnPoker, replace=True)
      self.assertIsInstance(registry.load_game("my_ttt"), kuhn_poker.KuhnPoker)
    self.assertNotIn("my_ttt", registry.registered_names())

  def test_register_game_refuses_unparseable_names(self):
    """register_game refuses an unparseable name."""
    with self.assertRaises(ValueError):
      registry.register_game("bad(name", tic_tac_toe.TicTacToe)

  @parameterized.parameters(
      "phantom_ttt_ir(gameversion=nonsense)",
      "phantom_ttt_ir(obstype=reveal-numturns)",
      "phantom_ttt(gameversion=abrupt)",
      "phantom_ttt(obstype=reveal-numturns)",
      "dark_hex(board_size=4)",
      "dark_hex_ir(board_size=4)",
      "dark_hex_ir(gameversion=nonsense)",
      "liars_dice(players=3)",
      "liars_dice(bidding_rule=increment-any)",
      "gin_rummy(backend=unknown)",
      # OpenSpiel's default bridge scores the auction with a double-dummy
      # solver; the JAX game plays the cards and refuses it.
      "bridge",
      "bridge(use_double_dummy_result=true)",
  )
  def test_unsupported_parameters_fail_loudly(self, game_str):
    """Unsupported parameters raise.

    Args:
      game_str: The game string.
    """
    with self.assertRaises(ValueError):
      registry.load_game(game_str)


# Games whose OpenSpiel defaults the JAX game refuses, played by the
# contract tests at the parameters they support.
_CONTRACT_GAME_STRINGS = {"bridge": "bridge(use_double_dummy_result=false)"}


def _load_for_contract(game_name):
  """Loads `game_name` with its contract-test parameters.

  Args:
    game_name: The game.

  Returns:
    The game.
  """
  return registry.load_game(_CONTRACT_GAME_STRINGS.get(game_name, game_name))


# Long enough for every game's opening deal (gin deals 21 cards) and its
# mid-game chance nodes.
_MAX_STEPS = 80
_SEEDS = (0, 1)


def _random_action(game, state, rng):
  """A random action: by chance probabilities, else uniform over legal.

  Args:
    game: The game.
    state: The state.
    rng: A random.Random.

  Returns:
    The action.
  """
  if bool(game.is_chance_node(state)):
    probs = np.asarray(game.chance_outcomes(state), dtype=np.float64)
    support = np.flatnonzero(probs > 0.0)
    return rng.choices(support.tolist(), probs[support].tolist())[0]
  legal = np.flatnonzero(np.asarray(game.legal_actions_mask(state)))
  return rng.choice(legal.tolist())


def _enable_x64():
  """A context in which jax uses 64-bit types.

  Returns:
    The context manager.
  """
  if hasattr(jax, "enable_x64"):
    return jax.enable_x64(True)
  # Lazy: an optional dependency, the fallback for JAX versions without
  # jax.enable_x64.
  from jax.experimental import enable_x64  # pylint: disable=g-import-not-at-top,import-outside-toplevel  # noqa: PLC0415

  return enable_x64()


class GameContractTest(parameterized.TestCase):
  """Every registered game honours the `game_lib.Game` contract."""

  @parameterized.named_parameters(
      (name, name) for name in registry.registered_names()
  )
  def test_playout_honours_the_game_contract(self, game_name):
    """A playout honours the Game contract.

    Args:
      game_name: The game.
    """
    game = _load_for_contract(game_name)
    self.assertIsInstance(game, game_lib.Game)
    self.assertGreater(game.num_actions, 0)
    n = game.max_chance_outcomes
    self.assertIsInstance(n, int)
    apply_fn = jax.jit(game.apply_action)
    mask_fn = jax.jit(game.legal_actions_mask)
    chance_fn = jax.jit(game.chance_outcomes)
    obs_fn = jax.jit(game.observation_tensor)
    info_fn = jax.jit(game.information_state_tensor)
    has_info_state = game.provides_information_state_tensor
    for seed in _SEEDS:
      rng = random.Random(seed)
      state = game.new_initial_state()
      chance_nodes = 0
      for step in range(_MAX_STEPS):
        where = f"{game_name} seed {seed} step {step}"
        if bool(game.is_terminal(state)):
          break
        outcomes = chance_fn(state)
        self.assertNotIsInstance(outcomes, tuple, where)
        self.assertEqual(outcomes.shape, (n,), where)
        self.assertEqual(outcomes.dtype, jnp.float32, where)
        mask = mask_fn(state)
        self.assertEqual(mask.shape, (game.num_actions,), where)
        self.assertEqual(mask.dtype, jnp.bool_, where)
        for player in range(game.num_players):
          obs = obs_fn(state, jnp.int32(player))
          self.assertEqual(obs.shape, game.obs_shape, where)
          if has_info_state:
            info = info_fn(state, jnp.int32(player))
            self.assertEqual(info.shape, game.info_state_shape, where)
        if bool(game.is_chance_node(state)):
          chance_nodes += 1
          self.assertEqual(
              int(game.current_player(state)), game_lib.CHANCE_PLAYER_ID, where
          )
          probs = np.asarray(outcomes, dtype=np.float64)
          self.assertTrue(np.all(probs >= 0.0), where)
          self.assertAlmostEqual(float(probs.sum()), 1.0, places=5, msg=where)
        else:
          player = int(game.current_player(state))
          self.assertIn(player, range(game.num_players), where)
          self.assertTrue(bool(np.any(np.asarray(mask))), where)
        state = apply_fn(state, jnp.int32(_random_action(game, state, rng)))
      returns = np.asarray(game.returns(state))
      self.assertEqual(returns.shape, (game.num_players,))
      if n > 0:
        self.assertGreater(chance_nodes, 0, f"{game_name}: no chance node")
      else:
        self.assertEqual(chance_nodes, 0, f"{game_name}: undeclared chance")

  @parameterized.named_parameters(
      (name, name) for name in registry.registered_names()
  )
  def test_step_compiles_with_observations(self, game_name):
    """A step with observations compiles.

    Args:
      game_name: The game.
    """
    game = _load_for_contract(game_name)
    state = game.new_initial_state()
    rng = random.Random(0)
    while bool(game.is_chance_node(state)):
      state = game.apply_action(
          state, jnp.int32(_random_action(game, state, rng))
      )

    @jax.jit
    def step(s):
      """Plays the first legal action and observes for every player.

      Args:
        s: The state.

      Returns:
        (next state, observations).
      """
      s = game.apply_action(s, jnp.argmax(game.legal_actions_mask(s)))
      obs = [
          game.observation_tensor(s, jnp.int32(p))
          for p in range(game.num_players)
      ]
      return s, obs

    _, obs = step(state)
    self.assertEqual(obs[0].dtype, jnp.float32)

  @parameterized.named_parameters(
      (name, name) for name in registry.registered_names()
  )
  def test_state_dtypes_hold_under_x64(self, game_name):
    """State dtypes hold under x64.

    Args:
      game_name: The game.
    """
    # With 64-bit types enabled process-wide, an int64 action (jnp.argmax's
    # result) and int64 literals must not change any state leaf's dtype.
    with _enable_x64():
      game = _load_for_contract(game_name)
      state = game.new_initial_state()
      want = jax.tree.map(lambda x: jnp.asarray(x).dtype, state)

      @jax.jit
      def step(s):
        """Plays the first legal action, observing with int64 players.

        Args:
          s: The state.

        Returns:
          (next state, observations, returns).
        """
        s = game.apply_action(s, jnp.argmax(game.legal_actions_mask(s)))
        obs = [
            game.observation_tensor(s, jnp.int64(p))
            for p in range(game.num_players)
        ]
        return s, obs, game.returns(s)

      for i in range(4):
        state, _, _ = step(state)
        got = jax.tree.map(lambda x: jnp.asarray(x).dtype, state)
        self.assertEqual(got, want, f"{game_name} step {i}")

  @parameterized.parameters(
      ("phantom_ttt_ir", [0, 3, 1, 4, 2], 8),
      ("phantom_ttt_ir(gameversion=abrupt)", [0, 3, 1, 4, 2], 8),
      (
          "liars_dice(numdice=2,dice_sides=5)",
          [4, 0, 3, 0, 0, 5, liars_dice.liar_action(2, 5)],
          6,
      ),
      ("liars_dice", [2, 3, 0, liars_dice.liar_action(1, 6)], 6),
  )
  def test_action_at_terminal_state_leaves_it_unchanged(
      self, game_name, actions, extra
  ):
    """An action at a terminal state leaves it unchanged.

    Args:
      game_name: The game.
      actions: Actions that end the game.
      extra: The action applied after the end.
    """
    game = registry.load_game(game_name)
    state = game.new_initial_state()
    for action in actions:
      state = game.apply_action(state, jnp.int32(action))
    self.assertTrue(bool(game.is_terminal(state)))
    after = game.apply_action(state, jnp.int32(extra))
    jax.tree.map(np.testing.assert_array_equal, after, state)


if __name__ == "__main__":
  absltest.main()
