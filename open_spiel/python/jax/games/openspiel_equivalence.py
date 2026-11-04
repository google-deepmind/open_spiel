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

"""The harness that plays every JAX game move for move against OpenSpiel.

Each game plays seeded random playouts in lockstep with its OpenSpiel
counterpart. At every state the harness compares terminality, chance nodes,
the player to move, returns, chance distributions, legal actions, and the
observation and information-state tensors (through `PyspielInterop`). A
playout that diverges fails with the game, playout, step and field.

The tests are openspiel_equivalence_test.py and its shard files
(openspiel_equivalence_shard<k>_test.py); this module holds no test.

The reference games come from `reference_loader`, by default
`pyspiel.load_game`. `run_playouts` is public so another test can check a
game against a different OpenSpiel build, for example gin rummy rule
variants that stock OpenSpiel does not implement:

  run_playouts(test_case, [GameConfig("gin_rummy", ...)], my_loader)
"""

from __future__ import annotations

import dataclasses
import os
import random
from typing import Any, Callable

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import bridge
from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import no_limit_holdem
from open_spiel.python.jax.games import optional_pyspiel_games
from open_spiel.python.jax.games import registry
import pyspiel


@dataclasses.dataclass(frozen=True)
class GameConfig:
  """One game of the harness.

  Attributes:
    game_str: The OpenSpiel game string of the reference game.
    playouts: Number of seeded random playouts.
    max_steps: Steps per playout before it must be terminal.
    jax_game_str: The JAX game string, when it differs from `game_str`.
    check_observation: Compare observation tensors.
    check_information_state: Compare information-state tensors.
    allow_truncation: Accept playouts still running after `max_steps`.
  """

  game_str: str
  playouts: int
  max_steps: int
  jax_game_str: str | None = None
  check_observation: bool = True
  check_information_state: bool = True
  allow_truncation: bool = False

  @property
  def name(self) -> str:
    """The registered game name."""
    return (self.jax_game_str or self.game_str).split("(")[0]


@dataclasses.dataclass(frozen=True)
class CompiledFns:
  """The jitted game functions of one config."""

  legal_actions_mask: Callable[[Any], Any]
  returns: Callable[[Any], Any]
  observation_tensor: Callable[[Any, Any], Any]
  information_state_tensor: Callable[[Any, Any], Any]


ReferenceLoader = Callable[[str], pyspiel.Game]


def default_reference_loader(game_str: str) -> pyspiel.Game:
  """Loads the reference game with `pyspiel.load_game`.

  Args:
    game_str: The pyspiel game string.

  Returns:
    The pyspiel game.
  """
  return pyspiel.load_game(game_str)


NO_LIMIT_HOLDEM_CONFIG = GameConfig(
    no_limit_holdem.UNIVERSAL_POKER_GAME_STRING,
    jax_game_str="no_limit_holdem",
    playouts=5000,
    max_steps=256,
)


# The no-limit hold'em reference game, universal_poker, needs ACPC, an
# optional OpenSpiel dependency: the tests that load it skip on a build
# without it.
_needs_universal_poker = optional_pyspiel_games.skip_unless_available(
    NO_LIMIT_HOLDEM_CONFIG.game_str
)
needs_universal_poker = _needs_universal_poker  # for the shard files


def needs_optional_game(config: GameConfig) -> bool:
  """Returns whether `config`'s game needs an optional OpenSpiel game.

  Args:
    config: The game config.

  Returns:
    True when the game is in `optional_pyspiel_games.OPTIONAL_DEPENDENCY`.
  """
  name = config.game_str.split("(")[0]
  return name in optional_pyspiel_games.OPTIONAL_DEPENDENCY


GAME_CONFIGS = (
    # Each game gets hundreds to a thousand playouts, split over shards by
    # TEST_TOTAL_SHARDS / TEST_SHARD_INDEX when they are set.
    GameConfig("bargaining", playouts=1000, max_steps=64),
    # Mid-game chance nodes, with OpenSpiel's continue / end outcome ids.
    GameConfig("bargaining(prob_end=0.1)", playouts=300, max_steps=64),
    GameConfig("connect_four", playouts=1000, max_steps=64),
    GameConfig("kuhn_poker", playouts=500, max_steps=8),
    GameConfig("leduc_poker", playouts=1000, max_steps=64),
    GameConfig("tic_tac_toe", playouts=200, max_steps=9),
    NO_LIMIT_HOLDEM_CONFIG,
    GameConfig("phantom_ttt", playouts=300, max_steps=17),
    # A game registered under an OpenSpiel name is checked at OpenSpiel's
    # default parameters (test_every_openspiel_name_is_checked_at_its_defaults).
    GameConfig("phantom_ttt_ir", playouts=300, max_steps=17),
    GameConfig(
        "phantom_ttt_ir(gameversion=abrupt)", playouts=300, max_steps=17
    ),
    GameConfig(
        "dark_hex(board_size=3)",
        jax_game_str="dark_hex",
        playouts=300,
        max_steps=81,
    ),
    GameConfig("dark_hex_ir", playouts=300, max_steps=17),
    GameConfig(
        "dark_hex_ir(board_size=3,gameversion=adh)", playouts=300, max_steps=17
    ),
    GameConfig("liars_dice", playouts=300, max_steps=16),
    GameConfig(
        "liars_dice(numdice=2,dice_sides=5)", playouts=300, max_steps=30
    ),
    # Larger boards and long games use smaller budgets.
    # Bridge: 2,400 boards, every dealer, checked tensor for tensor. The
    # auction view is dealer-relative, as in OpenSpiel's current bridge; the
    # other dealers are compared only with such a pyspiel (bridge_reference).
    GameConfig(
        "bridge(use_double_dummy_result=false)",
        playouts=1600,
        max_steps=bridge.NUM_CARDS + bridge.MAX_GAME_LENGTH,
    ),
    GameConfig(
        "bridge(use_double_dummy_result=false,dealer=1)",
        playouts=200,
        max_steps=bridge.NUM_CARDS + bridge.MAX_GAME_LENGTH,
    ),
    GameConfig(
        "bridge(use_double_dummy_result=false,dealer=1,dealer_vul=true)",
        playouts=200,
        max_steps=bridge.NUM_CARDS + bridge.MAX_GAME_LENGTH,
    ),
    GameConfig(
        "bridge(use_double_dummy_result=false,dealer=2,non_dealer_vul=true)",
        playouts=200,
        max_steps=bridge.NUM_CARDS + bridge.MAX_GAME_LENGTH,
    ),
    GameConfig(
        "bridge(use_double_dummy_result=false,dealer=3,dealer_vul=true)",
        playouts=200,
        max_steps=bridge.NUM_CARDS + bridge.MAX_GAME_LENGTH,
    ),
    GameConfig("breakthrough", playouts=200, max_steps=256),
    GameConfig("go", playouts=25, max_steps=200, allow_truncation=True),
    # Small boards play whole games, superko draws included.
    GameConfig("go(board_size=5)", playouts=200, max_steps=50),
    GameConfig("gomoku", playouts=100, max_steps=225, allow_truncation=True),
    GameConfig("hex", playouts=100, max_steps=121),
    GameConfig("hex(swap=True)", playouts=100, max_steps=122),
    GameConfig("hex(board_size=5)", playouts=100, max_steps=25),
    GameConfig(
        "lines_of_action",
        playouts=100,
        max_steps=99,
        allow_truncation=True,
    ),
    GameConfig(
        "othello", playouts=100, max_steps=130, check_information_state=False
    ),
    # Non-default game parameters.
    GameConfig("bargaining(discount=0.9)", playouts=100, max_steps=64),
    GameConfig("connect_four(rows=7,columns=8)", playouts=100, max_steps=56),
    GameConfig(
        "go(board_size=9)", playouts=10, max_steps=200, allow_truncation=True
    ),
    GameConfig(
        "go(board_size=13)", playouts=5, max_steps=200, allow_truncation=True
    ),
    GameConfig(
        "hex(board_size=11,swap=true)",
        playouts=10,
        max_steps=122,
        allow_truncation=True,
    ),
    GameConfig(
        "lines_of_action(max_game_length=100)",
        playouts=10,
        max_steps=100,
        allow_truncation=True,
    ),
)

# OpenSpiel names whose default parameters the JAX game refuses (it raises
# ValueError) rather than play another game under the name, with the reason.
REFUSES_OPENSPIEL_DEFAULTS = {
    "bridge": (
        "OpenSpiel's default use_double_dummy_result=true ends the board after "
        "the auction and scores it with a double-dummy solver; the JAX game "
        "plays the cards (use_double_dummy_result=false)"
    ),
}

# Games whose rules differ from released OpenSpiel's, so stock OpenSpiel is
# not their reference.
NEEDS_REFERENCE_BUILD = {
    "gin_rummy": (
        "one scoring rule differs from released OpenSpiel (2.0.2 and earlier): "
        "equal deadwood after layoffs is an undercut here (the defender scores "
        "0 plus the undercut bonus), while released OpenSpiel scores the tie 0 "
        "for "
        "both; check it with run_playouts and a loader for an OpenSpiel build "
        "with that rule"
    ),
}


def _shard_for_current_process(cases: list[tuple[GameConfig, int]]):
  """The cases this test shard runs (TEST_TOTAL_SHARDS, TEST_SHARD_INDEX).

  Args:
    cases: Every (config, playout) case.

  Returns:
    The shard's cases.

  Raises:
    ValueError: The shard environment variables are inconsistent.
  """
  total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
  shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))
  if not 0 <= shard_index < total_shards:
    raise ValueError(
        f"invalid shard env: TEST_SHARD_INDEX={shard_index}, "
        f"TEST_TOTAL_SHARDS={total_shards}"
    )
  return [
      case
      for index, case in enumerate(cases)
      if index % total_shards == shard_index
  ]


def bridge_other_dealer(config: GameConfig) -> bool:
  """Returns whether `config` is a bridge config whose dealer is not North.

  Args:
    config: The game config.

  Returns:
    True for bridge with a dealer other than 0.
  """
  name, params = registry.parse_game_string(config.game_str)
  return name == "bridge" and params.get("dealer", 0) != 0


def _playout_cases() -> list[tuple[GameConfig, int]]:
  """This shard's (config, playout) cases.

  Returns:
    The cases.
  """
  cases = []
  for config in GAME_CONFIGS:
    for playout in range(config.playouts):
      cases.append((config, playout))
  return _shard_for_current_process(cases)


def _context(config: GameConfig, playout: int, step: int, field: str) -> str:
  """The failure context of one compared field.

  Args:
    config: The game config.
    playout: The playout.
    step: The step.
    field: The field.

  Returns:
    The context string.
  """
  return f"game={config.game_str} playout={playout} step={step} field={field}"


def _as_bool(value: Any) -> bool:
  """A scalar array as a bool.

  Args:
    value: The scalar.

  Returns:
    The bool.
  """
  return bool(np.asarray(value).item())


def _as_int(value: Any) -> int:
  """A scalar array as an int.

  Args:
    value: The scalar.

  Returns:
    The int.
  """
  return int(np.asarray(value).item())


def _current_player(game: game_lib.Game, state: Any) -> int:
  """The JAX game's player to move.

  Args:
    game: The JAX game.
    state: The state.

  Returns:
    The player id.
  """
  return _as_int(game.current_player(state))


def _is_chance_node(game: game_lib.Game, state: Any) -> bool:
  """Whether `state` is a non-terminal chance node.

  Args:
    game: The JAX game.
    state: The state.

  Returns:
    True at a chance node.
  """
  return _as_bool(game.is_chance_node(state)) and not _as_bool(
      game.is_terminal(state)
  )


def _assert_equal(
    actual: Any,
    expected: Any,
    config: GameConfig,
    playout: int,
    step: int,
    field: str,
) -> None:
  if actual != expected:
    raise AssertionError(
        f"{_context(config, playout, step, field)} mismatch: "
        f"actual={actual!r} expected={expected!r}"
    )


def _assert_array_equal(
    actual: Any,
    expected: Any,
    config: GameConfig,
    playout: int,
    step: int,
    field: str,
) -> None:
  try:
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
  except AssertionError as exc:
    raise AssertionError(
        f"{_context(config, playout, step, field)} mismatch:\n{exc}"
    ) from exc


def _assert_array_close(
    actual: Any,
    expected: Any,
    config: GameConfig,
    playout: int,
    step: int,
    field: str,
    atol: float = 1e-5,
) -> None:
  """Asserts that two arrays are close within `atol`.

  Args:
    actual: The JAX value.
    expected: The OpenSpiel value.
    config: The game config.
    playout: The playout index.
    step: The step index.
    field: The compared field's name.
    atol: The absolute and relative tolerance.

  Raises:
    AssertionError: The arrays differ.
  """
  try:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=atol, rtol=atol
    )
  except AssertionError as exc:
    raise AssertionError(
        f"{_context(config, playout, step, field)} mismatch:\n{exc}"
    ) from exc


def _jax_chance_distribution(
    game: game_lib.Game, state: Any
) -> dict[int, float]:
  """The JAX game's chance distribution, zero entries dropped.

  Args:
    game: The JAX game.
    state: The state.

  Returns:
    {action: probability}.
  """
  probs = np.asarray(game.chance_outcomes(state), dtype=np.float64)
  return {
      int(action): float(prob)
      for action, prob in enumerate(probs)
      if float(prob) > 0.0
  }


def _pyspiel_chance_distribution(
    game: game_lib.Game, os_state: pyspiel.State
) -> dict[int, float]:
  """OpenSpiel's chance distribution in JAX action ids.

  Args:
    game: The JAX game.
    os_state: The OpenSpiel state.

  Returns:
    {JAX action: probability}.
  """
  distribution = {}
  for os_action, prob in os_state.chance_outcomes():
    jax_action = game.action_from_os(os_state, int(os_action))
    distribution[int(jax_action)] = float(prob)
  return distribution


def _legal_jax_actions(mask: Any) -> list[int]:
  """The legal actions of a mask.

  Args:
    mask: The legal mask.

  Returns:
    The legal action ids.
  """
  return [int(a) for a in np.flatnonzero(np.asarray(mask, dtype=bool))]


def _compare_chance_outcomes(
    game: game_lib.Game,
    os_state: pyspiel.State,
    jax_state: Any,
    config: GameConfig,
    playout: int,
    step: int,
) -> None:
  """Checks that both states have the same chance distribution.

  Args:
    game: The JAX game.
    os_state: The OpenSpiel state.
    jax_state: The JAX state.
    config: The game config.
    playout: The playout index.
    step: The step index.
  """
  jax_dist = _jax_chance_distribution(game, jax_state)
  os_dist = _pyspiel_chance_distribution(game, os_state)
  _assert_equal(
      set(jax_dist), set(os_dist), config, playout, step, "chance_actions"
  )
  for action in sorted(os_dist):
    _assert_array_close(
        jax_dist[action],
        os_dist[action],
        config,
        playout,
        step,
        f"chance_prob[{action}]",
    )


def _compare_legal_actions(
    game: game_lib.Game,
    os_state: pyspiel.State,
    jax_state: Any,
    fns: CompiledFns,
    config: GameConfig,
    playout: int,
    step: int,
) -> list[int]:
  """Checks that both states have the same legal actions.

  Compares the JAX legal mask with OpenSpiel's, and maps the legal actions
  both ways between the JAX and OpenSpiel action spaces.

  Args:
    game: The JAX game.
    os_state: The OpenSpiel state.
    jax_state: The JAX state.
    fns: The compiled JAX functions.
    config: The game config.
    playout: The playout index.
    step: The step index.

  Returns:
    The legal JAX actions.
  """
  mask = fns.legal_actions_mask(jax_state)
  legal_jax = _legal_jax_actions(mask)
  expected_mask = game.legal_actions_mask_from_os(os_state)
  _assert_array_equal(
      mask, expected_mask, config, playout, step, "legal_actions_mask"
  )

  mapped_to_os = {game.action_to_os(jax_state, action) for action in legal_jax}
  os_legal = set(os_state.legal_actions())
  _assert_equal(
      mapped_to_os, os_legal, config, playout, step, "legal_actions_to_os"
  )
  mapped_from_os = {
      game.action_from_os(os_state, os_action) for os_action in os_legal
  }
  _assert_equal(
      mapped_from_os,
      set(legal_jax),
      config,
      playout,
      step,
      "legal_actions_from_os",
  )
  return legal_jax


def _compare_tensors(
    game: game_lib.Game,
    os_game: pyspiel.Game,
    os_state: pyspiel.State,
    jax_state: Any,
    fns: CompiledFns,
    config: GameConfig,
    playout: int,
    step: int,
) -> None:
  """Checks every player's observation and information state tensors.

  Args:
    game: The JAX game.
    os_game: The OpenSpiel game.
    os_state: The OpenSpiel state.
    jax_state: The JAX state.
    fns: The compiled JAX functions.
    config: The game config.
    playout: The playout index.
    step: The step index.
  """
  if config.check_observation:
    for player in range(os_game.num_players()):
      _assert_array_equal(
          fns.observation_tensor(jax_state, jnp.array(player)),
          game.observation_from_os(os_state, player),
          config,
          playout,
          step,
          f"observation_tensor[player={player}]",
      )

  if config.check_information_state:
    for player in range(os_game.num_players()):
      _assert_array_equal(
          fns.information_state_tensor(jax_state, jnp.array(player)),
          game.information_state_from_os(os_state, player),
          config,
          playout,
          step,
          f"information_state_tensor[player={player}]",
      )


def compare_state(
    game: game_lib.Game,
    os_game: pyspiel.Game,
    os_state: pyspiel.State,
    jax_state: Any,
    fns: CompiledFns,
    config: GameConfig,
    playout: int,
    step: int,
) -> list[int]:
  """Compares one OpenSpiel state with its JAX twin, failing on a divergence.

  Args:
    game: The JAX game.
    os_game: The OpenSpiel game.
    os_state: The OpenSpiel state.
    jax_state: The JAX state.
    fns: The JAX game's compiled functions.
    config: The game config, for the failure message.
    playout: The playout index, for the failure message.
    step: The step in the playout, for the failure message.

  Returns:
    The legal JAX actions at a decision node; empty at a terminal or chance
    node.
  """
  os_terminal = os_state.is_terminal()
  jax_terminal = _as_bool(game.is_terminal(jax_state))
  _assert_equal(jax_terminal, os_terminal, config, playout, step, "terminal")

  os_chance = os_state.is_chance_node()
  jax_chance = _is_chance_node(game, jax_state)
  _assert_equal(jax_chance, os_chance, config, playout, step, "chance_node")

  # The player to move at a terminal state is game-specific in the JAX games, so
  # the turn is compared on non-terminal states.
  if not os_terminal:
    _assert_equal(
        _current_player(game, jax_state),
        os_state.current_player(),
        config,
        playout,
        step,
        "current_player",
    )

  _assert_array_close(
      fns.returns(jax_state),
      os_state.returns(),
      config,
      playout,
      step,
      "returns",
  )
  # pyspiel returns are cumulative at every state, so comparing them per step
  # also catches reward-timing divergence without relying on optional
  # game-specific reward APIs in the JAX wrappers.

  if os_terminal:
    return []
  if os_chance:
    _compare_chance_outcomes(game, os_state, jax_state, config, playout, step)
    return []

  legal_jax = _compare_legal_actions(
      game, os_state, jax_state, fns, config, playout, step
  )
  _compare_tensors(
      game, os_game, os_state, jax_state, fns, config, playout, step
  )
  return legal_jax


def _sample_chance_action(
    game: game_lib.Game, os_state: pyspiel.State, rng: random.Random
) -> int:
  """Samples a chance outcome from OpenSpiel's distribution.

  Args:
    game: The JAX game.
    os_state: The OpenSpiel state.
    rng: The random source.

  Returns:
    The sampled outcome as a JAX action.
  """
  outcomes = os_state.chance_outcomes()
  os_actions, probs = zip(*outcomes)
  os_action = rng.choices(os_actions, weights=probs)[0]
  return game.action_from_os(os_state, int(os_action))


def _sample_bridge_action(
    legal_jax: list[int], rng: random.Random, playout: int
) -> int:
  """Samples a bridge action, stratified by playout to vary the auctions.

  Args:
    legal_jax: The legal JAX actions.
    rng: The random number generator.
    playout: The playout index; selects the stratum.

  Returns:
    The JAX action.
  """
  # Uniform calls escalate to doubled seven-level contracts, which random
  # play nearly always defeats. Strata that pass more often, or that make the
  # cheapest call, reach part scores and games that random play can make,
  # and passed-out boards.
  stratum = playout % 4
  pass_prob = (0.0, 0.6, 0.8, 0.9)[stratum]
  if bridge.PASS_ACTION not in legal_jax:
    return rng.choice(legal_jax)
  if rng.random() < pass_prob:
    return bridge.PASS_ACTION
  if stratum == 2:
    return min(a for a in legal_jax if a != bridge.PASS_ACTION)
  return rng.choice(legal_jax)


def _sample_player_action(
    config: GameConfig, legal_jax: list[int], rng: random.Random, playout: int
) -> int:
  """Samples a player action, stratified by playout for bridge and NLHE.

  Args:
    config: The game config.
    legal_jax: The legal JAX actions.
    rng: The random number generator.
    playout: The playout index; selects the stratum.

  Returns:
    The JAX action.
  """
  if config.name == "bridge":
    return _sample_bridge_action(legal_jax, rng, playout)
  if config.name != "no_limit_holdem":
    return rng.choice(legal_jax)

  calls = [a for a in legal_jax if a == no_limit_holdem.CALL]
  folds = [a for a in legal_jax if a == no_limit_holdem.FOLD]
  raises = [a for a in legal_jax if a >= no_limit_holdem.MIN_RAISE_ACTION]
  stratum = playout % 10

  # Uniform sampling over the many raise sizes folds most betting trees
  # early. These strata keep some folds while favouring calls, minimum
  # raises, random raises and all-ins.
  if stratum == 0 and folds:
    return folds[0]
  if stratum in (1, 2, 3) and calls:
    return calls[0]
  if raises:
    if stratum in (4, 5):
      return raises[0]
    if stratum in (6, 7):
      return raises[-1]
    return rng.choice(raises)
  if calls:
    return calls[0]
  return rng.choice(legal_jax)


def card(rank: int, suit: int = 0) -> int:
  """Returns no_limit_holdem's card id of a rank and a suit.

  Args:
    rank: The rank, 0 (deuce) to 12 (ace).
    suit: The suit, 0 to no_limit_holdem.NUM_SUITS - 1.

  Returns:
    The card id.
  """
  return rank * no_limit_holdem.NUM_SUITS + suit


def run_playouts(
    test_case: absltest.TestCase,
    configs: tuple[GameConfig, ...] | list[GameConfig],
    reference_loader: ReferenceLoader = default_reference_loader,
    file_shard: tuple[int, int] = (0, 1),
) -> dict[str, int]:
  """Plays this process's shard of every config's playouts in lockstep.

  Args:
    test_case: The test case whose subTest reports each playout.
    configs: The games to check.
    reference_loader: Builds the OpenSpiel reference game from `game_str`.
    file_shard: (index, count): only the playouts at positions congruent to
      index mod count, before the process's own shard (TEST_SHARD_INDEX) is
      taken, so the playouts can be spread over test files.

  Returns:
    The number of truncated playouts per game string.

  Raises:
    AssertionError: through `test_case`, on the first divergence of each
      playout.
    ValueError: file_shard is not (index, count) with 0 <= index < count.
  """
  truncations = {config.game_str: 0 for config in configs}
  runtime = {}
  index, count = file_shard
  if not 0 <= index < count:
    raise ValueError(f"file_shard {file_shard}: want 0 <= index < count")
  cases = []
  for config in configs:
    cases += [(config, playout) for playout in range(config.playouts)]
  cases = cases[index::count]
  for config, playout in _shard_for_current_process(cases):
    with test_case.subTest(game=config.game_str, playout=playout):
      if config not in runtime:
        os_game = reference_loader(config.game_str)
        game = registry.load_game(config.jax_game_str or config.game_str)
        test_case.assertEqual(game.num_players, os_game.num_players())
        test_case.assertEqual(game.num_actions, os_game.num_distinct_actions())
        runtime[config] = (
            os_game,
            game,
            jax.jit(game.apply_action),
            CompiledFns(
                legal_actions_mask=jax.jit(game.legal_actions_mask),
                returns=jax.jit(game.returns),
                observation_tensor=jax.jit(game.observation_tensor),
                information_state_tensor=jax.jit(game.information_state_tensor),
            ),
        )
      os_game, game, apply_action, fns = runtime[config]
      truncated = _assert_playout_equivalent(
          config, playout, os_game, game, apply_action, fns
      )
      truncations[config.game_str] += int(truncated)
  return truncations


def _assert_playout_equivalent(
    config: GameConfig,
    playout: int,
    os_game: pyspiel.Game,
    game: game_lib.Game,
    apply_action: Callable[[Any, Any], Any],
    fns: CompiledFns,
) -> bool:
  """Plays one seeded playout in lockstep; returns True if truncated.

  Args:
    config: The game config.
    playout: The playout index (seeds the actions).
    os_game: The OpenSpiel game.
    game: The JAX game.
    apply_action: The jitted apply_action.
    fns: The jitted state functions.

  Returns:
    True when the playout hit max_steps before the end.

  Raises:
    AssertionError: The two games disagree.
  """
  rng = random.Random(137 + playout)
  os_state = os_game.new_initial_state()
  jax_state = game.new_initial_state()

  for step in range(config.max_steps + 1):
    legal_jax = compare_state(
        game, os_game, os_state, jax_state, fns, config, playout, step
    )
    if os_state.is_terminal():
      return False
    if step == config.max_steps:
      if config.allow_truncation:
        return True
      raise AssertionError(
          f"game={config.game_str} playout={playout} exceeded "
          f"max_steps={config.max_steps} without terminal state"
      )

    if os_state.is_chance_node():
      jax_action = _sample_chance_action(game, os_state, rng)
      os_action = game.action_to_os(jax_state, jax_action)
    else:
      jax_action = _sample_player_action(config, legal_jax, rng, playout)
      os_action = game.action_to_os(jax_state, jax_action)

    os_state.apply_action(int(os_action))
    jax_state = apply_action(jax_state, jnp.array(jax_action))
  return True


#: The playouts of test_move_for_move_equivalence and of its no_limit_holdem
#: variant are split over MOVE_SHARDS test files (this one is shard 0,
#: openspiel_equivalence_shard<k>_test.py shard k; run_playouts' file_shard), so
#: each file stays within a test runner's per-file time budget.
MOVE_SHARDS = 4


def move_for_move_configs() -> list[GameConfig]:
  """The configs of test_move_for_move_equivalence and its shard files.

  Returns:
    Every config but the optional-game and other-dealer bridge ones.
  """
  return [
      config
      for config in GAME_CONFIGS
      if not needs_optional_game(config) and not bridge_other_dealer(config)
  ]
