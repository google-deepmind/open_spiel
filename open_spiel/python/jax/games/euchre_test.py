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

"""Euchre known answers, and move for move against a corrected OpenSpiel.

The JAX game is OpenSpiel's corrected euchre (euchre module doc); released
OpenSpiel's euchre differs, so euchre is in openspiel_equivalence's
NEEDS_REFERENCE_BUILD and the move-for-move check here runs only when the
installed pyspiel is the corrected build (its euchre tensor has 959 entries).
EUCHRE_PLAYOUTS overrides the playouts of each of EUCHRE_CONFIGS.

Without that build, the JAX game replays games recorded from it
(`euchre_golden`), so runs using an older pre-built pyspiel wheel still check
it against the
reference: each node's player, legal actions or chance support, all four
seats' tensors at every decision node, and the returns. With the corrected
build installed, `python euchre_test.py --record_golden=PATH` re-records
them (a reference engine change).
"""

import dataclasses
import functools
import hashlib
import os
import sys
import textwrap
import types

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import euchre
from open_spiel.python.jax.games import euchre_golden
from open_spiel.python.jax.games import game as game_lib
from open_spiel.python.jax.games import openspiel_equivalence as equivalence
from open_spiel.python.jax.games import registry

try:
  import pyspiel  # pytype: disable=import-error

  _HAS_PYSPIEL = True
except ImportError:
  _HAS_PYSPIEL = False


def corrected_reference() -> bool:
  """Whether the installed pyspiel has the corrected euchre (959 entries).

  Returns:
    True when pyspiel is installed and its euchre tensor is f32[959].
  """
  if not _HAS_PYSPIEL:
    return False
  shape = pyspiel.load_game("euchre").information_state_tensor_shape()
  return list(shape) == [euchre.INFO_STATE_SIZE]


def _card(rank: str, suit: str) -> int:
  """A card id from its rank and suit letters.

  Args:
    rank: One of 9TJQKA.
    suit: One of CDHS.

  Returns:
    The card id.
  """
  return "9TJQKA".index(rank) * 4 + "CDHS".index(suit)


def _deal(dealer: int, hands: dict[int, list[int]], upcard: int) -> list[int]:
  """The chance outcomes that deal `hands` and turn `upcard`.

  Args:
    dealer: The dealer.
    hands: Seat -> its five cards.
    upcard: The up-card.

  Returns:
    The 22 chance outcomes (the dealer, 20 cards, the up-card).
  """
  queues = {seat: list(cards) for seat, cards in hands.items()}
  order = [queues[(dealer + i) % 4].pop(0) for i in range(20)]
  return [dealer] + order + [upcard]


# North holds the five highest hearts (right and left bowers, ace, king,
# queen); the up-card is the 9 of hearts. West deals, so North bids first.
_NORTH_TRUMPS = {
    0: [_card(*c) for c in ("JH", "JD", "AH", "KH", "QH")],
    1: [_card(*c) for c in ("9C", "TC", "JC", "QC", "KC")],
    2: [_card(*c) for c in ("AC", "9S", "TS", "JS", "QS")],
    3: [_card(*c) for c in ("KS", "AS", "TD", "QD", "KD")],
}
_NORTH_UPCARD = _card("9", "H")


@functools.lru_cache(maxsize=None)
def _load(game_str: str) -> types.SimpleNamespace:
  """A game and its jitted functions, compiled once per game string.

  Args:
    game_str: The game string.

  Returns:
    The game (`game`) and jitted `apply`, `mask`, `info` and `returns`.
  """
  game = registry.load_game(game_str)
  seats = np.arange(euchre.NUM_PLAYERS)
  return types.SimpleNamespace(
      game=game,
      apply=jax.jit(game.apply_action),
      mask=jax.jit(game.legal_actions_mask),
      info=jax.jit(game.information_state_tensor),
      infos=jax.jit(
          lambda s: jax.vmap(game.information_state_tensor, (None, 0))(s, seats)
      ),
      chance=jax.jit(game.chance_outcomes),
      returns=jax.jit(game.returns),
  )


def _legal(g, s) -> list[int]:
  """The legal actions of `s`.

  Args:
    g: A `_load` result.
    s: The state.

  Returns:
    The legal action ids.
  """
  return np.flatnonzero(np.asarray(g.mask(s))).tolist()


def _run(g, actions, then_first_legal=False):
  """Applies `actions`; optionally then the first legal action to the end.

  Args:
    g: A `_load` result.
    actions: The actions (chance outcomes included) to apply.
    then_first_legal: Continue with each state's lowest legal action.

  Returns:
    The state and the full action history.
  """
  s = g.game.new_initial_state()
  history = []
  for a in actions:
    legal = _legal(g, s)
    assert a in legal, (a, legal, history)
    s = g.apply(s, a)
    history.append(a)
  while then_first_legal and not bool(s.is_terminal):
    a = _legal(g, s)[0]
    s = g.apply(s, a)
    history.append(a)
  return s, history


def _pyspiel_replay(game_str, history):
  """Replays `history` in pyspiel; None without the corrected build.

  Args:
    game_str: The game string.
    history: The actions.

  Returns:
    The pyspiel state, or None.
  """
  if not corrected_reference():
    return None
  os_state = pyspiel.load_game(game_str).new_initial_state()
  for a in history:
    os_state.apply_action(a)
  return os_state


def _digests(trace) -> tuple[int, tuple[int, ...]]:
  """The golden digests of a game's nodes (euchre_golden module doc).

  Args:
    trace: Per non-terminal node, (player, bool mask, tensors): the legal
      actions (f[31]) at a decision node with the four seats' f32[959]
      tensors, the chance support (f[24]) and None at a chance node.

  Returns:
    The nodes digest and the four seats' tensor digests, each the first 8
    bytes of a sha256 as a big-endian integer.
  """
  nodes = hashlib.sha256()
  seats = [hashlib.sha256() for _ in range(euchre.NUM_PLAYERS)]
  for player, mask, tensors in trace:
    nodes.update(int(player).to_bytes(1, "little", signed=True))
    nodes.update(np.asarray(mask, np.bool_).tobytes())
    if tensors is not None:
      for h, t in zip(seats, tensors, strict=True):
        h.update(np.asarray(t, np.float32).tobytes())
  return _digest(nodes), tuple(_digest(h) for h in seats)


def _digest(h) -> int:
  """The first 8 bytes of a hash's digest, a big-endian integer.

  Args:
    h: A hashlib hash.

  Returns:
    The integer (euchre_golden writes it as a 16-digit 0x literal).
  """
  return int.from_bytes(h.digest()[:8], "big")


def _jax_replay(game_str, history):
  """Replays a recorded history in the JAX game, tracing every node.

  Args:
    game_str: The game string.
    history: The actions, chance outcomes included.

  Returns:
    The trace (`_digests`) and the integer returns.

  Raises:
    AssertionError: The game ends early or late, or an action is illegal.
  """
  g = _load(game_str)
  s = g.game.new_initial_state()
  trace = []
  for i, a in enumerate(history):
    assert not bool(s.is_terminal), (game_str, "terminal at", i)
    player = int(s.current_player)
    if player == game_lib.CHANCE_PLAYER_ID:
      mask = np.asarray(g.chance(s)) > 0
      trace.append((player, mask, None))
    else:
      mask = np.asarray(g.mask(s))
      trace.append((player, mask, np.asarray(g.infos(s))))
    assert mask[a], (game_str, "illegal", a, "at", i)
    s = g.apply(s, a)
  assert bool(s.is_terminal), (game_str, "not terminal after", len(history))
  return trace, tuple(int(r) for r in np.asarray(g.returns(s)))


def _record_game(game_str, rng):
  """One recorded game of the installed (corrected) pyspiel.

  Chance outcomes by their probability; a decision passes with probability
  GOLDEN_PASS_PROB when it may (so all-pass hands and round-two bids occur),
  else takes a uniform legal action.

  Args:
    game_str: The game string.
    rng: A numpy Generator.

  Returns:
    (history, returns, nodes digest, seat digests), a GAMES entry.
  """
  game = pyspiel.load_game(game_str)
  state = game.new_initial_state()
  history, trace = [], []
  while not state.is_terminal():
    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      support = np.zeros(game.max_chance_outcomes(), np.bool_)
      support[list(outcomes)] = True
      trace.append((-1, support, None))
      a = int(rng.choice(outcomes, p=np.asarray(probs) / sum(probs)))
    else:
      legal = state.legal_actions()
      mask = np.zeros(game.num_distinct_actions(), np.bool_)
      mask[legal] = True
      tensors = [
          np.asarray(state.information_state_tensor(q), np.float32)
          for q in range(euchre.NUM_PLAYERS)
      ]
      trace.append((state.current_player(), mask, tensors))
      if euchre.PASS_ACTION in legal and rng.random() < GOLDEN_PASS_PROB:
        a = euchre.PASS_ACTION
      else:
        a = int(rng.choice(legal))
    history.append(a)
    state.apply_action(a)
  returns = tuple(int(r) for r in state.returns())
  assert all(r == float(r) for r in state.returns()), state.returns()
  nodes, seats = _digests(trace)
  return tuple(history), returns, nodes, seats


GOLDEN_PASS_PROB = 0.6


def record_golden(path: str) -> None:
  """Writes euchre_golden.py from the installed corrected pyspiel.

  Args:
    path: The module to write.

  Raises:
    RuntimeError: The installed pyspiel is not the corrected build.
  """
  if not corrected_reference():
    raise RuntimeError("needs a pyspiel with the corrected euchre (959)")
  rng = np.random.default_rng(euchre_golden.SEED)
  games = {
      c.game_str: tuple(
          _record_game(c.game_str, rng)
          for _ in range(euchre_golden.GAMES_PER_CONFIG)
      )
      for c in equivalence.EUCHRE_CONFIGS
  }
  with open(euchre_golden.__file__, encoding="utf-8") as f:
    head = f.read().split("\nGAMES = ", 1)[0]
  with open(path, "w", encoding="utf-8") as f:
    f.write(head + "\n" + _format_games(games))


def _format_games(games) -> str:
  """GAMES as Python source, one game per block, within 80 columns.

  Args:
    games: Game string -> its recorded games (`_record_game`).

  Returns:
    The `GAMES = {...}` assignment.
  """
  lines = ["GAMES = {"]
  for game_str, entries in games.items():
    lines.append(f'    "{game_str}": (')
    for history, returns, nodes, seats in entries:
      lines.append("        (")
      lines.append("            (")
      lines += [
          f"                {line}"
          for line in textwrap.wrap(", ".join(map(str, history)) + ",", 60)
      ]
      lines.append("            ),")
      lines.append(f"            {tuple(returns)!r},")
      lines.append(f"            0x{nodes:016x},")
      lines.append(f"            (0x{seats[0]:016x}, 0x{seats[1]:016x},")
      lines.append(f"             0x{seats[2]:016x}, 0x{seats[3]:016x}),")
      lines.append("        ),")
    lines.append("    ),")
  lines.append("}")
  return "\n".join(lines) + "\n"


def _random_step(game, s, key):
  """One random step: a chance outcome by its probability, else a legal action.

  Args:
    game: The Euchre game.
    s: The state.
    key: The PRNG key.

  Returns:
    The next state.
  """
  probs = game.chance_outcomes(s)
  logits_c = jnp.where(probs > 0, jnp.log(jnp.maximum(probs, 1e-30)), -jnp.inf)
  logits_p = jnp.where(game.legal_actions_mask(s), 0.0, -jnp.inf)
  a = jnp.where(
      game.is_chance_node(s),
      jax.random.categorical(key, logits_c),
      jax.random.categorical(key, logits_p),
  )
  return game.apply_action(s, a)


def _random_playout(game, key):
  """A random hand: every chance node and the longest game's decisions.

  Args:
    game: The Euchre game.
    key: The PRNG key.

  Returns:
    The final state (terminal states are left unchanged by apply_action).
  """
  keys = jax.random.split(key, 22 + game.max_game_length)
  s, _ = jax.lax.scan(
      lambda s, k: (_random_step(game, s, k), None),
      game.new_initial_state(),
      keys,
  )
  return s


class EuchreTest(parameterized.TestCase):
  """Euchre known answers."""

  def test_registered_with_pyspiel_sizes(self):
    """Registered with OpenSpiel's sizes."""
    game = registry.load_game("euchre")
    self.assertIsInstance(game, euchre.Euchre)
    self.assertEqual(
        (game.num_players, game.num_actions, game.max_chance_outcomes),
        (4, 31, 24),
    )
    self.assertEqual(game.info_state_shape, (959,))
    self.assertEqual(game.max_game_length, 29)
    self.assertEqual(
        registry.load_game("euchre(allow_lone_defender=true)").max_game_length,
        31,
    )

  def test_rejects_non_bool_parameters(self):
    """Rejects non-bool parameters."""
    with self.assertRaises(TypeError):
      euchre.Euchre(allow_lone_defender=1)

  @parameterized.parameters(
      (euchre.GO_ALONE_ACTION, [4.0, -4.0, 4.0, -4.0]),
      (euchre.PLAY_WITH_PARTNER_ACTION, [2.0, -2.0, 2.0, -2.0]),
  )
  def test_known_answer_march(self, decision, expected):
    """North orders up hearts, takes every trick: 4 alone, else 2.

    Args:
      decision: North's go-alone decision.
      expected: The returns.
    """
    g = _load("euchre")
    deal = _deal(3, _NORTH_TRUMPS, _NORTH_UPCARD)
    s, _ = _run(g, deal + [euchre.HEARTS_TRUMP_ACTION, decision])
    # West deals and plays (North's partner sits out at most), so West takes
    # up the 9 of hearts and discards.
    self.assertEqual(int(s.phase), euchre.PHASE_DISCARD)
    self.assertEqual(int(s.current_player), 3)
    legal = _legal(g, s)
    self.assertEqual(legal, sorted(_NORTH_TRUMPS[3]))
    s, history = _run(
        g, deal + [euchre.HEARTS_TRUMP_ACTION, decision], then_first_legal=True
    )
    np.testing.assert_array_equal(np.asarray(g.returns(s)), expected)
    self.assertTrue(np.all(np.asarray(s.trick_winner) == 0))
    os_state = _pyspiel_replay("euchre", history)
    if os_state is not None:
      self.assertTrue(os_state.is_terminal())
      np.testing.assert_array_equal(os_state.returns(), expected)

  def test_dealer_sitting_out_takes_no_upcard(self):
    """The dealer's partner orders up alone: no pick-up, no discard."""
    g = _load("euchre")
    hands = {(seat + 2) % 4: cards for seat, cards in _NORTH_TRUMPS.items()}
    # North deals; South (seat 2) holds the hearts and bids second.
    deal = _deal(0, hands, _NORTH_UPCARD)
    actions = deal + [
        euchre.PASS_ACTION,
        euchre.HEARTS_TRUMP_ACTION,
        euchre.GO_ALONE_ACTION,
    ]
    s, history = _run(g, actions)
    self.assertEqual(int(s.phase), euchre.PHASE_PLAY)
    self.assertEqual(int(s.current_player), 1)
    self.assertEqual(int(s.holder[_NORTH_UPCARD]), -1)
    self.assertEqual(int(s.discard), -1)
    np.testing.assert_array_equal(
        np.asarray(s.active), [False, True, True, True]
    )
    tensor = np.asarray(g.info(s, 0))
    self.assertEqual(tensor[euchre.GO_ALONE_OFFSET], 1.0)
    self.assertFalse(tensor[euchre.DISCARD_OFFSET : euchre.HAND_OFFSET].any())
    os_state = _pyspiel_replay("euchre", history)
    if os_state is not None:
      self.assertEqual(os_state.current_player(), 1)
      for q in range(4):
        np.testing.assert_array_equal(
            tensor if q == 0 else g.info(s, q),
            os_state.information_state_tensor(q),
        )

  def test_discard_is_in_the_dealers_tensor_only(self):
    """The dealer's discard is set in the dealer's tensor, nobody else's."""
    g = _load("euchre")
    deal = _deal(3, _NORTH_TRUMPS, _NORTH_UPCARD)
    discard = sorted(_NORTH_TRUMPS[3])[0]
    s, _ = _run(
        g,
        deal
        + [
            euchre.HEARTS_TRUMP_ACTION,
            euchre.PLAY_WITH_PARTNER_ACTION,
            discard,
        ],
    )
    self.assertEqual(int(s.phase), euchre.PHASE_PLAY)
    self.assertEqual(int(s.current_player), 0)
    for q in range(4):
      block = np.asarray(g.info(s, q))[
          euchre.DISCARD_OFFSET : euchre.HAND_OFFSET
      ]
      self.assertEqual(
          np.flatnonzero(block).tolist(), [discard] if q == 3 else []
      )

  def test_pass_slots_and_stick_the_dealer(self):
    """Each pass sets its own Pass slot; the dealer is stuck at bid eight."""
    g = _load("euchre")
    deal = _deal(3, _NORTH_TRUMPS, _NORTH_UPCARD)
    s, _ = _run(g, deal + [euchre.PASS_ACTION] * 7)
    legal = _legal(g, s)
    self.assertNotIn(euchre.PASS_ACTION, legal)
    self.assertEqual(
        legal,
        [
            euchre.CLUBS_TRUMP_ACTION,
            euchre.DIAMONDS_TRUMP_ACTION,
            euchre.SPADES_TRUMP_ACTION,
        ],
    )
    bids = np.asarray(g.info(s, 1))[
        euchre.BIDDING_OFFSET : euchre.GO_ALONE_OFFSET
    ].reshape(8, 5)
    np.testing.assert_array_equal(bids[:, 4], [1] * 7 + [0])
    self.assertFalse(bids[:, :4].any())

  def test_eight_passes_end_the_hand_without_stick_the_dealer(self):
    """Eight passes end the hand at zero; the tensor stops after the bids."""
    g = _load("euchre(stick_the_dealer=false)")
    deal = _deal(3, _NORTH_TRUMPS, _NORTH_UPCARD)
    s, history = _run(g, deal + [euchre.PASS_ACTION] * 8)
    self.assertTrue(bool(s.is_terminal))
    np.testing.assert_array_equal(np.asarray(g.returns(s)), [0, 0, 0, 0])
    tensor = np.asarray(g.info(s, 0))
    self.assertFalse(tensor[euchre.GO_ALONE_OFFSET :].any())
    self.assertEqual(tensor[: euchre.GO_ALONE_OFFSET].sum(), 1 + 1 + 8)
    os_state = _pyspiel_replay("euchre(stick_the_dealer=false)", history)
    if os_state is not None:
      self.assertTrue(os_state.is_terminal())
      np.testing.assert_array_equal(
          tensor, os_state.information_state_tensor(0)
      )

  def test_lone_defender_order_and_score(self):
    """A defender goes alone against a lone declarer: two seats play."""
    g = _load("euchre(allow_lone_defender=true)")
    deal = _deal(3, _NORTH_TRUMPS, _NORTH_UPCARD)
    # North (the declarer) goes alone. The defender left of the dealer is
    # asked first, but that seat is North, so East is asked first; East
    # declines and West (the dealer) goes alone, so East and South sit out.
    s, _ = _run(
        g,
        deal
        + [
            euchre.HEARTS_TRUMP_ACTION,
            euchre.GO_ALONE_ACTION,
        ],
    )
    self.assertEqual(int(s.current_player), 1)
    s, history = _run(
        g,
        deal
        + [
            euchre.HEARTS_TRUMP_ACTION,
            euchre.GO_ALONE_ACTION,
            euchre.PLAY_WITH_PARTNER_ACTION,
        ],
    )
    self.assertEqual(int(s.current_player), 3)
    s, history = _run(
        g, history + [euchre.GO_ALONE_ACTION], then_first_legal=True
    )
    np.testing.assert_array_equal(
        np.asarray(s.active), [True, False, False, True]
    )
    # West, still playing, took up the up-card and discarded; North takes
    # every trick: a lone march, 4.
    np.testing.assert_array_equal(
        np.asarray(g.returns(s)), [4.0, -4.0, 4.0, -4.0]
    )
    os_state = _pyspiel_replay("euchre(allow_lone_defender=true)", history)
    if os_state is not None:
      np.testing.assert_array_equal(os_state.returns(), [4, -4, 4, -4])

  def test_batched_random_playouts_are_zero_sum_and_end(self):
    """vmap+jit random playouts end within max_game_length, zero-sum."""
    for game_str in (
        "euchre",
        "euchre(allow_lone_defender=true,stick_the_dealer=false)",
    ):
      game = registry.load_game(game_str)

      playouts = jax.jit(jax.vmap(functools.partial(_random_playout, game)))
      states = playouts(jax.random.split(jax.random.key(0), 256))
      self.assertTrue(bool(jnp.all(states.is_terminal)), game_str)
      rets = np.asarray(jax.vmap(game.returns)(states))
      np.testing.assert_array_equal(rets.sum(axis=1), 0.0)
      self.assertTrue(set(np.unique(rets).tolist()) <= {-4, -2, -1, 0, 1, 2, 4})

  def test_move_for_move_against_corrected_openspiel(self):
    """EUCHRE_CONFIGS agree move for move with the corrected OpenSpiel."""
    if not corrected_reference():
      self.skipTest(
          "needs a pyspiel with the corrected euchre (959-entry tensor); "
          "released OpenSpiel has 935 (openspiel_equivalence "
          "NEEDS_REFERENCE_BUILD)"
      )
    playouts = os.environ.get("EUCHRE_PLAYOUTS")
    configs = [
        dataclasses.replace(c, playouts=int(playouts)) if playouts else c
        for c in equivalence.EUCHRE_CONFIGS
    ]
    truncations = equivalence.run_playouts(self, configs)
    self.assertEqual(sum(truncations.values()), 0)

  def test_golden_games_cover_every_config(self):
    """The recorded games cover EUCHRE_CONFIGS, GAMES_PER_CONFIG each."""
    self.assertEqual(
        set(euchre_golden.GAMES),
        {c.game_str for c in equivalence.EUCHRE_CONFIGS},
    )
    for game_str, games in euchre_golden.GAMES.items():
      self.assertLen(games, euchre_golden.GAMES_PER_CONFIG, game_str)

  @parameterized.parameters(*(c.game_str for c in equivalence.EUCHRE_CONFIGS))
  def test_replays_games_recorded_from_corrected_openspiel(self, game_str):
    """The JAX game reproduces games recorded from the reference build.

    Args:
      game_str: The game string.
    """
    for i, (history, returns, nodes, seats) in enumerate(
        euchre_golden.GAMES[game_str]
    ):
      trace, jax_returns = _jax_replay(game_str, list(history))
      jax_nodes, jax_seats = _digests(trace)
      where = f"{game_str} game {i}"
      self.assertEqual(jax_returns, tuple(returns), where)
      self.assertEqual(jax_nodes, nodes, f"{where}: players or legal actions")
      self.assertEqual(jax_seats, tuple(seats), f"{where}: seat tensors")


if __name__ == "__main__":
  _RECORD = "--record_golden="
  if len(sys.argv) == 2 and sys.argv[1].startswith(_RECORD):
    record_golden(sys.argv[1][len(_RECORD) :])
  else:
    absltest.main()
