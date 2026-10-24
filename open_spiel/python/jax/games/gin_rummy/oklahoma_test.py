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

"""Tests for the Oklahoma flag on the gin rummy game.

Oklahoma gin: the first upcard sets each hand's knock limit; an ace means
the hand is played for gin (knock card 0), as OpenSpiel's `oklahoma`
parameter, which the scripted deals below compare against. Standard play is
`oklahoma=False`, with the knock card fixed at 10.

Layers:

1. **Standard play pinned**: with `oklahoma=False`, fixed-seed rollouts hash
   legal masks, observation tensors, actions and returns to a pinned digest.
2. **Knock-limit assignment**: both deal paths (chance-node deal and
   ``init_dealt_state``) set ``knock_card`` from the first upcard.
3. **Knock legality at several limits**: one scripted deal whose best
   line leaves deadwood 5, checked under upcards giving limits 0/3/7/10,
   each cross-checked against pyspiel's legal-action set.
"""

import hashlib

from absl.testing import absltest
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_STOCK,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_MELD_BASE,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_PASS
from open_spiel.python.jax.games.gin_rummy.simple_bot import (
    simple_bot_action_opt,
)
import pyspiel

# sha256 over (legal masks, player-0 obs tensors, actions, returns) of
# random rollouts at seeds (0, 7) with oklahoma=False. A change means
# standard (non-Oklahoma) play changed.
_STANDARD_ROLLOUT_GOLDEN = (
    "a5dc06fab23d449642aaf69a40aaf99d57af89283f731cd609d2ebb8ef2e1704"
)


class StandardRolloutDigestTest(absltest.TestCase):
  """The standard game's rollouts are unchanged."""

  def test_standard_rollout_digest(self):
    """The standard rollout digest equals its golden."""
    h = hashlib.sha256()
    game = gin_rummy.GinRummy()
    for seed in (0, 7):
      rng = np.random.default_rng(seed)
      state = game.new_initial_state()
      while not bool(state["done"]):
        if bool(game.is_chance_node(state)):
          p = np.asarray(game.chance_outcomes(state))
          action = int(rng.choice(p.size, p=p))
        else:
          mask = np.asarray(game.legal_actions_mask(state))
          h.update(mask.tobytes())
          obs = np.asarray(
              game.observation_tensor(state, jnp.int32(0)), dtype=np.float32
          )
          h.update(obs.tobytes())
          action = int(rng.choice(np.flatnonzero(mask)))
        state = game.apply_action(state, jnp.int32(action))
        h.update(np.int32(action).tobytes())
      h.update(np.asarray(game.returns(state), dtype=np.float32).tobytes())
    self.assertEqual(h.hexdigest(), _STANDARD_ROLLOUT_GOLDEN)


class KnockLimitAssignmentTest(absltest.TestCase):
  """Both deal paths must set knock_card from the first upcard."""

  def test_fast_init_knock_card_matches_upcard_value(self):
    """The fast deal's knock card is the upcard's value."""
    card_points = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
    seen = set()
    for seed in range(32):
      state, _ = gin_rummy.init_dealt_state(jax.random.PRNGKey(seed), True)
      upcard_value = int(card_points[int(state["upcard"]) % 13])
      expected = 0 if upcard_value == 1 else upcard_value
      self.assertEqual(int(state["knock_card"]), expected)
      self.assertTrue(bool(state["oklahoma"]))
      seen.add(expected)
    self.assertGreaterEqual(len(seen), 4, f"too little variety: {seen}")

  def test_standard_fast_init_unchanged(self):
    """The standard fast deal keeps knock card 10."""
    state, _ = gin_rummy.init_dealt_state(jax.random.PRNGKey(0))
    self.assertEqual(int(state["knock_card"]), 10)
    self.assertFalse(bool(state["oklahoma"]))


class KnockLegalityAtLimitsTest(absltest.TestCase):
  """One scripted deal, knock legality checked at limits 0/3/7/10.

  Deal (card = suit * 13 + rank; s=0, c=1, d=2, h=3; A=0..K=12):
    player 0: 2s-8s (a 7-card run, melded) + 2c + 3d + 4h (deadwood 9)
    player 1: arbitrary 10 cards
    upcard:   parameterized — sets the Oklahoma knock limit

  Both players pass the first upcard, player 0 draws 9s from stock (run
  grows to 2s-9s; best discard leaves deadwood 5). Knock must be legal
  iff 5 <= limit. Each case is cross-checked against pyspiel's
  ``oklahoma`` parameter (sanity check — the C++ reference has no
  doubling features to disable).
  """

  P0_CARDS = [1, 2, 3, 4, 5, 6, 7, 14, 28, 42]  # 2s..8s, 2c, 3d, 4h
  P1_CARDS = [18, 19, 20, 21, 22, 23, 24, 25, 31, 32]
  DRAWN_NINE_OF_SPADES = 8

  # (oklahoma, upcard card, expected knock_card, knock legal at deadwood 5)
  CASES = [
      (True, 0, 0, False),  # As => gin-only
      (True, 41, 3, False),  # 3h => limit 3
      (True, 45, 7, True),  # 7h => limit 7
      (True, 22, 10, True),  # 10c => value 10
      (False, 0, 10, True),  # standard: ace upcard is just a card
  ]

  def test_knock_legality_matches_limit_and_pyspiel(self):
    """Knock legality follows the limit, as in pyspiel."""
    for oklahoma, upcard, expected_limit, knock_legal in self.CASES:
      with self.subTest(oklahoma=oklahoma, upcard=upcard):
        if oklahoma:
          # P1 holds Jc (22) in the default hand; swap it out when Jc is
          # the upcard.
          assert upcard not in self.P0_CARDS
          p1 = [c for c in self.P1_CARDS if c != upcard] + (
              [33] if upcard in self.P1_CARDS else []
          )
        else:
          p1 = self.P1_CARDS
        actions = (
            self.P0_CARDS
            + p1
            + [upcard]
            + [ACTION_PASS, ACTION_PASS, ACTION_DRAW_STOCK]
            + [self.DRAWN_NINE_OF_SPADES]
        )

        game = gin_rummy.GinRummy(oklahoma=oklahoma)
        state = game.new_initial_state()
        for action in actions:
          state = game.apply_action(state, jnp.int32(action))
        self.assertEqual(int(state["knock_card"]), expected_limit)
        mask = np.asarray(game.legal_actions_mask(state))
        self.assertEqual(bool(mask[ACTION_KNOCK]), knock_legal)

        # pyspiel sanity cross-check: same deal, same legal-action set.
        os_state = pyspiel.load_game(
            "gin_rummy", {"oklahoma": oklahoma}
        ).new_initial_state()
        for action in actions:
          os_state.apply_action(action)
        self.assertEqual(
            set(np.flatnonzero(mask).tolist()),
            set(os_state.legal_actions()),
        )


class OklahomaSimpleBotTest(absltest.TestCase):
  """The simple bot must knock at <= the hand's active knock limit."""

  def _bot_action_at_golden_discard_node(self, oklahoma: bool) -> int:
    """The simple bot's action at the golden discard node.

    Args:
      oklahoma: Whether Oklahoma rules hold.

    Returns:
      The bot's action.
    """
    t = KnockLegalityAtLimitsTest
    actions = (
        t.P0_CARDS
        + t.P1_CARDS
        + [0]  # As upcard => gin-only under Oklahoma rules
        + [ACTION_PASS, ACTION_PASS, ACTION_DRAW_STOCK]
        + [t.DRAWN_NINE_OF_SPADES]
    )
    game = gin_rummy.GinRummy(oklahoma=oklahoma)
    state = game.new_initial_state()
    for action in actions:
      state = game.apply_action(state, jnp.int32(action))
    return int(simple_bot_action_opt(state))

  def test_bot_knocks_standard_but_not_oklahoma(self):
    """The bot knocks under standard rules but not under Oklahoma."""
    self.assertEqual(
        self._bot_action_at_golden_discard_node(oklahoma=False), ACTION_KNOCK
    )
    oklahoma_action = self._bot_action_at_golden_discard_node(oklahoma=True)
    self.assertNotEqual(oklahoma_action, ACTION_KNOCK)
    # The bot must still make a legal move: a discard (card index).
    self.assertLess(oklahoma_action, 52)


class OklahomaOutOfRangeTest(absltest.TestCase):
  """Non-card actions index oklahoma_knock_card out of range unless clipped.

  jnp.where evaluates both branches, so ACTION_KNOCK = 55 reads past the
  52-card value table even where the condition is False; the step clips the
  action first.
  """

  def test_unsafe_oklahoma_knock_card_raises_oob_with_action_knock(self):
    """An unclipped action makes oklahoma_knock_card index out of bounds."""

    def unsafe_branch(action):
      """Evaluates the knock card of an unclipped action.

      Args:
        action: The action.

      Returns:
        i8[] 10 (the knock card branch is never taken).
      """
      return jnp.where(
          False, gin_game.oklahoma_knock_card(action), jnp.int8(10)
      )

    err, _ = checkify.checkify(unsafe_branch, errors=checkify.index_checks)(
        ACTION_KNOCK
    )
    with self.assertRaisesRegex(Exception, "out-of-bounds"):
      err.throw()

  def test_safe_oklahoma_knock_card_clean_with_clipped_action(self):
    """A clipped action keeps oklahoma_knock_card in bounds."""

    def safe_branch(action):
      """Evaluates the knock card of a clipped action.

      Args:
        action: The action.

      Returns:
        i8[] 10 (the knock card branch is never taken).
      """
      safe_action = jnp.clip(action, 0, gin_game.NUM_CARDS - 1)
      return jnp.where(
          False, gin_game.oklahoma_knock_card(safe_action), jnp.int8(10)
      )

    err, _ = checkify.checkify(safe_branch, errors=checkify.index_checks)(
        ACTION_KNOCK
    )
    err.throw()

  def test_knock_card_unchanged_by_action_knock_in_dealt_oklahoma_state(self):
    """ACTION_KNOCK leaves a dealt Oklahoma state's knock card unchanged."""
    # Reaches private _step_body of the code under test.
    # pylint: disable=protected-access
    state, _ = gin_rummy.init_dealt_state(jax.random.PRNGKey(0), oklahoma=True)
    initial_knock_card = int(state["knock_card"])
    stepped = gin_game._step_body(state, ACTION_KNOCK)
    self.assertEqual(int(stepped["knock_card"]), initial_knock_card)

  def test_step_body_checkify_clean_for_strategic_non_card_actions(self):
    """_step_body is checkify index-clean for non-card actions.

    The non-card actions are PASS, KNOCK and the melds.

    jnp.where evaluates every card-indexed branch, so each must use the
    clipped action.
    """
    # Reaches private _step_body of the code under test.
    # pylint: disable=protected-access
    # In a dealt Oklahoma state, non-card actions still evaluate every
    # card-indexed branch of _step_body, because jnp.where is eager.
    state, _ = gin_rummy.init_dealt_state(jax.random.PRNGKey(0), oklahoma=True)

    # Test strategic non-card actions with full checkify index_checks.
    # These actions are illegal in PHASE_FIRST_UPCARD, but the card-indexed
    # branches still run; they must index with the clipped card action.
    checkified_step = checkify.checkify(
        gin_game._step_body, errors=checkify.index_checks
    )

    # ACTION_KNOCK = 55 is out of range for the 52-card arrays.
    err, _ = checkified_step(state, ACTION_KNOCK)
    err.throw()

    # ACTION_PASS has no card-indexed operation on its own path.
    err, _ = checkified_step(state, ACTION_PASS)
    err.throw()

    # ACTION_MELD_BASE = 56 is out of range for the 52-card arrays.
    err, _ = checkified_step(state, ACTION_MELD_BASE)
    err.throw()


if __name__ == "__main__":
  absltest.main()
