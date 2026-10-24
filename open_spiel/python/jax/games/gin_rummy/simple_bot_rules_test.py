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

"""simple_bot under the gin rule variants.

`simple_bot_action_opt` plays the rules the state carries. Under
`allow_upcard_throwback=False` the upcard taken this turn is never its
discard (at DISCARD or as the knocker's 11-card discard), and it knocks at
DISCARD and at the WALL exactly when the engine offers a knock. Under
`spade_doubles=True` only the score changes, and the engine computes it.

What the tests pin:

1. Default rules: a digest of the bot's action at every player node of seeded
   playouts (standard and Oklahoma; at each node the bot or a uniform legal
   move acts). If it moves, the bot's play under the default rules changed.
2. Seeded playouts under every accepted combination with a variant on: at
   FIRST_UPCARD / DRAW / DISCARD / WALL the bot's move is legal, it knocks
   iff KNOCK is legal, and its 11-card knock discard is never the taken
   upcard.
3. A constructed position whose optimal arrangement has the taken upcard as
   its only deadwood: without throwback the bot knocks and discards the other
   card leaving the least deadwood; under the default rules it throws the
   upcard back.
"""

import hashlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DISCARD
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DRAW
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_FIRST_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_WALL
from open_spiel.python.jax.games.gin_rummy.simple_bot import (
    simple_bot_action_opt,
)

DEFAULTS = dict(
    allow_upcard_throwback=True, wall_knock_pick=True, spade_doubles=False
)

# sha256 of `_default_rules_stream` (seeds 0..39, standard and Oklahoma).
_DEFAULT_RULES_BOT_DIGEST = (
    "4f025c5a576c2836d5fbe85dd4e82b22a1ba092b76c8b0e5e0e1021723dde5e4"
)

_step = jax.jit(gin_game.step)
_legal_mask = jax.jit(gin_game.legal_actions_mask)
_chance = jax.jit(gin_game.chance_outcomes)
_is_chance = jax.jit(gin_game.is_chance_node)
_bot = jax.jit(simple_bot_action_opt)


def _play(seed, oklahoma, rules, on_node, bot_share=1.0, on_action=None):
  """One hand from ``init_dealt_state(PRNGKey(seed))``.

  At each player node ``on_node(state, mask, bot_action)`` is called; the hand
  continues on the bot's action when legal and drawn to play it (probability
  ``bot_share``), else on a uniform legal move. ``on_action(action)``, when
  given, is called with the move played at each player node.

  Args:
    seed: The seed of the deal and of the uniform moves.
    oklahoma: The oklahoma rule.
    rules: The other rules, as init_dealt_state keywords.
    on_node: Called at each player node.
    bot_share: The probability of playing the bot's action.
    on_action: Called with each move played, or None.

  Returns:
    The terminal state.
  """
  rng = np.random.default_rng(seed)
  state, _ = gin_game.init_dealt_state(
      jax.random.PRNGKey(seed), oklahoma, False, **rules
  )
  for _ in range(400):
    if bool(state["done"]):
      return state
    if bool(_is_chance(state)):
      p = np.asarray(_chance(state), dtype=np.float64)
      action = int(rng.choice(p.size, p=p / p.sum()))
    else:
      mask = np.asarray(_legal_mask(state))
      bot_move = int(_bot(state))
      on_node(state, mask, bot_move)
      legal = np.flatnonzero(mask)
      if mask[bot_move] and rng.random() < bot_share:
        action = bot_move
      else:
        action = int(rng.choice(legal))
      if on_action is not None:
        on_action(action)
    state = _step(state, jnp.int32(action))
  raise AssertionError(f"seed {seed}: hand did not end in 400 steps")


def _default_rules_stream():
  """A digest of the simple bot's moves under the default rules.

  Returns:
    The hex digest.
  """
  h = hashlib.sha256()
  for oklahoma in (False, True):
    for seed in range(40):

      def record(state, mask, bot_move):
        del mask
        h.update(np.int32(int(state["phase"])).tobytes())
        h.update(np.int32(bot_move).tobytes())

      _play(seed, oklahoma, DEFAULTS, record, bot_share=0.7)
  return h.hexdigest()


# Every accepted combination with a variant on (spade_doubles is refused
# without Oklahoma); wall_knock_pick stays True.
_VARIANTS = (
    ("std_no_throwback", False, dict(DEFAULTS, allow_upcard_throwback=False)),
    ("okla_no_throwback", True, dict(DEFAULTS, allow_upcard_throwback=False)),
    ("okla_spade_doubles", True, dict(DEFAULTS, spade_doubles=True)),
    (
        "okla_no_throwback_spade_doubles",
        True,
        dict(DEFAULTS, allow_upcard_throwback=False, spade_doubles=True),
    ),
)


class SimpleBotRulesTest(parameterized.TestCase):
  """The simple bot plays the rule variants."""

  def test_default_rules_actions_match_digest(self):
    """The default-rules action stream equals its digest."""
    self.assertEqual(_default_rules_stream(), _DEFAULT_RULES_BOT_DIGEST)

  @parameterized.named_parameters(*_VARIANTS)
  def test_bot_plays_the_rules(self, oklahoma, rules):
    """The bot plays only legal moves under each rule variant.

    Args:
      oklahoma: Whether Oklahoma rules hold.
      rules: The rule variants.
    """
    seen = {"taken_nodes": 0, "knock11": 0, "doubled": 0, "nodes": 0}

    def check(state, mask, bot_move):
      seen["nodes"] += 1
      phase = int(state["phase"])
      taken = int(state["upcard_taken"])
      if not rules["allow_upcard_throwback"]:
        if phase in (PHASE_DISCARD, PHASE_KNOCK) and taken >= 0:
          seen["taken_nodes"] += 1
          self.assertNotEqual(bot_move, taken, "threw back the taken upcard")
          self.assertFalse(mask[taken])
      else:
        self.assertEqual(taken, -1)
      if phase in (PHASE_FIRST_UPCARD, PHASE_DRAW, PHASE_DISCARD, PHASE_WALL):
        self.assertTrue(
            mask[bot_move], f"illegal bot move {bot_move} at phase {phase}"
        )
      if phase in (PHASE_DISCARD, PHASE_WALL):
        self.assertEqual(bot_move == ACTION_KNOCK, bool(mask[ACTION_KNOCK]))
      if (
          phase == PHASE_KNOCK
          and int(
              np.sum(
                  np.asarray(gin_game.get_hand(state, state["current_player"]))
              )
          )
          == 11
      ):
        seen["knock11"] += 1
        self.assertLess(bot_move, NUM_CARDS)

    for seed in range(60):
      # Half the moves uniform, so upcards the bot would not take are taken
      # and the bot then plays the DISCARD / KNOCK node holding them.
      end = _play(seed, oklahoma, rules, check, bot_share=0.5)
      if int(end["score_multiplier"]) == 2:
        seen["doubled"] += 1
        self.assertTrue(rules["spade_doubles"])
        self.assertEqual(int(end["p0_score"]) % 2, 0)
    self.assertGreater(seen["nodes"], 1000)
    self.assertGreater(seen["knock11"], 0)
    if not rules["allow_upcard_throwback"]:
      self.assertGreater(seen["taken_nodes"], 0)
    if rules["spade_doubles"]:
      self.assertGreater(seen["doubled"], 0)

  def _witness(self, rules):
    """Player 0 holds gin (2s-4s, 5c-7c, 8d-Jd); the upcard is Kh.

    Args:
      rules: The rule variants.

    Returns:
      (the state, the Kh upcard).
    """
    game = gin_rummy.GinRummy(oklahoma=False, **rules)
    state = game.new_initial_state()
    p0 = [1, 2, 3, 17, 18, 19, 33, 34, 35, 36]
    kh = 3 * 13 + 12
    p1 = [c for c in range(NUM_CARDS) if c not in p0 and c != kh][:10]
    for a in (*p0, *p1, kh):
      state = _step(state, jnp.int32(a))
    self.assertEqual(int(state["phase"]), PHASE_FIRST_UPCARD)
    self.assertEqual(int(state["current_player"]), 0)
    return state, kh

  def test_no_throwback_knock_discard_is_not_the_taken_upcard(self):
    """Without throwback the knock discard is not the taken upcard."""
    state, kh = self._witness(dict(DEFAULTS, allow_upcard_throwback=False))
    # Kh plus a discard of Jd (or 8d) knocks for 10, so the bot takes it.
    self.assertEqual(int(_bot(state)), ACTION_DRAW_UPCARD)
    state = _step(state, jnp.int32(ACTION_DRAW_UPCARD))
    self.assertEqual(int(state["upcard_taken"]), kh)
    self.assertEqual(int(_bot(state)), ACTION_KNOCK)
    self.assertTrue(bool(np.asarray(_legal_mask(state))[ACTION_KNOCK]))
    state = _step(state, jnp.int32(ACTION_KNOCK))
    self.assertEqual(int(state["phase"]), PHASE_KNOCK)
    discard = int(_bot(state))
    # Least deadwood left (10: Kh) over the non-taken cards; ties to the
    # higher rank: Jd (card 36).
    self.assertEqual(discard, 36)
    self.assertTrue(bool(np.asarray(_legal_mask(state))[discard]))

  def test_default_rules_throw_the_upcard_back(self):
    """Under the default rules the bot throws the upcard back."""
    state, kh = self._witness(DEFAULTS)
    self.assertEqual(int(_bot(state)), ACTION_DRAW_UPCARD)
    state = _step(state, jnp.int32(ACTION_DRAW_UPCARD))
    self.assertEqual(int(state["upcard_taken"]), -1)
    self.assertEqual(int(_bot(state)), ACTION_KNOCK)
    state = _step(state, jnp.int32(ACTION_KNOCK))
    self.assertEqual(int(_bot(state)), kh)


if __name__ == "__main__":
  absltest.main()
