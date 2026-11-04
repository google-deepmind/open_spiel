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

"""The gin rummy rule variants and their defaults.

The JAX gin_rummy implements three optional rule variants
(`gin_game.DEFAULT_RULES`); the defaults are the rules of OpenSpiel's
`gin_rummy`, except that equal deadwood after layoffs is an undercut here
(released OpenSpiel, 2.0.2 and earlier, scores that tie 0 for both;
gin_rummy/game.py's module doc):

* `allow_upcard_throwback` (default True). When False, the upcard taken this
  turn (the first upcard, a draw-phase take, or the upcard a wall knock picks
  up) may not be discarded, nor be the discard after a knock. A knock is
  offered only if the minimum deadwood over the other discards is within the
  knock card. The repeated-move stop can then never be reached; the stop
  after 50 upcard draws still applies.
* `wall_knock_pick` (default True). When False, a non-knock discard by the
  player who drew the 50th card (two remain in the stock) ends the hand as a
  draw (0, 0) with no
  wall phase. A knock on that card stays legal. Standard gin only:
  `oklahoma=True` with `wall_knock_pick=False` is refused.
* `spade_doubles` (default False). Oklahoma only: a spade initial upcard
  doubles the hand score, bonuses included. Asking for it without `oklahoma`
  is refused.

What the tests pin:

1. Defaults: pinned rollout digests of standard and Oklahoma play, and the
   default value of every rule field of the state.
2. Scripted deals per variant (card = suit * 13 + rank; s=0, c=1, d=2, h=3;
   A=0 .. K=12), comparing the legal set at every node and the returns. When
   `REFERENCE_LOAD_GAME` is set to an OpenSpiel build that implements the
   variants, each line is also played through it and compared.
3. Seeded random playouts under the restrictive variants.
4. The observation planes: a non-default `allow_upcard_throwback` appends the
   52-float one-hot of `upcard_taken`, a non-default `spade_doubles` one float
   for `score_multiplier == 2`, in that order after the 644 default floats;
   the default tensor is unchanged. When the reference engine has the same
   planes (`REFERENCE_HAS_RULE_OBS`), every scripted line compares the full
   tensor of both seats with it.

Released OpenSpiel does not implement the variants, so by default this file
checks them against its own scripted lines and pinned digests only. The two
`REFERENCE_*` globals are the extension point for a stronger check: set them
to an OpenSpiel build that implements the variants and run these classes.
"""

import hashlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games import gin_rummy
from open_spiel.python.jax.games.gin_rummy import (
    dense_cache_game as gin_dense_cache_game,
)
from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_STOCK,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_DRAW_UPCARD,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    ACTION_MELD_BASE,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import ACTION_PASS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import get_meld_masks
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import NUM_CARDS
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_DISCARD
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import (
    PHASE_GAME_OVER,
)
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_KNOCK
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_LAYOFF
from open_spiel.python.jax.games.gin_rummy.cards_and_melds import PHASE_WALL

# The digest of the standard rollout stream at seeds (0, 7), the same value
# as oklahoma_test._STANDARD_ROLLOUT_GOLDEN.
_STANDARD_ROLLOUT_GOLDEN = (
    "a5dc06fab23d449642aaf69a40aaf99d57af89283f731cd609d2ebb8ef2e1704"
)

_MELD_SIZES = np.asarray(get_meld_masks()).sum(axis=1)

DEFAULTS = dict(
    allow_upcard_throwback=True, wall_knock_pick=True, spade_doubles=False
)
NO_THROWBACK = {**DEFAULTS, "allow_upcard_throwback": False}
NO_WALL_PICK = {**DEFAULTS, "wall_knock_pick": False}
SPADES = {**DEFAULTS, "spade_doubles": True}

# sha256 over the same stream as oklahoma_test's standard golden, for random
# rollouts at seeds (0, 7) with oklahoma=True. If it moves, a default changed.
_OKLAHOMA_ROLLOUT_GOLDEN = (
    "738f4d902ae08c6743a42a94b72f18af78b2a4ae528db7b1a93901f26f215c55"
)

# Optional reference engine: a callable(params: dict) -> pyspiel.Game for an
# OpenSpiel build that implements the rule variants, and whether its
# observer also carries the rule planes (644 + 52 floats under
# allow_upcard_throwback=False). None: no reference engine (module doc).
REFERENCE_LOAD_GAME = None
REFERENCE_HAS_RULE_OBS = False


def _rollout_digest(game, seeds=(0, 7)):
  """A digest of random rollouts: masks, observations, actions, returns.

  Args:
    game: The game.
    seeds: The rollout seeds.

  Returns:
    The hex digest.
  """
  h = hashlib.sha256()
  for seed in seeds:
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
  return h.hexdigest()


def _legal(state):
  """The legal actions of `state`, sorted.

  Args:
    state: The state.

  Returns:
    The sorted legal actions.
  """
  return sorted(np.flatnonzero(np.asarray(gin_game.legal_actions_mask(state))))


class _Line:
  """One hand played through the JAX game and (when set) the reference at once.

  Every ``apply`` first compares the two engines' legal sets (chance nodes
  included) and refuses an action either engine calls illegal, so a scripted
  line can only ever take legal moves.

  Attributes:
    game: The JAX game.
    ref: The pyspiel reference state, or None.
    state: The JAX state.
    test: The test case that asserts.
  """

  def __init__(self, test, oklahoma, rules):
    self.test = test
    self.game = gin_rummy.GinRummy(oklahoma=oklahoma, **rules)
    self.state = self.game.new_initial_state()
    self.ref = None
    if REFERENCE_LOAD_GAME is not None:
      params = dict(rules, oklahoma=oklahoma)
      self.ref = REFERENCE_LOAD_GAME(params).new_initial_state()

  def legal(self):
    """Returns the legal actions, checked against the reference.

    Returns:
      The sorted legal actions of the JAX state (chance outcomes at a chance
      node).
    """
    jax_legal = _legal(self.state)
    if self.ref is not None and not self.ref.is_terminal():
      if self.ref.is_chance_node():
        ref_legal = sorted(a for a, _ in self.ref.chance_outcomes())
      else:
        ref_legal = sorted(self.ref.legal_actions())
      self.test.assertEqual(jax_legal, ref_legal, "legal set vs reference")
    return jax_legal

  def compare_observations(self):
    """Both seats' full tensors vs the reference (when it has the planes)."""
    if self.ref is None or not REFERENCE_HAS_RULE_OBS:
      return
    for player in (0, 1):
      np.testing.assert_array_equal(
          np.asarray(
              self.game.observation_tensor(self.state, jnp.int32(player))
          ),
          np.asarray(self.ref.observation_tensor(player), dtype=np.float32),
          err_msg=f"observation tensor of player {player} vs reference",
      )

  def apply(self, *actions):
    """Applies `actions` in order, checking each against the reference.

    Args:
      *actions: The actions.

    Returns:
      self.
    """
    for a in actions:
      legal = self.legal()
      self.compare_observations()
      self.test.assertIn(a, legal, f"action {a} illegal; legal {legal}")
      self.state = gin_game.step(self.state, jnp.int32(a))
      if self.ref is not None:
        self.ref.apply_action(a)
    return self

  def deal(self, p0, p1, upcard):
    """Deals `p0`, `p1` and the upcard.

    Args:
      p0: Player 0's ten cards.
      p1: Player 1's ten cards.
      upcard: The upcard.

    Returns:
      self.
    """
    return self.apply(*p0, *p1, upcard)

  @property
  def phase(self):
    """The JAX state's phase."""
    return int(self.state["phase"])

  def done(self):
    """Returns whether the game is over, checked against the reference.

    Returns:
      True at a terminal state.
    """
    done = bool(self.state["done"])
    if self.ref is not None:
      self.test.assertEqual(
          done, self.ref.is_terminal(), "terminal vs reference"
      )
    return done

  def returns(self):
    """Returns the returns, checked against the reference.

    Returns:
      [player 0's, player 1's].
    """
    r = [float(x) for x in np.asarray(gin_game.get_returns(self.state))]
    if self.ref is not None:
      self.test.assertEqual(r, list(self.ref.returns()), "returns vs reference")
    return r

  def finish_endgame(self):
    """KNOCK (10 cards) / LAYOFF: lay the largest legal meld, else pass.

    Largest first (lowest id on a tie), so a four-card run is never split
    into a three-card meld plus deadwood. Never lays off a card, so every
    line's deadwood is fixed by the deal.

    Returns:
      self, at the end of the game.
    """
    for _ in range(64):
      if self.done():
        return self
      phase = self.phase
      legal = self.legal()
      if phase not in (PHASE_KNOCK, PHASE_LAYOFF):
        self.test.fail(f"finish_endgame at phase {phase}")
      melds = sorted(
          (a for a in legal if a >= ACTION_MELD_BASE),
          key=lambda a: (-int(_MELD_SIZES[a - ACTION_MELD_BASE]), a),
      )
      if phase == PHASE_LAYOFF and not bool(self.state["finished_layoffs"]):
        self.apply(ACTION_PASS)
      elif melds:
        self.apply(melds[0])
      else:
        self.apply(ACTION_PASS)
    self.test.fail("finish_endgame did not end the hand")


def _stock_line(line, p0, p1, upcard, draws=29):
  """Plays a line to the ``draws``-th stock draw.

  Both pass the first upcard, then each draw is from stock and its card is
  discarded, until ``draws`` stock draws are done (29 leaves the stock at 2,
  the wall). The stock is drawn in card order. Player 0 makes the even draws.

  Args:
    line: The _Line.
    p0: Player 0's ten cards.
    p1: Player 1's ten cards.
    upcard: The upcard.
    draws: The number of stock draws.

  Returns:
    (line, card): the line after the last draw, before its discard, and the
    card drawn.

  Raises:
    AssertionError: never; the loop always returns.
  """
  dealt = set(p0) | set(p1) | {upcard}
  stock = [c for c in range(NUM_CARDS) if c not in dealt]
  line.deal(p0, p1, upcard).apply(ACTION_PASS, ACTION_PASS)
  for i in range(draws):
    card = stock[i]
    line.apply(ACTION_DRAW_STOCK, card)
    if i == draws - 1:
      return line, card
    line.apply(card)
  raise AssertionError("unreachable")


# Witness hands (see the module docstring for the card encoding).
# H6: 2s-5s, 6c-8c melded; Ah 2d 3h loose (deadwood 6).
H6 = [1, 2, 3, 4, 18, 19, 20, 39, 27, 41]
# H1: 2s-4s, 5c-7c, 8d-Td melded; Ah loose (deadwood 1).
H1 = [1, 2, 3, 17, 18, 19, 33, 34, 35, 39]
KD = 38  # King of diamonds: matches nothing in H6 or H1.
FILLER_A = [8, 9, 10, 11, 21, 22, 30, 31, 32, 33]  # with H6
FILLER_B = [8, 9, 10, 11, 21, 22, 23, 24, 45, 46]  # with H1


class DefaultsTest(absltest.TestCase):
  """Default rules play OpenSpiel's gin; inconsistent rules are refused."""

  def test_standard_golden_unchanged(self):
    """The standard game's rollout digest equals its golden."""
    self.assertEqual(
        _rollout_digest(gin_rummy.GinRummy()), _STANDARD_ROLLOUT_GOLDEN
    )

  def test_explicit_default_rules_play_the_standard_golden(self):
    """Passing the default rules explicitly plays the standard golden."""
    self.assertEqual(
        _rollout_digest(gin_rummy.GinRummy(**DEFAULTS)),
        _STANDARD_ROLLOUT_GOLDEN,
    )

  def test_oklahoma_golden_unchanged(self):
    """The Oklahoma game's rollout digest equals its golden."""
    self.assertEqual(
        _rollout_digest(gin_rummy.GinRummy(oklahoma=True)),
        _OKLAHOMA_ROLLOUT_GOLDEN,
    )
    self.assertEqual(
        _rollout_digest(gin_rummy.GinRummy(oklahoma=True, **DEFAULTS)),
        _OKLAHOMA_ROLLOUT_GOLDEN,
    )

  def test_defaults_are_openspiel_rules(self):
    """The default rules are OpenSpiel's, in the game and in every state."""
    game = gin_rummy.GinRummy()
    self.assertEqual(game.rules, DEFAULTS)
    self.assertEqual(gin_game.DEFAULT_RULES, DEFAULTS)
    for state in (
        gin_game.init_state(),
        gin_game.init_dealt_state(jax.random.PRNGKey(3))[0],
        gin_game.init_dealt_state(jax.random.PRNGKey(3), True)[0],
    ):
      self.assertTrue(bool(state["allow_upcard_throwback"]))
      self.assertTrue(bool(state["wall_knock_pick"]))
      self.assertFalse(bool(state["spade_doubles"]))
      self.assertEqual(int(state["score_multiplier"]), 1)
      self.assertEqual(int(state["upcard_taken"]), -1)
      gin_game.assert_state_dtypes(state)

  def test_spade_doubles_requires_oklahoma(self):
    """spade_doubles without Oklahoma is refused at every entry point."""
    with self.assertRaisesRegex(ValueError, "spade_doubles"):
      gin_rummy.GinRummy(spade_doubles=True)
    with self.assertRaisesRegex(ValueError, "spade_doubles"):
      gin_game.init_state(oklahoma=False, spade_doubles=True)
    with self.assertRaisesRegex(ValueError, "spade_doubles"):
      gin_game.init_dealt_state(
          jax.random.PRNGKey(0), False, False, True, True, True
      )
    gin_rummy.GinRummy(oklahoma=True, spade_doubles=True)  # accepted

  def test_no_wall_pick_requires_standard(self):
    """wall_knock_pick=False with Oklahoma is refused at every entry point."""
    # Oklahoma gin always plays the wall; the JAX game refuses the combination
    # on every entry point.
    with self.assertRaisesRegex(ValueError, "wall_knock_pick"):
      gin_rummy.GinRummy(oklahoma=True, wall_knock_pick=False)
    with self.assertRaisesRegex(ValueError, "wall_knock_pick"):
      gin_rummy.GinRummy(oklahoma=True, **{**SPADES, "wall_knock_pick": False})
    with self.assertRaisesRegex(ValueError, "wall_knock_pick"):
      gin_game.init_state(oklahoma=True, wall_knock_pick=False)
    with self.assertRaisesRegex(ValueError, "wall_knock_pick"):
      gin_game.init_dealt_state(
          jax.random.PRNGKey(0), True, False, True, False, False
      )
    with self.assertRaisesRegex(ValueError, "wall_knock_pick"):
      gin_dense_cache_game.init_dealt_state(
          jax.random.PRNGKey(0), True, False, True, False, False
      )
    gin_rummy.GinRummy(oklahoma=False, wall_knock_pick=False)  # accepted

  def test_traced_oklahoma_keeps_the_wall_pick(self):
    """A traced Oklahoma keeps the wall pick whatever wall_knock_pick says."""
    # A traced (vmapped) oklahoma cannot be refused in Python; the step then
    # keeps the wall pick wherever oklahoma holds, so the refused
    # combination plays as Oklahoma with the wall (the default), exactly like
    # traced spade_doubles without oklahoma plays as standard gin.
    p0 = [KD, 8, 9, 10, 11, 21, 22, 30, 31, 32]
    _, drawn = _stock_line(_Line(self, False, DEFAULTS), p0, H6, 45)
    dealt = set(p0) | set(H6) | {45}
    stock = [c for c in range(NUM_CARDS) if c not in dealt]
    script = [*p0, *H6, 45, ACTION_PASS, ACTION_PASS]
    for i in range(29):
      script += [ACTION_DRAW_STOCK, stock[i], stock[i]]
    self.assertEqual(script[-1], drawn)

    def play(oklahoma, wall):
      """Plays the script under traced rules.

      Args:
        oklahoma: Whether Oklahoma rules hold.
        wall: wall_knock_pick.

      Returns:
        (phase, done) at the end of the script.
      """
      state = gin_game.init_state(oklahoma, True, wall, False)
      for a in script:
        state = gin_game.step(state, jnp.int32(a))
      return state["phase"], state["done"]

    oklahoma = jnp.array([False, True, False, True])
    wall = jnp.array([False, False, True, True])
    phase, done = jax.vmap(play)(oklahoma, wall)
    np.testing.assert_array_equal(
        np.asarray(phase),
        [PHASE_GAME_OVER, PHASE_WALL, PHASE_WALL, PHASE_WALL],
    )
    np.testing.assert_array_equal(np.asarray(done), [True, False, False, False])

  def test_unknown_rule_value_type_refused(self):
    """A rule value of the wrong type raises TypeError."""
    with self.assertRaises(TypeError):
      gin_rummy.GinRummy(allow_upcard_throwback="no")

  def test_mixed_batch_under_vmap(self):
    """A vmapped batch may mix rule settings."""
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    throwback = jnp.array([True, False, True, False])
    states, _ = jax.vmap(
        lambda k, t: gin_game.init_dealt_state(k, True, False, t, True, False)
    )(keys, throwback)
    np.testing.assert_array_equal(
        np.asarray(states["allow_upcard_throwback"]), np.asarray(throwback)
    )


class UpcardThrowbackTest(parameterized.TestCase):
  """allow_upcard_throwback."""

  @parameterized.product(oklahoma=(False, True), throwback=(True, False))
  def test_first_upcard_take(self, oklahoma, throwback):
    """Taking the first upcard: throwing it back is legal only with throwback.

    Args:
      oklahoma: Whether Oklahoma rules hold.
      throwback: allow_upcard_throwback.
    """
    rules = dict(DEFAULTS, allow_upcard_throwback=throwback)
    line = _Line(self, oklahoma, rules).deal(H6, FILLER_A, KD)
    line.apply(ACTION_DRAW_UPCARD)
    self.assertEqual(line.phase, PHASE_DISCARD)
    legal = line.legal()
    # H6 + Kd: only throwing the Kd back keeps deadwood within 10.
    self.assertEqual(KD in legal, throwback)
    self.assertEqual(ACTION_KNOCK in legal, throwback)
    self.assertEqual(int(line.state["upcard_taken"]), -1 if throwback else KD)
    for c in H6:
      self.assertIn(c, legal)
    if line.ref is not None:
      self.assertEqual(line.ref.upcard_taken(), None if throwback else KD)

  @parameterized.product(oklahoma=(False, True), throwback=(True, False))
  def test_first_upcard_pass_then_take(self, oklahoma, throwback):
    """The second player takes the passed first upcard under the same rule.

    Args:
      oklahoma: Whether Oklahoma rules hold.
      throwback: allow_upcard_throwback.
    """
    # The other draw_first path: the first player passes the first upcard
    # and the second player (H6) takes it.
    rules = dict(DEFAULTS, allow_upcard_throwback=throwback)
    line = _Line(self, oklahoma, rules).deal(FILLER_A, H6, KD)
    line.apply(ACTION_PASS)
    self.assertEqual(int(line.state["current_player"]), 1)
    self.assertIn(ACTION_DRAW_UPCARD, line.legal())
    line.apply(ACTION_DRAW_UPCARD)
    self.assertEqual(line.phase, PHASE_DISCARD)
    self.assertEqual(int(line.state["current_player"]), 1)
    legal = line.legal()
    self.assertEqual(KD in legal, throwback)
    self.assertEqual(ACTION_KNOCK in legal, throwback)
    self.assertEqual(int(line.state["upcard_taken"]), -1 if throwback else KD)
    for c in H6:
      self.assertIn(c, legal)
    if line.ref is not None:
      self.assertEqual(line.ref.upcard_taken(), None if throwback else KD)
    line.apply(H6[-1])
    self.assertEqual(int(line.state["upcard_taken"]), -1)

  @parameterized.parameters(False, True)
  def test_draw_phase_take_and_stock_draw(self, oklahoma):
    """A taken upcard may not be thrown back; a stock card may.

    Args:
      oklahoma: Whether Oklahoma rules hold.
    """
    # P0 passes, P1 passes, P0 draws the 2c (13) from stock: a stock card is
    # always a legal discard. P0 throws it; P1 takes it and may not throw it.
    line = _Line(self, oklahoma, NO_THROWBACK).deal(H6, FILLER_A, KD)
    line.apply(ACTION_PASS, ACTION_PASS, ACTION_DRAW_STOCK, 13)
    self.assertIn(13, line.legal())
    self.assertEqual(int(line.state["upcard_taken"]), -1)
    line.apply(13, ACTION_DRAW_UPCARD)
    self.assertEqual(line.phase, PHASE_DISCARD)
    self.assertNotIn(13, line.legal())
    self.assertEqual(int(line.state["upcard_taken"]), 13)
    line.apply(FILLER_A[0])
    self.assertEqual(int(line.state["upcard_taken"]), -1)

  @parameterized.parameters(False, True)
  def test_knock_discard_excludes_taken_card(self, oklahoma):
    """Without throwback, the knock discard excludes the taken upcard.

    Args:
      oklahoma: Whether Oklahoma rules hold.
    """
    for rules, want in ((DEFAULTS, [KD, 39]), (NO_THROWBACK, [39])):
      line = _Line(self, oklahoma, rules).deal(H1, FILLER_B, KD)
      line.apply(ACTION_DRAW_UPCARD)
      self.assertIn(ACTION_KNOCK, line.legal())
      line.apply(ACTION_KNOCK)
      self.assertEqual(line.phase, PHASE_KNOCK)
      self.assertEqual(line.legal(), want, rules)

  @parameterized.product(oklahoma=(False, True), throwback=(True, False))
  def test_wall_knock(self, oklahoma, throwback):
    """At the wall, a taken upcard is kept by the knock.

    Args:
      oklahoma: Whether Oklahoma rules hold.
      throwback: allow_upcard_throwback.
    """
    rules = dict(DEFAULTS, allow_upcard_throwback=throwback)
    # P1 holds H6; P0 holds the Kd and throws it on the 29th stock draw, so
    # P1 is at the wall with the Kd up. Taking it to knock keeps it.
    p0 = [KD, 8, 9, 10, 11, 21, 22, 30, 31, 32]
    line, _ = _stock_line(_Line(self, oklahoma, rules), p0, H6, 45)
    line.apply(KD)
    self.assertEqual(line.phase, PHASE_WALL)
    self.assertEqual(ACTION_KNOCK in line.legal(), throwback)
    # P1 holds H1 instead: the wall knock is legal either way, and the knock
    # discard may not be the Kd it picked up. The Th upcard keeps the
    # Oklahoma knock card at 10.
    p0 = [KD, 8, 9, 10, 11, 21, 22, 23, 24, 45]
    line, _ = _stock_line(_Line(self, oklahoma, rules), p0, H1, 48)
    line.apply(KD)
    self.assertEqual(line.phase, PHASE_WALL)
    self.assertIn(ACTION_KNOCK, line.legal())
    line.apply(ACTION_KNOCK)
    self.assertEqual(line.legal(), [KD, 39] if throwback else [39])
    self.assertEqual(int(line.state["upcard_taken"]), -1 if throwback else KD)
    line.apply(39)
    self.assertEqual(int(line.state["upcard_taken"]), -1)
    line.finish_endgame()
    line.returns()


class RandomPlayoutTest(parameterized.TestCase):
  """Random playouts under the rule variants keep their invariants."""

  def _playouts(self, game, seeds, visit):
    """Plays random games, calling `visit` at every node.

    Args:
      game: The game.
      seeds: One game per seed.
      visit: Called with (state, legal actions) at decisions and (state, None)
        after each action.

    Yields:
      Each game's final state.
    """
    for seed in seeds:
      rng = np.random.default_rng(seed)
      state = game.new_initial_state()
      while not bool(state["done"]):
        if bool(game.is_chance_node(state)):
          p = np.asarray(game.chance_outcomes(state))
          action = int(rng.choice(p.size, p=p))
        else:
          legal = np.flatnonzero(np.asarray(game.legal_actions_mask(state)))
          action = int(rng.choice(legal))
          visit(state, legal)
        state = game.apply_action(state, jnp.int32(action))
        visit(state, None)
      yield state

  @parameterized.parameters(False, True)
  def test_no_throwback_never_reaches_the_repeated_move_stop(self, oklahoma):
    """Without throwback no playout reaches the repeated-move stop.

    Args:
      oklahoma: Whether Oklahoma rules hold.
    """
    game = gin_rummy.GinRummy(oklahoma=oklahoma, **NO_THROWBACK)

    def visit(state, legal):
      self.assertFalse(
          bool(state["repeated_move"]),
          "repeated-move stop reached without throwback",
      )
      taken = int(state["upcard_taken"])
      if legal is not None and int(state["phase"]) == PHASE_DISCARD:
        self.assertNotIn(taken, legal)

    for _ in self._playouts(game, range(12), visit):
      pass

  def test_no_wall_pick_never_reaches_the_wall(self):
    """Without the wall pick no playout reaches the wall phase."""
    # Standard gin only (Oklahoma refuses wall_knock_pick=False).
    game = gin_rummy.GinRummy(oklahoma=False, **NO_WALL_PICK)

    def visit(state, legal):
      del legal
      self.assertNotEqual(int(state["phase"]), PHASE_WALL)

    for state in self._playouts(game, range(12), visit):
      if not bool(jnp.any(state["knocked"])):
        np.testing.assert_array_equal(np.asarray(game.returns(state)), [0, 0])

  def test_spade_doubles_random_returns(self):
    """Spade doubles doubles the returns exactly when the upcard is a spade."""
    plain = gin_rummy.GinRummy(oklahoma=True)
    doubled = gin_rummy.GinRummy(oklahoma=True, **SPADES)
    seen_spade = seen_other = 0
    for seed in range(16):
      rng = np.random.default_rng(seed)
      a = plain.new_initial_state()
      b = doubled.new_initial_state()
      while not bool(a["done"]):
        if bool(plain.is_chance_node(a)):
          p = np.asarray(plain.chance_outcomes(a))
          action = int(rng.choice(p.size, p=p))
        else:
          la = np.asarray(plain.legal_actions_mask(a))
          np.testing.assert_array_equal(la, plain.legal_actions_mask(b))
          action = int(rng.choice(np.flatnonzero(la)))
        a = plain.apply_action(a, jnp.int32(action))
        b = doubled.apply_action(b, jnp.int32(action))
      self.assertTrue(bool(b["done"]))
      ra, rb = np.asarray(plain.returns(a)), np.asarray(doubled.returns(b))
      spade_up = int(b["score_multiplier"]) == 2
      np.testing.assert_array_equal(rb, ra * (2 if spade_up else 1))
      seen_spade += spade_up
      seen_other += not spade_up
    self.assertGreater(seen_spade, 0)
    self.assertGreater(seen_other, 0)


# The four observation layouts: (allow_upcard_throwback, spade_doubles) under
# Oklahoma (spade_doubles needs it), named for the planes they add.
_OBS_COMBOS = (
    ("defaults", True, False, 644),
    ("upcard_taken", False, False, 644 + 52),
    ("score_multiplier", True, True, 644 + 1),
    ("both", False, True, 644 + 52 + 1),
)


class ObservationPlanesTest(parameterized.TestCase):
  """The rule planes appended to the observation tensor."""

  @parameterized.named_parameters(*_OBS_COMBOS)
  def test_size(self, throwback, spades, size):
    """The observation size follows the rule variants.

    Args:
      throwback: allow_upcard_throwback.
      spades: spade_doubles.
      size: The expected size.
    """
    # Every accepted (oklahoma, wall_knock_pick): Oklahoma keeps the wall
    # pick and spade_doubles needs Oklahoma, so wall_knock_pick=False
    # is checked under standard gin, where spade_doubles is absent.
    settings = [(True, True)]
    if not spades:
      settings += [(False, True), (False, False)]
    for oklahoma, wall in settings:
      rules = dict(
          allow_upcard_throwback=throwback,
          wall_knock_pick=wall,
          spade_doubles=spades,
      )
      game = gin_rummy.GinRummy(oklahoma=oklahoma, **rules)
      self.assertEqual(game.obs_shape, (size,))
      self.assertEqual(
          gin_game.observation_tensor_size(throwback, spades), size
      )
      obs = game.observation_tensor(game.new_initial_state(), jnp.int32(0))
      self.assertEqual(obs.shape, (size,))
      if REFERENCE_LOAD_GAME is not None and REFERENCE_HAS_RULE_OBS:
        # A reference build sets the module constant (module doc); None here.
        # pylint: disable-next=not-callable
        ref = REFERENCE_LOAD_GAME(dict(rules, oklahoma=oklahoma))
        self.assertEqual(ref.observation_tensor_size(), size)

  def test_defaults_keep_644(self):
    """The default rules keep OpenSpiel's 644-float observation."""
    self.assertEqual(gin_rummy.GinRummy().obs_shape, (644,))
    self.assertEqual(gin_rummy.GinRummy(**NO_WALL_PICK).obs_shape, (644,))
    self.assertEqual(gin_game.observation_tensor_size(), 644)

  @parameterized.product(
      oklahoma=(False, True), first_player_takes=(True, False)
  )
  def test_upcard_taken_plane(self, oklahoma, first_player_takes):
    """The upcard-taken plane one-hots the held-back upcard for both seats.

    Args:
      oklahoma: Whether Oklahoma rules hold.
      first_player_takes: Whether the first player takes the upcard.
    """
    # The one-hot of the held-back upcard, public to both seats; the 644
    # default floats equal the default game's at the same node.
    p0, p1 = (H6, FILLER_A) if first_player_takes else (FILLER_A, H6)
    pre = () if first_player_takes else (ACTION_PASS,)
    lines = {}
    for name, rules in (("default", DEFAULTS), ("no_throwback", NO_THROWBACK)):
      line = _Line(self, oklahoma, rules).deal(p0, p1, KD)
      line.apply(*pre)
      for player in (0, 1):
        obs = np.asarray(line.game.observation_tensor(line.state, player))
        if name == "no_throwback":
          np.testing.assert_array_equal(obs[644:], np.zeros(52))
      line.apply(ACTION_DRAW_UPCARD)
      lines[name] = line
    taker = int(lines["no_throwback"].state["current_player"])
    for player in (0, 1):
      default = np.asarray(
          lines["default"].game.observation_tensor(
              lines["default"].state, jnp.int32(player)
          )
      )
      no_throwback = np.asarray(
          lines["no_throwback"].game.observation_tensor(
              lines["no_throwback"].state, jnp.int32(player)
          )
      )
      self.assertEqual(default.shape, (644,))
      self.assertEqual(no_throwback.shape, (644 + 52,))
      np.testing.assert_array_equal(no_throwback[:644], default)
      want = np.zeros(52, dtype=np.float32)
      want[KD] = 1.0
      np.testing.assert_array_equal(
          no_throwback[644:], want, err_msg=str(player)
      )
    lines["no_throwback"].compare_observations()
    # The discard clears it.
    lines["no_throwback"].apply(H6[-1])
    self.assertNotEqual(
        int(lines["no_throwback"].state["current_player"]), taker
    )
    obs = np.asarray(
        lines["no_throwback"].game.observation_tensor(
            lines["no_throwback"].state, jnp.int32(0)
        )
    )
    np.testing.assert_array_equal(obs[644:], np.zeros(52))

  @parameterized.named_parameters(
      ("spade", 12, 1.0), ("heart", 48, 0.0), ("spade_ace", 0, 1.0)
  )
  def test_score_multiplier_plane(self, upcard, want):
    """The score-multiplier plane shows the hand's multiplier.

    Args:
      upcard: The upcard.
      want: The expected multiplier.
    """
    p0 = [1, 2, 14, 15, 16, 30, 31, 32, 33, 39]
    p1 = [13, 21, 23, 26, 28, 41, 43, 45, 49, 51]
    for throwback in (True, False):
      rules = dict(SPADES, allow_upcard_throwback=throwback)
      line = _Line(self, True, rules).deal(p0, p1, upcard)
      for player in (0, 1):
        obs = np.asarray(line.game.observation_tensor(line.state, player))
        self.assertEqual(obs.shape[0], 644 + (0 if throwback else 52) + 1)
        self.assertEqual(float(obs[-1]), want)
        plain = gin_rummy.GinRummy(
            oklahoma=True, **{**DEFAULTS, "allow_upcard_throwback": throwback}
        )
        np.testing.assert_array_equal(
            obs[:-1], np.asarray(plain.observation_tensor(line.state, player))
        )
      line.compare_observations()


class WallKnockPickTest(parameterized.TestCase):
  """wall_knock_pick."""

  def test_no_wall_pick_ends_drawn(self):
    """Without the wall pick the hand ends drawn at the wall."""
    # Standard gin only (Oklahoma refuses wall_knock_pick=False; see
    # DefaultsTest.test_no_wall_pick_requires_standard).
    p0 = [KD, 8, 9, 10, 11, 21, 22, 30, 31, 32]
    line, drawn = _stock_line(_Line(self, False, NO_WALL_PICK), p0, H6, 45)
    line.apply(drawn)
    self.assertTrue(line.done())
    self.assertEqual(line.phase, PHASE_GAME_OVER)
    self.assertEqual(line.returns(), [0.0, 0.0])

  @parameterized.parameters(False, True)
  def test_default_keeps_the_wall(self, oklahoma):
    """By default the hand reaches the wall phase.

    Args:
      oklahoma: Whether Oklahoma rules hold.
    """
    p0 = [KD, 8, 9, 10, 11, 21, 22, 30, 31, 32]
    line, drawn = _stock_line(_Line(self, oklahoma, DEFAULTS), p0, H6, 45)
    line.apply(drawn)
    self.assertFalse(line.done())
    self.assertEqual(line.phase, PHASE_WALL)

  @parameterized.parameters(DEFAULTS, NO_WALL_PICK)
  def test_knock_on_the_50th_card_stays_legal(self, **rules):
    """Knocking after the 29th stock draw stays legal.

    Args:
      **rules: The rule variants.
    """
    # P0 holds H1 (deadwood 1): after the 29th stock draw it may knock.
    line, _ = _stock_line(_Line(self, False, rules), H1, FILLER_B, 47)
    self.assertIn(ACTION_KNOCK, line.legal())
    line.apply(ACTION_KNOCK)
    self.assertEqual(line.phase, PHASE_KNOCK)


class SpadeDoublesTest(parameterized.TestCase):
  """spade_doubles."""

  def _play(self, rules, p0, p1, upcard, knock_discard):
    """Plays a hand where player 0 knocks off the upcard.

    Args:
      rules: The rule variants.
      p0: Player 0's cards.
      p1: Player 1's cards.
      upcard: The upcard.
      knock_discard: The knock discard.

    Returns:
      (the line, its returns).
    """
    line = _Line(self, True, rules).deal(p0, p1, upcard)
    line.apply(ACTION_DRAW_UPCARD, ACTION_KNOCK, knock_discard)
    line.finish_endgame()
    return line, line.returns()

  # (name, P0, P1, upcard, P0's knock discard, multiplier with spade_doubles)
  LINES = (
      # Js Qs + Ks taken, 2c-4c, 5d-8d: gin after throwing the Ah.
      (
          "gin_spade_king",
          [10, 11, 14, 15, 16, 30, 31, 32, 33, 39],
          [0, 21, 23, 26, 28, 41, 43, 45, 49, 51],
          12,
          39,
          2,
      ),
      # As taken (knock card 0, gin only): As-3s, 2c-4c, 5d-8d.
      (
          "gin_spade_ace",
          [1, 2, 14, 15, 16, 30, 31, 32, 33, 39],
          [13, 21, 23, 26, 28, 41, 43, 45, 49, 51],
          0,
          39,
          2,
      ),
      # H1 takes the Ts and throws it back: a knock win at deadwood 1.
      (
          "knock_win_spade",
          H1,
          [13, 21, 23, 26, 28, 41, 43, 45, 49, 51],
          9,
          9,
          2,
      ),
      # Same knock; the defender holds Tc-Qc, 3d-5d, 6h-8h and the As
      # (deadwood 1, no set to split the runs): an undercut tie.
      ("undercut_spade", H1, [0, 22, 23, 24, 28, 29, 30, 44, 45, 46], 9, 9, 2),
      # A non-spade upcard (Th) never doubles.
      (
          "knock_win_heart",
          H1,
          [13, 21, 23, 26, 28, 41, 43, 45, 49, 51],
          48,
          48,
          1,
      ),
  )

  @parameterized.named_parameters(*LINES)
  def test_multiplier(self, p0, p1, upcard, knock_discard, mult):
    """Spade doubles multiplies the score by the hand's multiplier.

    Args:
      p0: Player 0's cards.
      p1: Player 1's cards.
      upcard: The upcard.
      knock_discard: The knocker's discard.
      mult: The expected multiplier.
    """
    _, plain = self._play(DEFAULTS, p0, p1, upcard, knock_discard)
    line, doubled = self._play(SPADES, p0, p1, upcard, knock_discard)
    self.assertNotEqual(plain, [0.0, 0.0])
    self.assertEqual(doubled, [mult * x for x in plain])
    self.assertEqual(int(line.state["score_multiplier"]), mult)
    if line.ref is not None:
      self.assertEqual(line.ref.score_multiplier(), mult)

  def test_undercut_and_gin_signs(self):
    """Gin and undercut bonuses are doubled with the right signs."""
    _, gin = self._play(SPADES, *self.LINES[0][1:5])
    self.assertGreater(gin[0], 2 * 25)  # gin bonus doubled with the deadwood
    _, under = self._play(SPADES, *self.LINES[3][1:5])
    self.assertEqual(under, [-50.0, 50.0])  # undercut tie: 25 bonus, doubled

  def test_default_never_doubles(self):
    """Without spade doubles the multiplier stays 1."""
    state = gin_game.init_state(oklahoma=True)
    for a in self.LINES[0][1] + self.LINES[0][2] + [self.LINES[0][3]]:
      state = gin_game.step(state, jnp.int32(a))
    self.assertEqual(int(state["score_multiplier"]), 1)

  def test_fast_deal_sets_multiplier(self):
    """init_dealt_state sets the multiplier from the upcard's suit."""
    for seed in range(40):
      state, _ = gin_game.init_dealt_state(
          jax.random.PRNGKey(seed), True, False, True, True, True
      )
      spade = int(state["upcard"]) // 13 == 0
      self.assertEqual(int(state["score_multiplier"]), 2 if spade else 1)


if __name__ == "__main__":
  absltest.main()
