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

"""Gin positions with few unseen cards, for belief and search known answers.

`play` deals a chosen hand (seat 0 = the searcher "me", first to act) and
plays a scripted line through the raw env layer (every action is checked
legal, every stock draw is the lowest-index card left unless `stock_pick`
says otherwise):

  me   FIRST_UPCARD: pass; DRAW: draw from the stock; DISCARD: discard the
       card just drawn (so my hand stays the dealt ten).
  opp  FIRST_UPCARD: pass; DRAW: take the upcard (my discard), except for
       the `late` stock draws, which start once the stock is at most
       `late_at`; DISCARD after taking: shed one of its dealt (hidden) cards
       while more than `keep` remain, else a publicly known card (not the one
       just taken); DISCARD after a late stock draw: per `late[i]`, "known"
       (a known card), "orig" (a dealt hidden card) or "drawn" (the card it
       just drew).

The line stops when `stop(state)` holds at a decision of seat 0. Test
infrastructure only.
"""

from typing import Any, Callable, Optional, Sequence

import jax
import numpy as np

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.envs import gin

ME, OPP = 0, 1


class _Fns:

  def __init__(self, env):
    self.apply = jax.jit(env.apply)
    self.apply_chance = jax.jit(env.apply_chance)
    self.legal = jax.jit(env.legal_mask)
    self.is_chance = jax.jit(env.is_chance)
    self.terminal = jax.jit(env.terminal)


_FNS = {}


def _fns(env):
  """The jitted helpers of `env`, built once per env.

  Args:
    env: The env.

  Returns:
    The env's _Fns.
  """
  key = id(env)
  if key not in _FNS:
    _FNS[key] = (env, _Fns(env))
  return _FNS[key][1]


def at_my_turn(phase: int, stock: int) -> Callable[[Any], bool]:
  """A stop predicate for `play`: my turn in `phase` with `stock` in the stock.

  Args:
    phase: the phase to stop in.
    stock: the stock size to stop at.

  Returns:
    The predicate on a state.
  """

  def stop(s):
    """Whether `s` is my turn in the phase with the stock size.

    Returns:
      bool.

    Args:
      s: the state.
    """
    g = s.game
    return (
        int(g["current_player"]) == ME
        and int(g["phase"]) == phase
        and int(np.asarray(g["deck"]).sum()) == stock
    )

  return stop


def play(
    env: gin.GinEnv,
    hand0: Sequence[int],
    hand1: Sequence[int],
    upcard: int,
    stop: Callable[[Any], bool],
    *,
    keep: int = 0,
    late: Sequence[str] = (),
    late_at: int = 7,
    stock_pick: Optional[Callable[..., Any]] = None,
    max_plies: int = 400,
):
  """Plays a scripted line from a fixed deal until `stop`.

  Args:
    env: the Env.
    hand0: my dealt hand (seat 0).
    hand1: the opponent's dealt hand.
    upcard: the first upcard.
    stop: predicate on a state; checked at each of my decisions.
    keep: how many dealt cards the opponent keeps before shedding known ones.
    late: the opponent's discard mode after each late stock draw ("known",
      "orig" or "drawn").
    late_at: the stock size at or below which the opponent draws late.
    stock_pick: picks the stock card from the deck's card ids; None takes the
      lowest.
    max_plies: the ply budget.

  Returns:
    (state at the stop, raw history of actions and chance outcomes).

  Raises:
    AssertionError: the line leaves the script or never stops.
  """
  assert len(hand0) == len(hand1) == cm.HAND_SIZE
  assert len(set(hand0) | set(hand1) | {upcard}) == 2 * cm.HAND_SIZE + 1
  f = _fns(env)
  outcomes = [ME] + list(hand0) + list(hand1) + [upcard]
  s = env.deal(np.asarray(outcomes, np.int32))
  history = list(outcomes)
  orig = set(hand1)
  late_left = list(late)
  opp_mode = None  # how the opponent discards this turn
  drawn = {ME: None, OPP: None}
  for _ in range(max_plies):
    assert not bool(f.terminal(s)), "the scripted line ended the hand"
    g = s.game
    if bool(f.is_chance(s)):
      deck = np.flatnonzero(np.asarray(g["deck"]))
      c = int(deck[0]) if stock_pick is None else int(stock_pick(deck))
      drawer = int(g["prev_player"])
      assert drawer in (ME, OPP), drawer
      drawn[drawer] = c
      s = f.apply_chance(s, np.int32(c))
      history.append(c)
      continue
    p = int(g["current_player"])
    phase = int(g["phase"])
    if p == ME and stop(s):
      return s, history
    legal = np.asarray(f.legal(s))
    stock = int(np.asarray(g["deck"]).sum())
    hand = set(
        np.flatnonzero(
            np.asarray(g["player0_hand" if p == 0 else "player1_hand"])
        )
    )
    known = set(np.flatnonzero(np.asarray(g["known_cards"][p])))
    if phase == cm.PHASE_FIRST_UPCARD:
      a = cm.ACTION_PASS if legal[cm.ACTION_PASS] else cm.ACTION_DRAW_STOCK
    elif phase == cm.PHASE_DRAW:
      if p == ME:
        a = cm.ACTION_DRAW_STOCK
      elif late_left and stock <= late_at:
        a = cm.ACTION_DRAW_STOCK
        opp_mode = late_left.pop(0)
      else:
        a = cm.ACTION_DRAW_UPCARD
        opp_mode = "take"
    elif phase == cm.PHASE_DISCARD:
      if p == ME:
        assert drawn[ME] is not None
        a = drawn[ME]
      else:
        hidden_orig = sorted((hand & orig) - known)
        taken = int(g["drawn_from_discard"])
        known_ok = sorted(known - {taken})
        if opp_mode == "take":
          a = hidden_orig[0] if len(hidden_orig) > keep else known_ok[0]
        elif opp_mode == "known":
          a = known_ok[0]
        elif opp_mode == "orig":
          a = hidden_orig[0]
        elif opp_mode == "drawn":
          a = drawn[OPP]
        else:
          raise AssertionError(opp_mode)
      drawn[p] = None if p == ME else drawn[p]
    else:
      raise AssertionError(f"the scripted line reached phase {phase}")
    assert legal[a], (p, phase, a)
    s = f.apply(s, np.int32(a))
    history.append(int(a))
  raise AssertionError("no stop within max_plies")


# The hands of the known-answer positions (card ids; suits s c d h).
HEARTS_RUN = list(range(39, 49))  # Ah..Th: a 10-card gin (Ah-5h + 6h-Th)
# Ac 3c 5c 7c 9c Ad 3d 5d 7d 9d
OPP_HAND = [13, 15, 17, 19, 21, 26, 28, 30, 32, 34]
UPCARD = 23  # Jc


def case_a(env):
  """Known-answer position a: h = 2, 720 consistent worlds.

  Opponent: every dealt card shed, two late stock draws kept (known
  cards shed after them). Stops at my DRAW with 4 in the stock: h = 2,
  |U| = 6: C(6,2) = 15 hands x 4! = 24 orders x 2 labellings = 720.

  Args:
    env: the Env.

  Returns:
    `play`'s (state, history).
  """
  return play(
      env,
      HEARTS_RUN,
      OPP_HAND,
      UPCARD,
      at_my_turn(cm.PHASE_DRAW, 4),
      keep=0,
      late=("known", "known"),
      late_at=7,
  )


def case_c(env):
  """Known-answer position c: h = 3, 9 draw labellings.

  Opponent keeps two dealt cards, then two late stock draws: after the
  first it sheds a dealt card, after the second a known card. Stops at my
  DRAW with 4 in the stock: h = 3, labellings 3 x 3 = 9.

  Args:
    env: the Env.

  Returns:
    `play`'s (state, history).
  """
  return play(
      env,
      HEARTS_RUN,
      OPP_HAND,
      UPCARD,
      at_my_turn(cm.PHASE_DRAW, 4),
      keep=2,
      late=("orig", "known"),
      late_at=7,
  )


def wall_discard(env, hand0=None, hand1=None, upcard=UPCARD):
  """My DISCARD with 2 left in the stock.

  After it the hand is at the wall, so every action ends the hand through the
  scripted endgame; opponent as case_a (h = 2).

  Args:
    env: the Env.
    hand0: my dealt hand; None is HEARTS_RUN.
    hand1: the opponent's dealt hand; None is OPP_HAND.
    upcard: the first upcard.

  Returns:
    `play`'s (state, history).
  """
  return play(
      env,
      HEARTS_RUN if hand0 is None else hand0,
      OPP_HAND if hand1 is None else hand1,
      upcard,
      at_my_turn(cm.PHASE_DISCARD, 2),
      keep=0,
      late=("known", "known"),
      late_at=6,
  )
