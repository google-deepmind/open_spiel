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

"""replay_seat_steps alone.

replay_seat_steps alone: known answers, shapes, dtypes, masks, privacy of
the event stream, multi-hand flags, and fail-loud input checks.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import connect_four
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.envs import phantom_ttt
from open_spiel.python.scale.serve import replay

J, Q, K = 0, 1, 2
PASS, BET = 0, 1
FOLD, CALL, RAISE = 0, 1, 2


@pytest.fixture(name="kenv", scope="module")
def _kenv_fixture():
  """A Kuhn env.

  Returns:
    The env.
  """
  return kuhn.KuhnEnv()


def _np(steps):
  """The SeatSteps as numpy.

  Args:
    steps: The SeatSteps.

  Returns:
    The SeatSteps.
  """
  return jax.tree_util.tree_map(np.asarray, steps)


def test_kuhn_known_answer(kenv):
  """J to seat 0, Q to seat 1; seat 0 passes, seat 1 bets; seat 0's view."""
  r = replay.replay_seat_steps(kenv, [[J, Q, PASS, BET]], seat=0)
  st = _np(r.steps)
  assert r.num_plies == 3
  np.testing.assert_array_equal(st.seat, [[0, 0, 0]])
  np.testing.assert_array_equal(st.to_move, [[0, 1, 0]])
  np.testing.assert_array_equal(st.new_hand, [[True, False, False]])
  np.testing.assert_array_equal(st.new_match, [[True, False, False]])
  np.testing.assert_array_equal(st.last_action, [[types.NO_ACTION, PASS, BET]])
  np.testing.assert_array_equal(st.last_actor, [[types.NO_ACTOR, 0, 1]])
  # Another seat's legal set is never shown.
  np.testing.assert_array_equal(
      st.legal_mask, [[[True, True], [False, False], [True, True]]]
  )
  np.testing.assert_array_equal(st.info.card, [[J, J, J]])
  np.testing.assert_array_equal(st.info.num_bets, [[0, 1, 2]])
  np.testing.assert_array_equal(st.info.bets[0, 2], [PASS, BET, -1])
  # Seat 1 never sees seat 0's card.
  r1 = replay.replay_seat_steps(kenv, [[J, Q, PASS, BET]], seat=1)
  np.testing.assert_array_equal(np.asarray(r1.steps.info.card), [[Q, Q, Q]])
  np.testing.assert_array_equal(
      np.asarray(r1.steps.legal_mask[0, 1]), [True, True]
  )


def test_dtypes_and_shapes(kenv):
  """Dtypes and shapes."""
  r = replay.replay_seat_steps(kenv, [[K, J, BET]], seat=1)
  st = r.steps
  t = r.num_plies
  assert t == 2
  for name in ("seat", "to_move", "last_action", "last_actor"):
    x = getattr(st, name)
    assert x.dtype == jnp.int32 and x.shape == (1, t), (name, x.dtype, x.shape)
  for name in ("new_hand", "new_match"):
    x = getattr(st, name)
    assert x.dtype == jnp.bool_ and x.shape == (1, t), (name, x.dtype, x.shape)
  assert st.legal_mask.dtype == jnp.bool_ and st.legal_mask.shape == (1, t, 2)
  for leaf in jax.tree_util.tree_leaves(st.info):
    assert leaf.shape[:2] == (1, t)


def test_leduc_mid_hand_chance():
  """Leduc mid hand chance.

  Round 1: call, call; the public card (chance) is dealt; seat 1's view of the
  first round-2 ply shows the card and the last PLAYER action.
  """
  env = leduc.LeducEnv()
  hist = [
      0,
      2,
      CALL,
      CALL,
      5,
  ]  # cards 0 (J) and 2 (Q), call call, public card 5 (K)
  r = replay.replay_seat_steps(env, [hist], seat=1)
  st = _np(r.steps)
  assert r.num_plies == 3
  np.testing.assert_array_equal(st.to_move, [[0, 1, 0]])
  np.testing.assert_array_equal(st.last_action, [[types.NO_ACTION, CALL, CALL]])
  np.testing.assert_array_equal(st.last_actor, [[types.NO_ACTOR, 0, 1]])
  np.testing.assert_array_equal(st.info.public_card, [[-1, -1, 5]])
  np.testing.assert_array_equal(st.info.card, [[2, 2, 2]])
  np.testing.assert_array_equal(st.info.round, [[1, 1, 2]])


def test_phantom_ttt_hidden_actions_stay_hidden():
  """Phantom ttt hidden actions stay hidden.

  Seat 1 does not see seat 0's move: last_action is NO_ACTION, the actor is
  public.
  """
  env = phantom_ttt.PhantomTTTEnv()
  r = replay.replay_seat_steps(env, [[4, 0]], seat=1)
  st = _np(r.steps)
  np.testing.assert_array_equal(st.last_actor, [[types.NO_ACTOR, 0, 1]])
  np.testing.assert_array_equal(
      st.last_action, [[types.NO_ACTION, types.NO_ACTION, 0]]
  )
  r0 = replay.replay_seat_steps(env, [[4, 0]], seat=0)
  np.testing.assert_array_equal(
      np.asarray(r0.steps.last_action), [[types.NO_ACTION, 4, types.NO_ACTION]]
  )


def test_multi_hand_flags_and_terminal_ply(kenv):
  """Multi hand flags and terminal ply.

  A completed hand contributes its decision plies plus ONE terminal ply; the
  next hand's first ply has new_hand (not new_match) and no last action.

  Args:
    kenv: the kuhn env (fixture).
  """
  r = replay.replay_seat_steps(kenv, [[J, Q, BET, PASS], [Q, K]], seat=0)
  st = _np(r.steps)
  assert r.num_plies == 4
  np.testing.assert_array_equal(st.to_move, [[0, 1, -1, 0]])
  np.testing.assert_array_equal(st.new_hand, [[True, False, False, True]])
  np.testing.assert_array_equal(st.new_match, [[True, False, False, False]])
  np.testing.assert_array_equal(
      st.last_action, [[types.NO_ACTION, BET, PASS, types.NO_ACTION]]
  )
  np.testing.assert_array_equal(
      st.last_actor, [[types.NO_ACTOR, 0, 1, types.NO_ACTOR]]
  )
  assert not st.legal_mask[0, 2].any()
  np.testing.assert_array_equal(st.info.card, [[J, J, J, Q]])
  assert r.hand_starts == (0, 3)


def test_terminal_last_hand_replays(kenv):
  """Terminal last hand replays."""
  r = replay.replay_seat_steps(kenv, [[J, Q, BET, BET]], seat=0)
  assert int(r.steps.to_move[0, -1]) == -1


def test_deterministic(kenv):
  """Deterministic."""
  a = replay.replay_seat_steps(kenv, [[K, Q, PASS]], seat=1)
  b = replay.replay_seat_steps(kenv, [[K, Q, PASS]], seat=1)
  for x, y in zip(
      jax.tree_util.tree_leaves(a.steps), jax.tree_util.tree_leaves(b.steps)
  ):
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_connect_four_no_chance():
  """Connect four no chance."""
  env = connect_four.ConnectFourEnv()
  r = replay.replay_seat_steps(env, [[3, 3, 4]], seat=1)
  st = _np(r.steps)
  np.testing.assert_array_equal(st.to_move, [[0, 1, 0, 1]])
  np.testing.assert_array_equal(st.last_action, [[types.NO_ACTION, 3, 3, 4]])
  assert st.legal_mask[0, 3].all()


@pytest.mark.parametrize(
    "hands,match",
    [
        ([[J, J]], "chance outcome 0 has probability 0"),  # J twice
        ([[J, Q, 2]], "illegal action 2"),
        ([[J]], "ends at a chance node"),
        ([[J, Q, BET, PASS, PASS]], "after the hand is over"),
        ([[J, Q, BET], [Q, K]], "is not terminal"),
        ([], "no hands"),
        ([[J, Q, -1]], "illegal action -1"),
    ],
)
def test_bad_histories_raise(kenv, hands, match):
  """Bad histories raise."""
  with pytest.raises(ValueError, match=match):
    replay.replay_seat_steps(kenv, hands, seat=0)


def test_bad_seat_raises(kenv):
  """Bad seat raises."""
  with pytest.raises(ValueError, match="seat"):
    replay.replay_seat_steps(kenv, [[J, Q]], seat=2)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
