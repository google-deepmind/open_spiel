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

"""league.sampler at full generality: per-seat fills and uneven sides.

* Per-seat fills on bridge sides and team_pennies_2v2's (0, 1, 0, 1): a
  learner + fixed partner against a pool pair; the learner's one seat is
  the only perspective seat, the partner seat holds the fixed member, the
  opponent side one drawn member on both seats; the fill rotates with its
  side (cyclic). Two per-seat pool draws on one side are independent and
  have exact stratified marginals.
* A per-seat lineup made only of whole-side-equivalent fills is the
  whole-side lineup's table bit for bit when it has no pool seat.
* Uneven sides: 2v1 (team_pennies_2v1's (0, 1, 0)) seats the learner on the
  pair or on the lone seat (lineup_from_mix(learner_side)), nothing rotates
  (the sides' sizes differ); 3v1v1 rotates the learner over the two lone
  sides only, and the team of three keeps its member on all three seats.
* The perspective width is the same on every row.
"""

import sys

import numpy as np
import pytest

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.league import sampler

SideFill, Lineup, Sides = sides_lib.SideFill, sides_lib.Lineup, sides_lib.Sides

WHERE = {"a@1": (1, 0), "b@2": (1, 1), "c@3": (2, 0), "partner@0": (3, 0)}
POOL = sampler.Mix(0.0, {"a@1": 0.5, "b@2": 0.25, "c@3": 0.25})
BRIDGE = Sides.partnerships(4, 2, True)  # also team_pennies_2v2's sides
TWO_V_ONE = Sides((0, 1, 0), True)  # team_pennies_2v1
THREE_V_ONE_V_ONE = Sides((0, 1, 0, 2, 0), True)


def _arr(t):
  """A table's arrays on host.

  Args:
    t: A SlotTable.

  Returns:
    Its arrays, as numpy.
  """
  return tuple(np.asarray(x) for x in t)


@pytest.mark.parametrize("stratified", [True, False])
def test_learner_with_fixed_partner_against_a_pool_pair_on_bridge_sides(
    stratified,
):
  """Learner with fixed partner against a pool pair on bridge sides."""
  lu = sides_lib.lineup((("learner", "fixed"), "pool"), members=("partner@0",))
  b = 40
  t, counts = sampler.table_from_lineup(
      lu, POOL, WHERE, b, BRIDGE, rng.key(3), stratified
  )
  slot, member, persp = _arr(t)
  assert persp.shape == (b,)  # one learner seat: the [B] perspective
  for r in range(b):
    mine = BRIDGE.seats(r % 2)  # cyclic: position 0 on side r % 2
    theirs = BRIDGE.seats(1 - r % 2)
    assert persp[r] == mine[0]
    assert (slot[r, mine[0]], member[r, mine[0]]) == (0, 0)
    assert (slot[r, mine[1]], member[r, mine[1]]) == (3, 0)
    pair = {(int(slot[r, q]), int(member[r, q])) for q in theirs}
    assert len(pair) == 1 and pair.pop() in set(WHERE.values()) - {(3, 0)}
  assert counts["partner@0"] == b and sum(counts.values()) == 2 * b


def test_per_seat_pool_draws_are_independent_with_exact_marginals():
  """Per seat pool draws are independent with exact marginals."""
  # Learner pair against two seats drawn independently (mixed opponents).
  lu = Lineup((SideFill("learner"), (SideFill("pool"), SideFill("pool"))))
  b = 400
  t, counts = sampler.table_from_lineup(lu, POOL, WHERE, b, BRIDGE, rng.key(7))
  slot, member, persp = _arr(t)
  assert persp.shape == (b, 2)
  name_of = {v: k for k, v in WHERE.items()}
  # per shift class, per seat: exact largest-remainder marginals
  for c in range(2):
    rows = np.arange(c, b, 2)
    theirs = BRIDGE.seats(1 - c)
    for q in theirs:
      got = {}
      for r in rows:
        n = name_of[(int(slot[r, q]), int(member[r, q]))]
        got[n] = got.get(n, 0) + 1
      assert got == {"a@1": 100, "b@2": 50, "c@3": 50}, (c, q, got)
  assert counts == {"a@1": 400, "b@2": 200, "c@3": 200}
  # The two seats are not one draw: partners differ on many rows.
  differ = sum(
      int(
          (slot[r, (1 - r % 2)], member[r, 1 - r % 2])
          != (slot[r, 3 - r % 2], member[r, 3 - r % 2])
      )
      for r in range(b)
  )
  assert differ > b // 4


def test_per_seat_fills_equal_to_whole_sides_give_the_same_table():
  """Per seat fills equal to whole sides give the same table."""
  for sides in (BRIDGE, Sides.partnerships(6, 3, True)):
    t = sides.num_sides
    whole = Lineup(
        (SideFill("learner"), SideFill("self"))
        + (SideFill("fixed", member="c@3"),) * (t - 2)
    )
    per_seat = Lineup(
        tuple(
            f if q == 0 else (f,) * sides.side_sizes[q]
            for q, f in enumerate(whole.fills)
        )
    )
    a, ca = sampler.table_from_lineup(whole, POOL, WHERE, 24, sides, rng.key(1))
    b, cb = sampler.table_from_lineup(
        per_seat, POOL, WHERE, 24, sides, rng.key(1)
    )
    for x, y in zip(_arr(a), _arr(b)):
      np.testing.assert_array_equal(x, y)
    assert (
        ca[sampler.SELF] * 2 == cb[sampler.SELF]
    )  # self counts per unit (per seat)


@pytest.mark.parametrize("learner_side", [0, 1])
def test_two_v_one_seats_the_learner_on_either_side(learner_side):
  """Two v one seats the learner on either side."""
  lu = sampler.lineup_from_mix(TWO_V_ONE, learner_side)
  # rotatable, but the two sides differ in size: nothing moves
  assert lu.rotate == "cyclic"
  t, counts = sampler.table_from_lineup(
      lu, POOL, WHERE, 30, TWO_V_ONE, rng.key(2)
  )
  slot, member, persp = _arr(t)
  mine = TWO_V_ONE.seats(learner_side)
  theirs = TWO_V_ONE.seats(1 - learner_side)
  want_persp = np.broadcast_to(np.asarray(mine, np.int32), (30, len(mine)))
  np.testing.assert_array_equal(
      persp if persp.ndim == 2 else persp[:, None], want_persp
  )
  assert (slot[:, list(mine)] == 0).all() and (member[:, list(mine)] == 0).all()
  for r in range(30):
    assert len({(int(slot[r, q]), int(member[r, q])) for q in theirs}) == 1
  assert sum(counts.values()) == 30


def test_two_v_one_mixed_pair_learner_plus_fixed_partner_against_pool():
  """Two v one mixed pair learner plus fixed partner against pool."""
  lu = sides_lib.lineup((("learner", "fixed"), "pool"), members=("partner@0",))
  t, _ = sampler.table_from_lineup(lu, POOL, WHERE, 12, TWO_V_ONE, rng.key(0))
  slot, _, persp = _arr(t)
  np.testing.assert_array_equal(persp, np.zeros(12, np.int32))
  assert (slot[:, 2] == 3).all() and (slot[:, 1] != 0).all()


def test_three_v_one_v_one_rotates_only_the_lone_sides():
  """Three v one v one rotates only the lone sides."""
  sides = THREE_V_ONE_V_ONE
  lu = sampler.lineup_from_mix(sides, learner_side=1)
  b = 40
  t, counts = sampler.table_from_lineup(lu, POOL, WHERE, b, sides, rng.key(5))
  slot, member, persp = _arr(t)
  assert persp.shape == (b,)
  for r in range(b):
    lone = sides.seats(1 + r % 2)[0]  # learner on side 1, then side 2, ...
    other = sides.seats(2 - r % 2)[0]
    assert persp[r] == lone and slot[r, lone] == 0
    assert slot[r, other] != 0
    team = {(int(slot[r, q]), int(member[r, q])) for q in sides.seats(0)}
    # one member for the team of three
    assert len(team) == 1 and team.pop() != (
        0,
        0,
    )
  assert sum(counts.values()) == 2 * b
  # The learner on the team of three: M = 3, the same on every row.
  t3, _ = sampler.table_from_lineup(
      sampler.lineup_from_mix(sides), POOL, WHERE, b, sides, rng.key(5)
  )
  np.testing.assert_array_equal(
      np.asarray(t3.perspective_seat),
      np.broadcast_to(np.array([0, 2, 4], np.int32), (b, 3)),
  )
  # A per-seat team: learner + self + fixed on the three seats of side 0.
  lu3 = sides_lib.lineup(
      (("learner", "self", "fixed"), "pool", "pool"), members=("c@3",)
  )
  t4, _ = sampler.table_from_lineup(lu3, POOL, WHERE, b, sides, rng.key(6))
  s4, _, p4 = _arr(t4)
  np.testing.assert_array_equal(p4, np.zeros(b, np.int32))
  assert (s4[:, 2] == 0).all() and (s4[:, 4] == 2).all()


def test_refusals():
  """Refusals."""
  # A per-seat fill sized for another side is refused (rotation keeps sizes, so
  # this is the one way a per-seat fill could meet a side of another size).
  with pytest.raises(
      ValueError, match="per-seat fill of 2 seats for a side of 1"
  ):
    sampler.table_from_lineup(
        Lineup((SideFill("pool"), (SideFill("learner"), SideFill("pool")))),
        POOL,
        WHERE,
        8,
        TWO_V_ONE,
        rng.key(0),
    )
  with pytest.raises(NotImplementedError, match="learning slot 1.*fixed"):
    sampler.table_from_lineup(
        Lineup((
            (SideFill("learner"), SideFill("learner", learner=1)),
            SideFill("pool"),
        )),
        POOL,
        WHERE,
        8,
        BRIDGE,
        rng.key(0),
    )
  with pytest.raises(ValueError, match="rotatable"):
    sampler.table_from_lineup(
        sides_lib.lineup(("learner", "pool")),
        POOL,
        WHERE,
        8,
        Sides((0, 1, 0), True, rotatable=False),
        rng.key(0),
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
