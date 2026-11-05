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

"""league.sampler seating by sides.

table_from_lineup, lineup_from_mix, draw_table, and sample with
LeagueSlots.sides.

  * Two players: lineup_from_mix's table is table_from_mix's, bit for bit,
    over a grid of mixes, batches, seeds and both draw modes; draw_table and
    sample with sides at P = 2 are table_from_mix.
  * One perspective width M per Arena: the same on every row and every
    table of one Lineup and Sides, whatever the Mix (self-play mass and pool
    mass together included): the learner's seats, [B] when that is one seat.
  * Bridge: no drawn member ever sits on the learner's side; the learner
    holds both partner seats (M = 2); each opponent side is one member.
  * Every side "learner" (and self_play_seats="all") is
    core.arena.self_play_table(all_seats=True).
  * Independence: on a 3-player stub env (one side per seat) the two
    opponent sides' joint draws match the product of the Mix's marginals
    (chi-square, iid draws), and stratified draws have exact marginals per
    side and learner position and pass a chi-square test of independence.
  * Without sides, sample is table_from_mix at every P (the seating before
    sides, for callers that do not pass them yet).
  * Refusals: a second learning slot; a learner side out of range; a lineup
    that does not fit the sides; "all" with a non-self-play mix or a lineup;
    an unknown fixed member.

Runs without jax: keys are core.rng keys, pools hold a stub policy.
"""

import sys

import numpy as np
import pytest
from scipy import stats

from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import stub
from open_spiel.python.scale.league import sampler

SideFill, Lineup, Sides = sides_lib.SideFill, sides_lib.Lineup, sides_lib.Sides

WHERE = {"a@1": (1, 0), "b@2": (1, 1), "c@3": (2, 0), "d@4": (1, 2)}
MIXES = {
    "one": sampler.Mix(0.0, {"a@1": 1.0}),
    "self_two": sampler.Mix(0.25, {"a@1": 0.5, "b@2": 0.25}),
    "self": sampler.Mix(1.0, {}),
    "thirds": sampler.Mix(0.1, {"a@1": 0.3, "b@2": 0.3, "c@3": 0.3}),
    "uneven": sampler.Mix(
        0.05, {"a@1": 0.61, "b@2": 0.07, "c@3": 0.19, "d@4": 0.08}
    ),
}
BRIDGE = Sides.partnerships(4, 2, True)
SIDES = {
    "p2": Sides.individual(2, True),
    "p3": Sides.individual(3, True),
    "p4": Sides.individual(4, False),
    "bridge": BRIDGE,
    "p6_three_pairs": Sides.partnerships(6, 3, True),
}


def _eq_table(t1, t2):
  """Asserts two SlotTables are equal, dtype and shape included.

  Args:
    t1: A table.
    t2: A table.
  """
  for name, x, y in zip(
      ("seat_slot", "seat_member", "perspective_seat"), t1, t2
  ):
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype and x.shape == y.shape, (
        name,
        x.dtype,
        x.shape,
        y.dtype,
        y.shape,
    )
    np.testing.assert_array_equal(x, y, err_msg=name)


# -- two players: table_from_mix bit for bit -----------------------------------
@pytest.mark.parametrize("mix_name", sorted(MIXES))
@pytest.mark.parametrize("stratified", [True, False])
def test_two_player_lineup_table_is_table_from_mix(mix_name, stratified):
  """Two player lineup table is table from mix."""
  sides = SIDES["p2"]
  for batch in (1, 2, 37, 64):
    for seed in range(4):
      for learner in ((0, 0), (0, 3)):
        want, want_c = sampler.table_from_mix(
            MIXES[mix_name], WHERE, batch, 2, rng.key(seed), stratified, learner
        )
        got, got_c = sampler.table_from_lineup(
            sampler.lineup_from_mix(sides),
            MIXES[mix_name],
            WHERE,
            batch,
            sides,
            rng.key(seed),
            stratified,
            learner,
        )
        _eq_table(got, want)
        assert list(got_c.items()) == list(want_c.items())
        got, got_c = sampler.draw_table(
            MIXES[mix_name],
            WHERE,
            batch,
            sides,
            rng.key(seed),
            stratified=stratified,
            learner=learner,
        )
        _eq_table(got, want)
        assert list(got_c.items()) == list(want_c.items())


class _StubPolicy:

  def __init__(self):
    self.spec = types.PolicySpec(
        "kuhn_poker",
        "kuhn_poker_info_index",
        2,
        2,
        False,
        types.Capability.NATIVE_JAX,
        "jax",
        "tabular",
    )


def _pools():
  """Two stub pools: three members, then one.

  Returns:
    The pools.
  """
  pol = _StubPolicy()
  return (
      sampler.Pool(pol, types.ARGMAX, ("a@1", "b@2", "d@4"), np.zeros((3, 1))),
      sampler.Pool(pol, types.ARGMAX, ("c@3",), np.zeros((1, 1))),
  )


@pytest.mark.parametrize("mix_name", sorted(MIXES))
def test_sample_with_two_player_sides_is_unchanged(mix_name):
  """Sample with two player sides is unchanged."""
  for stratified in (True, False):
    for seed in range(3):
      base = sampler.sample(
          sampler.LeagueSlots(2, 40, _pools(), MIXES[mix_name], stratified),
          rng.key(seed),
      )
      with_sides = sampler.sample(
          sampler.LeagueSlots(
              2, 40, _pools(), MIXES[mix_name], stratified, sides=SIDES["p2"]
          ),
          rng.key(seed),
      )
      _eq_table(with_sides.table, base.table)
      assert with_sides.counts == base.counts


# -- one perspective width per Arena -------------------------------------------
def _lineups(sides):
  """Lineups of `sides` with the perspective rows each seats.

  Args:
    sides: The Sides.

  Yields:
    (Lineup, perspective rows per lineup).
  """
  t = sides.num_sides
  yield sampler.lineup_from_mix(sides), sides.side_sizes[0]
  yield Lineup((SideFill("learner"),) * t, "cyclic"), sides.num_players
  if t >= 3:
    yield (
        Lineup(
            (SideFill("learner"), SideFill("self"))
            + (SideFill("pool"),) * (t - 2),
            "cyclic",
        ),
        sides.side_sizes[0],
    )
    yield (
        Lineup(
            (SideFill("learner"), SideFill("learner"))
            + (SideFill("pool"),) * (t - 2),
            "none",
        ),
        2 * sides.side_sizes[0],
    )
  yield (
      Lineup(
          (SideFill("learner"),) + (SideFill("fixed", member="c@3"),) * (t - 1),
          "none",
      ),
      sides.side_sizes[0],
  )


@pytest.mark.parametrize("sides_name", sorted(SIDES))
def test_perspective_width_is_one_per_lineup_and_sides(sides_name):
  """Perspective width is one per lineup and sides."""
  sides = SIDES[sides_name]
  for lineup, m in _lineups(sides):
    shapes = set()
    for mix in MIXES.values():
      for stratified in (True, False):
        for seed in range(3):
          t, _ = sampler.table_from_lineup(
              lineup, mix, WHERE, 24, sides, rng.key(seed), stratified
          )
          persp = np.asarray(t.perspective_seat)
          shapes.add(persp.shape)
          assert persp.dtype == np.int32
          persp2 = persp[:, None] if persp.ndim == 1 else persp
          assert all(len(set(row)) == m for row in persp2.tolist())
          np.testing.assert_array_equal(np.sort(persp2, axis=1), persp2)
          slot = np.asarray(t.seat_slot)
          assert (slot[np.arange(24)[:, None], persp2] == 0).all()
    want = (24,) if m == 1 else (24, m)
    assert shapes == {want}, (lineup, shapes)


def test_mixed_self_and_pool_mix_gives_one_width_at_p4():
  """A Mix with self-play mass and pool mass at P = 4.

  Every row emits the learner side's seats, self-play rows included.
  """
  mix = MIXES["self_two"]
  for sides in (SIDES["p4"], BRIDGE):
    t, counts = sampler.draw_table(mix, WHERE, 64, sides, rng.key(3))
    assert counts[sampler.SELF] > 0 and counts["a@1"] > 0
    m = sides.side_sizes[0]
    assert np.shape(t.perspective_seat) == ((64,) if m == 1 else (64, m))


# -- bridge --------------------------------------------------------------------
@pytest.mark.parametrize("mix_name", ["one", "self_two", "thirds", "uneven"])
@pytest.mark.parametrize("stratified", [True, False])
def test_bridge_no_drawn_member_at_the_learners_partner_seat(
    mix_name, stratified
):
  """Bridge no drawn member at the learners partner seat."""
  learner = (0, 0)
  for seed in range(5):
    t, counts = sampler.draw_table(
        MIXES[mix_name], WHERE, 48, BRIDGE, rng.key(seed), stratified=stratified
    )
    slot, member, persp = (np.asarray(x) for x in t)
    assert persp.shape == (48, 2)
    for r in range(48):
      mine = BRIDGE.seats(r % 2)
      theirs = BRIDGE.seats(1 - r % 2)
      # the learner holds both partner seats
      np.testing.assert_array_equal(persp[r], mine)
      for s in mine:
        assert (slot[r, s], member[r, s]) == learner
      # The opposing partnership is one entrant: both its seats hold one draw.
      assert (slot[r, theirs[0]], member[r, theirs[0]]) == (
          slot[r, theirs[1]],
          member[r, theirs[1]],
      )
    assert sum(counts.values()) == 48


def test_bridge_learner_rotates_over_both_partnerships():
  """Bridge learner rotates over both partnerships."""
  t, _ = sampler.draw_table(MIXES["one"], WHERE, 8, BRIDGE, rng.key(0))
  np.testing.assert_array_equal(
      np.asarray(t.perspective_seat), [[0, 2], [1, 3]] * 4
  )
  np.testing.assert_array_equal(
      np.asarray(t.seat_slot), [[0, 1, 0, 1], [1, 0, 1, 0]] * 4
  )


# -- every side the learner is self_play_table ---------------------------------
def _self_play_table(batch, p):
  """The all-seats self-play table, as arrays.

  Args:
    batch: The rows.
    p: The players.

  Returns:
    (seat_slot, seat_member, perspective_seat).
  """
  # core.arena.self_play_table(batch, p, all_seats=True), spelled out (this
  # test is jax-free; sampler_sides_arena_test compares with the function).
  return (
      np.zeros((batch, p), np.int32),
      np.zeros((batch, p), np.int32),
      np.broadcast_to(np.arange(p, dtype=np.int32), (batch, p)).copy(),
  )


@pytest.mark.parametrize("sides_name", sorted(SIDES))
def test_all_learner_lineup_and_all_seats_are_self_play_table(sides_name):
  """All learner lineup and all seats are self play table."""
  sides = SIDES[sides_name]
  want = _self_play_table(30, sides.num_players)
  for rotate in ("cyclic", "none"):
    lineup = Lineup((SideFill("learner"),) * sides.num_sides, rotate)
    for mix in (MIXES["self"], MIXES["uneven"]):
      t, counts = sampler.table_from_lineup(
          lineup, mix, WHERE, 30, sides, rng.key(1)
      )
      _eq_table(t, want)
      assert not counts
  t, counts = sampler.draw_table(
      MIXES["self"], {}, 30, sides, rng.key(1), self_play_seats="all"
  )
  _eq_table(t, want)
  assert counts == {sampler.SELF: 30}


# -- independence of the pool sides --------------------------------------------
P3 = stub.StubEnv(
    "stub3",
    num_players=3,
    num_actions=2,
    max_len=4,
    info_like=np.zeros((2,), np.float32),
)


def _joint(t, learner_pos, side_a, side_b, codes):
  """The codes the learner's rows seat at two sides.

  Args:
    t: The SlotTable.
    learner_pos: The learner's position (rows r with r % 3 equal to it).
    side_a: A seat.
    side_b: A seat.
    codes: {(slot, member): code}.

  Returns:
    (codes at side_a, codes at side_b).
  """
  slot, member = np.asarray(t.seat_slot), np.asarray(t.seat_member)
  rows = np.flatnonzero(np.arange(slot.shape[0]) % 3 == learner_pos)

  def code(s):
    """The (slot, member) code of seat `s` on the learner's rows.

    Args:
      s: The seat.

    Returns:
      int codes, one per row.
    """
    return np.asarray(
        [codes[(int(slot[r, s]), int(member[r, s]))] for r in rows]
    )

  return code(side_a), code(side_b)


def test_stub_p3_iid_pool_sides_match_the_product_of_marginals():
  """Stub p3 iid pool sides match the product of marginals."""
  sides = sides_lib.sides_of(P3)
  assert sides == Sides.individual(3, True)
  mix = sampler.Mix(0.2, {"a@1": 0.5, "b@2": 0.3})
  codes = {(0, 0): 0, (1, 0): 1, (1, 1): 2}
  probs = np.asarray([0.2, 0.5, 0.3])
  b = 3 * 6000
  t, _ = sampler.draw_table(mix, WHERE, b, sides, rng.key(11), stratified=False)
  for c in range(3):
    others = [s for s in range(3) if s != c]
    x, y = _joint(t, c, others[0], others[1], codes)
    obs = np.zeros((3, 3))
    np.add.at(obs, (x, y), 1)
    exp = np.outer(probs, probs) * obs.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p_value = stats.chi2.sf(chi2, df=8)
    assert p_value > 1e-3, (c, chi2, p_value, obs)


def test_stub_p3_stratified_pool_sides_exact_marginals_and_independent():
  """Stub p3 stratified pool sides exact marginals and independent."""
  sides = sides_lib.sides_of(P3)
  mix = sampler.Mix(0.2, {"a@1": 0.5, "b@2": 0.3})
  codes = {(0, 0): 0, (1, 0): 1, (1, 1): 2}
  b = 3 * 3000
  for seed in range(3):
    t, counts = sampler.draw_table(mix, WHERE, b, sides, rng.key(seed))
    assert counts == {
        sampler.SELF: 2 * 600 * 3,
        "a@1": 2 * 1500 * 3,
        "b@2": 2 * 900 * 3,
    }
    for c in range(3):
      others = [s for s in range(3) if s != c]
      x, y = _joint(t, c, others[0], others[1], codes)
      for v in (x, y):  # exact marginals per side and learner position
        np.testing.assert_array_equal(
            np.bincount(v, minlength=3), [600, 1500, 900]
        )
      assert not np.array_equal(x, y)
      obs = np.zeros((3, 3))
      np.add.at(obs, (x, y), 1)
      _, p_value, _, _ = stats.chi2_contingency(obs, correction=False)
      assert p_value > 1e-3, (seed, c, p_value, obs)


def test_every_pool_side_draws_its_own_member():
  """Four individual seats.

  The three opponent sides are three draws, not three copies (hearts).
  """
  t, _ = sampler.draw_table(
      sampler.uniform(["a@1", "b@2", "c@3", "d@4"]),
      WHERE,
      64,
      SIDES["p4"],
      rng.key(2),
  )
  slot, member = np.asarray(t.seat_slot), np.asarray(t.seat_member)
  ident = slot * 10 + member
  differ = 0
  for r in range(64):
    others = [ident[r, s] for s in range(4) if s != r % 4]
    differ += len(set(others)) > 1
  assert differ > 48


# -- counts, rotation, fixed and self sides ------------------------------------
def test_counts_fixed_and_self_sides_and_rotate_none():
  """Counts fixed and self sides and rotate none."""
  sides = SIDES["p4"]
  lineup = sides_lib.lineup(
      ("learner", "self", "fixed", "pool"), rotate="none", members=("c@3",)
  )
  t, counts = sampler.table_from_lineup(
      lineup, MIXES["self_two"], WHERE, 20, sides, rng.key(4)
  )
  slot, member, persp = (np.asarray(x) for x in t)
  np.testing.assert_array_equal(persp, np.zeros(20, np.int32))
  assert (slot[:, 1] == 0).all() and (member[:, 1] == 0).all()
  assert (slot[:, 2] == 2).all() and (member[:, 2] == 0).all()
  assert counts["c@3"] == 20
  assert sum(counts.values()) == 3 * 20
  # the self side plus the pool side's self-play draws
  assert counts[sampler.SELF] == 20 + 5


def test_sample_with_sides_seats_fixed_members_from_the_pools():
  """Sample with sides seats fixed members from the pools."""
  sides = SIDES["p3"]
  lineup = sides_lib.lineup(("learner", "fixed", "pool"), members=("c@3",))
  st = sampler.LeagueSlots(
      3, 30, _pools(), MIXES["one"], sides=sides, lineup=lineup
  )
  d = sampler.sample(st, rng.key(0))
  slot = np.asarray(d.table.seat_slot)
  for r in range(30):
    fixed_side = (1 + r) % 3
    assert slot[r, fixed_side] == 2


def test_sample_without_sides_is_table_from_mix_at_any_player_count():
  """Sample without sides is table from mix at any player count."""
  for p in (2, 3, 4):
    for mix_name in ("one", "self_two", "self"):
      d = sampler.sample(
          sampler.LeagueSlots(p, 24, _pools(), MIXES[mix_name]), rng.key(p)
      )
      want, want_c = sampler.table_from_mix(
          MIXES[mix_name], WHERE, 24, p, rng.key(p)
      )
      _eq_table(d.table, want)
      assert d.counts == want_c


# -- refusals ------------------------------------------------------------------
def test_refusals():
  """Refusals."""
  with pytest.raises(ValueError, match="needs sides"):
    sampler.LeagueSlots(4, 8, _pools(), MIXES["self"], self_play_seats="all")
  with pytest.raises(ValueError, match="seats, the table"):
    sampler.LeagueSlots(4, 8, _pools(), MIXES["self"], sides=SIDES["p3"])
  with pytest.raises(NotImplementedError, match="learning slot 1"):
    sampler.table_from_lineup(
        Lineup((SideFill("learner"), SideFill("learner", learner=1))),
        MIXES["one"],
        WHERE,
        8,
        BRIDGE,
        rng.key(0),
    )
  with pytest.raises(ValueError, match="learner_side 3"):
    sampler.lineup_from_mix(Sides((0, 1, 1), True), learner_side=3)
  with pytest.raises(ValueError, match="fills for"):
    sampler.table_from_lineup(
        sampler.lineup_from_mix(SIDES["p3"]),
        MIXES["one"],
        WHERE,
        8,
        BRIDGE,
        rng.key(0),
    )
  with pytest.raises(ValueError, match="pure self-play"):
    sampler.draw_table(
        MIXES["one"], WHERE, 8, BRIDGE, rng.key(0), self_play_seats="all"
    )
  with pytest.raises(ValueError, match="one or the other"):
    sampler.draw_table(
        MIXES["self"],
        WHERE,
        8,
        BRIDGE,
        rng.key(0),
        self_play_seats="all",
        lineup=sampler.lineup_from_mix(BRIDGE),
    )
  with pytest.raises(ValueError, match="self_play_seats"):
    sampler.draw_table(
        MIXES["self"], WHERE, 8, BRIDGE, rng.key(0), self_play_seats="side"
    )
  with pytest.raises(KeyError, match="fixed member"):
    sampler.table_from_lineup(
        sides_lib.lineup(("learner", "fixed"), members=("zz@9",)),
        MIXES["one"],
        WHERE,
        8,
        BRIDGE,
        rng.key(0),
    )
  with pytest.raises(KeyError, match="no slot holds member"):
    sampler.draw_table(
        sampler.Mix(0.0, {"zz@9": 1.0}), WHERE, 8, BRIDGE, rng.key(0)
    )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
