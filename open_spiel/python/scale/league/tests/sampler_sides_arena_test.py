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

"""Seating by sides against the recorded two-player behaviour and the Arena.

sampler_sides_test is the jax-free half.

  * draw_table at P = 2 reproduces every recorded table_from_mix digest of
    the two-player goldens (testing/goldens/two_player.json), bit for bit.
  * Every side "learner", and self_play_seats="all", equal
    core.arena.self_play_table(all_seats=True).
  * A bridge table drawn with a pool runs in the Arena: it validates, every
    ply's actor is the member the table seats there, the learner's
    partnership is slot 0 at every ply it acts, and the batch has B * 2 rows
    (both partner seats are perspective seats).
  * A per-seat lineup (learner + fixed partner against a pool side) runs in
    the Arena on bridge and on team_pennies_2v1's uneven sides: the learner's
    seat is slot 0, the partner's the fixed member's slot, the rest the pool.
  * kinds.base.Driver.check_table keeps one perspective shape per driver.
"""

import json
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import rng
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import bridge
from open_spiel.python.scale.envs import nplayer_matrix
from open_spiel.python.scale.league import sampler
from open_spiel.python.scale.league.kinds import base as kinds_base
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing.goldens import two_player


def _two_player_table_cases():
  """The two-player goldens and their P = 2 table_from_mix cases.

  Returns:
    (goldens, sorted case names).
  """
  with open(two_player.GOLDEN, encoding="utf-8") as f:
    golden = json.load(f)
  return golden, sorted(
      n for n in golden if n.startswith("sampler/table_from_mix/p2/")
  )


_GOLDEN, _P2_CASES = _two_player_table_cases()


def test_there_are_recorded_two_player_table_cases():
  """There are recorded two player table cases."""
  assert len(_P2_CASES) == 5 * 2 * 2 * 3 + 1, len(_P2_CASES)


@pytest.mark.parametrize("name", _P2_CASES)
def test_draw_table_at_two_players_reproduces_the_recorded_digest(name):
  """Draw table at two players reproduces the recorded digest."""
  parts = name.split("/")
  if parts[3] == "learner_slot":
    mix_name, batch, strat, seed, learner = "self_two", 40, True, 5, (0, 3)
  else:
    mix_name, batch, strat, seed = (
        parts[3],
        int(parts[4][1:]),
        bool(int(parts[5][5:])),
        int(parts[6][1:]),
    )
    learner = (0, 0)
  sides = sides_lib.Sides.individual(2, True)
  # pylint: disable=protected-access
  mix, where, digest = (
      two_player._MIXES[mix_name],
      two_player._WHERE,
      two_player._table,
  )
  # pylint: enable=protected-access
  for lineup in (None, sampler.lineup_from_mix(sides)):
    t, counts = sampler.draw_table(
        mix,
        where,
        batch,
        sides,
        rng.key(seed),
        stratified=strat,
        learner=learner,
        lineup=lineup,
    )
    got = {}
    digest("table", t, got)
    got["counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
    assert got == _GOLDEN[name]


@pytest.mark.parametrize("p,t", [(2, 2), (3, 3), (4, 4), (4, 2), (6, 3)])
def test_all_learner_sides_equal_self_play_table(p, t):
  """All learner sides equal self play table."""
  sides = (
      sides_lib.Sides.individual(p, True)
      if p == t
      else sides_lib.Sides.partnerships(p, t, True)
  )
  want = arena_lib.self_play_table(20, p, all_seats=True)
  got, _ = sampler.table_from_lineup(
      sides_lib.Lineup((sides_lib.SideFill("learner"),) * t),
      sampler.Mix(1.0, {}),
      {},
      20,
      sides,
      rng.key(0),
  )
  got_all, _ = sampler.draw_table(
      sampler.Mix(1.0, {}), {}, 20, sides, rng.key(0), self_play_seats="all"
  )
  for x, y, z in zip(got, got_all, want):
    for a in (x, y):
      assert np.asarray(a).dtype == np.asarray(z).dtype
      np.testing.assert_array_equal(a, z)


def test_bridge_pool_table_runs_in_the_arena():
  """Bridge pool table runs in the arena."""
  env = bridge.BridgeEnv()
  sides = sides_lib.sides_of(env)
  assert sides is bridge.SIDES
  pol = uniform.UniformPolicy("bridge", env.num_actions, 4)
  k, b = 3, 8
  arena = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.SAMPLE),
          arena_lib.Slot(pol, k, types.SAMPLE),
      ],
  )
  where = {"x@1": (1, 0), "y@2": (1, 1), "z@3": (1, 2)}
  mix = sampler.Mix(0.0, {"x@1": 0.5, "y@2": 0.25, "z@3": 0.25})
  table, counts = sampler.draw_table(mix, where, b, sides, rng.key(5))
  assert sum(counts.values()) == b
  params = [{}, {}]
  arena.validate(params, table)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 4)
  _, batch, log = arena.collect(params, table, dk, pk, 24)
  assert np.asarray(batch.valid).shape[0] == 2 * b
  actor, slot, member = (
      np.asarray(x) for x in (log.actor, log.slot, log.member)
  )
  seat_slot, seat_member = np.asarray(table.seat_slot), np.asarray(
      table.seat_member
  )
  acted = actor >= 0
  assert acted.sum() > 0
  rows = np.broadcast_to(np.arange(b)[:, None], actor.shape)
  np.testing.assert_array_equal(
      slot[acted], seat_slot[rows[acted], actor[acted]]
  )
  np.testing.assert_array_equal(
      member[acted], seat_member[rows[acted], actor[acted]]
  )
  side_of = np.asarray(sides.side_of_seat)
  # row r: the learner holds side r % 2
  mine = acted & (side_of[np.maximum(actor, 0)] == rows % 2)
  assert mine.any() and (slot[mine] == 0).all() and (member[mine] == 0).all()
  assert (
      slot[acted & ~mine] == 1
  ).all()  # the drawn side: a pool member, never the learner


@pytest.mark.parametrize("game", ["bridge", "team_pennies_2v1"])
def test_per_seat_lineup_runs_in_the_arena(game):
  """A mixed side.

  The learner and a fixed partner (slot 2) against a pool side (slot 1); bridge
  sides, and team_pennies_2v1's uneven (0, 1, 0).

  Args:
    game: the env's name.
  """
  if game == "bridge":
    env = bridge.BridgeEnv()
  else:
    env = nplayer_matrix.NPlayerMatrixEnv(game)
  sides = sides_lib.sides_of(env)
  p = env.num_players
  pol = uniform.UniformPolicy(env.name, env.num_actions, p)
  k, b = 3, 8
  arena = arena_lib.Arena(
      env,
      [
          arena_lib.Slot(pol, 1, types.SAMPLE),
          arena_lib.Slot(pol, k, types.SAMPLE),
          arena_lib.Slot(pol, 1, types.SAMPLE),
      ],
  )
  where = {"x@1": (1, 0), "y@2": (1, 1), "z@3": (1, 2), "partner@4": (2, 0)}
  mix = sampler.Mix(0.0, {"x@1": 0.5, "y@2": 0.25, "z@3": 0.25})
  lu = sides_lib.lineup((("learner", "fixed"), "pool"), members=("partner@4",))
  table, _ = sampler.draw_table(mix, where, b, sides, rng.key(5), lineup=lu)
  params = [{}, {}, {}]
  arena.validate(params, table)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, p)
  _, batch, log = arena.collect(params, table, dk, pk, 4 * (env.max_len + 1))
  assert np.asarray(batch.valid).shape[0] == b  # one perspective seat per row
  actor, slot = np.asarray(log.actor), np.asarray(log.slot)
  persp = np.asarray(table.perspective_seat)
  acted = actor >= 0
  rows = np.broadcast_to(np.arange(b)[:, None], actor.shape)
  learner_ply = acted & (actor == persp[rows])
  partner_seat = np.asarray([sides.seats(sides.side(int(q)))[1] for q in persp])
  partner_ply = acted & (actor == partner_seat[rows])
  assert learner_ply.any() and partner_ply.any()
  assert (slot[learner_ply] == 0).all() and (slot[partner_ply] == 2).all()
  assert (slot[acted & ~learner_ply & ~partner_ply] == 1).all()


class _Probe(kinds_base.Driver):
  kind = "probe"


def test_check_table_keeps_one_perspective_shape():
  """Check table keeps one perspective shape."""
  d = _Probe()
  t1 = arena_lib.self_play_table(8, 4)
  d.check_table(t1)
  d.check_table(t1)
  with pytest.raises(ValueError, match="one seating per Arena"):
    d.check_table(arena_lib.self_play_table(8, 4, all_seats=True))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
