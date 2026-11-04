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

"""Subprocess body of arena_shard_map_test (8 forced host devices).

1. The shard_map Arena at 8 and 2 devices equals the 1-device Arena bit for
   bit (state, TrajectoryBatch, PlyLog), on: kuhn self-play with a stateful
   policy; kuhn with a CALLBACK_BATCHED K=2 slot; leduc duplicate deals with a
   K=3 population slot and 2-hand matches.
2. The compiled reset and rollout at 8 devices contain no collective.
3. Placement (known answer): device d holds global rows [d*R/8, (d+1)*R/8) of
   the TrajectoryBatch, equal to those rows of the 1-device run.
4. No recompiles: an ArenaSource over 8 devices compiles on its first next()
   and never again over 4 more calls with new params each time (XLA compile
   events, parallel/compiles.py).
5. A batch not divisible by 8 is refused.
"""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import frameworks
from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.envs import leduc
from open_spiel.python.scale.parallel import compiles
from open_spiel.python.scale.parallel import data_parallel as dpl
from open_spiel.python.scale.testing import fixtures
from open_spiel.python.scale.testing import multidevice

tree_map = jax.tree_util.tree_map
KUHN_INDEX = f"{kuhn.__name__}:info_index"


def stack(params, k=1):
  """Params stacked K times on a new leading member axis.

  Args:
    params: one member's params.
    k: K, the member count.

  Returns:
    the stacked params.
  """
  return tree_map(lambda x: jnp.stack([x] * k), params)


def kuhn_self_play():
  """A kuhn HistoryPolicy self-play case.

  Returns:
    (env, slots, params, table, deal keys, policy keys, T, hands per match).
  """
  env = kuhn.KuhnEnv()
  pol = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  params = [
      stack(
          pol.make_params(
              b=np.array([0.0, 0.7], np.float32),
              w=np.array([[0.3, -0.2], [0.1, 0.4]], np.float32),
          )
      )
  ]
  b = 64
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(0), b, 2)
  return (
      env,
      [arena_lib.Slot(pol, 1, types.SAMPLE)],
      params,
      table,
      dk,
      pk,
      24,
      1,
  )


def kuhn_callback():
  """A kuhn case with a CALLBACK_BATCHED slot.

  Returns:
    (env, slots, params, table, deal keys, policy keys, T, hands per match).
  """
  env = kuhn.KuhnEnv()
  hist = fixtures.HistoryPolicy("kuhn_poker", 2, 2)
  cb = fixtures.CallbackTabularPolicy(
      "kuhn_poker", kuhn.NUM_INFO_STATES, 2, 2, KUHN_INDEX
  )
  cb.set_host_tables(
      np.random.default_rng(0)
      .normal(size=(2, kuhn.NUM_INFO_STATES, 2))
      .astype(np.float32)
  )
  b = 32
  member = (np.arange(b) % 2).astype(np.int32)
  table = arena_lib.SlotTable(
      seat_slot=np.tile(np.array([[0, 1]], np.int32), (b, 1)),
      seat_member=np.stack([np.zeros(b, np.int32), member], axis=1),
      perspective_seat=np.zeros(b, np.int32),
  )
  params = [
      stack(hist.make_params(b=np.array([0.2, -0.1], np.float32))),
      {"member": jnp.arange(2, dtype=jnp.int32)},
  ]
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 2)
  slots = [
      arena_lib.Slot(hist, 1, types.SAMPLE),
      arena_lib.Slot(cb, 2, types.SAMPLE),
  ]
  return env, slots, params, table, dk, pk, 20, 1


def leduc_population():
  """A leduc case with a population slot of 3 members.

  Returns:
    (env, slots, params, table, deal keys, policy keys, T, hands per match).
  """
  env = leduc.LeducEnv()
  a = env.num_actions
  learner = fixtures.HistoryPolicy("leduc_poker", a, 2, cross_hand=True)
  pop = fixtures.HistoryPolicy("leduc_poker", a, 2, cross_hand=True)
  rng = np.random.default_rng(2)
  members = [
      pop.make_params(
          b=rng.normal(size=a).astype(np.float32),
          w=rng.normal(size=(a, a)).astype(np.float32) * 0.3,
      )
      for _ in range(3)
  ]
  pop_params = tree_map(lambda *xs: jnp.stack(xs), *members)
  n_deals = 24  # x 2 rotations = 48 rows
  slots_line = [0, 1]
  table = arena_lib.rotated_table(n_deals, slots_line, [0, 0])
  # member of the population seat cycles over the 3 members per deal
  seat_member = np.asarray(table.seat_member).copy()
  seat_slot = np.asarray(table.seat_slot)
  for row in range(seat_member.shape[0]):
    seat_member[row][seat_slot[row] == 1] = (row // 2) % 3
  table = table._replace(seat_member=seat_member)
  dk, pk = arena_lib.duplicate_keys(jax.random.key(3), n_deals, 2)
  params = [
      stack(
          learner.make_params(b=np.linspace(-0.5, 0.5, a).astype(np.float32))
      ),
      pop_params,
  ]
  slots = [
      arena_lib.Slot(learner, 1, types.SAMPLE),
      arena_lib.Slot(pop, 3, types.SAMPLE),
  ]
  return env, slots, params, table, dk, pk, 40, 2


def collect(case, n):
  """Collects `case` on an n-device mesh.

  Args:
    case: a case tuple.
    n: the device count.

  Returns:
    (Arena, its collect output).
  """
  env, slots, params, table, dk, pk, t, hpm = case
  ar = arena_lib.Arena(env, slots, hands_per_match=hpm, mesh=dpl.batch_mesh(n))
  return ar, ar.collect(params, table, dk, pk, t)


def main():
  """Checks every case on 8 and 2 devices against 1 device.

  Raises:
    AssertionError: a batch that does not split over the devices was accepted.
  """
  multidevice.expect_devices(8)
  for name, build in (
      ("kuhn_self_play", kuhn_self_play),
      ("kuhn_callback", kuhn_callback),
      ("leduc_population", leduc_population),
  ):
    case = build()
    env, slots, params, table, dk, pk, t, hpm = case
    _, out1 = collect(case, 1)
    for n in (8, 2):
      ar, out = collect(case, n)
      multidevice.assert_trees_equal(out, out1, f"{name}: {n} devices vs 1")
      state, batch, log = out
      assert len(batch.action.sharding.device_set) == n, (
          name,
          batch.action.sharding,
      )
      assert len(log.action.sharding.device_set) == n
      assert (
          len(jax.tree_util.tree_leaves(state.env_state)[0].sharding.device_set)
          == n
      )
    # 2. no collective in the compiled 8-device programs.
    ar8 = arena_lib.Arena(
        env, slots, hands_per_match=hpm, mesh=dpl.batch_mesh(8)
    )
    # Reaches private members of the arena under test.
    # pylint: disable=protected-access
    jp, jt = ar8._replicate(list(params)), ar8._shard_rows(table)
    reset_txt = (
        ar8._reset_jit.lower(jp, jt, ar8._shard_rows(dk), ar8._shard_rows(pk))
        .compile()
        .as_text()
    )
    st = ar8.reset(params, table, dk, pk)
    roll_txt = (
        ar8._rollout_jit.lower(jp, jt, st, num_steps=t).compile().as_text()
    )
    # pylint: enable=protected-access
    for what, txt in (("reset", reset_txt), ("rollout", roll_txt)):
      found = {k: v for k, v in multidevice.collectives_in(txt).items() if v}
      assert not found, f"{name}: collectives in the sharded {what}: {found}"
    # 3. placement: device d holds rows [d*R/8, (d+1)*R/8).
    _, (_, batch8, _) = collect(case, 8)
    _, batch1, _ = out1
    r = batch1.action.shape[0]
    per = r // 8
    dev_index = {d: i for i, d in enumerate(dpl.batch_mesh(8).devices.flat)}
    for shard in batch8.action.addressable_shards:
      d = dev_index[shard.device]
      assert range(r)[shard.index[0]] == range(d * per, (d + 1) * per), (
          shard.index,
          d,
          per,
      )
      np.testing.assert_array_equal(
          np.asarray(shard.data),
          np.asarray(batch1.action)[d * per : (d + 1) * per],
      )
    print(
        f"{name}: 8 and 2 devices == 1 device; no collectives; rows placed by "
        "device"
    )

  # 4. no recompiles across iterations (new params every call).
  env, slots, params, table, dk, pk, t, hpm = kuhn_self_play()
  ar = arena_lib.Arena(env, slots, mesh=dpl.batch_mesh(8))
  src = arena_lib.ArenaSource(ar, params, table, t)
  with compiles.Counter() as first:
    src.next(jax.random.key(10))
  assert (
      first.count >= 2
  ), first.count  # reset and rollout (and small eager helpers)
  new_params = [
      [tree_map(lambda x, i=i: x + jnp.float32(0.01 * i), params[0])]
      for i in range(1, 5)
  ]  # built outside the counted block (eager adds compile)
  keys = [jax.random.key(10 + i) for i in range(1, 5)]
  with compiles.assert_no_compiles("ArenaSource.next, iterations 2-5"):
    for p, k in zip(new_params, keys):
      src.set_params(p)
      src.next(k)
  print(
      f"ArenaSource: iteration 1 compiled {first.count} programs, iterations "
      "2-5 compiled none"
  )

  # 5. batch not divisible by the device count.
  try:
    ar.validate(params, arena_lib.self_play_table(12, 2))
  except ValueError:
    pass
  else:
    raise AssertionError("batch 12 on 8 devices was accepted")
  print(multidevice.OK)


if __name__ == "__main__":
  # A script: x64 on before any array is made (here, not at import, which
  # changes no process-global state).
  frameworks.enable_x64()
  main()
