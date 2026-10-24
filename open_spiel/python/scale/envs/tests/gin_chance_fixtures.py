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

"""gin_test's engine-chance known answer, run as SHARDS files.

The check runs once per oklahoma_frac in OKLAHOMA_FRACS, each about half of
gin_test's time, so gin_test runs one of them and gin_shard<k>_test.py the
others: each file stays within the downstream per-file test budget.
"""

import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.jax.games.gin_rummy import game as gin_game
from open_spiel.python.scale.envs import gin

#: Every oklahoma_frac the engine-chance check covers.
OKLAHOMA_FRACS = (None, 1.0)

#: Shard k's oklahoma_fracs: shard 0 is gin_test, shard k is
#: gin_shard<k>_test.py.
SHARDS = {0: (None,), 1: (1.0,)}


def _two_step_chance(g, o, completes, first_player):
  """The two-step reference form of an engine chance step.

  Step, then install_first_player when the deal completes.

  Args:
    g: The engine game state.
    o: The chance outcome.
    completes: Whether the outcome completes the deal.
    first_player: The first player to install.

  Returns:
    The engine game state after the step.
  """
  g2 = gin_game.step(g, o)
  return jax.tree_util.tree_map(
      lambda a, b: jnp.where(completes, a, b),
      gin.install_first_player(g2, first_player),
      g2,
  )


def check_one_step_engine_chance(okla):
  """One step engine chance equals step then install.

  Known answer for the throughput path: every engine chance event (each
  deal card, the deal-completing upcard for both first players, stock draws)
  through _engine_chance_step equals step + install_first_player, every
  field bit for bit, over random hands played on with uniform actions.

  Args:
    okla: The env's oklahoma_frac.
  """
  env = gin.GinEnv(oklahoma_frac=okla)
  checked = {"deal": 0, "complete": 0, "stock": 0}
  firsts = set()
  for seed in range(12):
    key = jax.random.key(100 + seed)
    s = env.root()
    for ply in range(400):
      key, k = jax.random.split(key)
      if bool(env.terminal(s)):
        break
      if bool(env.is_chance(s)):
        o = jax.random.categorical(
            k, jnp.log(env.chance_probs(s) + 1e-30)
        ).astype(jnp.int32)
        if int(s.stage) == gin.STAGE_ENGINE:
          g = s.game
          completes = (g["prev_player"] < 0) & (g["deal_cards_dealt"] == 20)
          # Reaches private _engine_chance_step of the code under test.
          got = gin._engine_chance_step(  # pylint: disable=protected-access
              g, o, completes, s.first_player
          )
          want = _two_step_chance(g, o, completes, s.first_player)
          assert set(got) == set(want)
          for name in want:
            a, b = np.asarray(got[name]), np.asarray(want[name])
            assert a.dtype == b.dtype and a.shape == b.shape, name
            np.testing.assert_array_equal(
                a, b, err_msg=f"seed {seed} ply {ply} {name}"
            )
          if bool(completes):
            kind = "complete"
          elif int(g["prev_player"]) < 0:
            kind = "deal"
          else:
            kind = "stock"
          checked[kind] += 1
          if bool(completes):
            firsts.add(int(s.first_player))
        s = env.apply_chance(s, o)
      else:
        legal = np.nonzero(np.asarray(env.legal_mask(s)))[0]
        s = env.apply(
            s, jnp.int32(legal[int(jax.random.randint(k, (), 0, len(legal)))])
        )
  assert firsts == {0, 1}, firsts
  assert (
      checked["complete"] == 12
      and checked["deal"] == 12 * 20
      and checked["stock"] > 50
  ), checked
