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

"""Arena integration over liars_dice, phantom_ttt_ir and bargaining.

Integration over liars_dice, phantom_ttt_ir and bargaining (each env
passes check_env alone, each encoder its contract): the core Arena with the
uniform policy completes hands with sane returns (zero-sum where declared;
bargaining in [0, 10] per seat), and a NetworkPolicy built by registry name
on each game's research encoder passes check_policy on real SeatSteps.
"""

import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.encoders import registry as enc_registry
from open_spiel.python.scale.envs import bargaining
from open_spiel.python.scale.envs import liars_dice
from open_spiel.python.scale.envs import phantom_ttt_ir
from open_spiel.python.scale.policies import registry as policy_registry
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.testing import check_policy
from open_spiel.python.scale.testing import steps as steps_lib

ENVS = {
    "liars_dice": (liars_dice.LiarsDiceEnv, "liars_dice_info_state"),
    "phantom_ttt_ir": (
        phantom_ttt_ir.PhantomTTTIREnv,
        "phantom_ttt_ir_observation",
    ),
    "bargaining": (bargaining.BargainingEnv, "bargaining_info_state"),
}


@pytest.mark.parametrize("game", sorted(ENVS))
def test_uniform_self_play_hands_complete(game):
  """Uniform self play hands complete.

  Args:
    game: The parametrized case.
  """
  env = ENVS[game][0]()
  pol = uniform.UniformPolicy(env.name, env.num_actions, env.num_players)
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 64, 3 * (env.max_len + 1)
  table = arena_lib.self_play_table(b, 2)
  dk, pk = arena_lib.independent_keys(jax.random.key(1), b, 2)
  params = [
      jax.tree_util.tree_map(
          lambda x: x[None], pol.init_params(jax.random.key(0))
      )
  ]
  _, batch, log = ar.collect(params, table, dk, pk, t)
  done = np.asarray(log.done)
  assert done.sum() >= b
  assert not np.asarray(log.bad).any()
  r = np.asarray(log.rewards)[done]
  if env.zero_sum:
    np.testing.assert_allclose(r.sum(-1), 0.0)
    assert np.isin(r, [-1.0, 0.0, 1.0]).all()
  else:
    assert ((r >= 0.0) & (r <= 10.0)).all() and (r > 0).any()
  assert batch.action.dtype == np.int32


@pytest.mark.parametrize("game", sorted(ENVS))
def test_network_policy_on_the_research_encoder(game):
  """Network policy on the research encoder.

  Args:
    game: The parametrized case.
  """
  ctor, enc_name = ENVS[game]
  env = ctor()
  enc = enc_registry.get(enc_name)
  net = {
      "compute_dtype": "float32",
      "torso": {"kind": "mlp", "widths": [16, 16], "layout": "shared"},
      "policy_head": {"kind": "dense"},
      "value_head": {"kind": "scalar"},
  }
  pol = policy_registry.build(
      "network",
      dict(
          game=env.name,
          obs_format=enc.name,
          num_actions=env.num_actions,
          num_players=env.num_players,
          encoder=enc.name,
          obs_shape=list(enc.shape),
          network=net,
          value_layout="seat",
      ),
  )
  pa = pol.init_params(jax.random.key(1))
  pb = jax.tree.map(lambda x: x + 0.5, pa)
  check_policy.check_policy(
      pol, pa, pb, *steps_lib.two_ply_steps(env, batch=8, seat=1)
  )


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
