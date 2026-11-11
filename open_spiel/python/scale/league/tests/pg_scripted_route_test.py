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

"""kinds.PG(scripted_route=...): the route a run's scripted endgame plays on.

* "" (the default) is the endgame's own route and is left out of the config
  hash, so every config built before the field hashes as before;
* a route must be one the endgame declares, and needs an endgame;
* the driver routes the learner slot and bare fixed opponents on it, keeps an
  already routed opponent as it is, and counts the learner decisions the
  endgame's own route would have scripted.
"""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.jax.games.gin_rummy import cards_and_melds as cm
from open_spiel.python.scale.core import arena as arena_lib
from open_spiel.python.scale.core import types
from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import rule_bot as gin_rule_bot
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league.kinds import pg as pg_kind
from open_spiel.python.scale.learners import pg
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform
from open_spiel.python.scale.sources import decisions

AGENT = gin_endgame.AGENT_KNOCK_ROUTE


def test_default_route_is_not_hashed():
  """An empty scripted_route hashes as the class without the field."""
  spec = kinds.PG(
      pg.PGConfig(num_updates=10), scripted_endgame=gin_endgame.ARCH
  )
  d = common.config_dict(spec)
  assert "scripted_route" not in d
  default_route = common.plain(spec)["scripted_route"]
  assert isinstance(default_route, str) and not default_route, default_route
  agent = dataclasses.replace(spec, scripted_route=AGENT)
  assert common.config_dict(agent)["scripted_route"] == AGENT
  assert common.config_hash(agent) != common.config_hash(spec)
  # The route the endgame plays by default, written out, is a different config.
  own = dataclasses.replace(spec, scripted_route=gin_endgame.ROUTE)
  assert common.config_hash(own) != common.config_hash(spec)


def test_routes_are_validated():
  """Only a declared route of the run's endgame is accepted."""
  assert pg_kind.scripted_routes(gin_endgame.ARCH) == gin_endgame.ROUTES
  with pytest.raises(ValueError, match="is not a route of"):
    kinds.PG(
        pg.PGConfig(num_updates=10),
        scripted_endgame=gin_endgame.ARCH,
        scripted_route=f"{gin_endgame.__name__}:is_knock_discard",
    )
  with pytest.raises(ValueError, match="needs a scripted_endgame"):
    kinds.PG(pg.PGConfig(num_updates=10), scripted_route=AGENT)


def _driver(route, base=None):
  """A gin PGDriver on scripted route `route`.

  Args:
    route: The scripted route.
    base: The base policy (default uniform).

  Returns:
    (env, base, driver).
  """
  env = gin.GinEnv(oklahoma_frac=1.0)
  if base is None:
    base = uniform.UniformPolicy("gin_rummy", cm.NUM_ACTIONS, 2)
  return (
      env,
      base,
      pg_kind.PGDriver(
          base, object(), gin_endgame.ARCH, env=env, scripted_route=route
      ),
  )


def test_driver_routes_slots_on_the_run_route():
  """Learner slot and bare opponents on the run's route; routed ones kept."""
  _, base, d = _driver(AGENT)
  assert d.slot_policy.spec.arch_kwargs["route"] == AGENT
  opp, _ = d.wrap_opponent(base, {})
  assert opp.spec.arch_kwargs["route"] == AGENT
  default = routed.with_scripted_phase(base, gin_endgame.ARCH)
  assert d.wrap_opponent(default, default.make_params({}, {}))[0] is default
  _, base, d0 = _driver("")
  assert d0.slot_policy.spec.arch_kwargs["route"] == gin_endgame.ROUTE
  opp0, _ = d0.wrap_opponent(base, {})
  assert opp0.spec.arch_kwargs["route"] == gin_endgame.ROUTE
  with pytest.raises(ValueError, match="needs a scripted_endgame"):
    pg_kind.PGDriver(base, object(), "", scripted_route=AGENT)


def test_driver_counts_the_agent_endgame_decisions():
  """train_step's agent_endgame_decisions = kept knock-discard rows."""
  # The rule bot knocks whenever it may, so the rollout has knock discards.
  env, _, d = _driver(AGENT, gin_rule_bot.GinRuleBotPolicy())
  pol = d.slot_policy
  ar = arena_lib.Arena(env, [arena_lib.Slot(pol, 1, types.SAMPLE)])
  b, t = 32, 300
  table = arena_lib.self_play_table(b, 2, all_seats=True)
  dk, pk = arena_lib.independent_keys(jax.random.key(5), b, 2)
  params = pol.make_params({}, {})
  _, tb, _ = ar.collect(
      [jax.tree_util.tree_map(lambda x: x[None], params)], table, dk, pk, t
  )
  knock11 = np.asarray(gin_endgame.is_knock_discard(tb.steps.info))
  valid = np.asarray(tb.valid)
  assert (valid & knock11).sum() > 0
  # The count train_step reports, computed as it does.
  out, _ = decisions.exclude_override(pol, tb)
  # The rows the endgame's own route scripts, as the driver builds that route.
  default = routed.with_scripted_phase(d.policy, gin_endgame.ARCH)
  own = np.asarray(routed.override_rows(default, out.steps.info))
  kept = int((np.asarray(out.valid) & own).sum())
  assert kept == int((valid & knock11).sum())


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
