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

"""LeagueConfig.fixed_layout: how the League seats its fixed opponents.

"stacked" (the default) seats fixed opponents of one (spec, rule) in one
Arena slot of stacked members; "slots" seats each in its own slot of one
member, so no slot gathers params per row. The field is left out of the
config hash at its default (core.hash_compat), a run trains under either
layout, and an unknown layout is refused.
"""

import dataclasses
import sys

import pytest

from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import loop
from open_spiel.python.scale.league import refs
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.league.tests import league_fixtures


def _uniform(name: str) -> refs.FixedOpponent:
  """A uniform-random kuhn opponent named `name`.

  Args:
    name: the opponent's name.

  Returns:
    The refs.FixedOpponent.
  """
  return refs.FixedOpponent(
      name,
      refs.PolicyRef(
          arch_name="uniform",
          arch_kwargs={
              "game": "kuhn_poker",
              "num_actions": 2,
              "num_players": 2,
          },
      ),
  )


def _config(layout: str = "stacked") -> runner.LeagueRunConfig:
  """A kuhn run against three uniform opponents of one spec.

  Args:
    layout: the LeagueConfig.fixed_layout.

  Returns:
    The run config.
  """
  learner = league_fixtures.br_learner(league_fixtures.kuhn_policy(8)).cfg
  cfg = runner.one_slot(
      "fixed_layout",
      "kuhn_poker",
      kinds.PG(learner),
      num_envs=8,
      num_steps=4,
      num_updates=1,
      policy_kwargs=league_fixtures.kuhn_policy_kwargs(8),
      self_play=False,
      opponents=tuple(_uniform(n) for n in ("u_a", "u_b", "u_c")),
  )
  if layout == "stacked":
    return cfg
  return dataclasses.replace(
      cfg, league=dataclasses.replace(cfg.league, fixed_layout=layout)
  )


def test_stacked_is_one_slot_and_slots_one_per_opponent(tmp_path):
  stacked = runner.build(_config(), str(tmp_path / "a"))
  assert [p.names for p in stacked.fixed] == [("u_a", "u_b", "u_c")]
  per = runner.build(_config("slots"), str(tmp_path / "b"))
  assert [p.names for p in per.fixed] == [("u_a",), ("u_b",), ("u_c",)]
  assert len(per.opponent_slots()) == len(stacked.opponent_slots()) + 2


def test_the_default_is_left_out_of_the_hash():
  cfg, per = _config(), _config("slots")
  assert "fixed_layout" not in loop.hashed_config(cfg.league)
  assert loop.hashed_config(per.league)["fixed_layout"] == "slots"
  assert "fixed_layout" not in common.config_dict(cfg)["league"]
  assert common.config_hash(cfg) != common.config_hash(per)


def test_a_run_trains_under_the_slots_layout(tmp_path):
  res = runner.run(_config("slots"), str(tmp_path / "ck"))
  ls = res.league.learners["main"]
  assert int(ls.updates) == 1
  assert set(ls.mix.weights) == {"u_a", "u_b", "u_c"}


def test_an_unknown_layout_is_refused():
  with pytest.raises(ValueError, match="fixed_layout"):
    _config("per_name")


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-p", "no:cacheprovider"]))
