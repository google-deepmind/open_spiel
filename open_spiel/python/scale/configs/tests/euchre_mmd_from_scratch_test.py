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

"""configs.euchre_mmd_from_scratch: the recipe, rules and from-scratch tests."""

import dataclasses
import sys

import pytest

from open_spiel.python.scale import deps

# The euchre config imports the optional JAX games package.
pytest.importorskip(deps.GAMES_PACKAGE)

# pylint: disable=g-import-not-at-top,wrong-import-position
from open_spiel.python.scale.configs import euchre_mmd_from_scratch as ems
from open_spiel.python.scale.core import sides as sides_lib
from open_spiel.python.scale.envs import euchre
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import kinds
from open_spiel.python.scale.league import refs

# pylint: enable=g-import-not-at-top,wrong-import-position


def test_recipe():
  """The registered config is the module's recipe."""
  cfg = common.load_config("euchre_mmd_from_scratch")
  assert cfg == ems.get_config()
  assert cfg.env == "euchre" and dict(cfg.env_kwargs) == euchre.OPTIONS
  assert cfg.controller == "scheduled_promotion"
  assert cfg.run_updates() == ems.NUM_UPDATES
  assert cfg.league.num_envs == ems.NUM_ENVS
  assert cfg.league.num_steps == ems.NUM_STEPS >= euchre.MAX_LEN
  lc = cfg.learner.learner
  assert lc.loss == "mmd" and lc.num_updates == ems.NUM_UPDATES
  assert lc.value_scale == ems.VALUE_SCALE == 4.0 and lc.gamma == 1.0
  assert lc.max_decision_share is None
  assert [(e.name, e.every, e.deals, e.design) for e in cfg.eval] == [
      ("vs_uniform", ems.UNIFORM_EVERY, ems.UNIFORM_DEALS, "auto")
  ]
  assert cfg.metadata["rules_sha"] == euchre.rules_sha(euchre.OPTIONS)
  assert cfg.metadata["euchre_rules"] == "standard"
  env = common.build_env(cfg)
  assert sides_lib.sides_of(env) == sides_lib.Sides.partnerships(
      4, 2, constant_sum=True
  )


def test_run_length_override_moves_the_lr_anneal():
  """--num_updates sets the run length and the LR anneal's horizon."""
  cfg = common.with_overrides(
      common.load_config("euchre_mmd_from_scratch"), num_updates=1234
  )
  assert cfg.run_updates() == 1234
  assert cfg.learner.learner.num_updates == 1234


def test_rules_check_refuses_other_options():
  """A non-default rule option fails check_standard_rules."""
  cfg = ems.get_config()
  for k, v in ems.RULES.items():
    other = dataclasses.replace(cfg, env_kwargs={**ems.RULES, k: not v})
    with pytest.raises(AssertionError):
      ems.check_standard_rules(other)


def test_from_scratch_check_refuses_a_learner_without_a_magnet():
  """A PPO learner (no magnet re-taken from its own params) is refused."""
  cfg = ems.get_config()
  lc = dataclasses.replace(
      cfg.learner.learner, loss="ppo", kl_coef=ems.schedules.constant(0.0)
  )
  ppo = dataclasses.replace(
      cfg, learner=dataclasses.replace(cfg.learner, learner=lc)
  )
  assert isinstance(ppo.learner, kinds.PG)
  with pytest.raises(AssertionError):
    ems.check_from_scratch(ppo)


_EXTERNAL = "external/euchre@1"


def _learner(cfg, **kw):
  """The config with its learner config's fields `kw` replaced.

  Args:
    cfg: a league config.
    **kw: learner config fields.

  Returns:
    The new config.
  """
  return dataclasses.replace(
      cfg, learner=dataclasses.replace(cfg.learner, **kw)
  )


def _pg(cfg, **kw):
  """The config with its inner (PG) learner config's fields `kw` replaced.

  Args:
    cfg: a league config.
    **kw: PG learner config fields.

  Returns:
    The new config.
  """
  return _learner(cfg, learner=dataclasses.replace(cfg.learner.learner, **kw))


#: Configs refused by check_from_scratch, one per guarantee it asserts
#: (the PPO learner, no magnet, is the test above).
_NOT_FROM_SCRATCH = {
    "warm_start": lambda c: _learner(
        c, init=refs.PolicyRef(artifact=_EXTERNAL)
    ),
    # parents without init cannot be built (kinds.PG), so init covers both.
    "warm_start_with_parents": lambda c: _learner(
        c, init=refs.PolicyRef(artifact=_EXTERNAL), parents=(_EXTERNAL,)
    ),
    "num_initial": lambda c: dataclasses.replace(c, num_initial=1),
    "fixed_opponents": lambda c: dataclasses.replace(
        c,
        fixed_opponents=(
            refs.FixedOpponent("ext", refs.PolicyRef(artifact=_EXTERNAL)),
        ),
    ),
    "controller_kwargs": lambda c: dataclasses.replace(
        c, controller_kwargs={**c.controller_kwargs, "pool_source": "ext"}
    ),
    # PGConfig allows magnet_from_behavior only at reference_update_every 1.
    "magnet_from_behavior": lambda c: _pg(
        c, magnet_from_behavior=True, reference_update_every=1
    ),
    "kl_init_to_uniform": lambda c: _pg(
        c,
        pg=dataclasses.replace(c.learner.learner.pg, kl_init_to_uniform=True),
    ),
}


@pytest.mark.parametrize("case", sorted(_NOT_FROM_SCRATCH))
def test_from_scratch_check_refuses(case):
  """Each from-scratch assertion refuses the one config that breaks it."""
  cfg = ems.get_config()
  ems.check_from_scratch(cfg)
  bad = _NOT_FROM_SCRATCH[case](cfg)
  with pytest.raises(AssertionError):
    ems.check_from_scratch(bad)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
