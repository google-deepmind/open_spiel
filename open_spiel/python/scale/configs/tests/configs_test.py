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

"""The shipped example configs load, hash and build.

The shipped example configs: exactly this set ships, and each loads as a run
config of the stated learner kind on its game, hashes, and builds its env and
policy.
"""

import sys

import pytest

from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.policies import registry

SHIPPED = {
    "mmd_kuhn_poker_jax": ("pg", "kuhn_poker_jax"),
    "mmd_kuhn_poker": ("pg", "kuhn_poker"),
    "mmd_leduc_poker": ("pg", "leduc_poker"),
    "mmd_gin_rummy": ("pg", "gin_rummy"),
    "ppo_tic_tac_toe": ("pg", "tic_tac_toe"),
    "az_connect_four": ("az", "connect_four"),
    "bc_connect_four": ("bc", "connect_four"),
    "psro_connect_four": ("pg", "connect_four"),
}


def test_the_shipped_configs_are_exactly_these():
  """The shipped configs are exactly these."""
  assert set(SHIPPED) <= set(common.config_names())
  pkg = common.__name__.rsplit(".", 2)[0] + ".configs."
  own = {n for n, m in common.config_modules().items() if m.startswith(pkg)}
  assert own == set(SHIPPED), own


def test_discovery_ignores_a_generated_file_beside_the_configs(
    tmp_path, monkeypatch
):
  """A .py file without get_config in a configs package is not a config.

  A test runner may generate wrapper files next to the configs; discovery
  reads each file for a top-level get_config instead of listing every .py.

  Args:
    tmp_path: where the configs package is written.
    monkeypatch: restores the registered packages and sys.path.
  """
  pkg = tmp_path / "gen_cfgs"
  pkg.mkdir()
  (pkg / "__init__.py").write_text("")
  (pkg / "gen_kuhn.py").write_text("def get_config():\n  return None\n")
  (pkg / "gen_alias.py").write_text(
      "from gen_cfgs.gen_kuhn import get_config\n"
  )
  (pkg / "gen_assigned.py").write_text("get_config = lambda: None\n")
  (pkg / "configs_test_wrapper.py").write_text(
      "import sys\nimport pytest\nsys.exit(pytest.main([__file__]))\n"
  )
  (pkg / "mentions_only.py").write_text(
      "def f():\n  def get_config():\n    pass\n# get_config\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  # Reaches private _PACKAGES of the code under test.
  packages = list(common._PACKAGES)  # pylint: disable=protected-access
  monkeypatch.setattr(common, "_PACKAGES", packages)
  common.register_config_package("gen_cfgs")
  own = {n for n, m in common.config_modules().items() if m.startswith("gen_")}
  assert own == {"gen_kuhn", "gen_alias", "gen_assigned"}, own


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_config_loads_hashes_and_builds(name):
  """Shipped config loads hashes and builds."""
  cfg = common.load_config(name)
  kind, env = SHIPPED[name]
  assert isinstance(cfg, runner.LeagueRunConfig) and cfg.name == name
  assert common.learner_kind(cfg) == kind and cfg.env == env
  assert not cfg.metadata
  assert (
      len(common.config_hash(cfg)) == 64 and len(common.resume_hash(cfg)) == 64
  )
  if kind in ("pg", "bc"):

    pol = registry.build(cfg.policy_arch, dict(cfg.policy_kwargs))
    assert pol.spec.game == env
  common.build_env(cfg)


def test_psro_config_runs_the_psro_controller():
  """Psro config runs the psro controller."""
  assert common.load_config("psro_connect_four").controller == "psro"


def test_overrides_set_the_run_length_geometry_and_seed():
  """Overrides set the run length geometry and seed."""
  cfg = common.with_overrides(
      common.load_config("mmd_kuhn_poker_jax"),
      num_updates=7,
      num_envs=8,
      num_steps=4,
  )
  assert (
      cfg.run_updates() == 7
      and cfg.league.num_envs == 8
      and cfg.league.num_steps == 4
  )
  assert common.apply_overrides(cfg, ["seed=3"]).seed == 3
  with pytest.raises(ValueError, match="num_updates"):
    common.apply_overrides(cfg, ["num_iterations=9"])


def test_overrides_widen_int_to_float_only():
  """Overrides widen int to float only."""
  cfg = common.load_config("mmd_kuhn_poker_jax")
  got = common.apply_overrides(cfg, ["learner.learner.lr=1"]).learner.learner.lr
  assert got == 1.0 and isinstance(got, float)
  with pytest.raises(TypeError, match="league.num_envs"):
    common.apply_overrides(cfg, ["league.num_envs=64.5"])


def test_az_config_overrides_reach_the_az_settings():
  """Az config overrides reach the az settings."""
  cfg = common.with_overrides(
      common.load_config("az_connect_four"), num_envs=8, seed=5
  )
  assert (
      cfg.learner.az.batch_size == 8
      and cfg.learner.az.seed == 5
      and cfg.league.num_envs == 8
  )
  with pytest.raises(ValueError, match="num_steps"):
    common.with_overrides(cfg, num_steps=4)
  with pytest.raises(ValueError, match="num_updates"):
    common.apply_overrides(cfg, ["learner.az.num_iterations=3"])


def test_unknown_env_lists_the_registered_ones():
  """Unknown env lists the registered ones."""
  with pytest.raises(KeyError, match="kuhn_poker_jax"):
    common.build_env_by_name("no_such_game")


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
