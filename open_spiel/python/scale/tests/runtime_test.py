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

"""The runtime target holds every module the registries import by name.

The registries (envs, encoders, policies, belief samplers, slot factories,
configs, JAX games) import their built-in modules from string tables, so a
missing build dependency only shows when a name is first used. This test
resolves every built-in name, with only the runtime target (and the
registries) as dependencies: a module added to a table but not to the
runtime target fails here, not in a user's binary. The GUI's checkpoint
controller, which OpenSpiel's game GUI imports by module name, is checked the
same way.
"""

from open_spiel.python.jax.games import registry as jax_games
from open_spiel.python.scale.belief import registry as belief
from open_spiel.python.scale.encoders import registry as encoders
from open_spiel.python.scale.envs import registry as envs
from open_spiel.python.scale.league import config_common
from open_spiel.python.scale.league import slot_factories
from open_spiel.python.scale.policies import registry as policies


def test_every_env_resolves_with_its_extras_and_components():
  for name in envs.builtin_names():
    g = envs.game(name)
    envs.resolve(g.ctor)
    envs.load_extras(name)
    for field in (g.canon, g.pyspiel_history_to_raw, g.raw_history_to_pyspiel):
      if field is not None:
        envs.resolve(field)
    for kind, component, _ in g.components:
      if kind == belief.COMPONENT_KIND:
        belief.get(component)
      elif kind == slot_factories.COMPONENT_KIND:
        slot_factories.get(component, {})


def test_every_encoder_policy_and_sampler_resolves():
  for name in encoders.names():
    if name not in encoders.families():
      encoders.get(name)
  for name in policies.registered():
    policies.policy_class(name)
  for name in belief.names():
    belief.get(name)


def test_every_config_and_jax_game_imports():
  for name in config_common.config_names():
    assert config_common.config_module(name) is not None, name
  for name in jax_games.registered_names():
    # Every built-in game lives in the module of its name. Importing it is
    # enough: some games need non-default parameters to build.
    jax_games.load_game_module(name)


def test_the_gui_controller_resolves_by_name():
  # OpenSpiel's game GUI imports the checkpoint controller by module name
  # (--controllers <package>.serve.controller), so it is in the runtime too.
  package = envs.__name__.rpartition(".envs.")[0]
  register_into = envs.resolve(f"{package}.serve.controller:register_into")
  assert callable(register_into)
