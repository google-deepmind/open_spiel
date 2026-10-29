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

"""The gin reference evaluation.

An entrant against the rule bot, both routed to the game's scripted endgame,
greedy, on duplicate deals.
"""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.envs.gin import endgame as gin_endgame
from open_spiel.python.scale.envs.gin import (
    reference_eval as gin_reference_eval,
)
from open_spiel.python.scale.envs.gin import rule_bot as rule_bot_policy
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.eval import registry
from open_spiel.python.scale.policies import routed
from open_spiel.python.scale.policies import uniform


@pytest.fixture(name="ev", scope="module")
def _ev_fixture():
  """The gin rule-bot evaluator.

  Returns:
    The evaluator.
  """
  return registry.build("gin_rummy.rule_bot", gin.GinEnv(oklahoma_frac=1.0))


def test_config_shape(ev):
  """Config shape.

  Args:
    ev: The module `ev` fixture.
  """
  cfg = ev.config
  assert cfg.name == "gin_rummy.rule_bot"
  assert cfg.duplicate and cfg.hands_per_match == 1 and cfg.rule == types.ARGMAX
  assert isinstance(ev.env, gin.GinEnv) and ev.env.oklahoma_frac == 1.0
  ref = cfg.reference()
  assert isinstance(ref.policy, rule_bot_policy.GinRuleBotPolicy)
  wrapped = cfg.wrap(ref)
  assert isinstance(wrapped.policy, routed.RoutedPolicy)
  assert isinstance(
      wrapped.policy.override, gin_endgame.GinScriptedEndgamePolicy
  )
  assert (
      wrapped.policy.override.spec.arch_name
      == env_registry.game("gin_rummy").endgame
  )
  assert wrapped.name == ref.name


def test_refuses_another_env():
  """Refuses another env."""
  with pytest.raises(TypeError, match="GinEnv"):
    gin_reference_eval.evaluator(env_registry.make("kuhn_poker"))


def test_rule_bot_against_itself_is_exactly_zero(ev):
  """Rule bot against itself is exactly zero.

  Deterministic play on duplicate deals is mirrored game by game, so every
  pair sums to 0 (known answer).

  Args:
    ev: The module rule-bot reference evaluator fixture.
  """
  r = ev.vs_reference(
      gin_reference_eval.rule_bot_entrant(), 4, jax.random.key(0)
  )
  assert r.b == "gin_rule_bot"
  np.testing.assert_array_equal(r.a_payoff.sum(1), 0.0)
  assert r.mean.mean == 0.0 and r.mean.se == 0.0
  assert np.any(r.a_payoff != 0.0), "some hand must score"


def test_uniform_subject_loses_to_the_rule_bot(ev):
  """Uniform subject loses to the rule bot.

  Args:
    ev: The module `ev` fixture.
  """
  # keep the subject's SAMPLE rule
  cfg = dataclasses.replace(ev.config, rule=None)
  ev2 = match.Evaluator(ev.env, cfg)
  pol = uniform.UniformPolicy("gin_rummy", ev.env.num_actions, 2)
  r = ev2.vs_reference(
      match.Entrant("uniform", pol, {}, types.SAMPLE), 8, jax.random.key(1)
  )
  assert r.mean.hi < 0.0, r.mean


def test_route_is_idempotent_for_an_already_routed_entrant():
  """Route is idempotent for an already routed entrant.

  An entrant that already routes the scripted endgame (a gin training
  artifact) is left as is instead of routed twice.
  """
  spec = env_registry.game("gin_rummy")
  once = gin_reference_eval.route_endgame(
      gin_reference_eval.rule_bot_entrant(), spec
  )
  twice = gin_reference_eval.route_endgame(once, spec)
  assert twice.policy is once.policy and twice.params is once.params
  assert not isinstance(once.policy.base, routed.RoutedPolicy)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
