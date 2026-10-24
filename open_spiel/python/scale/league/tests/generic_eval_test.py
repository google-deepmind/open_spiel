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

"""Game-agnostic in-loop evals (league.runner.GENERIC_EVALS).

vs_uniform and vs_mcts on any 2-player game, InLoopEval.simulations (part of
the config hash), and known answers on tic-tac-toe.
"""

import dataclasses
import sys

import jax
import numpy as np
import pytest

from open_spiel.python.scale.core import types
from open_spiel.python.scale.envs import registry as env_registry
from open_spiel.python.scale.eval import match
from open_spiel.python.scale.league import config_common as common
from open_spiel.python.scale.league import runner
from open_spiel.python.scale.policies import rollout
from open_spiel.python.scale.policies import uniform


def test_inloop_eval_rules_and_keys():
  """Inloop eval rules and keys."""
  u = runner.InLoopEval("vs_uniform", every=10, deals=64)
  assert u.key == "vs_uniform" and u.simulations == 0
  m = runner.InLoopEval("vs_mcts", every=10, deals=8, simulations=1000)
  assert m.key == "vs_mcts1000"
  assert (
      runner.InLoopEval(
          "vs_mcts", every=1, deals=2, simulations=5, label="m"
      ).key
      == "m"
  )
  with pytest.raises(ValueError, match="simulations"):
    runner.InLoopEval("vs_mcts", every=1, deals=2)
  with pytest.raises(ValueError, match="simulations"):
    runner.InLoopEval("vs_uniform", every=1, deals=2, simulations=3)
  with pytest.raises(ValueError, match="simulations"):
    runner.InLoopEval("gin_rummy.rule_bot", every=1, deals=2, simulations=3)
  with pytest.raises(TypeError, match="simulations"):
    runner.InLoopEval("vs_mcts", every=1, deals=2, simulations=True)
  with pytest.raises(AssertionError):
    runner.InLoopEval("vs_nobody", every=1, deals=2)
  assert not set(runner.GENERIC_EVALS) & set(runner.eval_names())


def test_simulations_is_part_of_the_config_hash():
  """Simulations is part of the config hash."""
  ttt = common.load_config("ppo_tic_tac_toe")
  m = dataclasses.replace(
      ttt, eval=(runner.InLoopEval("vs_mcts", every=5, deals=2, simulations=7),)
  )
  assert common.config_dict(m)["eval"][0]["simulations"] == 7
  assert common.config_hash(m) != common.config_hash(
      dataclasses.replace(
          ttt,
          eval=(runner.InLoopEval("vs_mcts", every=5, deals=2, simulations=8),),
      )
  )


@pytest.mark.parametrize(
    "name", ["mmd_kuhn_poker", "ppo_tic_tac_toe", "bc_connect_four"]
)
def test_generic_evals_go_on_any_game(name):
  """Generic evals go on any game."""
  cfg = common.load_config(name)
  got = dataclasses.replace(
      cfg, eval=(runner.InLoopEval("vs_uniform", every=1, deals=2),)
  )
  assert [e.key for e in got.evals()] == ["vs_uniform"]


def _ttt():
  """A tic-tac-toe env.

  Returns:
    The env.
  """
  return env_registry.make("tic_tac_toe")


def test_vs_uniform_known_answer_uniform_is_even():
  """Uniform vs the uniform reference averages zero.

  Uniform vs the uniform reference on duplicate deals: E[mean] = 0 by
  symmetry (zero-sum), so the estimate is within 4 SE of 0.
  """
  env = _ttt()
  ev = runner.build_generic_eval(
      runner.InLoopEval("vs_uniform", every=1, deals=512), env
  )
  me = match.Entrant(
      "me", uniform.UniformPolicy("tic_tac_toe", 9, 2), {}, types.SAMPLE
  )
  r = ev.vs_reference(me, 512, jax.random.key(3))
  assert r.n_deals == 512 and r.b == "uniform"
  assert abs(r.mean.mean) < 4 * r.mean.se + 1e-9, (r.mean.mean, r.mean.se)


def test_vs_mcts_known_answer_uniform_loses():
  """A uniform player loses to the MCTS-50 bot.

  The random-rollout MCTS-50 bot beats a uniform player at tic-tac-toe
  (it never misses an immediate win at 50 simulations here): the uniform
  subject's mean is well below 0.
  """
  env = _ttt()
  ev = runner.build_generic_eval(
      runner.InLoopEval("vs_mcts", every=1, deals=32, simulations=50), env
  )
  assert ev.config.name == "vs_mcts50"
  me = match.Entrant(
      "me", uniform.UniformPolicy("tic_tac_toe", 9, 2), {}, types.SAMPLE
  )
  r = ev.vs_reference(me, 32, jax.random.key(4))
  assert r.mean.mean < -0.4, r.mean


def test_rollout_bot_env_kwargs_pass_through():
  """RandomRolloutPolicy passes env_kwargs through.

  RandomRolloutPolicy records env_kwargs only when given (a default bot's
  spec is unchanged) and builds the env with them (unknown options raise).
  """
  pol = rollout.RandomRolloutPolicy("tic_tac_toe")
  assert pol.spec.arch_kwargs == {"env": "tic_tac_toe"}
  assert rollout.RandomRolloutPolicy("tic_tac_toe", {}).spec.arch_kwargs == {
      "env": "tic_tac_toe"
  }
  with pytest.raises(Exception, match="nope"):
    rollout.RandomRolloutPolicy("tic_tac_toe", {"nope": 1})
  sp, params = rollout.mcts_bot("tic_tac_toe", 4)
  assert params is not None and sp.spec.game == "tic_tac_toe"
  assert np.isfinite(np.float64(sp.cfg.num_simulations))


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
