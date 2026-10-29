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

"""A short CPU smoke run of MMD self-play on Kuhn poker.

A short CPU smoke run: MMD self-play on Kuhn poker through train.run brings
the exact exploitability of the learned policy below the uniform policy's.
"""

import dataclasses
import sys

import pytest

from open_spiel.python.scale import train
from open_spiel.python.scale.envs import kuhn
from open_spiel.python.scale.eval import exploitability as expl_lib
from open_spiel.python.scale.league import config_common as common

UNIFORM_KUHN = 0.4583333333  # NashConv 11/12, /2 players


def _exploitability(policy, params) -> float:
  """The Kuhn exploitability of `policy` at `params`.

  Args:
    policy: The policy.
    params: Its params.

  Returns:
    The exploitability.
  """
  return expl_lib.exploitability(kuhn.KuhnEnv(), policy, params).exploitability


def test_mmd_kuhn_smoke_beats_uniform(tmp_path):
  """MMD kuhn smoke beats uniform."""
  cfg = common.load_config("mmd_kuhn_poker")
  cfg = common.with_overrides(cfg, num_envs=128, num_steps=16, num_updates=60)
  pgc = dataclasses.replace(cfg.learner.learner, lr=3e-3)
  cfg = dataclasses.replace(
      cfg,
      name=cfg.name + "_smoke",
      keep_every=60,
      learner=dataclasses.replace(cfg.learner, learner=pgc),
  )
  res = train.run(cfg, checkpoint_dir=str(tmp_path / "ckpt"))
  expl = _exploitability(res.learner.policy(), res.state.params)
  print(
      f"kuhn smoke: exploitability {expl:.4f} after {cfg.num_iterations} "
      "updates "
      f"(uniform {UNIFORM_KUHN:.4f}); last metrics {res.metrics[-1]}"
  )
  assert expl < 0.8 * UNIFORM_KUHN
  assert int(res.state.update_step) == 60


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
