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

"""Tests for the ESCHER agent."""

from absl import app
from absl.testing import absltest
from absl.testing import parameterized

import pyspiel
from open_spiel.python.pytorch import escher


class EscherTest(parameterized.TestCase):

  @parameterized.parameters("kuhn_poker", "leduc_poker")
  def test_escher_runs(self, game_name):
    game = pyspiel.load_game(game_name)

    cfg = escher.Config()
    cfg.value_traversals = 2
    cfg.value_net = [2]
    cfg.value_batch_size = 2
    cfg.value_batch_steps = 2
    cfg.regret_traversals = 2
    cfg.regret_net = [2]
    cfg.regret_batch_size = 2
    cfg.regret_batch_steps = 2
    cfg.avg_policy_net = [2]
    cfg.avg_policy_batch_size = 2
    cfg.avg_policy_batch_steps = 2
    agent = escher.Agent(game, cfg)

    train_cfg = escher.TrainConfig(game)
    train_cfg.iterations = 2
    train_cfg.nashconv = True
    train_cfg.games_vs_random = 1
    escher.train(train_cfg, agent)


def main(_):
  absltest.main()


if __name__ == "__main__":
  app.run(main)
