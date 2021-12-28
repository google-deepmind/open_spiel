# Copyright 2019 DeepMind Technologies Limited
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

"""Tests for open_spiel.python.referee."""

import os
from absl import flags
from absl.testing import absltest
import pyspiel


flags.DEFINE_string("bot_dir",
                    os.path.dirname(__file__) + "/../bots",
                    "Path to python implementation of bots.")
FLAGS = flags.FLAGS


class RefereeTest(absltest.TestCase):

  def test_playing_tournament(self):
    ref = pyspiel.Referee(
        "kuhn_poker", [f"python {FLAGS.bot_dir}/higc_random_bot_test.py"] * 2,
        settings=pyspiel.TournamentSettings(
            timeout_ready=2000, timeout_start=500))
    results = ref.play_tournament(num_matches=1)
    self.assertLen(results.matches, 1)


if __name__ == "__main__":
  absltest.main()
