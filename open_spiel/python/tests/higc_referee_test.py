# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for open_spiel.python.referee."""
import os

from absl.testing import absltest
import pyspiel


class RefereeTest(absltest.TestCase):

  def test_playing_tournament(self):
    base = os.path.dirname(__file__) + "/../bots"
    ref = pyspiel.Referee(
        "kuhn_poker", [f"{base}/higc_random_bot_test.py"] * 2,
        settings=pyspiel.TournamentSettings(
            timeout_ready=2000, timeout_start=500))
    self.assertTrue(ref.started_successfully())
    results = ref.play_tournament(num_matches=1)
    self.assertLen(results.matches, 1)


if __name__ == "__main__":
  absltest.main()
