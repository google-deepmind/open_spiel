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

import pyspiel
from absl.testing import absltest


class RefereeTest(absltest.TestCase):
    def test_playing_tournament(self):
        base = os.path.dirname(__file__) + "/../../higc/bots"
        ref = pyspiel.Referee(
            "kuhn_poker",
            [f"{base}/random_bot_py.sh", f"{base}/random_bot_cpp.sh"],
            settings=pyspiel.TournamentSettings(timeout_ready=2000,
                                                timeout_start=2000)
        )
        results = ref.play_tournament(num_matches=1)
        self.assertEqual(len(results.matches), 1)


if __name__ == "__main__":
    absltest.main()
