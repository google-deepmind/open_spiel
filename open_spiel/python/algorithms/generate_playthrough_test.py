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

"""Tests for google3.third_party.open_spiel.python.algorithms.playthrough."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from open_spiel.python.algorithms import generate_playthrough


class PlaythroughTest(unittest.TestCase):

  def test_runs(self):
    result = generate_playthrough.playthrough("tic_tac_toe", seed=1234)
    self.assertGreater(len(result), 0)


if __name__ == "__main__":
  unittest.main()
