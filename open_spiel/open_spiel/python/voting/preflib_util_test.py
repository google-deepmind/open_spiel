# Copyright 2023 DeepMind Technologies Limited
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

"""Tests for open_spiel.python.voting.util."""

from absl.testing import absltest
from open_spiel.python.voting import preflib_util

TEST_DATA = """
# FILE NAME: 00004-00000050.soc
# TITLE: Netflix Prize Data
# DESCRIPTION:
# DATA TYPE: soc
# MODIFICATION TYPE: induced
# RELATES TO:
# RELATED FILES:
# PUBLICATION DATE: 2013-08-17
# MODIFICATION DATE: 2022-09-16
# NUMBER ALTERNATIVES: 3
# NUMBER VOTERS: 391
# NUMBER UNIQUE ORDERS: 6
# ALTERNATIVE NAME 1: The Amityville Horror
# ALTERNATIVE NAME 2: Mars Attacks!
# ALTERNATIVE NAME 3: Lean on Me
186: 3,1,2
71: 1,3,2
58: 3,2,1
45: 2,3,1
18: 1,2,3
13: 2,1,3
"""


class UtilTest(absltest.TestCase):
  def test_load_preflib(self):
    print(TEST_DATA)
    profile = preflib_util.parse_preflib_data(TEST_DATA)
    print(profile)
    self.assertEqual(profile.num_alternatives(), 3)
    self.assertEqual(profile.num_votes(), 391)
    self.assertListEqual(profile.alternatives, [
        "The Amityville Horror", "Mars Attacks!", "Lean on Me"
    ])
    print(profile.alternatives)
    print(profile.margin_matrix())


if __name__ == "__main__":
  absltest.main()
