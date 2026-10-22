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

"""Tests for the health body."""

from absl.testing import absltest

from open_spiel.python.gui import contract
from open_spiel.python.gui import health

_HUMAN = contract.Controller(contract.HUMAN, "Human", {})
_RANDOM = contract.Controller("random", "Random", {})


class CapabilitiesTest(absltest.TestCase):

  def test_the_human_controller_is_listed_first(self) -> None:
    policies = (contract.ChancePolicy.AUTO,)
    health.Capabilities((_HUMAN, _RANDOM), policies, False)
    with self.assertRaisesRegex(ValueError, "human"):
      health.Capabilities((_RANDOM, _HUMAN), policies, False)

  def test_at_least_one_chance_policy_is_accepted(self) -> None:
    with self.assertRaisesRegex(ValueError, "chance policy"):
      health.Capabilities((_HUMAN,), (), False)


if __name__ == "__main__":
  absltest.main()
