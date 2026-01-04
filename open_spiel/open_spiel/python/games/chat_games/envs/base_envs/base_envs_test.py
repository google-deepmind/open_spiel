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

"""Tests for base environments."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.games.chat_games.envs.base_envs import email_plain
from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone
from open_spiel.python.games.chat_games.envs.base_envs import email_with_tone_info
from open_spiel.python.games.chat_games.envs.base_envs import schedule_meeting_with_info
from open_spiel.python.games.chat_games.envs.base_envs import trade_fruit_with_info
from open_spiel.python.games.chat_games.envs.utils import header


class BaseEnvsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(base_env=email_plain),
      dict(base_env=email_with_tone),
      dict(base_env=email_with_tone_info),
      dict(base_env=schedule_meeting_with_info),
      dict(base_env=trade_fruit_with_info),
  ])
  def test_give_me_a_name(self, base_env):
    self.assertTrue(header.plain_header_is_valid(base_env.HEADER))


if __name__ == '__main__':
  absltest.main()
