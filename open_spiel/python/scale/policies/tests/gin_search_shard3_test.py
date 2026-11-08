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

"""Shard 3 of gin_search_test: search beats or equals the raw network.

At the positions gin_search_fixtures.SEARCH_SHARDS[3].
"""

import sys

import pytest

from open_spiel.python.scale.envs import gin
from open_spiel.python.scale.policies.tests import gin_search_fixtures


@pytest.fixture(name="env", scope="module")
def _env_fixture():
  """The gin env, once per module.

  Returns:
    The env.
  """
  return gin.GinEnv()


@pytest.mark.parametrize("pos", gin_search_fixtures.SEARCH_SHARDS[3])
def test_search_beats_or_equals_the_raw_network(env, pos):
  """Search beats or equals the raw network at this shard's positions."""
  gin_search_fixtures.check_search_beats_or_equals_the_raw_network(env, pos)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
