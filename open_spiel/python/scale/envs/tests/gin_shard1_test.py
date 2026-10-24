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

"""Shard 1 of gin_test: the engine-chance known answer (gin_chance_fixtures)."""

import sys

import pytest

from open_spiel.python.scale.core import x64
from open_spiel.python.scale.envs.tests import gin_chance_fixtures


@pytest.mark.parametrize("okla", gin_chance_fixtures.SHARDS[1])
def test_one_step_engine_chance_equals_step_then_install(okla):
  """One step engine chance equals step then install (shard 1)."""
  gin_chance_fixtures.check_one_step_engine_chance(okla)


if __name__ == "__main__":
  # The package runs with x64 on, as its entry points and conftest.py set
  # it; set here for `python <file>`, never at import (which changes no
  # process-global state).
  x64.enable_x64()
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
