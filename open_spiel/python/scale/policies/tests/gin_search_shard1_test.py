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

"""Shard 1 of gin_search_test: serve == eval through HostAgent."""

import sys

import pytest

from open_spiel.python.scale.policies.tests import gin_search_fixtures


def test_gin_search_serve_equals_eval():
  """Serve equals eval (gin_search_test)."""
  gin_search_fixtures.check_gin_search_serve_equals_eval()


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
