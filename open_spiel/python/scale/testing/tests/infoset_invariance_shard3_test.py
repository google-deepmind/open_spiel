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

"""Shard 3 of infoset_invariance_test (of the fixtures' NUM_SHARDS)."""

import sys

import pytest

from open_spiel.python.scale.policies import operator_vocab
from open_spiel.python.scale.testing.tests import infoset_invariance_fixtures


def pytest_generate_tests(metafunc):
  """Gives the shard test shard 3's items, at collection (not import)."""
  infoset_invariance_fixtures.parametrize_shard(metafunc, 3)


def test_every_fixture_is_infoset_invariant(key, case, plies_):
  """Every item of this shard is infoset-invariant."""
  del key
  infoset_invariance_fixtures.check_invariant(case, plies_)


def test_check_operator_gin_pimc():
  """check_operator holds for the gin PIMC operator."""
  family, shard = infoset_invariance_fixtures.SHARDED_OPERATOR_FAMILY
  assert shard == 3, shard  # this file
  rep = infoset_invariance_fixtures.check_operator_family(family)
  assert rep["decider"] == "search" and rep["acting"] > 0
  assert rep["value_bias"] in operator_vocab.VALUE_BIASES


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
