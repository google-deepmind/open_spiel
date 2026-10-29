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

"""stats: known answers for the estimate behind every Evaluator number."""

import math
import sys

import numpy as np
import pytest

from open_spiel.python.scale.eval import stats


def test_known_answer():
  """`estimate` has its known mean, SE and CI."""
  e = stats.estimate(np.array([1.0, 2.0, 3.0, 4.0]))
  sd = math.sqrt(((1.5**2) * 2 + (0.5**2) * 2) / 3)  # ddof=1
  assert e.n == 4
  assert e.mean == 2.5
  assert e.se == pytest.approx(sd / 2.0, abs=1e-15)
  assert e.lo == pytest.approx(2.5 - stats.Z95 * sd / 2.0, abs=1e-15)
  assert e.hi == pytest.approx(2.5 + stats.Z95 * sd / 2.0, abs=1e-15)
  assert stats.Z95 == pytest.approx(1.959963984540054, abs=1e-15)


def test_constant_sample_has_zero_se():
  """A constant sample has SE 0."""
  e = stats.estimate(np.full(10, -0.25))
  assert (e.mean, e.se, e.lo, e.hi) == (-0.25, 0.0, -0.25, -0.25)


def test_dtype_is_f64():
  """`estimate` computes in f64."""
  e = stats.estimate(np.array([1, 2, 4], np.float32))
  assert isinstance(e.mean, float) and isinstance(e.se, float)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0]),
        np.array([]),
        np.array([1.0, np.nan]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_fails_loud(bad):
  """Bad samples raise ValueError."""
  with pytest.raises(ValueError):
    stats.estimate(bad)


if __name__ == "__main__":
  sys.exit(
      pytest.main(
          [__file__, "--import-mode=importlib", "-p", "no:cacheprovider"]
      )
  )
