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

"""The one estimate behind every Evaluator number: mean, SE and 95% CI.

f64 numpy. The SE is the sample standard deviation (ddof=1) over sqrt(n), and
the interval is the normal one, mean +- Z95 * SE (the unit of a sample is one
duplicate deal pair, n is in the hundreds or more). Raises ValueError on
fewer than two samples or a non-finite one.
"""

from typing import NamedTuple

import numpy as np

Z95 = 1.959963984540054  # two-sided 95% normal quantile


class Estimate(NamedTuple):
  """A mean with its standard error and normal 95% interval over n samples.

  Attributes:
    mean: the sample mean.
    se: its standard error.
    lo: mean - Z95 * se.
    hi: mean + Z95 * se.
    n: the sample size.
  """

  mean: float
  se: float
  lo: float  # mean - Z95 * se
  hi: float  # mean + Z95 * se
  n: int


def estimate(x) -> Estimate:
  """The Estimate of a 1-D f64 sample (at least 2 values, all finite).

  Args:
    x: the sample.

  Returns:
    the Estimate (mean, SE, 95% CI, n).

  Raises:
    ValueError: the sample is not 1-D, has fewer than 2 values, or is not
        finite.
  """
  x = np.asarray(x, np.float64)
  if x.ndim != 1:
    raise ValueError(f"estimate takes a 1-D sample, got shape {x.shape}")
  n = int(x.shape[0])
  if n < 2:
    raise ValueError(f"an SE needs at least 2 samples, got {n}")
  if not np.all(np.isfinite(x)):
    raise ValueError("estimate: non-finite sample")
  mean = float(x.mean())
  se = float(x.std(ddof=1) / np.sqrt(n))
  return Estimate(mean=mean, se=se, lo=mean - Z95 * se, hi=mean + Z95 * se, n=n)
