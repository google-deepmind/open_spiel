# Copyright 2019 DeepMind Technologies Limited
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

"""Elo rating system.

There are two ways to compute the Elo ratings:
  1. From a win matrix and a draw matrix.
  2. From a list of match records.

The function here is a wrapper to method (1) that simply converts numpy arrays
to lists of lists and then calls the pybind11 wrapper. For examples of (2),
please see elo_test.py.

Both are wrappers around the C++ implementation in evaluation/elo.cc, which is
based on the algorithm of Hunter, MM Algorithms for Generalized Bradley-Terry
Models, The Annals of Statistics 2004, Vol. 32, No. 1, 384--406.
"""

from typing import Optional
import numpy as np
import pyspiel


def compute_ratings_from_matrices(
    win_matrix: np.ndarray,
    draw_matrix: Optional[np.ndarray] = None,
    smoothing_factor: float = pyspiel.elo.DEFAULT_SMOOTHING_FACTOR,
    max_iterations: int = pyspiel.elo.DEFAULT_MAX_ITERATIONS,
    convergence_delta: float = pyspiel.elo.DEFAULT_CONVERGENCE_DELTA,
) -> np.ndarray:
  """Compute Elo ratings from a win matrix and a draw matrix."""
  options = pyspiel.elo.default_elo_options()
  options.smoothing_factor = smoothing_factor
  options.max_iterations = max_iterations
  options.convergence_delta = convergence_delta
  return pyspiel.elo.compute_ratings_from_matrices(
      win_matrix=win_matrix.tolist(),
      draw_matrix=(draw_matrix.tolist() if draw_matrix is not None else []),
      options=options,
  )
