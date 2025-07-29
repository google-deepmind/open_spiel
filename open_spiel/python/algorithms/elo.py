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
# Modified: 2023 James Flynn
# Original:
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py

"""Elo rating system."""

import numpy as np
import pyspiel


def compute_elo(
    win_matrix: np.ndarray,
    draw_matrix: np.ndarray | None = None,
    smoothing_factor: float = pyspiel.elo.DEFAULT_SMOOTHING_FACTOR,
    max_iterations: int = pyspiel.elo.DEFAULT_MAX_ITERATIONS,
    convergence_delta: float = pyspiel.elo.DEFAULT_CONVERGENCE_DELTA,
) -> np.ndarray:
  """Compute Elo ratings from a win matrix and a draw matrix."""
  return pyspiel.elo.compute_elo(
      win_matrix=win_matrix.tolist(),
      draw_matrix=(draw_matrix.tolist() if draw_matrix is not None else []),
      smoothing_factor=smoothing_factor,
      max_iterations=max_iterations,
      convergence_delta=convergence_delta
  )
