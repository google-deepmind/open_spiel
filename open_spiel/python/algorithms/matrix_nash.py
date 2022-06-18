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


"""Find Nash equilibria for constant- or general-sum 2-player games.

Non-matrix games are handled by computing the normal (bimatrix) form.
The algorithms used are:
* direct computation of pure equilibria.
* linear programming to find equilibria for constant-sum games.
* iterated dominance to reduce the action space.
* reverse search vertex enumeration (if using lrsnash) to find all general-sum
  equilibria.
* support enumeration (if using nashpy) to find all general-sum equilibria.
* Lemke-Howson enumeration (if using nashpy) to find one general-sum
  equilibrium.
The general-sum mixed-equilibrium algorithms are likely to work well for tens of
actions, but less likely to scale beyond that.
"""

import fractions
import os
import subprocess
import tempfile
import warnings

import nashpy
import numpy as np


@np.vectorize
def to_fraction_str(x, lrsnash_max_denom):
  return str(fractions.Fraction(x).limit_denominator(lrsnash_max_denom))


def lrs_solve(row_payoffs, col_payoffs, lrsnash_max_denom, lrsnash_path):
  """Find all Nash equilibria using the lrsnash solver.

  `lrsnash` uses reverse search vertex enumeration on rational polytopes.
  For more info, see: http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash

  Args:
    row_payoffs: payoffs for row player
    col_payoffs: payoffs for column player
    lrsnash_max_denom: maximum denominator
    lrsnash_path: path for temporary files

  Yields:
    (row_mixture, col_mixture), numpy vectors of float64s.
  """
  num_rows, num_cols = row_payoffs.shape
  game_file, game_file_path = tempfile.mkstemp()
  try:
    game_file = os.fdopen(game_file, "w")

    # write dimensions
    game_file.write("%d %d\n\n" % (num_rows, num_cols))

    # write row-player payoff matrix as fractions
    for row in range(num_rows):
      game_file.write(
          " ".join(to_fraction_str(row_payoffs[row], lrsnash_max_denom)) + "\n")
    game_file.write("\n")

    # write col-player payoff matrix as fractions
    for row in range(num_rows):
      game_file.write(
          " ".join(to_fraction_str(col_payoffs[row], lrsnash_max_denom)) + "\n")
    game_file.write("\n")
    game_file.close()
    lrs = subprocess.Popen([lrsnash_path or "lrsnash", "-s", game_file_path],
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    col_mixtures = []
    for line in lrs.stdout:
      if len(line) <= 1 or line[:1] == b"*":
        continue
      line = np.asfarray([fractions.Fraction(x) for x in line.decode().split()])
      if line[0] == 2:  # col-player
        col_mixtures.append(line[1:-1])
      else:  # row-player
        row_mixture = line[1:-1]
        # row-mixture forms a Nash with every col-mixture listed directly above
        for col_mixture in col_mixtures:
          yield (row_mixture, col_mixture)
        col_mixtures = []
  finally:
    os.remove(game_file_path)


def lemke_howson_solve(row_payoffs, col_payoffs):
  """Find Nash equilibria using the Lemke-Howson algorithm.

  The algorithm is not guaranteed to find all equilibria. Also it can yield
  wrong answers if the game is degenerate (but raises warnings in that case).
  Args:
    row_payoffs: payoffs for row player
    col_payoffs: payoffs for column player
  Yields:
    (row_mixture, col_mixture), numpy vectors of float64s.
  """

  showwarning = warnings.showwarning
  warned_degenerate = [False]

  def showwarning_check_degenerate(message, *args, **kwargs):
    if "Your game could be degenerate." in str(message):
      warned_degenerate[0] = True
    showwarning(message, *args, **kwargs)

  try:
    warnings.showwarning = showwarning_check_degenerate
    for row_mixture, col_mixture in nashpy.Game(
        row_payoffs, col_payoffs).lemke_howson_enumeration():
      if warned_degenerate[0]:
        # attempt to discard obviously-wrong results
        if (row_mixture.shape != row_payoffs.shape[:1] or
            col_mixture.shape != row_payoffs.shape[1:]):
          warnings.warn("Discarding ill-shaped solution.")
          continue
        if (not np.isfinite(row_mixture).all() or
            not np.isfinite(col_mixture).all()):
          warnings.warn("Discarding non-finite solution.")
          continue
      yield row_mixture, col_mixture
  finally:
    warnings.showwarning = showwarning
