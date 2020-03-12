# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script provides a general Nash equilibrium solver that can solve general-sum many-player games.
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import logging.info_function

import fractions
import itertools
import os
import subprocess
import tempfile
import warnings

import nashpy
import numpy as np

import logging
from open_spiel.python.algorithms import lp_solver
import pyspiel
from open_spiel.python.algorithms.nash_solver.gambit_tools import do_gambit_analysis
from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

def renormalize(probabilities):
  """Replaces all non-zero entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities

@np.vectorize
def _to_fraction_str(x, lrsnash_max_denom=1000):
    return str(fractions.Fraction(x).limit_denominator(lrsnash_max_denom))


def lrs_solve(row_payoffs, col_payoffs, lrsnash_path):
    """Find all Nash equilibria using the lrsnash solver.
    `lrsnash` uses reverse search vertex enumeration on rational polytopes.
    For more info, see: http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash
    Args:
      row_payoffs: payoffs for row player
      col_payoffs: payoffs for column player
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
            game_file.write(" ".join(_to_fraction_str(row_payoffs[row])) + "\n")
        game_file.write("\n")

        # write col-player payoff matrix as fractions
        for row in range(num_rows):
            game_file.write(" ".join(_to_fraction_str(col_payoffs[row])) + "\n")
        game_file.write("\n")
        game_file.close()

        lrs = subprocess.Popen(
            [lrsnash_path or "lrsnash", "-s", game_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

        lrs_result = lrs.communicate()[0]
        equilibria = []
        col_mixtures = []
        for line in lrs_result.split(b'\n'):
            if len(line) <= 1 or line[:1] == b"*":
                continue
            line = np.asfarray([fractions.Fraction(x) for x in line.decode().split()])
            if line[0] == 2:  # col-player
                col_mixtures.append(line[1:-1])
            else:  # row-player
                row_mixture = line[1:-1]
                # row-mixture forms a Nash with every col-mixture listed directly above
                for col_mixture in col_mixtures:
                    equilibria.append([row_mixture, col_mixture])
                col_mixtures = []
        return equilibria
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
                print('wrong results')
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

def gambit_solve(meta_games, mode):
    """
    Find NE using gambit.
    :param meta_games: meta-games in PSRO.
    :param mode: options "all", "one", "pure"
    :return: a list of NE.
    """
    return do_gambit_analysis(meta_games, mode)

def pure_ne_solve(meta_games, tol=1e-7):
    """
    Find pure NE. Only work for two-player game. For more than 2 player case,
    the nash_solver will call gambit to find pure NE.
    :param meta_games: meta-games in PSRO.
    :param tol: Error allowed.
    :return: pure NE
    """
    row_payoffs, col_payoffs = meta_games[0], meta_games[1]
    pure_nash = list(
        zip(*((row_payoffs >= row_payoffs.max(0, keepdims=True) - tol)
              & (col_payoffs >= col_payoffs.max(1, keepdims=True) - tol)
              ).nonzero()))
    p1_num_str, p2_num_str = np.shape(meta_games[0])
    pure_ne = []
    for i, j in pure_nash:
        p1_ne = np.zeros(p1_num_str)
        p2_ne = np.zeros(p2_num_str)
        p1_ne[i] = 1
        p2_ne[j] = 1
        pure_ne.append([p1_ne, p2_ne])
    return pure_ne

def nash_solver(meta_games,
                solver,
                mode="one",
                gambit_path=None,
                lrsnash_path=None):
    """
    Solver for NE.
    :param meta_games: meta-games in PSRO.
    :param solver: options "gambit", "nashpy", "linear", "lrsnash", "replicator".
    :param mode: options "all", "one", "pure"
    :param lrsnash_path: path to lrsnash solver.
    :return: a list of NE.
    WARNING:
    opening up a subprocess in every iteration eventually
    leads the os to block the subprocess. Not usable.
    """
    num_players = len(meta_games)
    if solver == "gambit":
        return normalize_ne(gambit_solve(meta_games, mode))
    elif solver == "replicator":
        return normalize_ne([replicator_dynamics(meta_games)])
    else:
        assert num_players == 2

        num_rows, num_cols = np.shape(meta_games[0])
        row_payoffs, col_payoffs = meta_games[0], meta_games[1]

        if num_rows == 1 or num_cols == 1:
            equilibria = itertools.product(np.eye(num_rows), np.eye(num_cols))
        elif mode == 'pure':
            return pure_ne_solve(meta_games)

        elif solver == "linear":
            meta_games = [x.tolist() for x in meta_games]
            nash_prob_1, nash_prob_2, _, _ = (
                lp_solver.solve_zero_sum_matrix_game(
                    pyspiel.create_matrix_game(*meta_games)))
            return [
                renormalize(np.array(nash_prob_1).reshape(-1)),
                renormalize(np.array(nash_prob_2).reshape(-1))
            ]
        elif solver == "lrsnash":
            logging.info("Using lrsnash solver.")
            equilibria = lrs_solve(row_payoffs, col_payoffs, lrsnash_path)
        elif solver == "nashpy":
            if mode == "all":
                logging.info("Using nashpy vertex enumeration.")
                equilibria = nashpy.Game(row_payoffs, col_payoffs).vertex_enumeration()
            else:
                logging.info("Using nashpy Lemke-Howson solver.")
                equilibria = lemke_howson_solve(row_payoffs, col_payoffs)
        else:
            raise ValueError("Please choose a valid NE solver.")

        equilibria = iter(equilibria)
        # check that there's at least one equilibrium
        try:
            equilibria = itertools.chain([next(equilibria)], equilibria)
        except StopIteration:
            logging.warning("degenerate game!")
#            pklfile = open('/home/qmaai/degenerate_game.pkl','wb')
#            pickle.dump([row_payoffs,col_payoffs],pklfile)
#            pklfile.close()
            # degenerate game apply support enumeration
            equilibria = nashpy.Game(row_payoffs, col_payoffs).support_enumeration()
            try:
                equilibria = itertools.chain([next(equilibria)], equilibria)
            except StopIteration:
                logging.warning("no equilibrium!")
        
        equilibria = list(equilibria)
        if mode == 'all':
            return normalize_ne(equilibria)
        elif mode == 'one':
            return normalize_ne([equilibria[0]])
        else:
            raise ValueError("Please choose a valid mode.")

def normalize_ne(eq):
    for p in range(len(eq)):
        for i, str in enumerate(eq[p]):
            eq[p][i] = renormalize(str)
    return eq
