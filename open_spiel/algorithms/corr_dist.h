// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_CORR_DIST_H_
#define OPEN_SPIEL_ALGORITHMS_CORR_DIST_H_

#include <optional>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// This file provides a set of functions for computing the distance of a
// distribution of joint policies to a correlated equilibrium. It is the
// analogue of NashConv for correlated equilibria, i.e. sum of the incentives to
// deviate to a best response over all players, as an empirical metric that
// summarizes how far the distribution is from an equilibrium.
//
// The functions compute these metrics for extensive-form correlated equilibria
// (EFCE) and extensive-form coarse-correlated equilibria (EFCCE). The
// algorithms work by constructing an auxiliary game (described in Def 2.2 of
// von Stengel and Forges 2008) where chance initially samples a joint policy,
// then lets the players decide to follow or not follow the recommendations.
// If they follow, they continue to receive recommendations. Otherwise, they
// stop receiving recommendations. The incentive for a player to deviate toward
// a best response can be computed by the existing best response algorithm in
// this new game.
//
// In both cases of EFCE and EFCCE, the algorithms compute the normal-form
// equivalents, and two wrapper functions are provided specifically for the
// normal-form setting (CEDist and CCEDist).
//
// **WARNING**: the implementations of the metrics for the extensive-form
// versions of the correlated equilibria have only been lightly tested (on a
// few simple examples). We plan to add more thorough tests as we implement
// more benchmark general-sum games and more worked-out examples.
//
// For formal definitions and algorithms, please refer to:
//   - von Stengel and Forges, 2008. Extensive-Form Correlated Equilibrium:
//     Definition and Computational Complexity, Mathematics of Operations
//     Research, vol 33, no. 4.
//   - Farina, Bianchi, and Sandholm, 2019. Coarse Correlation in Extensive-Form
//     Games. https://arxiv.org/abs/1908.09893
//   - Dudik & Gordon, https://arxiv.org/abs/1205.2649
//   - Gordon, Greenwald, and Marks. No-Regret Learning in Convex Games.
//     https://www.cs.cmu.edu/~ggordon/gordon-greenwald-marks-icml-phi-regret.pdf

// A CorrelationDevice represents a distribution over joint policies (name is
// from von Stengel & Forges 2008). Note, however, that unlike von Stengel &
// Forges 2008, the joint policies can be mixed. In this case, an equivalent
// joint distribution over deterministic joint policies could be reconstructed
// (if the game is small enough) or the metrics below can be approximated via
// Monte Carlo sampling of deterministic joint policies from the mixtures.
using CorrelationDevice = std::vector<std::pair<double, TabularPolicy>>;

// A helper class for the normal-form functions.
struct NormalFormJointPolicyWithProb {
  // Probability of this joint policy.
  double probability;

  // The action taken by each player.
  std::vector<Action> actions;
};

using NormalFormCorrelationDevice = std::vector<NormalFormJointPolicyWithProb>;

// A configuration object for the metrics.
struct CorrDistConfig {
  // Are the underlying policies deterministic (pure)? If false, then a Monte
  // Carlo approximation is done by sampling deterministic joint policies
  // Only 'true' is currently supported.
  bool deterministic = true;

  // Should the distribution over mixed policies be converted to an equivalent
  // one over determinstic policies before computing the metric?
  // Only 'false' is currently supported.
  bool convert_policy = false;

  // A tag used to delimit recommendation sequences from the normal part of the
  // information state string.
  std::string recommendation_delimiter = " R-*-=-*-R ";

  // If determinstic is false, how many deterministic joint policies should be
  // samples in order to assemble an approximate empirical joint distribution?
  // Note: not yet implemented, but coming soon.
  // int num_samples = 1000;
};

// Return the expected values (one per player) of a correlation device.
std::vector<double> ExpectedValues(const Game& game,
                                   const CorrelationDevice& mu);
std::vector<double> ExpectedValues(const Game& game,
                                   const NormalFormCorrelationDevice& mu);

// Compute the sum of individual incentives to deviate (from the joint
// distribution) to a best response, over all players. The auxiliary game
// constructed is with accordance to the EFCE concept, which means players see
// their recommendations once they reach the information states (unless they've
// chosen not to follow at some point).
double EFCEDist(const Game& game, CorrDistConfig config,
                const CorrelationDevice& mu);

// Compute the sum of individual incentives to deviate (from the joint
// distribution) to a best response, over all players. The auxiliary game
// constructed is with accordance to the EFCCE concept, which means players see
// their recommendations at their information states only after they've decided
// whether or not to follow them.
double EFCCEDist(const Game& game, CorrDistConfig config,
                 const CorrelationDevice& mu);

// Analog to the functions above but for normal-form games. The game can be a
// normal-form game *or* a TurnBasedSimultaneousGame wrapping a normal-form
// game.
double CEDist(const Game& game, const NormalFormCorrelationDevice& mu);
double CCEDist(const Game& game, const NormalFormCorrelationDevice& mu);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_CORR_DIST_H_
