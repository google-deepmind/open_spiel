// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_ALGORITHMS_MINIMAX_H_
#define OPEN_SPIEL_ALGORITHMS_MINIMAX_H_

#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

// The actions that achieve the optimal value at the search root, in
// legal-action order. Empty only when the search root is terminal.
class BestActions {
 private:
  std::vector<Action> actions;

 public:
  // The first optimal action, which is the one the search returned before tie
  // collection was added. kInvalidAction when the set is empty.
  Action Single() const;

  // A uniformly drawn optimal action, kInvalidAction when the set is empty.
  Action SampleUniformly(std::mt19937& rng) const;

  size_t Size() const;
  bool ContainsAction(Action action) const;

  // Set equality: same size and same members, ignoring order.
  bool Equals(const std::vector<Action>& action_list) const;

  void Clear();
  void Add(Action action);
};

// Solves deterministic, 2-players, perfect-information 0-sum game.
//
// Arguments:
//   game: The game to analyze, as returned by `LoadGame`.
//   state: The state to start from. If nullptr, starts from initial state.
//   value_function: An optional function mapping a Spiel `State` to a
//     numerical value to the maximizing player, to be used as the value for a
//     node when we reach `depth_limit` and the node is not terminal. Use
//     `nullptr` for no value function.
//   depth_limit: The maximum depth to search over. When this depth is
//     reached, an exception will be raised.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN. Passing in kInvalidPlayer will set this to the search root's
//     current player.
//   collect_actions: collect all actions that achieve the best value.
//   Returns:
//     A pair of the value of the game for the maximizing player when both
//     players play optimally, along with the actions that achieve this value.

std::pair<double, BestActions> AlphaBetaSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player, bool use_undo = true);

// Solves stochastic, 2-players, perfect-information 0-sum game.
//
// Arguments:
//   game: The game to analyze, as returned by `LoadGame`.
//   state: The state to start from. If nullptr, starts from initial state.
//   value_function: An optional function mapping a Spiel `State` to a
//     numerical value to the maximizing player, to be used as the value for a
//     node when we reach `depth_limit` and the node is not terminal. Use
//     `nullptr` or {} for no value function.
//   depth_limit: The maximum depth to search over (not counting chance nodes).
//     When this depth is reached, an exception will be raised.
//   maximizing_player_id: The id of the MAX player. The other player is assumed
//     to be MIN. Passing in kInvalidPlayer will set this to the search root's
//     current player (which must not be a chance node).

//   Returns:
//     A pair of the value of the game for the maximizing player when both
//     players play optimally, along with the action that achieves this value.

std::pair<double, BestActions> ExpectiminimaxSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_MINIMAX_H_
