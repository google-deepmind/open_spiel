// Copyright 2023 DeepMind Technologies Limited
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

#include "open_spiel/games/abalone/abalone_core_ab.h"

#include <limits.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace abalone_core {

const int RingWeights[] = { 9, 7, 6, 4, 1 };
const int RFactor = 5;
const int BFactor = 61;

int Heuristic(const core_state& _state, CellState _player) {
  int ballCount[2] = { 0, 0 };
  int scoreCount[2] = { 0, 0 };

  for (auto row = 0; row < kNumRows; ++row) {
    for (auto column = 0; column < kNumCols; ++column) {
      auto player = _state.board_[row][column];
      if (player >= 0) {
        auto y = row;
        auto x = column;

        auto dY = y - kNumRows / 2;
        auto dX = x - kNumCols / 2;

        auto distToCenter = 0;
        if (dX * dY >= 0)
          distToCenter = std::max(std::abs(dX), std::abs(dY));
        else
          distToCenter = std::abs(dX - dY);

        scoreCount[player] += RingWeights[distToCenter];
        ballCount[player]++;
      }
    }
  }

  CellState opponent = CellState(1 - _player);
  return (scoreCount[_player] * RFactor + (kMarblesPerPlayer - ballCount[opponent]) * BFactor
      - scoreCount[opponent] * RFactor - (kMarblesPerPlayer - ballCount[_player]) * BFactor);
}

// Default values (see header): alpha = -1 (worst case), beta = 1 (best
// case).
float _AlphaBeta(float _alpha, float _beta, const core_state& _state,
                 int _depth, CellState _maximize_player) {
  if (_state.outcome_ != CellState::Invalid) {
    if (_state.outcome_ == CellState::Empty)
      return 0;
    return _state.outcome_ == _maximize_player ? 1.f : -1.f;
  }

  if (_depth <= 0) {
    return Heuristic(_state, _maximize_player) * 0.001f;
  }

  auto player = _state.ToPlay();
  float score;
  core_state child_state;
  if (player == _maximize_player) {
    score = -1.f;
    for (core_Action action_id = kActionMin;
         action_id < kActionMax; ++action_id) {
      child_state = _state;
      auto move = Move::ActionToMove(action_id);
      if (!move.IsValid(_state))
        continue;
      move.Apply(child_state);
      // Note: we could precompute the move's heuristic delta and apply it
      // to the previous evaluation.
      auto child_score = _AlphaBeta(
          _alpha, _beta, child_state, _depth - 1, _maximize_player);

      if (child_score > score) {
        score = child_score;
      }
      _alpha = std::max(_alpha, score);
      if (_alpha >= _beta)  // cut
        break;
    }
    return score;
  } else {
    score = 1.f;
    for (core_Action action_id = kActionMin;
         action_id < kActionMax; ++action_id) {
      child_state = _state;
      auto move = Move::ActionToMove(action_id);
      if (!move.IsValid(_state))
        continue;
      move.Apply(child_state);
      auto child_score = _AlphaBeta(
          _alpha, _beta, child_state, _depth - 1, _maximize_player);

      if (child_score < score) {
        score = child_score;
      }
      _beta = std::min(_beta, score);
      if (_beta <= _alpha)  // cut
        break;
    }
    return score;
  }

  return score;
}

std::pair<core_Action, float> AlphaBeta(
    const core_state& _state, int _depth, float _alpha, float _beta,
    std::vector<std::pair<core_Action, float>>* _all_moves) {
  float score = -1.f - 0.001f;
  Move best_move;

  CellState maximize_player = _state.ToPlay();

  std::array<core_Action, kActionMax> actions;
  std::iota(actions.begin(), actions.end(), static_cast<core_Action>(0));
  static std::minstd_rand rng;
  std::shuffle(actions.begin(), actions.end(), rng);

  for (auto action_it = actions.begin();
       action_it != actions.end(); ++action_it) {
    core_Action action_id = *action_it;
    auto child_state = _state;
    auto move = Move::ActionToMove(action_id);
    if (!move.IsValid(_state))
      continue;
    move.Apply(child_state);
    auto child_score = _AlphaBeta(
        _alpha, _beta, child_state, _depth - 1, maximize_player);
    if (_all_moves) {
      _all_moves->push_back(
          std::make_pair(Move::MoveToAction(move), child_score));
    }
    if (child_score > score) {
      score = child_score;
      best_move = move;
    }
    _alpha = std::max(_alpha, score);
    if (_alpha >= _beta)  // cut... but won't happen as beta is not updated
      break;
  }
  return std::make_pair(Move::MoveToAction(best_move), score);
}

}  // namespace abalone_core
