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

#include "open_spiel/games/pig.h"

#include <sys/types.h>

#include <utility>

namespace open_spiel {
namespace pig {

namespace {
// Moves.
enum ActionType { kRoll = 0, kStop = 1 };

// Bin size for the information state vectors: how many score values to put
// into one bin. Higher means more coarser.
constexpr int kBinSize = 1;

// Default parameters.
constexpr int kDefaultDiceOutcomes = 6;
constexpr int kDefaultHorizon = 1000;
constexpr int kDefaultPlayers = 2;
constexpr int kDefaultWinScore = 100;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"pig",
    /*long_name=*/"Pig",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/true,
    /*provides_observation=*/false,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/
    {
        {"players", {GameParameter::Type::kInt, false}},
        {"horizon", {GameParameter::Type::kInt, false}},
        {"winscore", {GameParameter::Type::kInt, false}},
        {"diceoutcomes", {GameParameter::Type::kInt, false}},
    }};

static std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new PigGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string PigState::ActionToString(int player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("chance outcome ", move_id);
  } else if (move_id == kRoll) {
    return "roll";
  } else {
    return "stop";
  }
}

bool PigState::IsTerminal() const {
  if (total_moves_ >= horizon_) {
    return true;
  }

  for (int p = 0; p < num_players_; p++) {
    if (scores_[p] >= win_score_) {
      return true;
    }
  }
  return false;
}

std::vector<double> PigState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(0.0, num_players_);
  }

  // For (n>2)-player games, must keep it zero-sum.
  std::vector<double> returns(num_players_, -1.0 / (num_players_ - 1));

  for (int player = 0; player < num_players_; ++player) {
    if (scores_[player] >= win_score_) {
      returns[player] = 1.0;
      return returns;
    }
  }

  // Nobody has won? (e.g. over horizon length.) Then everyone gets 0.
  return std::vector<double>(0.0, num_players_);
}

std::string PigState::InformationState(int player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

std::vector<int> PigGame::InformationStateNormalizedVectorShape() const {
  int num_bins = (win_score_ / kBinSize) + 1;
  return {1 + num_players_, num_bins};
}

void PigState::InformationStateAsNormalizedVector(
    int player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // One extra bin for when value is >= max.
  // So for win_score_ 100, binSize = 1 -> 0, 1, ..., 99, >= 100.
  int num_bins = (win_score_ / kBinSize) + 1;

  // One-hot encoding: turn total (#bin) followed by p1, p2, ...
  values->resize(num_bins + num_players_ * num_bins);
  std::fill(values->begin(), values->end(), 0.);
  int pos = 0;

  // One-hot encoding:
  //  - turn total (#bins)
  //  - player 0 (#bins)
  //  - player 1 (#bins)
  //      .
  //      .
  //      .

  int bin = turn_total_ / kBinSize;
  if (bin >= num_bins) {
    // When the value is too large, use last bin.
    (*values)[pos + (num_bins - 1)] = 1;
  } else {
    (*values)[pos + bin] = 1;
  }

  pos += num_bins;

  // Find the right bin for each player.
  for (int p = 0; p < num_players_; p++) {
    bin = scores_[p] / kBinSize;
    if (bin >= num_bins) {
      // When the value is too large, use last bin.
      (*values)[pos + (num_bins - 1)] = 1;
    } else {
      (*values)[pos + bin] = 1;
    }

    pos += num_bins;
  }
}

PigState::PigState(int num_distinct_actions, int num_players, int dice_outcomes,
                   int horizon, int win_score)
    : State(num_distinct_actions, num_players),
      dice_outcomes_(dice_outcomes),
      horizon_(horizon),
      win_score_(win_score) {
  total_moves_ = 0;
  cur_player_ = 0;
  turn_player_ = 0;
  scores_.resize(num_players, 0);
  turn_total_ = 0;
}

int PigState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void PigState::DoApplyAction(Action move) {
  // For decision node: 0 means roll, 1 means stop.
  // For chance node: outcome of the dice (1-x).
  if (cur_player_ >= 0 && move == kRoll) {
    // Player roll -> chance node.
    cur_player_ = kChancePlayerId;
    total_moves_++;
  } else if (cur_player_ >= 0 && move == kStop) {
    // Player stops. Take turn total and pass to next player.
    scores_[turn_player_] += turn_total_;
    turn_total_ = 0;
    turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
    cur_player_ = turn_player_;
    total_moves_++;
  } else if (IsChanceNode()) {
    // Resolve chance node outcome. If 1, reset turn total and change players;
    // else, add to total and keep going.
    if (move == 1) {
      // Reset turn total and loses turn!
      turn_total_ = 0;
      turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
      cur_player_ = turn_player_;
    } else {
      // Add to the turn total.
      turn_total_ += (move + 1);
      cur_player_ = turn_player_;
    }
  } else {
    SpielFatalError(absl::StrCat("Move ", move, " is invalid."));
  }
}

std::vector<Action> PigState::LegalActions() const {
  if (IsChanceNode())
    return LegalChanceOutcomes();
  else
    return {kRoll, kStop};
}

std::vector<std::pair<Action, double>> PigState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  // All the chance outcomes come after roll and stop.
  outcomes.reserve(dice_outcomes_);
  for (int i = 0; i < dice_outcomes_; i++) {
    outcomes.push_back(std::make_pair(i + 1, 1.0 / dice_outcomes_));
  }

  return outcomes;
}

std::string PigState::ToString() const {
  return absl::StrCat("Scores: ", absl::StrJoin(scores_, " "),
                      ", Turn total:", turn_total_,
                      "\nCurrent player: ", turn_player_,
                      (cur_player_ == 0 ? ", rolling\n" : "\n"));
}

std::unique_ptr<State> PigState::Clone() const {
  return std::unique_ptr<State>(new PigState(*this));
}

PigGame::PigGame(const GameParameters& params)
    : Game(kGameType, params),
      dice_outcomes_(ParameterValue<int>("diceoutcomes", kDefaultDiceOutcomes)),
      horizon_(ParameterValue<int>("horizon", kDefaultHorizon)),
      num_players_(ParameterValue<int>("players", kDefaultPlayers)),
      win_score_(ParameterValue<int>("winscore", kDefaultWinScore)) {}

}  // namespace pig
}  // namespace open_spiel
