// Copyright 2019 DeepMind Technologies Limited
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

#include "open_spiel/games/pig.h"

#include <sys/types.h>

#include <utility>

#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/tensor_view.h"

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
constexpr bool kDefaultPiglet = false;

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
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"players", GameParameter(kDefaultPlayers)},
        {"horizon", GameParameter(kDefaultHorizon)},
        {"winscore", GameParameter(kDefaultWinScore)},
        {"diceoutcomes", GameParameter(kDefaultDiceOutcomes)},
        {"piglet", GameParameter(kDefaultPiglet)},
    }};

static std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PigGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

std::string PigState::ActionToString(Player player, Action move_id) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Roll ", piglet_ ? move_id : 1 + move_id);
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

  for (auto p = Player{0}; p < num_players_; p++) {
    if (scores_[p] >= win_score_) {
      return true;
    }
  }
  return false;
}

std::vector<double> PigState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  // For (n>2)-player games, must keep it zero-sum.
  std::vector<double> returns(num_players_, -1.0 / (num_players_ - 1));

  for (auto player = Player{0}; player < num_players_; ++player) {
    if (scores_[player] >= win_score_) {
      returns[player] = 1.0;
      return returns;
    }
  }

  // Nobody has won? (e.g. over horizon length.) Then everyone gets 0.
  return std::vector<double>(num_players_, 0.0);
}

std::string PigState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

std::vector<int> PigGame::ObservationTensorShape() const {
  int num_bins = (win_score_ / kBinSize) + 1;
  return {1 + num_players_, num_bins};
}

void PigState::ObservationTensor(Player player,
                                 absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // One extra bin for when value is >= max.
  // So for win_score_ 100, binSize = 1 -> 0, 1, ..., 99, >= 100.
  int num_bins = (win_score_ / kBinSize) + 1;

  // One-hot encoding: turn total (#bin) followed by p1, p2, ...
  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {1 + num_players_, num_bins}, true);

  // One-hot encoding:
  //  - turn total (#bins)
  //  - player 0 (#bins)
  //  - player 1 (#bins)
  //      .
  //      .
  //      .

  // turn total
  view[{0, std::min(turn_total_ / kBinSize, num_bins - 1)}] = 1;

  for (auto p = Player{0}; p < num_players_; p++) {
    // score of each player
    view[{1 + p, std::min(scores_[p] / kBinSize, num_bins - 1)}] = 1;
  }
}

PigState::PigState(std::shared_ptr<const Game> game, int dice_outcomes,
                   int horizon, int win_score, bool piglet)
    : State(game),
      dice_outcomes_(dice_outcomes),
      horizon_(horizon),
      win_score_(win_score),
      piglet_(piglet) {
  total_moves_ = 0;
  cur_player_ = 0;
  turn_player_ = 0;
  scores_.resize(game->NumPlayers(), 0);
  turn_total_ = 0;
}

int PigState::CurrentPlayer() const {
  return IsTerminal() ? kTerminalPlayerId : cur_player_;
}

void PigState::DoApplyAction(Action move) {
  // For decision node: 0 means roll, 1 means stop.
  // For chance node: outcome of the dice (x-1, piglet: [x != 1]).
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
    if (move == 0) {
      // Reset turn total and loses turn!
      turn_total_ = 0;
      turn_player_ = NextPlayerRoundRobin(turn_player_, num_players_);
      cur_player_ = turn_player_;
    } else {
      // Add to the turn total.
      turn_total_ += (piglet_ ? 1 : move + 1);
      cur_player_ = turn_player_;
    }
  } else {
    SpielFatalError(absl::StrCat("Move ", move, " is invalid."));
  }
}

std::vector<Action> PigState::LegalActions() const {
  if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else if (IsTerminal()) {
    return {};
  } else {
    if (scores_[cur_player_] + turn_total_ >= win_score_) {
      return {kStop};
    } else {
      return {kRoll, kStop};
    }
  }
}

std::vector<std::pair<Action, double>> PigState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  if (piglet_) {
    // Chance outcomes are labelled 0 or 1, corresponding to rolling 1 or not 1
    // respectively
    outcomes.reserve(2);
    outcomes.push_back(std::make_pair(0, 1.0 / dice_outcomes_));
    outcomes.push_back(std::make_pair(1, 1.0 - (1.0 / dice_outcomes_)));
  } else {
    // Chance outcomes are labelled 0+, corresponding to rolling 1+x.
    outcomes.reserve(dice_outcomes_);
    for (int i = 0; i < dice_outcomes_; i++) {
      outcomes.push_back(std::make_pair(i, 1.0 / dice_outcomes_));
    }
  }

  return outcomes;
}

std::string PigState::ToString() const {
  return absl::StrCat("Scores: ", absl::StrJoin(scores_, " "),
                      ", Turn total: ", turn_total_,
                      "\nCurrent player: ", turn_player_,
                      (cur_player_ == kChancePlayerId ? " (rolling)\n" : "\n"));
}

std::unique_ptr<State> PigState::Clone() const {
  return std::unique_ptr<State>(new PigState(*this));
}

PigGame::PigGame(const GameParameters& params)
    : Game(kGameType, params),
      dice_outcomes_(ParameterValue<int>("diceoutcomes")),
      horizon_(ParameterValue<int>("horizon")),
      num_players_(ParameterValue<int>("players")),
      win_score_(ParameterValue<int>("winscore")),
      piglet_(ParameterValue<bool>("piglet")) {}

}  // namespace pig
}  // namespace open_spiel
