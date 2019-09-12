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

#include "open_spiel/games/go.h"

#include <sstream>

#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel_optional.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace go {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"go",
    /*long_name=*/"Go",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/false,
    /*parameter_specification=*/
    {
        {"komi", GameType::ParameterSpec{GameParameter::Type::kDouble, false}},
        {"board_size",
         GameType::ParameterSpec{GameParameter::Type::kInt, false}},
    },
};

std::unique_ptr<Game> Factory(const GameParameters& params) {
  return std::unique_ptr<Game>(new GoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::vector<GoPoint> HandicapStones(int num_handicap) {
  if (num_handicap < 2 || num_handicap > 9) return {};

  static std::array<GoPoint, 9> placement = {
      {MakePoint("d4"), MakePoint("q16"), MakePoint("d16"), MakePoint("q4"),
       MakePoint("d10"), MakePoint("q10"), MakePoint("k4"), MakePoint("k16"),
       MakePoint("k10")}};
  static GoPoint center = MakePoint("k10");

  std::vector<GoPoint> points;
  points.reserve(num_handicap);
  for (int i = 0; i < num_handicap; ++i) {
    points.push_back(placement[i]);
  }

  if (num_handicap >= 5 && num_handicap % 2 == 1) {
    points[num_handicap - 1] = center;
  }

  return points;
}

}  // namespace

GoState::GoState(int board_size, float komi, int handicap)
    : State(go::NumDistinctActions(board_size), go::NumPlayers()),
      board_(board_size),
      komi_(komi),
      handicap_(handicap),
      to_play_(GoColor::kBlack) {
  ResetBoard();
}

std::string GoState::InformationState(int player) const {
  return HistoryString();
}

std::string GoState::Observation(int player) const {
  return ToString();
}

std::vector<Action> GoState::LegalActions() const {
  std::vector<Action> actions = {kPass};

  for (GoPoint p : BoardPoints(board_.board_size())) {
    if (board_.IsLegalMove(p, to_play_)) {
      actions.push_back(p);
    }
  }

  return actions;
}

std::string GoState::ActionToString(int player, Action action) const {
  return absl::StrCat(GoColorToString(static_cast<GoColor>(player)), " ",
                      GoPointToString(action));
}

std::string GoState::ToString() const {
  std::stringstream ss;
  ss << "GoState(komi=" << komi_ << ", to_play=" << GoColorToString(to_play_)
     << "history.size()=" << history_.size() << ")\n";
  ss << board_;
  return ss.str();
}

bool GoState::IsTerminal() const {
  if (history_.size() < 2) return false;
  return (history_.size() >= MaxGameLength(board_.board_size())) || superko_ ||
         (history_[history_.size() - 1] == kPass &&
          history_[history_.size() - 2] == kPass);
}

std::vector<double> GoState::Returns() const {
  if (!IsTerminal()) return {0.0, 0.0};

  if (superko_) {
    // Superko rules (https://senseis.xmp.net/?Superko) are complex and vary
    // between rulesets.
    // For simplicity and because superkos are very rare, we just treat them as
    // a draw.
    return {DrawUtility(), DrawUtility()};
  }

  // Score with Tromp-Taylor.
  float black_score = TrompTaylorScore(board_, komi_, handicap_);

  std::vector<double> returns(go::NumPlayers());
  if (black_score > 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = WinUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = LossUtility();
  } else if (black_score < 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = LossUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = WinUtility();
  } else {
    returns[ColorToPlayer(GoColor::kBlack)] = DrawUtility();
    returns[ColorToPlayer(GoColor::kWhite)] = DrawUtility();
  }
  return returns;
}

std::unique_ptr<State> GoState::Clone() const {
  return std::unique_ptr<State>(new GoState(*this));
}

void GoState::UndoAction(int player, Action action) {
  // We don't have direct undo functionality, but copying the board and
  // replaying all actions is still pretty fast (> 1 million undos/second).
  history_.pop_back();
  ResetBoard();
  for (Action action : history_) {
    DoApplyAction(action);
  }
}

void GoState::DoApplyAction(Action action) {
  SPIEL_CHECK_TRUE(board_.PlayMove(action, to_play_));
  to_play_ = OppColor(to_play_);

  bool was_inserted = repetitions_.insert(board_.HashValue()).second;
  if (!was_inserted && action != kPass) {
    // We have encountered this position before.
    superko_ = true;
  }
}

void GoState::ResetBoard() {
  board_.Clear();
  if (handicap_ < 2) {
    to_play_ = GoColor::kBlack;
  } else {
    for (GoPoint p : HandicapStones(handicap_)) {
      board_.PlayMove(p, GoColor::kBlack);
    }
    to_play_ = GoColor::kWhite;
  }

  repetitions_.clear();
  repetitions_.insert(board_.HashValue());
  superko_ = false;
}

GoGame::GoGame(const GameParameters& params)
    : Game(kGameType, params),
      komi_(ParameterValue<double>("komi", 7.5)),
      board_size_(ParameterValue<int>("board_size", 19)),
      handicap_(ParameterValue<int>("handicap", 0)) {}

}  // namespace go
}  // namespace open_spiel
