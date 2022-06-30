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

#include "open_spiel/games/cursor_go.h"

#include <sstream>

#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/go/go_board.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace cursor_go {
namespace {

using go::BoardPoints;
using go::MakePoint;
using go::VirtualPoint;
using go::VirtualPointFrom2DPoint;
using go::VirtualPointToString;

// Facts about the game
const GameType kGameType{
    /*short_name=*/"cursor_go",
    /*long_name=*/"Cursor Go",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"komi", GameParameter(7.5)},
        {"board_size", GameParameter(19)},
        {"handicap", GameParameter(0)},
        {"max_cursor_moves", GameParameter(100)},
    },
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CursorGoGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::vector<VirtualPoint> HandicapStones(int num_handicap) {
  if (num_handicap < 2 || num_handicap > 9) return {};

  static std::array<VirtualPoint, 9> placement = {
      {MakePoint("d4"), MakePoint("q16"), MakePoint("d16"), MakePoint("q4"),
       MakePoint("d10"), MakePoint("q10"), MakePoint("k4"), MakePoint("k16"),
       MakePoint("k10")}};
  static VirtualPoint center = MakePoint("k10");

  std::vector<VirtualPoint> points;
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

CursorGoState::CursorGoState(std::shared_ptr<const Game> game, int board_size,
                             float komi, int handicap, int max_cursor_moves)
    : State(game),
      board_(board_size),
      komi_(komi),
      handicap_(handicap),
      max_cursor_moves_(max_cursor_moves),
      to_play_(GoColor::kBlack) {
  ResetBoard();
}

std::string CursorGoState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string CursorGoState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void CursorGoState::ObservationTensor(Player player,
                                      absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int num_cells = board_.board_size() * board_.board_size();
  SPIEL_CHECK_EQ(values.size(), num_cells * (kCellStates + 3));
  std::fill(values.begin(), values.end(), 0.);

  // Add planes: black, white, empty.
  int cell = 0;
  for (VirtualPoint p : BoardPoints(board_.board_size())) {
    int color_val = static_cast<int>(board_.PointColor(p));
    values[num_cells * color_val + cell] = 1.0;
    ++cell;
  }
  SPIEL_CHECK_EQ(cell, num_cells);

  // Fourth plane for cursor position.
  const auto [row, col] = cursor_[ColorToPlayer(to_play_)];
  const int cursor_cell = row * board_.board_size() + col;
  values[num_cells * kCellStates + cursor_cell] = 1.0;

  // Add a fifth binary plane for komi (whether white is to play).
  std::fill(values.begin() + ((1 + kCellStates) * num_cells),
            values.begin() + ((2 + kCellStates) * num_cells),
            (to_play_ == GoColor::kWhite ? 1.0 : 0.0));

  // Add a sixth binary plane for the number of cursor moves.
  std::fill(values.begin() + ((2 + kCellStates) * num_cells), values.end(),
            static_cast<float>(cursor_moves_count_) / max_cursor_moves_);
}

std::vector<Action> CursorGoState::LegalActions() const {
  std::vector<Action> actions{};
  if (is_terminal_) return actions;
  const auto cursor = cursor_[ColorToPlayer(to_play_)];
  if (cursor_moves_count_ < max_cursor_moves_) {
    const auto [row, col] = cursor;
    if (row < board_.board_size() - 1) actions.push_back(kActionUp);
    if (row > 0) actions.push_back(kActionDown);
    if (col > 0) actions.push_back(kActionLeft);
    if (col < board_.board_size() - 1) actions.push_back(kActionRight);
  }
  if (board_.IsLegalMove(VirtualPointFrom2DPoint(cursor), to_play_))
    actions.push_back(kActionPlaceStone);
  actions.push_back(kActionPass);
  return actions;
}

std::string CursorGoState::ActionToString(Player player, Action action) const {
  static constexpr std::array<absl::string_view, kNumDistinctActions>
      kActionNames{"Up", "Down", "Left", "Right", "Place Stone", "Pass"};
  if (action < 0 || action >= kActionNames.size()) {
    return absl::StrFormat("invalid action %d", action);
  }
  return std::string(kActionNames[action]);
}

std::string CursorGoState::ToString() const {
  std::stringstream ss;
  ss << "CursorGoState(komi=" << komi_;
  if (!is_terminal_) ss << ", to_play=" << GoColorToString(to_play_);
  ss << ", history.size()=" << history_.size();
  if (!is_terminal_) ss << ", cursor_moves_count=" << cursor_moves_count_;
  ss << ")\n" << board_;
  if (!is_terminal_)
    ss << "\nCursor: "
       << VirtualPointToString(
              VirtualPointFrom2DPoint(cursor_[ColorToPlayer(to_play_)]));
  return ss.str();
}

std::vector<double> CursorGoState::Returns() const {
  if (!is_terminal_) return {0.0, 0.0};

  if (superko_) {
    // Superko rules (https://senseis.xmp.net/?Superko) are complex and vary
    // between rulesets.
    // For simplicity and because superkos are very rare, we just treat them as
    // a draw.
    return {kDrawUtility, kDrawUtility};
  }

  // Score with Tromp-Taylor.
  float black_score = TrompTaylorScore(board_, komi_, handicap_);

  std::vector<double> returns(kNumPlayers);
  if (black_score > 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = kWinUtility;
    returns[ColorToPlayer(GoColor::kWhite)] = kLossUtility;
  } else if (black_score < 0) {
    returns[ColorToPlayer(GoColor::kBlack)] = kLossUtility;
    returns[ColorToPlayer(GoColor::kWhite)] = kWinUtility;
  } else {
    returns[ColorToPlayer(GoColor::kBlack)] = kDrawUtility;
    returns[ColorToPlayer(GoColor::kWhite)] = kDrawUtility;
  }
  return returns;
}

std::unique_ptr<State> CursorGoState::Clone() const {
  return std::unique_ptr<State>(new CursorGoState(*this));
}

void CursorGoState::DoApplyAction(Action action) {
  if (action == kActionPlaceStone || action == kActionPass) {
    VirtualPoint point =
        (action == kActionPass)
            ? go::kVirtualPass
            : VirtualPointFrom2DPoint(cursor_[ColorToPlayer(to_play_)]);
    SPIEL_CHECK_TRUE(board_.PlayMove(point, to_play_));
    is_terminal_ = last_move_was_pass_ && (action == kActionPass);
    last_move_was_pass_ = (action == kActionPass);
    to_play_ = OppColor(to_play_);
    cursor_moves_count_ = 0;

    bool was_inserted = repetitions_.insert(board_.HashValue()).second;
    if (!was_inserted && action == kActionPlaceStone) {
      // We have encountered this position before.
      superko_ = true;
    }
  } else {
    switch (action) {
      case kActionUp:
        cursor_[ColorToPlayer(to_play_)].first++;
        break;
      case kActionDown:
        cursor_[ColorToPlayer(to_play_)].first--;
        break;
      case kActionLeft:
        cursor_[ColorToPlayer(to_play_)].second--;
        break;
      case kActionRight:
        cursor_[ColorToPlayer(to_play_)].second++;
        break;
      default:
        SpielFatalError(absl::StrCat("Invalid action ", action));
    }
    ++cursor_moves_count_;
  }
}

void CursorGoState::ResetBoard() {
  board_.Clear();
  const int middle = board_.board_size() / 2;
  cursor_[0] = {middle, middle};
  cursor_[1] = {middle, middle};
  cursor_moves_count_ = 0;
  if (handicap_ < 2) {
    to_play_ = GoColor::kBlack;
  } else {
    for (VirtualPoint p : HandicapStones(handicap_)) {
      board_.PlayMove(p, GoColor::kBlack);
    }
    to_play_ = GoColor::kWhite;
  }

  repetitions_.clear();
  repetitions_.insert(board_.HashValue());
  superko_ = false;
  is_terminal_ = false;
  last_move_was_pass_ = false;
}

CursorGoGame::CursorGoGame(const GameParameters& params)
    : Game(kGameType, params),
      komi_(ParameterValue<double>("komi")),
      board_size_(ParameterValue<int>("board_size")),
      handicap_(ParameterValue<int>("handicap")),
      max_cursor_moves_(ParameterValue<int>("max_cursor_moves")) {}

}  // namespace cursor_go
}  // namespace open_spiel
