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

#include "open_spiel/games/capture_the_flag/capture_the_flag.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/simultaneous_move_game.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace capture_the_flag {
namespace {

// Default parameter values.
constexpr int kDefaultHorizon = 1000;
constexpr bool kDefaultZeroSum = true;
constexpr int kDefaultScoreLimit = 1;

const GameType kGameType{/*short_name=*/"capture_the_flag",
                         /*long_name=*/"Capture the Flag",
                         GameType::Dynamics::kSimultaneous,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kPerfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/false,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"horizon", GameParameter(kDefaultHorizon)},
                          {"zero_sum", GameParameter(kDefaultZeroSum)},
                          {"score_limit", GameParameter(kDefaultScoreLimit)},
                          {"grid", GameParameter(std::string(kDefaultGrid))}}};

GameType GameTypeForParams(const GameParameters& params) {
  GameType game_type = kGameType;
  bool is_zero_sum = kDefaultZeroSum;
  auto it = params.find("zero_sum");
  if (it != params.end()) is_zero_sum = it->second.bool_value();
  game_type.utility = is_zero_sum ? GameType::Utility::kZeroSum
                                  : GameType::Utility::kGeneralSum;
  return game_type;
}

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CaptureTheFlagGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

Grid ParseGrid(const std::string& grid_string) {
  Grid grid;
  int row = 0;
  int col = 0;
  int count_a = 0;
  int count_b = 0;
  for (char c : grid_string) {
    if (c == '\n') {
      row += 1;
      col = 0;
      continue;
    }
    if (row >= grid.num_rows) grid.num_rows = row + 1;
    if (col >= grid.num_cols) grid.num_cols = col + 1;
    switch (c) {
      case '.':
        break;
      case '*':
        grid.obstacles.emplace_back(row, col);
        break;
      case 'a':
        grid.a_base = {row, col};
        ++count_a;
        break;
      case 'b':
        grid.b_base = {row, col};
        ++count_b;
        break;
      default:
        SpielFatalError(absl::StrCat("Invalid char '", std::string(1, c),
                                     "' at grid (", row, ",", col, ")"));
    }
    col += 1;
  }
  SPIEL_CHECK_EQ(count_a, 1);
  SPIEL_CHECK_EQ(count_b, 1);
  SPIEL_CHECK_GE(grid.num_cols, 3);  // Need at least one column per territory.
  return grid;
}

// Action -> (row offset, col offset). 'Stay' is (0, 0).
constexpr std::array<int, kNumDistinctActions> kRowOffset = {-1, 0, 1, 0, 0};
constexpr std::array<int, kNumDistinctActions> kColOffset = {0, 1, 0, -1, 0};

const char* ActionName(int action) {
  switch (action) {
    case kMoveNorth:
      return "North";
    case kMoveEast:
      return "East";
    case kMoveSouth:
      return "South";
    case kMoveWest:
      return "West";
    case kStay:
      return "Stay";
    default:
      return "Invalid";
  }
}

}  // namespace

CaptureTheFlagState::CaptureTheFlagState(std::shared_ptr<const Game> game,
                                         const Grid& grid, int horizon,
                                         bool zero_sum, int score_limit)
    : SimMoveState(game),
      grid_(grid),
      horizon_(horizon),
      zero_sum_(zero_sum),
      score_limit_(score_limit) {
  cur_player_ = kSimultaneousPlayerId;
  player_row_[0] = grid_.a_base.first;
  player_col_[0] = grid_.a_base.second;
  player_row_[1] = grid_.b_base.first;
  player_col_[1] = grid_.b_base.second;
  flag_row_[0] = grid_.a_base.first;
  flag_col_[0] = grid_.a_base.second;
  flag_row_[1] = grid_.b_base.first;
  flag_col_[1] = grid_.b_base.second;
}

std::string CaptureTheFlagState::ActionToString(Player player,
                                                Action action_id) const {
  if (player == kSimultaneousPlayerId) {
    return FlatJointActionToString(action_id);
  }
  if (player == kChancePlayerId) {
    if (action_id == kChanceInit0Action) return "(A resolves first)";
    if (action_id == kChanceInit1Action) return "(B resolves first)";
    SpielFatalError(absl::StrCat("Invalid chance action: ", action_id));
  }
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, kNumDistinctActions);
  return ActionName(action_id);
}

bool CaptureTheFlagState::InBounds(int r, int c) const {
  return r >= 0 && c >= 0 && r < grid_.num_rows && c < grid_.num_cols;
}

bool CaptureTheFlagState::IsObstacle(int r, int c) const {
  for (const auto& obs : grid_.obstacles) {
    if (obs.first == r && obs.second == c) return true;
  }
  return false;
}

bool CaptureTheFlagState::InHomeTerritory(Player player, int r, int c) const {
  // A's home territory is the columns strictly left of the centre; B's home
  // territory is the columns strictly right of the centre. With an
  // odd-width grid the centre column is neutral; with an even-width grid
  // there is no neutral column and the split is exactly in half.
  const int centre = grid_.num_cols / 2;
  if (grid_.num_cols % 2 == 1) {
    if (player == 0) return c < centre;
    return c > centre;
  }
  if (player == 0) return c < centre;
  return c >= centre;
}

void CaptureTheFlagState::ResetFlagToHome(Player flag_owner) {
  flag_holder_[flag_owner] = -1;
  if (flag_owner == 0) {
    flag_row_[0] = grid_.a_base.first;
    flag_col_[0] = grid_.a_base.second;
  } else {
    flag_row_[1] = grid_.b_base.first;
    flag_col_[1] = grid_.b_base.second;
  }
}

void CaptureTheFlagState::RespawnPlayer(Player player) {
  if (player == 0) {
    player_row_[0] = grid_.a_base.first;
    player_col_[0] = grid_.a_base.second;
  } else {
    player_row_[1] = grid_.b_base.first;
    player_col_[1] = grid_.b_base.second;
  }
}

void CaptureTheFlagState::ResolveMove(Player player, int move) {
  SPIEL_CHECK_GE(move, 0);
  SPIEL_CHECK_LT(move, kNumDistinctActions);

  if (move == kStay) return;

  const int new_r = player_row_[player] + kRowOffset[move];
  const int new_c = player_col_[player] + kColOffset[move];
  if (!InBounds(new_r, new_c)) return;
  if (IsObstacle(new_r, new_c)) return;

  const Player opponent = 1 - player;
  if (player_row_[opponent] == new_r && player_col_[opponent] == new_c) {
    // Cell is blocked by the opponent: no tag, no move.
    return;
  }

  // Move succeeds. Update player position. If the player is carrying a flag,
  // move the flag with them.
  player_row_[player] = new_r;
  player_col_[player] = new_c;
  const Player opponent_flag = opponent;
  if (flag_holder_[opponent_flag] == player) {
    flag_row_[opponent_flag] = new_r;
    flag_col_[opponent_flag] = new_c;
  }

  // Pickup: stepping onto the opponent's flag while the flag is loose (no
  // current carrier) and at its home base.
  const int opp_flag_home_r =
      (opponent == 0) ? grid_.a_base.first : grid_.b_base.first;
  const int opp_flag_home_c =
      (opponent == 0) ? grid_.a_base.second : grid_.b_base.second;
  if (flag_holder_[opponent_flag] == -1 && new_r == opp_flag_home_r &&
      new_c == opp_flag_home_c) {
    flag_holder_[opponent_flag] = player;
    flag_row_[opponent_flag] = new_r;
    flag_col_[opponent_flag] = new_c;
  }

  // Capture: carrier arrives at own base, and own flag is at its home base.
  const int own_base_r =
      (player == 0) ? grid_.a_base.first : grid_.b_base.first;
  const int own_base_c =
      (player == 0) ? grid_.a_base.second : grid_.b_base.second;
  const bool own_flag_home =
      (flag_holder_[player] == -1 && flag_row_[player] == own_base_r &&
       flag_col_[player] == own_base_c);
  if (flag_holder_[opponent_flag] == player && new_r == own_base_r &&
      new_c == own_base_c && own_flag_home) {
    score_[player] += 1;
    ResetFlagToHome(opponent_flag);
    if (score_[player] >= score_limit_) {
      winner_ = player;
      rewards_[player] += zero_sum_ ? 1.0 : 1.0;
      rewards_[opponent] += zero_sum_ ? -1.0 : 0.0;
      returns_[player] += zero_sum_ ? 1.0 : 1.0;
      returns_[opponent] += zero_sum_ ? -1.0 : 0.0;
    }
  }
}

void CaptureTheFlagState::ResolveTags() {
  // A carrier in the defender's home territory and Manhattan-adjacent to the
  // defender is tagged. Both players carrying are handled, but in practice
  // at most one flag is loose per side at any moment in 1v1 CTF.
  for (Player carrier = 0; carrier < 2; ++carrier) {
    const Player flag_owner = 1 - carrier;
    if (flag_holder_[flag_owner] != carrier) continue;
    const Player defender = flag_owner;
    if (!InHomeTerritory(defender, player_row_[carrier],
                         player_col_[carrier])) {
      continue;
    }
    const int dr = std::abs(player_row_[carrier] - player_row_[defender]);
    const int dc = std::abs(player_col_[carrier] - player_col_[defender]);
    if (dr + dc == 1) {
      ResetFlagToHome(flag_owner);
      RespawnPlayer(carrier);
    }
  }
}

void CaptureTheFlagState::DoApplyActions(const std::vector<Action>& moves) {
  SPIEL_CHECK_EQ(moves.size(), 2);
  SPIEL_CHECK_EQ(cur_player_, kSimultaneousPlayerId);
  moves_[0] = moves[0];
  moves_[1] = moves[1];
  cur_player_ = kChancePlayerId;
}

void CaptureTheFlagState::DoApplyAction(Action action_id) {
  if (IsSimultaneousNode()) {
    ApplyFlatJointAction(action_id);
    return;
  }
  SPIEL_CHECK_TRUE(IsChanceNode());
  SPIEL_CHECK_GE(action_id, 0);
  SPIEL_CHECK_LT(action_id, kNumChanceOutcomes);

  rewards_ = {0.0, 0.0};
  if (action_id == kChanceInit0Action) {
    ResolveMove(0, moves_[0]);
    if (winner_ == kNoWinnerYetId) ResolveMove(1, moves_[1]);
  } else {
    ResolveMove(1, moves_[1]);
    if (winner_ == kNoWinnerYetId) ResolveMove(0, moves_[0]);
  }
  if (winner_ == kNoWinnerYetId) ResolveTags();
  total_moves_ += 1;
  cur_player_ = kSimultaneousPlayerId;
}

std::vector<Action> CaptureTheFlagState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  std::vector<Action> actions(kNumDistinctActions);
  for (int i = 0; i < kNumDistinctActions; ++i) actions[i] = i;
  return actions;
}

ActionsAndProbs CaptureTheFlagState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  return {{kChanceInit0Action, 0.5}, {kChanceInit1Action, 0.5}};
}

std::string CaptureTheFlagState::ToString() const {
  std::string result;
  for (int r = 0; r < grid_.num_rows; ++r) {
    for (int c = 0; c < grid_.num_cols; ++c) {
      char ch = '.';
      if (IsObstacle(r, c)) {
        ch = '*';
      }
      // Players overwrite obstacles in the rendering, but obstacles can't
      // share cells with players, so this is just for safety.
      if (player_row_[0] == r && player_col_[0] == c) ch = 'A';
      if (player_row_[1] == r && player_col_[1] == c) ch = 'B';
      // Loose flags render with their owner's lowercase letter.
      if (flag_holder_[0] == -1 && flag_row_[0] == r && flag_col_[0] == c) {
        if (ch == 'A') {
          ch = 'A';  // A is standing on their own flag at base; show player.
        } else {
          ch = 'a';
        }
      }
      if (flag_holder_[1] == -1 && flag_row_[1] == r && flag_col_[1] == c) {
        if (ch == 'B') {
          ch = 'B';
        } else {
          ch = 'b';
        }
      }
      result += ch;
    }
    absl::StrAppend(&result, "\n");
  }
  absl::StrAppend(&result, "Carrier(A's flag): ", flag_holder_[0],
                  " Carrier(B's flag): ", flag_holder_[1], "\n");
  absl::StrAppend(&result, "Score: A=", score_[0], " B=", score_[1], "\n");
  absl::StrAppend(&result, "Moves: ", total_moves_, "/",
                  horizon_ < 0 ? -1 : horizon_, "\n");
  if (IsChanceNode()) absl::StrAppend(&result, "Chance Node\n");
  return result;
}

std::string CaptureTheFlagState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

bool CaptureTheFlagState::IsTerminal() const {
  if (winner_ != -1) return true;
  if (horizon_ >= 0 && total_moves_ >= horizon_) return true;
  return false;
}

std::vector<double> CaptureTheFlagState::Rewards() const { return rewards_; }

std::vector<double> CaptureTheFlagState::Returns() const { return returns_; }

int CaptureTheFlagState::ObservationPlaneForPlayer(int r, int c,
                                                   Player player) const {
  if (player_row_[player] == r && player_col_[player] == c) return player;
  return -1;
}

int CaptureTheFlagState::ObservationPlaneForFlag(int r, int c,
                                                 Player flag_owner) const {
  if (flag_row_[flag_owner] == r && flag_col_[flag_owner] == c) {
    return 2 + flag_owner;
  }
  return -1;
}

void CaptureTheFlagState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  TensorView<3> view(values, {kCellStates, grid_.num_rows, grid_.num_cols},
                     /*reset=*/true);
  for (int r = 0; r < grid_.num_rows; ++r) {
    for (int c = 0; c < grid_.num_cols; ++c) {
      if (IsObstacle(r, c)) view[{4, r, c}] = 1.0;
    }
  }
  for (Player p = 0; p < 2; ++p) {
    view[{p, player_row_[p], player_col_[p]}] = 1.0;
  }
  for (Player f = 0; f < 2; ++f) {
    view[{2 + f, flag_row_[f], flag_col_[f]}] = 1.0;
  }
}

std::unique_ptr<State> CaptureTheFlagState::Clone() const {
  return std::unique_ptr<State>(new CaptureTheFlagState(*this));
}

CaptureTheFlagGame::CaptureTheFlagGame(const GameParameters& params)
    : SimMoveGame(GameTypeForParams(params), params),
      grid_(ParseGrid(ParameterValue<std::string>("grid"))),
      horizon_(ParameterValue<int>("horizon")),
      zero_sum_(ParameterValue<bool>("zero_sum")),
      score_limit_(ParameterValue<int>("score_limit")) {
  SPIEL_CHECK_GE(score_limit_, 1);
}

std::unique_ptr<State> CaptureTheFlagGame::NewInitialState() const {
  return std::unique_ptr<State>(new CaptureTheFlagState(
      shared_from_this(), grid_, horizon_, zero_sum_, score_limit_));
}

double CaptureTheFlagGame::MinUtility() const {
  if (zero_sum_) return -1.0;
  return 0.0;
}

double CaptureTheFlagGame::MaxUtility() const { return 1.0; }

absl::optional<double> CaptureTheFlagGame::UtilitySum() const {
  if (zero_sum_) return 0.0;
  return std::nullopt;
}

std::vector<int> CaptureTheFlagGame::ObservationTensorShape() const {
  return {kCellStates, grid_.num_rows, grid_.num_cols};
}

}  // namespace capture_the_flag
}  // namespace open_spiel
