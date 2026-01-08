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

#include "open_spiel/games/nine_mens_morris/nine_mens_morris.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace nine_mens_morris {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"nine_mens_morris",
    /*long_name=*/"Nine men's morris",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new NineMensMorrisGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

enum kDirection : int { kNorth = 0, kEast = 1, kSouth = 2, kWest = 3 };

//     0      7      14
// 0:  .------.------.     0, 1, 2
// 1:  |      |      |
// 2:  | .----.----. |     3, 4, 5
// 3:  | |    |    | |
// 4:  | | .--.--. | |     6, 7, 8
// 5:  | | |     | | |
// 6:  .-.-.     .-.-.     9, 10, 11, 12, 13, 14
// 7:  | | |     | | |
// 8:  | | .--.--. | |     15, 16, 17
// 9:  | |    |    | |
// 10: | .----.----. |     18, 19, 20
// 11: |      |      |
// 12: .------.------.     21, 22, 23

constexpr std::array<std::array<int, 2>, kNumPoints> kPointStrCoords = {
    {{0, 0},  {0, 7},  {0, 14}, {2, 2},  {2, 7},   {2, 12}, {4, 4},  {4, 7},
     {4, 10}, {6, 0},  {6, 2},  {6, 4},  {6, 10},  {6, 12}, {6, 14}, {8, 4},
     {8, 7},  {8, 10}, {10, 2}, {10, 7}, {10, 12}, {12, 0}, {12, 7}, {12, 14}}};

constexpr std::array<std::array<int, 4>, kNumPoints> kPointNeighbors = {{
    // N, E, S, W
    {-1, 1, 9, -1},    // 0
    {-1, 2, 4, 0},     // 1
    {-1, -1, 14, 1},   // 2
    {-1, 4, 10, -1},   // 3
    {1, 5, 7, 3},      // 4
    {-1, -1, 13, 4},   // 5
    {-1, 7, 11, -1},   // 6
    {4, 8, -1, 6},     // 7
    {-1, -1, 12, 7},   // 8
    {0, 10, 21, -1},   // 9
    {3, 11, 18, 9},    // 10
    {6, -1, 15, 10},   // 11
    {8, 13, 17, -1},   // 12
    {5, 14, 20, 12},   // 13
    {2, -1, 23, 13},   // 14
    {11, 16, -1, -1},  // 15
    {-1, 17, 19, 15},  // 16
    {12, -1, -1, 16},  // 17
    {10, 19, -1, -1},  // 18
    {16, 20, 22, 18},  // 19
    {13, -1, -1, 19},  // 20
    {9, 22, -1, -1},   // 21
    {19, 23, -1, 21},  // 22
    {14, -1, -1, 22}   // 23
}};

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kWhite;
    case 1:
      return CellState::kBlack;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

const char* PlayerToStr(Player player) {
  switch (player) {
    case 0:
      return "W";
    case 1:
      return "B";
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return "";
  }
}

char StateToChar(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return '.';
    case CellState::kWhite:
      return 'W';
    case CellState::kBlack:
      return 'B';
    default:
      SpielFatalError("Unknown state.");
  }
}

Player StateToPlayer(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return kInvalidPlayer;
    case CellState::kWhite:
      return 0;
    case CellState::kBlack:
      return 1;
    default:
      SpielFatalError("Unknown state.");
  }
}

Action ToMoveAction(int source, int dest) {
  return kNumPoints + (source * kNumPoints + dest);
}

void FromMoveAction(Action action, int* source, int* dest) {
  action -= kNumPoints;
  *source = action / kNumPoints;
  *dest = action % kNumPoints;
}

void NineMensMorrisState::GetCurrentLegalActions() {
  cur_legal_actions_.clear();

  if (capture_) {
    Player opp = 1 - current_player_;
    bool all_mills = CheckAllMills(opp);
    for (int p = 0; p < kNumPoints; ++p) {
      if (StateToPlayer(board_[p]) == opp) {
        if (all_mills || !CheckInMill(p)) {
          cur_legal_actions_.push_back(p);
        }
      }
    }
  } else {
    if (men_to_deploy_[current_player_] > 0) {
      // Still in phase 1.
      for (int p = 0; p < kNumPoints; ++p) {
        if (board_[p] == CellState::kEmpty) {
          cur_legal_actions_.push_back(p);
        }
      }
    } else if (num_men_[current_player_] > 3) {
      // Phase 2.
      for (int p = 0; p < kNumPoints; ++p) {
        Player player = StateToPlayer(board_[p]);
        if (player == current_player_) {
          for (int dir = 0; dir < 4; ++dir) {
            int np = kPointNeighbors[p][dir];
            if (np > 0 && board_[np] == CellState::kEmpty) {
              cur_legal_actions_.push_back(ToMoveAction(p, np));
            }
          }
        }
      }
      absl::c_sort(cur_legal_actions_);
    } else {
      // Phase 3.
      for (int p = 0; p < kNumPoints; ++p) {
        Player player = StateToPlayer(board_[p]);
        if (player == current_player_) {
          for (int np = 0; np < kNumPoints; ++np) {
            if (p == np) {
              continue;
            }

            if (board_[np] == CellState::kEmpty) {
              cur_legal_actions_.push_back(ToMoveAction(p, np));
            }
          }
        }
      }
      absl::c_sort(cur_legal_actions_);
    }
  }
}

bool NineMensMorrisState::CheckAllMills(Player player) const {
  for (int p = 0; p < kNumPoints; ++p) {
    if (StateToPlayer(board_[p]) == player) {
      if (!CheckInMill(p)) {
        return false;
      }
    }
  }
  return true;
}

bool NineMensMorrisState::CheckInMill(int pos) const {
  Player player = StateToPlayer(board_[pos]);
  if (player == kInvalidPlayer) {
    return false;
  }

  int cp = pos;

  // Direction base: North or East.
  for (int dir_base = 0; dir_base < 2; ++dir_base) {
    int total_matches = 0;

    // Try North + South, then East + West
    for (int dir : {dir_base, dir_base + 2}) {
      cp = pos;

      for (int i = 0; i < 2; ++i) {
        cp = kPointNeighbors[cp][dir];
        if (cp < 0 || StateToPlayer(board_[cp]) != player) {
          break;
        } else {
          total_matches++;
        }
      }
    }

    if (total_matches == 2) {
      return true;
    }
  }

  return false;
}

void NineMensMorrisState::DoApplyAction(Action move) {
  cur_legal_actions_.clear();
  if (move < kNumPoints) {
    if (capture_) {
      // Capture move: choosing which piece to remove.
      SPIEL_CHECK_TRUE(board_[move] != CellState::kEmpty);
      Player opp = StateToPlayer(board_[move]);
      SPIEL_CHECK_TRUE(opp == 1 - current_player_);
      num_men_[opp]--;
      board_[move] = CellState::kEmpty;
      capture_ = false;
      current_player_ = 1 - current_player_;
      num_turns_++;
    } else {
      // Regular move in phase 1 (deployment)
      SPIEL_CHECK_TRUE(board_[move] == CellState::kEmpty);
      board_[move] = PlayerToState(current_player_);
      SPIEL_CHECK_GT(men_to_deploy_[current_player_], 0);
      men_to_deploy_[current_player_]--;
      bool mill = CheckInMill(move);
      if (mill) {
        capture_ = true;
      } else {
        current_player_ = 1 - current_player_;
        num_turns_++;
      }
    }
  } else {
    // Movement move (phase 2 or 3).
    int from_pos = -1, to_pos = -1;
    FromMoveAction(move, &from_pos, &to_pos);
    SPIEL_CHECK_TRUE(StateToPlayer(board_[from_pos]) == current_player_);
    SPIEL_CHECK_TRUE(board_[to_pos] == CellState::kEmpty);
    board_[to_pos] = board_[from_pos];
    board_[from_pos] = CellState::kEmpty;
    bool mill = CheckInMill(to_pos);
    if (mill) {
      capture_ = true;
    } else {
      current_player_ = 1 - current_player_;
      num_turns_++;
    }
  }

  if (cur_legal_actions_.empty()) {
    GetCurrentLegalActions();
  }
}

std::vector<Action> NineMensMorrisState::LegalActions() const {
  if (IsTerminal()) return {};
  return cur_legal_actions_;
}

std::string NineMensMorrisState::ActionToString(Player player,
                                                Action action_id) const {
  return game_->ActionToString(player, action_id);
}

NineMensMorrisState::NineMensMorrisState(std::shared_ptr<const Game> game)
    : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
  GetCurrentLegalActions();
}

std::string NineMensMorrisState::ToString() const {
  std::string str =
      ".------.------.\n"
      "|      |      |\n"
      "| .----.----. |\n"
      "| |    |    | |\n"
      "| | .--.--. | |\n"
      "| | |     | | |\n"
      ".-.-.     .-.-.\n"
      "| | |     | | |\n"
      "| | .--.--. | |\n"
      "| |    |    | |\n"
      "| .----.----. |\n"
      "|      |      |\n"
      ".------.------.\n\n";
  absl::StrAppend(&str, "Current player: ", PlayerToStr(current_player_), "\n");
  absl::StrAppend(&str, "Turn number: ", num_turns_, "\n");
  absl::StrAppend(&str, "Men to deploy: ", men_to_deploy_[0], " ",
                  men_to_deploy_[1], "\n");
  absl::StrAppend(&str, "Num men: ", num_men_[0], " ", num_men_[1], "\n");
  if (capture_) {
    absl::StrAppend(&str, "Last move formed a mill. Capture time!");
  }

  for (int i = 0; i < kNumPoints; ++i) {
    int row = kPointStrCoords[i][0];
    int col = kPointStrCoords[i][1];
    int idx = row * 16 + col;
    str[idx] = StateToChar(board_[i]);
  }
  return str;
}

bool NineMensMorrisState::IsTerminal() const {
  return num_turns_ >= kMaxNumTurns || num_men_[0] <= 2 || num_men_[1] <= 2 ||
         cur_legal_actions_.empty();
}

std::vector<double> NineMensMorrisState::Returns() const {
  std::vector<double> returns = {0.0, 0.0};
  if (cur_legal_actions_.empty()) {
    Player opp = 1 - current_player_;
    returns[current_player_] = -1.0;
    returns[opp] = 1.0;
  } else if (num_men_[0] <= 2) {
    returns[0] = -1.0;
    returns[1] = 1.0;
  } else if (num_men_[1] <= 2) {
    returns[0] = 1.0;
    returns[1] = -1.0;
  }

  return returns;
}

std::string NineMensMorrisState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string NineMensMorrisState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void NineMensMorrisState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::string templ =
      ".--.--.\n"
      "|.-.-.|\n"
      "||...||\n"
      "... ...\n"
      "||...||\n"
      "|.-.-.|\n"
      ".--.--.\n";
  int pos = 0;
  TensorView<3> view(
      values, {kCellStates + 2, kObservationSize, kObservationSize}, true);
  for (int r = 0; r < kObservationSize; ++r) {
    for (int c = 0; c < kObservationSize; ++c) {
      int char_idx = r * 8 + c;
      int plane = -1;
      if (templ[char_idx] == '.') {
        if (board_[pos] == CellState::kWhite) {
          plane = 0;
        } else if (board_[pos] == CellState::kBlack) {
          plane = 1;
        } else {
          plane = 2;
        }
        pos++;
      } else if (templ[char_idx] == '-') {
        plane = 3;
      } else if (templ[char_idx] == '|') {
        plane = 4;
      }

      if (plane >= 0) {
        view[{plane, r, c}] = 1.0;
      }
    }
  }
}

std::unique_ptr<State> NineMensMorrisState::Clone() const {
  return std::unique_ptr<State>(new NineMensMorrisState(*this));
}

std::string NineMensMorrisGame::ActionToString(Player player,
                                               Action action_id) const {
  if (action_id < kNumPoints) {
    return absl::StrCat("Point ", action_id);
  } else {
    int from_pos = 0, to_pos = 0;
    FromMoveAction(action_id, &from_pos, &to_pos);
    return absl::StrCat("Move ", from_pos, " -> ", to_pos);
  }
}

int NineMensMorrisGame::NumDistinctActions() const {
  return kNumPoints + kNumPoints * kNumPoints;
}

NineMensMorrisGame::NineMensMorrisGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace nine_mens_morris
}  // namespace open_spiel
