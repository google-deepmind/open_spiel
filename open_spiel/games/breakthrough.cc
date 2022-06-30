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

#include "open_spiel/games/breakthrough.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/game_parameters.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace breakthrough {
namespace {

// Number of unique directions each piece can take.
constexpr int kNumDirections = 6;

// Numbers of rows needed to have 2 full rows of pieces.
constexpr int kNumRowsForFullPieces = 6;

// Direction offsets for black, then white.
constexpr std::array<int, kNumDirections> kDirRowOffsets = {
    {1, 1, 1, -1, -1, -1}};

constexpr std::array<int, kNumDirections> kDirColOffsets = {
    {-1, 0, 1, -1, 0, 1}};

// Facts about the game
const GameType kGameType{/*short_name=*/"breakthrough",
                         /*long_name=*/"Breakthrough",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kDeterministic,
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
                         {{"rows", GameParameter(kDefaultRows)},
                          {"columns", GameParameter(kDefaultColumns)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BreakthroughGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

int StateToPlayer(CellState state) {
  switch (state) {
    case CellState::kBlack:
      return 0;
    case CellState::kWhite:
      return 1;
    default:
      SpielFatalError("No player id for this cell state");
  }
}

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kBlack;
    case 1:
      return CellState::kWhite;
    default:
      SpielFatalError("No cell state for this player id");
  }
}

std::string CellToString(CellState state) {
  switch (state) {
    case CellState::kBlack:
      return "b";
    case CellState::kWhite:
      return "w";
    case CellState::kEmpty:
      return ".";
    default:
      SpielFatalError("Unrecognized cell state");
  }
}

CellState OpponentState(CellState state) {
  return PlayerToState(1 - StateToPlayer(state));
}

std::string RowLabel(int rows, int row) {
  std::string label = "";
  label += static_cast<char>('1' + (rows - 1 - row));
  return label;
}

std::string ColLabel(int col) {
  std::string label = "";
  label += static_cast<char>('a' + col);
  return label;
}

}  // namespace

BreakthroughState::BreakthroughState(std::shared_ptr<const Game> game, int rows,
                                     int cols)
    : State(game), rows_(rows), cols_(cols) {
  SPIEL_CHECK_GT(rows_, 1);
  SPIEL_CHECK_GT(cols_, 1);

  board_ = std::vector<CellState>(rows_ * cols_, CellState::kEmpty);
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      // Only use two rows if there are at least 6 rows.
      if (r == 0 || (rows_ >= kNumRowsForFullPieces && r == 1)) {
        SetBoard(r, c, CellState::kBlack);
      } else if (r == (rows_ - 1) ||
                 (rows_ >= kNumRowsForFullPieces && r == (rows_ - 2))) {
        SetBoard(r, c, CellState::kWhite);
      }
    }
  }

  winner_ = kInvalidPlayer;
  pieces_[0] = pieces_[1] = cols_ * (rows_ >= kNumRowsForFullPieces ? 2 : 1);
  cur_player_ = 0;
  total_moves_ = 0;
}

int BreakthroughState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

void BreakthroughState::DoApplyAction(Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
  int r1 = values[0];
  int c1 = values[1];
  int dir = values[2];
  bool capture = values[3] == 1;
  int r2 = r1 + kDirRowOffsets[dir];
  int c2 = c1 + kDirColOffsets[dir];

  SPIEL_CHECK_TRUE(InBounds(r1, c1));
  SPIEL_CHECK_TRUE(InBounds(r2, c2));

  // Remove pieces if captured.
  if (board(r2, c2) == CellState::kWhite) {
    pieces_[StateToPlayer(CellState::kWhite)]--;
    SPIEL_CHECK_EQ(board(r1, c1), CellState::kBlack);
    SPIEL_CHECK_EQ(cur_player_, StateToPlayer(CellState::kBlack));
  } else if (board(r2, c2) == CellState::kBlack) {
    pieces_[StateToPlayer(CellState::kBlack)]--;
    SPIEL_CHECK_EQ(board(r1, c1), CellState::kWhite);
    SPIEL_CHECK_EQ(cur_player_, StateToPlayer(CellState::kWhite));
  }

  // Move the piece.
  if (capture) {
    SPIEL_CHECK_EQ(board(r2, c2), OpponentState(board(r1, c1)));
  }
  SetBoard(r2, c2, board(r1, c1));
  SetBoard(r1, c1, CellState::kEmpty);

  // Check for winner.
  if (cur_player_ == 0 && r2 == (rows_ - 1)) {
    winner_ = 0;
  } else if (cur_player_ == 1 && r2 == 0) {
    winner_ = 1;
  }

  cur_player_ = NextPlayerRoundRobin(cur_player_, kNumPlayers);
  total_moves_++;
}

std::string BreakthroughState::ActionToString(Player player,
                                              Action action) const {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
  int r1 = values[0];
  int c1 = values[1];
  int dir = values[2];
  bool capture = values[3] == 1;
  int r2 = r1 + kDirRowOffsets[dir];
  int c2 = c1 + kDirColOffsets[dir];

  std::string action_string = "";
  absl::StrAppend(&action_string, ColLabel(c1));
  absl::StrAppend(&action_string, RowLabel(rows_, r1));
  absl::StrAppend(&action_string, ColLabel(c2));
  absl::StrAppend(&action_string, RowLabel(rows_, r2));
  if (capture) {
    absl::StrAppend(&action_string, "*");
  }

  return action_string;
}

std::vector<Action> BreakthroughState::LegalActions() const {
  std::vector<Action> movelist;
  if (IsTerminal()) return movelist;
  const Player player = CurrentPlayer();
  CellState mystate = PlayerToState(player);
  std::vector<int> action_bases = {rows_, cols_, kNumDirections, 2};
  std::vector<int> action_values = {0, 0, 0, 0};

  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      if (board(r, c) == mystate) {
        for (int o = 0; o < kNumDirections / 2; o++) {
          int dir = player * kNumDirections / 2 + o;
          int rp = r + kDirRowOffsets[dir];
          int cp = c + kDirColOffsets[dir];

          if (InBounds(rp, cp)) {
            action_values[0] = r;
            action_values[1] = c;
            action_values[2] = dir;
            if (board(rp, cp) == CellState::kEmpty) {
              // Regular move.
              action_values[3] = 0;
              movelist.push_back(
                  RankActionMixedBase(action_bases, action_values));
            } else if ((o == 0 || o == 2) &&
                       board(rp, cp) == OpponentState(mystate)) {
              // Capture move (can only capture diagonally)
              action_values[3] = 1;
              movelist.push_back(
                  RankActionMixedBase(action_bases, action_values));
            }
          }
        }
      }
    }
  }

  return movelist;
}

bool BreakthroughState::InBounds(int r, int c) const {
  return (r >= 0 && r < rows_ && c >= 0 && c < cols_);
}

std::string BreakthroughState::ToString() const {
  std::string result = "";

  for (int r = 0; r < rows_; r++) {
    absl::StrAppend(&result, RowLabel(rows_, r));

    for (int c = 0; c < cols_; c++) {
      absl::StrAppend(&result, CellToString(board(r, c)));
    }

    result.append("\n");
  }

  absl::StrAppend(&result, " ");
  for (int c = 0; c < cols_; c++) {
    absl::StrAppend(&result, ColLabel(c));
  }
  absl::StrAppend(&result, "\n");

  return result;
}

int BreakthroughState::observation_plane(int r, int c) const {
  int plane = -1;
  switch (board(r, c)) {
    case CellState::kBlack:
      plane = 0;
      break;
    case CellState::kWhite:
      plane = 1;
      break;
    case CellState::kEmpty:
      plane = 2;
      break;
    default:
      std::cerr << "Invalid character on board: " << CellToString(board(r, c))
                << std::endl;
      plane = -1;
      break;
  }

  return plane;
}

bool BreakthroughState::IsTerminal() const {
  return (winner_ >= 0 || (pieces_[0] == 0 || pieces_[1] == 0));
}

std::vector<double> BreakthroughState::Returns() const {
  if (winner_ == 0 || pieces_[1] == 0) {
    return {1.0, -1.0};
  } else if (winner_ == 1 || pieces_[0] == 0) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string BreakthroughState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void BreakthroughState::ObservationTensor(Player player,
                                          absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<3> view(values, {kCellStates, rows_, cols_}, true);

  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      int plane = observation_plane(r, c);
      SPIEL_CHECK_TRUE(plane >= 0 && plane < kCellStates);
      view[{plane, r, c}] = 1.0;
    }
  }
}

void BreakthroughState::UndoAction(Player player, Action action) {
  std::vector<int> values =
      UnrankActionMixedBase(action, {rows_, cols_, kNumDirections, 2});
  int r1 = values[0];
  int c1 = values[1];
  int dir = values[2];
  bool capture = values[3] == 1;
  int r2 = r1 + kDirRowOffsets[dir];
  int c2 = c1 + kDirColOffsets[dir];

  cur_player_ = PreviousPlayerRoundRobin(cur_player_, 2);
  total_moves_--;

  // Undo win status.
  winner_ = kInvalidPlayer;

  // Move back the piece, and put back the opponent's piece if necessary.
  // The move is (r1, c1) -> (r2, c2) where r is row and c is column.
  SetBoard(r1, c1, board(r2, c2));
  SetBoard(r2, c2, CellState::kEmpty);
  if (capture) {
    if (board(r1, c1) == CellState::kWhite) {
      // It was a white move: put back the black piece.
      SetBoard(r2, c2, CellState::kBlack);
      pieces_[kBlackPlayerId]++;
    } else if (board(r1, c1) == CellState::kBlack) {
      // It was a black move: put back the white piece.
      SetBoard(r2, c2, CellState::kWhite);
      pieces_[kWhitePlayerId]++;
    }
  }
  history_.pop_back();
  --move_number_;
}

std::unique_ptr<State> BreakthroughState::Clone() const {
  return std::unique_ptr<State>(new BreakthroughState(*this));
}

BreakthroughGame::BreakthroughGame(const GameParameters& params)
    : Game(kGameType, params),
      rows_(ParameterValue<int>("rows")),
      cols_(ParameterValue<int>("columns")) {}

int BreakthroughGame::NumDistinctActions() const {
  return rows_ * cols_ * kNumDirections * 2;
}

std::string BreakthroughState::Serialize() const {
  std::string str = "";
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      absl::StrAppend(&str, CellToString(board(r, c)));
    }
  }
  return str;
}

std::unique_ptr<State> BreakthroughGame::DeserializeState(
    const std::string& str) const {
  std::unique_ptr<State> state = NewInitialState();

  if (str.length() != rows_ * cols_) {
    SpielFatalError("Incorrect number of characters in string.");
    return std::unique_ptr<State>();
  }

  BreakthroughState* bstate = dynamic_cast<BreakthroughState*>(state.get());

  bstate->SetPieces(0, 0);
  bstate->SetPieces(1, 0);
  int i = 0;
  for (int r = 0; r < rows_; r++) {
    for (int c = 0; c < cols_; c++) {
      if (str.at(i) == 'b') {
        bstate->SetPieces(0, bstate->pieces(0) + 1);
        bstate->SetBoard(r, c, CellState::kBlack);
      } else if (str.at(i) == 'w') {
        bstate->SetPieces(1, bstate->pieces(1) + 1);
        bstate->SetBoard(r, c, CellState::kWhite);
      } else if (str.at(i) == '.') {
        bstate->SetBoard(r, c, CellState::kEmpty);
      } else {
        std::string error = "Invalid character in std::string: ";
        error += str.at(i);
        SpielFatalError(error);
        return std::unique_ptr<State>();
      }

      i++;
    }
  }

  return state;
}

}  // namespace breakthrough
}  // namespace open_spiel
