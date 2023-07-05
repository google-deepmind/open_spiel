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

#include "open_spiel/games/squadro.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace squadro {
namespace {

// Facts about the game
const GameType kGameType{
    /*short_name=*/"squadro",
    /*long_name=*/"Squadro",
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
  return std::shared_ptr<const Game>(new SquadroGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string PlayerToString(Player player) {
  switch (player) {
    case 0:
      return "P0";
    case 1:
      return "P1";
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
  }
}

std::string CellToString(int row, int col, 
                         const std::array<std::array<Position, kRows>, kNumPlayers>& board) {
  if (board[0][col].position == row) {
    if (board[0][col].direction == TokenState::forward) {
      return "^";
    } else if (board[0][col].direction == TokenState::backward) {
      return "v";
    }
  }
  if (board[1][row].position == col) {
    if (board[1][row].direction == TokenState::forward) {
      return ">";
    } else if (board[1][row].direction == TokenState::backward) {
      return "<";
    }
  }
  return ".";
}

}  // namespace

int SquadroState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return current_player_;
  }
}

bool SquadroState::OverpassOpponent(int opponent, int player_position, Action move) {
  bool overpassOpponent = false;

  while (board_[opponent][player_position].position == move + 1) {
      overpassOpponent = true;
      // Send opponent player back to default position
      if (board_[opponent][player_position].direction == TokenState::forward) {
        board_[opponent][player_position].position = 0;
      } else {
        board_[opponent][player_position].position = 6;
      }
      player_position += board_[current_player_][move + 1].direction == TokenState::forward ? 1 : -1;
    }
  board_[current_player_][move + 1].position = player_position;
  return overpassOpponent;
}

void SquadroState::DoApplyAction(Action move) {
  int playerPosition = board_[current_player_][move + 1].position;
  TokenState playerDirection = board_[current_player_][move + 1].direction;
  int steps = playerDirection == TokenState::forward ? movements_[current_player_][move].forward : movements_[current_player_][move].backward;
  int other_player = 1 - current_player_;
  bool overpassOpponent = false;
  int unit_move = playerDirection == TokenState::forward ? 1 : -1;
  int finalPosition = playerPosition + steps;

  do {
    playerPosition += unit_move;
    board_[current_player_][move + 1].position = playerPosition;
    overpassOpponent = OverpassOpponent(other_player, playerPosition, move);
  } while (playerPosition > 0 && playerPosition < kRows - 1 && unit_move * playerPosition < unit_move * finalPosition && !overpassOpponent);

  if (board_[current_player_][move + 1].position == 0) {
    board_[current_player_][move + 1].direction = TokenState::missing; // Token removed from the board
    ++missing_tokens_[current_player_];
  } else if (board_[current_player_][move + 1].position == 6) {
    board_[current_player_][move + 1].direction = TokenState::backward; // Invert token direction when it reaches the end of the board
  }

  if (missing_tokens_[current_player_] == 4) {
    outcome_ = current_player_ == 0 ? Outcome::kPlayer1 : Outcome::kPlayer2;
  }

  ++moves_made_;
  if (moves_made_ >= 200) {
    outcome_ = Outcome::kDraw;
  }

  current_player_ = other_player;
}

std::vector<Action> SquadroState::LegalActions() const {
  std::vector<Action> moves;
  if (IsTerminal()) return moves;
  for (int pos = 0; pos < kNumActions; ++pos) {
    if (board_[current_player_][pos + 1].direction != TokenState::missing) moves.push_back(pos);
  }
  return moves;
}

std::string SquadroState::ActionToString(Player player,
                                             Action action_id) const {
  return absl::StrCat(PlayerToString(player), action_id);
}

SquadroState::SquadroState(std::shared_ptr<const Game> game)
    : State(game) {
      for (int player = 0; player <= 1; ++player) {
        for (int pos = 0; pos < kRows; ++pos) {
          if (pos == 0 || pos == 6) {
            // There are only 5 pieces per player. No pieces are present in the corners.
            board_[player][pos] = {0, TokenState::missing};
          } else {
            board_[player][pos] = {0, TokenState::forward};
          }
        }
      }
}

std::string SquadroState::ToString() const {
  std::string str;
  for (int row = kRows - 1; row >= 0; --row) {
    for (int col = 0; col < kCols; ++col) {
      str.append(CellToString(row, col, board_));
    }
    str.append("\n");
  }
  str.append("C");
  int current_player = CurrentPlayer();
  if (current_player == kTerminalPlayerId) {
    str.append("2");
  } else {
    str.append(std::to_string(CurrentPlayer()));
  }
  return str;
}

bool SquadroState::IsTerminal() const {
  return outcome_ != Outcome::kUnknown;
}

std::vector<double> SquadroState::Returns() const {
  if (outcome_ == Outcome::kPlayer1) return {1.0, -1.0};
  if (outcome_ == Outcome::kPlayer2) return {-1.0, 1.0};
  return {0.0, 0.0};
}

std::string SquadroState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string SquadroState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

int SquadroState::CellToInt(int row, int col) const{
  std::string str = CellToString(row, col, board_);
  return cell_state_map_.at(str);
}

void SquadroState::ObservationTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  TensorView<2> view(values, {kCellStates, kRows*kCols}, true);
  for (int row = kRows - 1; row >= 0; --row) {
    for (int col = 0; col < kCols; ++col) {
      int cell = row * kCols + col;
      view[{CellToInt(row, col), cell}] = 1.0;
    }
  }
  
}

std::unique_ptr<State> SquadroState::Clone() const {
  return std::unique_ptr<State>(new SquadroState(*this));
}

SquadroGame::SquadroGame(const GameParameters& params)
    : Game(kGameType, params) {}

SquadroState::SquadroState(std::shared_ptr<const Game> game,
                                   const std::string& str)
    : State(game) {

  for (int player = 0; player <= 1; ++player) {
    for (int pos = 0; pos < kRows; ++pos) {
      board_[player][pos] = {0, TokenState::missing};
    }
  }

  int xs = 0;
  int os = 0;
  int r = 6;
  int c = 0;
  for (const char ch : str) {
    switch (ch) {
      case '.':
        break;
      case '^':
        board_[0][c].position = r;
        board_[0][c].direction = TokenState::forward;
        break;
      case '>':
        board_[1][r].position = c;
        board_[1][r].direction = TokenState::forward;
        break;
      case 'v':
        board_[0][c].position = r;
        board_[0][c].direction = TokenState::backward;
        break;
      case '<':
        board_[1][r].position = c;
        board_[1][r].direction = TokenState::backward;
        break;
      case '0':
        current_player_ = 0;
        break;
      case '1':
        current_player_ = 1;
        break;
      case '2':
        current_player_ = kTerminalPlayerId;
        break;
    }
    if (ch == '.' || ch == '^' || ch == '>' || ch == 'v' || ch == '<' || ch == 'C') {
      ++c;
      if (c >= kCols) {
        r--;
        c = 0;
      }
    }
  }
  SPIEL_CHECK_TRUE(r == -1 && ("Problem parsing state (incorrect rows)."));
  SPIEL_CHECK_TRUE(c == 1 &&
                   ("Problem parsing state (column value should be 0)"));

  int count_p0_tokens = 0;
  int count_p1_tokens = 0;
  for (int i = 0; i < kNumActions; ++i) {
    count_p0_tokens += board_[0][i + 1].direction == TokenState::missing ? 0 : 1;
    count_p1_tokens += board_[1][i + 1].direction == TokenState::missing ? 0 : 1;
  }

  if (count_p0_tokens == 1) {
    outcome_ = Outcome::kPlayer1;
  } else if (count_p1_tokens == 1) {
    outcome_ = Outcome::kPlayer2;
  }
  SPIEL_CHECK_FALSE(count_p0_tokens == 1 && count_p1_tokens == 1 && 
  ("P1 and P2 cannot both have a single piece."));
}

}  // namespace squadro
}  // namespace open_spiel
