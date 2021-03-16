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

#include "open_spiel/games/amazons.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

using namespace std;

namespace open_spiel
{
  namespace amazons
  {
    namespace
    {
      // Facts about the game.
      const GameType kGameType{
          /*short_name=*/"amazons",
          /*long_name=*/"Amazons",
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
          /*parameter_specification=*/{} // no parameters
      };

      std::shared_ptr<const Game> Factory(const GameParameters &params)
      {
        return std::shared_ptr<const Game>(new AmazonsGame(params));
      }

      REGISTER_SPIEL_GAME(kGameType, Factory);

    } // namespace

    CellState PlayerToState(Player player)
    {
      switch (player)
      {
      case 0:
        return CellState::kCross;
      case 1:
        return CellState::kNought;
      default:
        SpielFatalError(absl::StrCat("Invalid player id ", player));
        return CellState::kEmpty;
      }
    }

    std::string StateToString(CellState state)
    {
      switch (state)
      {
      case CellState::kEmpty:
        return ".";
      case CellState::kNought:
        return "O";
      case CellState::kCross:
        return "X";
      case CellState::kBlock:
        return "@";
      default:
        SpielFatalError("Unknown state.");
      }
    }


    /* Action Encoding / Decoding */
    Action AmazonsState::EncodeAction(std::vector<unsigned char> v) const
    {
      Action ret = 0;

      for (auto const &value : v)
      {
        ret |= value;
        ret <<= 8;
      }
      ret >>= 8;

      return ret;
    }

    std::vector<unsigned char> AmazonsState::DecodeAction(Action action) const
    {
      std::vector<unsigned char> ret;
      unsigned char to, from, shoot;
      unsigned char mask = 0b11111111;

      from = action & mask;
      action >>= 8;
      to = action & mask;
      action >>= 8;
      shoot = action & mask;
      action >>= 8;

      ret.push_back(shoot);
      ret.push_back(to);
      ret.push_back(from);

      return ret;
    }

    // Takes an action and applies it to the GameState
    void AmazonsState::DoApplyAction(Action move)
    {
      std::vector<unsigned char> parts = DecodeAction(move);

      unsigned char from = parts[0];
      unsigned char to = parts[1];
      unsigned char shoot = parts[2];

      // Checking needs to be more vigorous??
      SPIEL_CHECK_EQ(board_[from], PlayerToState(CurrentPlayer())); // Check if the move can be performed
      SPIEL_CHECK_EQ(board_[to], CellState::kEmpty);                // Check if the move can be performed
      
      
      // SPIEL_CHECK_EQ(board_[shoot], CellState::kEmpty);             // Check if the move can be performed

      // Adjust the state
      board_[from] = CellState::kEmpty;
      board_[to] = PlayerToState(CurrentPlayer());
      board_[shoot] = CellState::kBlock;

      // This is expensive af
      if (LegalActions().size() == 0)
      {
        outcome_ = 1 - current_player_;
      }

      current_player_ = 1 - current_player_;
      num_moves_ += 1;
    }

    /* Move generation functions */
    std::vector<unsigned char> AmazonsState::GetHorizontalMoves(unsigned char cell, std::array<CellState, kNumCells> board) const
    {
      std::vector<unsigned char> horizontalMoves;

      unsigned char col = cell % kNumRows;      // The column the cell is in
      unsigned char left = col;                 // The maximum amount of spaces to check left of given cell
      unsigned char right = kNumCols - col - 1; // The maximal amount of spaces to check right of given cell
      unsigned char focus;

      // <-----X
      // Walk until we encounter a blocking piece or end of row
      int count = 1;
      while (count <= left)
      {
        focus = cell - count;
        if (board[focus] == CellState::kEmpty)
        {
          horizontalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      // X---->
      // Walk until we encounter a blocking piece or end of row
      count = 1;
      while (count <= right)
      {
        focus = cell + count;
        if (board[focus] == CellState::kEmpty)
        {
          horizontalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      return horizontalMoves;
    }

    std::vector<unsigned char> AmazonsState::GetVerticalMoves(unsigned char cell, std::array<CellState, kNumCells> board) const
    {
      std::vector<unsigned char> verticalMoves;

      unsigned char row = cell / kNumRows;     // The row the cell is in
      unsigned char up = row;                  // The maximum amount of spaces to check up of given cell
      unsigned char down = kNumRows - row - 1; // The maximal amount of spaces to check down of given cell
      unsigned char focus;

      // ^
      // |
      // |
      // X
      // Walk until we encounter a blocking piece or end of column
      int count = 1;
      focus = cell;
      while (count <= up)
      {
        focus -= kNumRows;
        if (board[focus] == CellState::kEmpty)
        {
          verticalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      // X
      // |
      // |
      // V
      // Walk until we encounter a blocking piece or end of column
      count = 1;
      focus = cell;
      while (count <= down)
      {
        focus += kNumRows;
        if (board[focus] == CellState::kEmpty)
        {
          verticalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      return verticalMoves;
    }

    std::vector<unsigned char> AmazonsState::GetDiagonalMoves(unsigned char cell, std::array<CellState, kNumCells> board) const
    {
      std::vector<unsigned char> diagonalMoves;

      unsigned char col = cell % kNumCols;                                    // The column the cell is in
      unsigned char row = cell / kNumRows;                                    // The row the cell is in
      unsigned char upLeft = min(row, col);                                   // The maximum amount of spaces to check up and left of given cell
      unsigned char upRight = min(row, (unsigned char)(kNumCols - col - 1));  // The maximum amount of spaces to check up and right of given cell
      unsigned char downLeft = min((unsigned char)(kNumRows - row - 1), col); // The maximum amount of spaces to check down and left of given cell
      unsigned char downRight = min((unsigned char)(kNumRows - row - 1),
                                    (unsigned char)(kNumCols - col - 1)); // The maximum amount of spaces to check down and right of given cell
      unsigned char focus;

      // Up and left
      int count = 1;
      focus = cell;
      while (count <= upLeft)
      {
        focus -= (kNumRows + 1);
        if (board[focus] == CellState::kEmpty)
        {
          diagonalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      // Up and right
      count = 1;
      focus = cell;
      while (count <= upRight)
      {
        focus -= (kNumRows - 1);
        if (board[focus] == CellState::kEmpty)
        {
          diagonalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      // Down and left
      count = 1;
      focus = cell;
      while (count <= downLeft)
      {
        focus += (kNumRows - 1);
        if (board[focus] == CellState::kEmpty)
        {
          diagonalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      // Down and right
      count = 1;
      focus = cell;
      while (count <= downRight)
      {
        focus += (kNumRows + 1);
        if (board[focus] == CellState::kEmpty)
        {
          diagonalMoves.push_back(focus);
          count++;
        }

        // We have encountered a blocking piece
        else
        {
          break;
        }
      }

      return diagonalMoves;
    }

    std::vector<unsigned char> AmazonsState::GetAllMoves(unsigned char cell, std::array<CellState, kNumCells> board) const
    {
      std::vector<unsigned char> horizontals = GetHorizontalMoves(cell, board);
      std::vector<unsigned char> verticals = GetVerticalMoves(cell, board);
      std::vector<unsigned char> diagonals = GetDiagonalMoves(cell, board);
      std::vector<unsigned char> acc = horizontals;

      acc.insert(acc.end(), verticals.begin(), verticals.end());
      acc.insert(acc.end(), diagonals.begin(), diagonals.end());

      return acc;
    }

    // Looks okay
    std::vector<Action> AmazonsState::LegalActions() const
    {
      std::array<CellState,kNumCells> board = board_;

      if (IsTerminal())
        return {};

      std::vector<Action> actions;

      // find all amazons
      for (unsigned char cell = 0; cell < kNumCells; ++cell)
      {
        if (board[cell] == PlayerToState(CurrentPlayer()))
        {
          // find all valid moves for this amazon
          std::vector<unsigned char> moves = GetAllMoves(cell, board);

          // find all legal shot locations for each move
          board[cell] = CellState::kEmpty;
          for (auto const &move : moves)
          {
            std::vector<unsigned char> shots = GetAllMoves(move, board);

            // build a legal action from each shot
            for (auto const &shot : shots)
            {
              std::vector<unsigned char> triple;

              triple.push_back(cell);
              triple.push_back(move);
              triple.push_back(shot);

              Action action = EncodeAction(triple);

              actions.push_back(action);
            }
          }
          board[cell] = PlayerToState(CurrentPlayer());
        }
      }
      sort(actions.begin(), actions.end());
      return actions;
    }

    // Looks okay
    std::string AmazonsState::ActionToString(Player player, Action action) const
    {
      std::vector<unsigned char> decoded = DecodeAction(action);
      unsigned char from, to, shoot;
      from = decoded[0];
      to = decoded[1];
      shoot = decoded[2];

      return absl::StrCat(StateToString(PlayerToState(player)), "( ",
                          from, " | ", to, " | ", shoot," )");
    }

    
    // Looks okay
    AmazonsState::AmazonsState(std::shared_ptr<const Game> game) : State(game)
    {
      std::fill(begin(board_), end(board_), CellState::kEmpty);
      switch (kNumRows)
      {
      case 8:
        board_[2] = board_[5] = board_[16] = board_[23] = CellState::kCross;
        board_[40] = board_[47] = board_[58] = board_[61] = CellState::kNought;
        break;
      
      default:
        break;
      }

    }

    // Stringify the current state of the game
    // This should be okay
    std::string AmazonsState::ToString() const
    {
      std::string str;
      for (int r = 0; r < kNumRows; ++r)
      {
        for (int c = 0; c < kNumCols; ++c)
        {
          absl::StrAppend(&str, StateToString(BoardAt(r, c)));
        }
        if (r < (kNumRows - 1))
        {
          absl::StrAppend(&str, "\n");
        }
      }
      return str;
    }

    // This should also be okay
    bool AmazonsState::IsTerminal() const
    {
      return outcome_ != kInvalidPlayer;
    }

    // This seems reasonable
    std::vector<double> AmazonsState::Returns() const
    {
      if (outcome_ == (Player{0}))
      {
        return {1.0, -1.0};
      }
      else if (outcome_ == (Player{1}))
      {
        return {-1.0, 1.0};
      }
      else
      {
        return {0.0, 0.0};
      }
    }

    // Looks okay, unclear though
    std::string AmazonsState::InformationStateString(Player player) const
    {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);
      return HistoryString();
    }

    // Looks okay, unclear though
    std::string AmazonsState::ObservationString(Player player) const
    {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);
      return ToString();
    }

    // Looks okay, unclear though
    void AmazonsState::ObservationTensor(Player player,
                                         absl::Span<float> values) const
    {
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, num_players_);

      // Treat `values` as a 2-d tensor.
      TensorView<2> view(values, {kCellStates, kNumCells}, true);
      for (int cell = 0; cell < kNumCells; ++cell)
      {
        view[{static_cast<int>(board_[cell]), cell}] = 1.0;
      }
    }

    // This looks okay
    void AmazonsState::UndoAction(Player player, Action move)
    {
      std::vector<unsigned char> decoded = DecodeAction(move);
      unsigned char from, to, shoot;
      from = decoded[0];
      to = decoded[1];
      shoot = decoded[2];
      
      board_[from] = PlayerToState(player);
      board_[to] = CellState::kEmpty;
      board_[shoot] = CellState::kEmpty;

      current_player_ = player;
      outcome_ = kInvalidPlayer;
      num_moves_ -= 1;
      history_.pop_back();
    }

    // Looks okay
    std::unique_ptr<State> AmazonsState::Clone() const
    {
      return std::unique_ptr<State>(new AmazonsState(*this));
    }

    // Looks okay
    AmazonsGame::AmazonsGame(const GameParameters &params)
        : Game(kGameType, params) {}

  } // namespace amazons
} // namespace open_spiel
