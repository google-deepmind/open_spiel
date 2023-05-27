// Copyright 2022 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_MAEDN_H_
#define OPEN_SPIEL_GAMES_MAEDN_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of Mensch-Aergere-Dich-Nicht (see
// https://en.wikipedia.org/wiki/Mensch_%C3%A4rgere_Dich_nicht)
//
// Rules used:
// - start field must be cleared as soon as possible
// - throwing out own pieces is not possible
// - only one dice roll even if no move is possible except if dice roll was
//   a six, in this case, same player may roll again
// - pieces may jump over each other on four final fields
//
// Parameters:
// - players: Number of Players (2 to 4)
// - twoPlayersOpposite:
//   If two players play, two different settings are possible:
//   Either players can play side by side or they can play on opposite sides.
//   Since opposite sides are more fair, default value is true.

namespace open_spiel {
namespace maedn {

inline constexpr const int kMaxNumPlayers = 4;
inline constexpr const int kNumChanceOutcomes = 6;
inline constexpr const int kRedPlayerId = 0;
inline constexpr const int kBluePlayerId = 1;
inline constexpr const int kGreenPlayerId = 2;
inline constexpr const int kYellowPlayerId = 3;
// Board consists of 40 common fields for all
// players and 4 separate goal fields for each player.
inline constexpr const int kNumCommonFields = 40;
inline constexpr const int kNumGoalFields = 16;
inline constexpr const int kNumGoalFieldsPerPlayer = 4;
inline constexpr const int kNumFields = kNumCommonFields + kNumGoalFields;

// Number of pieces per player in the standard game.
inline constexpr const int kNumPiecesPerPlayer = 4;

// position of pieces not yet in game
inline constexpr const int kOutPos = -1;

// Action modelling (with ideas from Marc Lancot):
// The first action [0] is to pass (necessary if player cannot move any
// piece). The second action is to bring in a new piece. Once a piece is
// on the field, there are 43 fields a piece can stand on and be moved away
// from that field. Actions are coded as the field a move starts from, from
// each player's own PoV. That means that action 2 means to move a piece on
// field 0 for player 0 but a piece on field 10 for player 1 and so on. So
// there are 43 actions for moves, one action to bring in a new piece and
// one action to pass. Total number of possible actions is 45
// ({ 0, 1, 2, ..., 44 }).
inline constexpr const int kNumDistinctActions = 45;

inline constexpr const Action kPassAction = 0;
inline constexpr const Action kBringInAction = 1;
inline constexpr const Action kFieldActionsOffset = 2;

// See ObservationTensorShape for details.
inline constexpr const int kBoardEncodingSize = 4 * kNumFields;
inline constexpr const int kStateEncodingSize =
    kMaxNumPlayers + kBoardEncodingSize + kMaxNumPlayers + kNumChanceOutcomes;

struct Coords {
  int x;
  int y;
};

const Coords kFieldToBoardString[]{
    // Common fields.
    {0, 4},
    {2, 4},
    {4, 4},
    {6, 4},
    {8, 4},
    {8, 3},
    {8, 2},
    {8, 1},
    {8, 0},
    {10, 0},
    {12, 0},
    {12, 1},
    {12, 2},
    {12, 3},
    {12, 4},
    {14, 4},
    {16, 4},
    {18, 4},
    {20, 4},
    {20, 5},
    {20, 6},
    {18, 6},
    {16, 6},
    {14, 6},
    {12, 6},
    {12, 7},
    {12, 8},
    {12, 9},
    {12, 10},
    {10, 10},
    {8, 10},
    {8, 9},
    {8, 8},
    {8, 7},
    {8, 6},
    {6, 6},
    {4, 6},
    {2, 6},
    {0, 6},
    {0, 5},
    // Goal fields.
    {2, 5},
    {4, 5},
    {6, 5},
    {8, 5},
    {10, 1},
    {10, 2},
    {10, 3},
    {10, 4},
    {18, 5},
    {16, 5},
    {14, 5},
    {12, 5},
    {10, 9},
    {10, 8},
    {10, 7},
    {10, 6},
    // Off the board fields.
    {0, 0},
    {2, 0},
    {2, 1},
    {0, 1},
    {18, 0},
    {20, 0},
    {20, 1},
    {18, 1},
    {18, 10},
    {20, 10},
    {20, 9},
    {18, 9},
    {0, 10},
    {2, 10},
    {2, 9},
    {0, 9},
};

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  int dice;
  int prev_dice;
  Action action;
  int thrown_out_player;
  TurnHistoryInfo(int _player, int _prev_player, int _dice, int _prev_dice,
                  int _action, int _thrown_out_player)
      : player(_player),
        prev_player(_prev_player),
        dice(_dice),
        prev_dice(_prev_dice),
        action(_action),
        thrown_out_player(_thrown_out_player) {}
};

class MaednGame;

class MaednState : public State {
 public:
  MaednState(const MaednState&) = default;
  MaednState(std::shared_ptr<const Game>, bool two_players_opposite);

  Player CurrentPlayer() const override;
  void UndoAction(Player player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests.
  // History is not set by this method, so calls to UndoAction will cause
  // undefined behaviour!
  void SetState(int cur_player, int dice, int prev_player, int prev_dice,
                const std::vector<int>& board,
                const std::vector<int>& out);
  // Some values are not part of ObservationTensor (like prev_player_ and
  // prev_dice_) and so have to be given from outside. History is not part
  // of ObservationTensor either, so calls to UndoAction will cause undefined
  // behaviour!
  void FromObservationTensor(Player player, absl::Span<float> values,
                             Player prev_player, int prev_dice);

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);
  std::pair<int, int> GetFieldsFromAction(Action action, Player player,
                                          int dice) const;
  int RelPosToAbsPos(int relative_position, int position) const;
  int AbsPosToRelPos(int absolute_position, int position) const;
  int GetPlayersFirstField(Player player) const;

  int PlayerToPosition(Player player) const {
    // Position is equal to player except if two players play on opposite
    // sides, in this case position of player 1 is 2. For completeness,
    // in this case position of player 2 is 1, so that even for iterations
    // over 4 players no position is used twice.
    return num_players_ == 2 && two_players_opposite_ &&
                   (player == 1 || player == 2)
               ? 3 - player
               : player;
  }

  bool AllInGoal(Player player) const;
  Player cur_player_;
  Player prev_player_;
  const bool two_players_opposite_;
  int turns_;
  int dice_;              // Current dice roll.
  int prev_dice_;         // Last dice roll.
  std::vector<int> out_;  // Number of pieces of each player outside of field.

  // Board consists of 40 common fields, starting with the set-in field of
  // player 0. After that, four goal fields of each player follow, beginning
  // with player 0 again.
  // Player 0 starts on field 0, goes up to field 39 and continues into
  // goal fields 40-43.
  // Player 1 starts on field 10, goes up to field 39, continues from 0 to 9
  // and jumps from 9 to 44-47.
  // Player 2 starts on field 20, goes up to field 39, continues from 0 to 19
  // and jumps from 19 to 48-51.
  // Player 3 starts on field 30, goes up to field 39, continues from 0 to 29
  // and jumps from 29 to 52-55.
  std::vector<int> board_;
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
};

class MaednGame : public Game {
 public:
  explicit MaednGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new MaednState(shared_from_this(), two_player_opposite_));
  }

  // Classic six sided dice.
  int MaxChanceOutcomes() const override { return kNumChanceOutcomes; }

  // Arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  // Upper bound: chance node per move, with an initial chance node for
  // determining starting player.
  int MaxChanceNodesInHistory() const override { return MaxGameLength() + 1; }

  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return -MaxUtility(); }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 3; }

  std::vector<int> ObservationTensorShape() const override {
    // Encode each field on the board as four doubles:
    // - One double for whether there is a piece of player 1 (1 or 0).
    // - One double for whether there is a piece of player 2 (1 or 0).
    // - One double for whether there is a piece of player 3 (1 or 0).
    // - One double for whether there is a piece of player 4 (1 or 0).
    // (effectively that is one-hot encoded player number)
    //
    // Return a vector encoding:
    // - Every field.
    // - One double for the number of pieces outside the board for player 1.
    // - One double for the number of pieces outside the board for player 2.
    // - One double for the number of pieces outside the board for player 3.
    // - One double for the number of pieces outside the board for player 4.
    // - One double for whether it's player 1's turn (1 or 0).
    // - One double for whether it's player 2's turn (1 or 0).
    // - One double for whether it's player 3's turn (1 or 0).
    // - One double for whether it's player 4's turn (1 or 0).
    //   (If it's chance player's turn, all four doubles are 0.)
    // - One double for whether dice roll is a 1 (1 or 0).
    // - One double for whether dice roll is a 2 (1 or 0).
    // - One double for whether dice roll is a 3 (1 or 0).
    // - One double for whether dice roll is a 4 (1 or 0).
    // - One double for whether dice roll is a 5 (1 or 0).
    // - One double for whether dice roll is a 6 (1 or 0).
    //   (If it's chance player's turn, all six doubles are 0.)

    return {kStateEncodingSize};
  }

 private:
  bool two_player_opposite_;
  int num_players_;
};

}  // namespace maedn
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MAEDN_H_
