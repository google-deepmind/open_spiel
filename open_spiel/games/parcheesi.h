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

#ifndef OPEN_SPIEL_GAMES_PARCHEESI_H_
#define OPEN_SPIEL_GAMES_PARCHEESI_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of the classic parcheesi game
// Wiki: https://en.wikipedia.org/wiki/Parcheesi
// Android: https://play.google.com/store/apps/details?id=com.superking.parchisi.star

namespace open_spiel {
namespace parcheesi {

inline constexpr const int kNumBoardTiles = 68;
// After 64 moves, tokens go into the ladder
inline constexpr const int kNumBoardPos = 64;
inline const std::vector<std::string> kTokens = {"r", "g", "y", "b"};
// Position where token enters the board for each player
inline const std::vector<int> kStartPos = {0, 17, 34, 51};
// Safe positions on the board where tokens cannot be hit by another player.
// StartPos is not safe if a token enters the board from base
inline const std::vector<int> kSafePos = {0, 7, 12, 17, 24, 29, 34, 41, 46, 51, 58, 63};
inline constexpr const int kHomePos = 71;

inline constexpr const int kBonusMovesForHittingOpponentToken = 20;
inline constexpr const int kBonusMovesForGettingTokenHome = 10;

inline constexpr const int kActionNoValidMoves = 0;
inline constexpr const int kActionNoBonusMoves = 1;

inline constexpr const int kNumPlayers = 4;
inline constexpr const int kNumTokens = 4;

// die_index values
// '0': first die used
// '1': second die used
// '2': both die used
inline constexpr const int kTokenMoveDieIndexMax = 3;

// pos values
//  -1   : base
//  0-63 : 64 grid positions
// 64-70 : 7 ladder positions
//  71   : home
inline constexpr const int kTokenMovePosMax = 73;

// token_index values
// (4 + 1) since we assign -1 for the token moving from base
inline constexpr const int kTokenMoveTokenIndexMax = 5;

// breaking_block values
// true & false
inline constexpr const int kTokenMoveBreakingBlockMax = 2;

inline constexpr const int kNumDistinctActions = kTokenMoveDieIndexMax * kTokenMovePosMax * kTokenMovePosMax * kTokenMoveTokenIndexMax * kTokenMoveBreakingBlockMax;
inline constexpr const int kStateEncodingSize = 1;

struct TokenMove {
  int die_index;
  int old_pos;
  int new_pos;
  int token_index;
  bool breaking_block;
  TokenMove(int _die_index, int _old_pos, int _new_pos, int _token_index, bool _breaking_block)
      : die_index(_die_index), old_pos(_old_pos), new_pos(_new_pos), token_index(_token_index), breaking_block(_breaking_block) {}
};

class ParcheesiGame;

class ParcheesiState : public State {
 public:
  ParcheesiState(const ParcheesiState&) = default;
  ParcheesiState(std::shared_ptr<const Game>);

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

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& bar, const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  TokenMove SpielMoveToTokenMove(Action move) const;
  std::vector<Action> MultipleTokenMoveToSpielMove(std::vector<TokenMove> tokenMoves) const;
  Action TokenMoveToSpielMove(TokenMove tokenMoves) const;
  

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void SetupInitialBoard();
  void RollDice(int outcome);

  void PrintMove(TokenMove move) const;
  int GetPlayerFromToken(std::string token) const;
  int GetGridPosForPlayer(int pos, int player) const;
  std::string GetHumanReadablePosForPlayer(int pos, int player) const;
  std::vector<TokenMove> GetTokenMoves(int player, std::vector<int> dice) const;
  std::vector<TokenMove> GetGridMoves(std::vector<int> player_token_pos, int player, std::vector<int> dice, bool breaking_block) const;
  bool DestinationOccupiedBySafeToken(int destination,int player) const;
  bool BlocksInRoute(int start, int end, int player) const;
  void PenalisePlayer(Player player);
  void NextTurn();
  
  Player cur_player_;
  Player prev_player_;
  int turns_;
  int extra_turn_;
  int bonus_move_;
  TokenMove illegal_move_;
  bool player_forced_to_move_block_;
  std::vector<int> dice_;    // Current dice.

  // Starting position for tokens
  std::vector<std::vector<std::string>> base_;
  
  // Final position for tokens. If all of the token of a player reaches home_, he wins
  std::vector<std::vector<std::string>> home_;

  //  -1   : base
  //  0-63 : 64 grid positions
  // 64-70 : 7 ladder positions
  //  71   : home
  std::vector<std::vector<int>> token_pos_;

  // board_ designates the common 68 tiles all players can occuppy. This excludes the ladder and home tiles.
  // Each player starts at different positions on the board and token_pos_ marks these starting positions 
  // relatively to zero. So board_ makes it easy to check if there's a hit or blocks while moving tokens
  std::vector<std::vector<std::string>> board_;  
  
};

class ParcheesiGame : public Game {
 public:
  explicit ParcheesiGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ParcheesiState(
        shared_from_this()));
  }

  int MaxChanceOutcomes() const override { return 30; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return -MaxUtility(); }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override;

  std::vector<int> ObservationTensorShape() const override {
    return {kStateEncodingSize};
  }

  int NumCheckersPerPlayer() const;

 private:
  int num_players_;
};

}  // namespace parcheesi
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PARCHEESI_H_
