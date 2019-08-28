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

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_BACKGAMMON_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_BACKGAMMON_H_

#include <array>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// An implementation of the classic: https://en.wikipedia.org/wiki/Backgammon
// using rule set from
// http://usbgf.org/learn-backgammon/backgammon-rules-and-terms/rules-of-backgammon/
// where red -> 'x' (player 0) and white -> 'o' (player 1).
//
// Currently does not support the doubling cube nor "matches" (multiple games
// where outcomes are scored and tallied to 21).
//
// Parameters:
//   "scoring_type"    string  Type of scoring for the game: "winloss_scoring"
//                             (default), "enable_gammons", or "full_scoring"

namespace open_spiel {
namespace backgammon {

constexpr const int kNumPlayers = 2;
constexpr const int kNumChanceOutcomes = 21;
constexpr const int kNumPoints = 24;
constexpr const int kNumDiceOutcomes = 6;
constexpr const int kNumCheckersPerPlayer = 15;
constexpr const int kXPlayerId = 0;
constexpr const int kOPlayerId = 1;
constexpr const int kBarPos = 100;
constexpr const int kScorePos = 101;
constexpr const int kPassPos = -1;

// The action encoding stores a number in { 0, 1, ..., 1351 }. If the high
// roll is to move first, then the number is encoded as a 2-digit number in
// base 26 ({0, 1, .., 23, kBarPos, Pass}) (=> first 676 numbers). Otherwise,
// the low die is to move first and, 676 is subtracted and then again the
// number is encoded as a 2-digit number in base 26.
constexpr const int kNumDistinctActions = 1352;

// See InformationStateNormalizedVectorShape for details.
constexpr const int kBoardEncodingSize = 4 * kNumPoints * kNumPlayers;
constexpr const int kStateEncodingSize = 3 * kNumPlayers + kBoardEncodingSize;
constexpr const char* kDefaultScoringType = "winloss_scoring";

// Game scoring type, whether to score gammons/backgammons specially.
enum class ScoringType {
  kWinLossScoring,  // "winloss_scoring": Score only 1 point per player win.
  kEnableGammons,   // "enable_gammons": Score 2 points for a "gammon".
  kFullScoring,     // "full_scoring": Score gammons as well as 3 points for a
                    // "backgammon".
};

struct CheckerMove {
  // Pass is encoded as (pos, num, hit) = (-1, -1, false).
  int pos;  // 0-24  (0-23 for locations on the board and kBarPos)
  int num;  // 1-6
  bool hit;
  CheckerMove(int _pos, int _num, bool _hit)
      : pos(_pos), num(_num), hit(_hit) {}
  bool operator<(const CheckerMove& rhs) const {
    return (pos * 6 + (num - 1)) < (rhs.pos * 6 + rhs.num - 1);
  }
};

// This is a small helper to track historical turn info not stored in the moves.
// It is only needed for proper implementation of Undo.
struct TurnHistoryInfo {
  int player;
  int prev_player;
  std::vector<int> dice;
  Action action;
  bool double_turn;
  bool first_move_hit;
  bool second_move_hit;
  TurnHistoryInfo(int _player, int _prev_player, std::vector<int> _dice,
                  int _action, bool _double_turn, bool fmh, bool smh)
      : player(_player),
        prev_player(_prev_player),
        dice(_dice),
        action(_action),
        double_turn(_double_turn),
        first_move_hit(fmh),
        second_move_hit(smh) {}
};

class BackgammonGame;

class BackgammonState : public State {
 public:
  BackgammonState(const BackgammonState&) = default;
  BackgammonState(int num_distinct_actions, int num_players,
                  ScoringType scoring_type);

  int CurrentPlayer() const override;
  void UndoAction(int player, Action action) override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(int player, Action move_id) const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(int player) const override;
  void InformationStateAsNormalizedVector(
      int player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;

  // Setter function used for debugging and tests. Note: this does not set the
  // historical information properly, so Undo likely will not work on states
  // set this way!
  void SetState(int cur_player, bool double_turn, const std::vector<int>& dice,
                const std::vector<int>& bar, const std::vector<int>& scores,
                const std::vector<std::vector<int>>& board);

  // Returns the opponent of the specified player.
  int Opponent(int player) const;

  // Compute a distance between 'from' and 'to'. The from can be kBarPos. The
  // to can be a number below 0 or above 23, but do not use kScorePos directly.
  int GetDistance(int player, int from, int to) const;

  // Is this position off the board, i.e. >23 or <0?
  bool IsOff(int player, int pos) const;

  // Returns whether pos2 is further (closer to scoring) than pos1 for the
  // specifed player.
  bool IsFurther(int player, int pos1, int pos2) const;

  // Is this a legal from -> to checker move? Here, the to_pos can be a number
  // that is outside {0, ..., 23}; if so, it is counted as "off the board" for
  // the corresponding player (i.e. >23 is a bear-off move for XPlayerId, and
  // <0 is a bear-off move for OPlayerId).
  bool IsLegalFromTo(int player, int from_pos, int to_pos, int my_checkers_from,
                     int opp_checkers_to) const;

  // Get the To position for this play given the from position and number of
  // pips on the die. This function simply adds the values: the return value
  // will be a position that might be off the the board (<0 or >23).
  int GetToPos(int player, int from_pos, int pips) const;

  // Accessor functions for some of the specific data.
  int player_turns() const { return turns_; }
  int player_turns(int player) const {
    return (player == kXPlayerId ? x_turns_ : o_turns_);
  }
  int bar(int player) const { return bar_[player]; }
  int score(int player) const { return scores_[player]; }
  int dice(int i) const { return dice_[i]; }
  bool double_turn() const { return double_turn_; }

  // Get the number of checkers on the board in the specified position belonging
  // to the specified player. The position can be kBarPos or any valid position
  // on the main part of the board, but kScorePos (use score() to get the number
  // of checkers born off).
  int board(int player, int pos) const;

  // Action encoding / decoding functions.
  Action CheckerMovesToSpielMove(const std::vector<CheckerMove>& moves) const;
  std::vector<CheckerMove> SpielMoveToCheckerMoves(int player,
                                                   Action spiel_move) const;
  Action TranslateAction(int from1, int from2, bool use_high_die_first) const;

 protected:
  void DoApplyAction(Action move_id) override;

 private:
  void RollDice(int outcome);
  bool IsPosInHome(int player, int pos) const;
  bool AllInHome(int player) const;
  int CheckersInHome(int player) const;
  bool UsableDiceOutcome(int outcome) const;
  int PositionFromBar(int player, int spaces) const;
  int PositionFrom(int player, int pos, int spaces) const;
  int NumOppCheckers(int player, int pos) const;
  std::string DiceToString(int outcome) const;
  int CountTotalCheckers(int player) const;
  int IsGammoned(int player) const;
  int IsBackgammoned(int player) const;
  int DiceValue(int i) const;
  int HighestUsableDiceOutcome() const;
  Action EncodedPassMove() const;
  Action EncodedBarMove() const;

  // Returns the position of the furthest checker in the home of this player.
  // Returns -1 if none found.
  int FurthestCheckerInHome(int player) const;

  bool ApplyCheckerMove(int player, const CheckerMove& move);
  void UndoCheckerMove(int player, const CheckerMove& move);
  std::set<CheckerMove> LegalCheckerMoves(int player) const;
  int RecLegalMoves(std::vector<CheckerMove> moveseq,
                    std::set<std::vector<CheckerMove>>* movelist);
  std::vector<Action> ProcessLegalMoves(
      int max_moves, const std::set<std::vector<CheckerMove>>& movelist) const;

  ScoringType scoring_type_;  // Which rules apply when scoring the game.

  int cur_player_;
  int prev_player_;
  int turns_;
  int x_turns_;
  int o_turns_;
  bool double_turn_;
  std::vector<int> dice_;    // Current dice.
  std::vector<int> bar_;     // Checkers of each player in the bar.
  std::vector<int> scores_;  // Checkers returned home by each player.
  std::vector<std::vector<int>> board_;  // Checkers for each player on points.
  std::vector<TurnHistoryInfo> turn_history_info_;  // Info needed for Undo.
};

class BackgammonGame : public Game {
 public:
  explicit BackgammonGame(const GameParameters& params);

  int NumDistinctActions() const override { return kNumDistinctActions; }

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new BackgammonState(NumDistinctActions(), kNumPlayers, scoring_type_));
  }

  int MaxChanceOutcomes() const override { return kNumChanceOutcomes; }

  // There is arbitrarily chosen number to ensure the game is finite.
  int MaxGameLength() const override { return 1000; }

  int NumPlayers() const override { return 2; }
  double MinUtility() const override { return -MaxUtility(); }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override;
  std::unique_ptr<Game> Clone() const override {
    return std::unique_ptr<Game>(new BackgammonGame(*this));
  }

  std::vector<int> InformationStateNormalizedVectorShape() const override {
    // Encode each point on the board as four doubles:
    // - One double for whether there is one checker or not (1 or 0).
    // - One double for whether there are two checkers or not (1 or 0).
    // - One double for whether there are three checkers or not (1 or 0).
    // - One double if there are more than 3 checkers, the number of checkers.
    //   more than three that are on that point.
    //
    // Return a vector encoding:
    // Every point listed for the current player.
    // Every point listed for the opponent.
    // One double for the number of checkers on the bar for the current player.
    // One double for the number of checkers scored for the current player.
    // One double for whether it's the current player's turn (1 or 0).
    // One double for the number of checkers on the bar for the opponent.
    // One double for the number of checkers scored for the opponent.
    // One double for whether it's the opponent's turn (1 or 0).

    return {kStateEncodingSize};
  }

 private:
  ScoringType scoring_type_;  // Which rules apply when scoring the game.
};

}  // namespace backgammon
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_BACKGAMMON_H_
