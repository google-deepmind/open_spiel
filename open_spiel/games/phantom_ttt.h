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

#ifndef OPEN_SPIEL_GAMES_PHANTOM_TTT_H_
#define OPEN_SPIEL_GAMES_PHANTOM_TTT_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"

// Phantom Tic-Tac-Toe is a phantom version of the classic game of Tic-Tac-Toe
// (Noughts and Crosses). For some perfect information game X", the game
// "phantom X" is a version of the game X where the players do not observe the
// other players' pieces. Only a referee knows the full state of the board.
// So, on a player's turn, a chosen moves may fail because it is illegal given
// the true state of the board; in this case, a player can continue to try moves
// until one succeeds.
//
// Common phantom games include Kriegspiel (Phantom chess), e.g. see
// https://en.wikipedia.org/wiki/Kriegspiel_(chess), and Phantom Go.
// See also http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, Ch 3.
//
// Parameters:
///    "obstype", string, "reveal-nothing" (default) or "reveal-numturns"

namespace open_spiel {
namespace phantom_ttt {

inline constexpr const char* kDefaultObsType = "reveal-nothing";

// Longest sequence is 17 moves, e.g. 0011223344556677889
inline constexpr int kLongestSequence = 2 * tic_tac_toe::kNumCells - 1;
inline constexpr int kBitsPerAction = 10;  // Reserve 9 as "I don't know."

enum class ObservationType {
  kRevealNothing,
  kRevealNumTurns,
};

// State of an in-play game.
class PhantomTTTState : public State {
 public:
  PhantomTTTState(std::shared_ptr<const Game> game, ObservationType obs_type);

  // Forward to underlying game state
  Player CurrentPlayer() const override { return state_.CurrentPlayer(); }
  std::string ActionToString(Player player, Action action_id) const override {
    return state_.ActionToString(player, action_id);
  }
  std::string ToString() const override { return state_.ToString(); }
  bool IsTerminal() const override { return state_.IsTerminal(); }
  std::vector<double> Returns() const override { return state_.Returns(); }
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;

  // These are implemented for phantom games
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;
  std::string ViewToString(Player player) const;

 private:
  std::string ActionSequenceToString(Player player) const;

  tic_tac_toe::TicTacToeState state_;
  ObservationType obs_type_;
  // TODO(author2): Use the base class history_ instead.
  std::vector<std::pair<int, Action>> action_sequence_;
  std::array<tic_tac_toe::CellState, tic_tac_toe::kNumCells> x_view_;
  std::array<tic_tac_toe::CellState, tic_tac_toe::kNumCells> o_view_;
};

// Game object.
class PhantomTTTGame : public Game {
 public:
  PhantomTTTGame(const GameParameters& params, GameType game_type);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new PhantomTTTState(shared_from_this(), obs_type_));
  }
  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }
  double MaxUtility() const override { return game_->MaxUtility(); }

  // These will depend on the obstype parameter.
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override { return kLongestSequence; }

  ObservationType obs_type() const { return obs_type_; }

 private:
  std::shared_ptr<const tic_tac_toe::TicTacToeGame> game_;
  ObservationType obs_type_;
};

// Implements the FOE abstraction from Lanctot et al. '12
// http://mlanctot.info/files/papers/12icml-ir.pdf
class ImperfectRecallPTTTState : public PhantomTTTState {
 public:
  ImperfectRecallPTTTState(std::shared_ptr<const Game> game,
                           ObservationType obs_type)
      : PhantomTTTState(game, obs_type) {}
  std::string InformationStateString(Player player) const override {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, num_players_);
    return absl::StrCat("P", player, " ", ViewToString(player));
  }
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new ImperfectRecallPTTTState(*this));
  }
};

class ImperfectRecallPTTTGame : public PhantomTTTGame {
 public:
  explicit ImperfectRecallPTTTGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new ImperfectRecallPTTTState(shared_from_this(), obs_type()));
  }
};

inline std::ostream& operator<<(std::ostream& stream,
                                const ObservationType& obs_type) {
  switch (obs_type) {
    case ObservationType::kRevealNothing:
      return stream << "Reveal Nothing";
    case ObservationType::kRevealNumTurns:
      return stream << "Reveal Num Turns";
    default:
      SpielFatalError("Unknown observation type");
  }
}

}  // namespace phantom_ttt
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PHANTOM_TTT_H_
