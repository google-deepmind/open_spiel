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

#ifndef OPEN_SPIEL_GAMES_DARK_HEX_H_
#define OPEN_SPIEL_GAMES_DARK_HEX_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/hex.h"
#include "open_spiel/spiel.h"

// Dark Hex (Some versions also called Phantom Hex or Kriegspiel Hex) is an
// imperfect information version of the classic game of Hex. Players are not
// exposed to oppsite sides piece information. Only a refree has the full
// information of the board. When a move fails due to collision the player gets
// the information of the cell (stone exists), and is allowed to make another
// move until success.
//
// There are slightly different versions of the game exists depending on the
// level of information being exposed to the opponent and what happens in the
// event of a collision. Right now we only have the Classic Dark Hex (Phantom
// Hex) implemented with player:
//        - Replays after a collision
//        - Exposed(information on opponent) only to number of turns or nothing
//
// Common phantom games include Kriegspiel (Phantom chess), e.g. see
// https://en.wikipedia.org/wiki/Kriegspiel_(chess), and Phantom Go.
// See also http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf, Ch 3.
//
// Parameters:
//    "obstype"       string      If the player is informed of the
//                                number of moves attempted
//                                ['reveal-nothing' (def.), 'reveal-numturns']
//    "board_size"    int         Size of the board (default 11)

namespace open_spiel {
namespace dark_hex {

inline constexpr const char* kDefaultObsType = "reveal-nothing";

// black - white - empty
inline constexpr int kPosStates = hex::kNumPlayers + 1;

// Add here if anything else is needed to be revealed
enum class ObservationType {
  kRevealNothing,
  kRevealNumTurns,
};

class DarkHexState : public State {
 public:
  DarkHexState(std::shared_ptr<const Game> game, int board_size,
               ObservationType obs_type);

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

  // Dark games funcs.
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;

 private:
  std::string ViewToString(Player player) const;
  std::string ActionSequenceToString(Player player) const;

  hex::HexState state_;
  ObservationType obs_type_;
  const int board_size_;
  const int num_cells_;
  const int bits_per_action_;
  const int longest_sequence_;

  // Change this to _history on base class
  std::vector<std::pair<int, Action>> action_sequence_;
  std::vector<hex::CellState> black_view_;
  std::vector<hex::CellState> white_view_;
};

class DarkHexGame : public Game {
 public:
  explicit DarkHexGame(const GameParameters& params);
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new DarkHexState(shared_from_this(), board_size_, obs_type_));
  }
  int NumDistinctActions() const override {
    return game_->NumDistinctActions();
  }
  int NumPlayers() const override { return game_->NumPlayers(); }
  double MinUtility() const override { return game_->MinUtility(); }
  double UtilitySum() const override { return game_->UtilitySum(); }
  double MaxUtility() const override { return game_->MaxUtility(); }

  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    return board_size_ * board_size_ * 2 - 1;
  }

 private:
  std::shared_ptr<const hex::HexGame> game_;
  ObservationType obs_type_;
  const int board_size_;
  const int num_cells_;
  const int bits_per_action_;
  const int longest_sequence_;
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

}  // namespace dark_hex
}  // namespace open_spiel

#endif
