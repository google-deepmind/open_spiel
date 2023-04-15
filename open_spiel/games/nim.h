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

#ifndef OPEN_SPIEL_GAMES_NIM_H_
#define OPEN_SPIEL_GAMES_NIM_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Nim:
// * Two players take turns removing objects from distinct piles;
// * On each turn, a player must remove at least one object,
//      and may remove any number of objects provided they all come from the
//      same heap or pile;
// * Depending on the version, the goal of the game is either to avoid taking
// the last object or to take it. Please see https://en.wikipedia.org/wiki/Nim
// for more

namespace open_spiel {
namespace nim {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kDefaultNumPiles = 3;
inline constexpr bool kDefaultIsMisere = true;

// State of an in-play game.
class NimState : public State {
 public:
  explicit NimState(std::shared_ptr<const Game> game, int num_piles,
                    std::vector<int> piles, bool is_misere,
                    int max_num_per_pile);

  NimState(const NimState &) = default;
  NimState &operator=(const NimState &) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  Player outcome() const { return outcome_; }

 protected:
  void DoApplyAction(Action move) override;
  int num_piles_ = kDefaultNumPiles;
  std::vector<int> piles_;

 private:
  bool IsEmpty() const;
  std::pair<int, int> UnpackAction(Action action_id) const;
  Player current_player_ = 0;  // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  bool is_misere_ = kDefaultIsMisere;
  const int max_num_per_pile_;
};

// Game object.
class NimGame : public Game {
 public:
  explicit NimGame(const GameParameters &params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new NimState(shared_from_this(), num_piles_, piles_, is_misere_,
                     max_num_per_pile_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  absl::optional<double> UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {
        2 +                     // Turn
        1 +                     // Is terminal?
        num_piles_ +            // One-hot bit for the number `num_piles_`
        // One hot representation of the quantity in each pile.
        num_piles_ * (max_num_per_pile_ + 1)
    };
  };
  int MaxGameLength() const override;

 private:
  std::vector<int> piles_;
  int num_piles_ = kDefaultNumPiles;
  bool is_misere_ = kDefaultIsMisere;
  int max_num_per_pile_;
};

}  // namespace nim
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_NIM_H_
