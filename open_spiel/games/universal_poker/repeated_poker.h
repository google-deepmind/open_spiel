// Copyright 2021 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_UNIVERSAL_POKER_REPEATED_POKER_H_
#define OPEN_SPIEL_GAMES_UNIVERSAL_POKER_REPEATED_POKER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/universal_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"


// Wrapper around universal_poker for playing multiple hands within the same
// game episode. This enables simulating both cash games and tournaments.
//
// Parameters:
//  "universal_poker_game_string" string (required)
//      Specifies the underlying ACPC game to begin play. Note that the params
//      will be updated to reflect the current state of the repeated game (e.g.,
//      number of players, bet limits, etc.).
//  "max_num_hands" int (required)
//      The maximum number of hands to play in the episode. This is a required
//      parameter because it should be set deliberately in relation to the blind
//      schedule when playing a tournament.
//  "reset_stacks" bool (required)
//      Whether to reset the stacks at the start of each hand. Required
//  "rotate_dealer" bool (optional, default=true)
//      Whether to rotate the dealer at the start of each hand. This defaults to
//      true as it is always done in practice.
//  "blind_schedule" string (optional)
//    Specifies the blind schedule for playing a tournament. The format is:
//    <blind_level_1>;<blind_level_2>;...<blind_level_n> where each blind level
//    is of the form <num_hands>:<small_blind>/<big_blind>. If play continues
//    beyond the number of hands specified in the last blind level, the last
//    blind level will continue to be used.

// Returns are calculated by summing the returns for each hand.
// TODO(jhtschultz): Support payout structures for tournaments.

// Note that this implementation imposes some slightly stricter assumptions on
// the game definition than the ACPC implementation:
//  1. Exactly two blinds.
//  2. At least 2 rounds.
// Both of these are very standard assumptions for poker as played in practice.

// Note: we use simplified moving button rules. See
// https://en.wikipedia.org/wiki/Betting_in_poker#When_a_player_in_the_blinds_leaves_the_game
// This is common in online poker games generally speaking, and it is used here
// because the logic for remapping each hand to an ACPC gamedef becomes quite
// complex and highly error-prone when using dead button rules.

// WARNING: This implementation diverges from standard tournament play in the
// following respect: players who cannot post a big blind are eliminated.
// This stems from the ACPC implementation, which requires all players' starting
// stacks to be at least one big blind.
// This can result in the total rewards not adding up to the number of total
// chips to start the tournament, so users should not rely on this assumption.

namespace open_spiel {
namespace universal_poker {
namespace repeated_poker {

inline constexpr int kDefaultNumHands = 1;
inline constexpr int kInactivePlayerSeat = -1;
inline constexpr int kInvalidBlindValue = -1;

struct BlindLevel {
  int num_hands;
  int small_blind;
  int big_blind;
};

class RepeatedPokerState : public State {
 public:
  RepeatedPokerState(std::shared_ptr<const Game> game,
                     std::string universal_poker_game_string, int max_num_hands,
                     bool reset_stacks, bool rotate_dealer,
                     std::string blind_schedule);
  RepeatedPokerState(const RepeatedPokerState& other);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override {
    return universal_poker_state_->ActionToString(player, action_id);
  }
  std::string ToString() const override;
  bool IsTerminal() const override { return is_terminal_; };
  std::vector<double> Returns() const override;
  std::string ObservationString(Player player) const override;
  std::unique_ptr<State> Clone() const override;
  std::vector<Action> LegalActions() const override {
    return universal_poker_state_->LegalActions();
  }
  ActionsAndProbs ChanceOutcomes() const override {
    return universal_poker_state_->ChanceOutcomes();
  }

  std::unique_ptr<UniversalPokerState> GetUniversalPokerState() const {
    return std::make_unique<UniversalPokerState>(*universal_poker_state_);
  }
  int Dealer() const { return dealer_; }
  int SmallBlind() const { return small_blind_; }
  int BigBlind() const { return big_blind_; }
  std::vector<int> Stacks() const { return stacks_; }
  int PlayerToSeat(Player player) const { return player_to_seat_.at(player); }
  Player SeatToPlayer(int seat) const { return seat_to_player_.at(seat); }
  int DealerSeat() const { return player_to_seat_.at(dealer_); }
  int SmallBlindSeat() const { return small_blind_seat_; }
  int BigBlindSeat() const { return big_blind_seat_; }

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::string universal_poker_game_string_;
  GameParameters universal_poker_game_params_;
  std::unique_ptr<UniversalPokerState> universal_poker_state_;
  int hand_number_ = 0;
  int max_num_hands_ = 0;
  bool is_terminal_ = false;
  // Represents the stack sizes at the start of the current hand. To access the
  // stacks for the hand in progress, use the underlying UniversalPokerState.
  std::vector<int> stacks_;
  bool reset_stacks_;
  bool rotate_dealer_;
  std::string blind_schedule_str_;
  std::vector<BlindLevel> blind_schedule_levels_;
  std::map<Player, int> player_to_seat_;
  Player dealer_ = kInvalidPlayer;
  std::map<int, Player> seat_to_player_;
  int num_active_players_;
  int small_blind_ = kInvalidBlindValue;
  int big_blind_ = kInvalidBlindValue;
  int small_blind_seat_ = kInactivePlayerSeat;
  int big_blind_seat_ = kInactivePlayerSeat;
  std::vector<std::string> acpc_hand_histories_{};
  std::vector<std::vector<double>> hand_returns_{
      std::vector<double>(num_players_, 0.0)};

  void UpdateStacks();
  void UpdateSeatAssignments();
  void UpdateDealer();
  void UpdateBlinds();
  void UpdateUniversalPoker();
};

class RepeatedPokerGame : public Game {
 public:
  explicit RepeatedPokerGame(const GameParameters& params);
  std::string ToString() const;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxGameLength() const override {
    return max_num_hands_ * base_game_->MaxGameLength();
  }
  int NumPlayers() const override { return base_game_->NumPlayers(); }
  int NumDistinctActions() const override;
  int MaxChanceOutcomes() const override {
    return base_game_->MaxChanceOutcomes();
  }
  int MaxChanceNodesInHistory() const override {
    return max_num_hands_ * base_game_->MaxChanceNodesInHistory();
  }
  double MinUtility() const override;
  double MaxUtility() const override;
  absl::optional<double> UtilitySum() const override { return 0; };

  const UniversalPokerGame* BaseGame() const {
    return dynamic_cast<const UniversalPokerGame*>(base_game_.get());
  }

 private:
  std::string universal_poker_game_string_;
  int max_num_hands_;
  bool reset_stacks_;
  bool rotate_dealer_;
  std::string blind_schedule_;
  std::shared_ptr<const UniversalPokerGame> base_game_;
};

}  // namespace repeated_poker
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_REPEATED_POKER_H_
