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

// A generalized version of a Leduc poker, a simple but non-trivial poker game
// described in http://poker.cs.ualberta.ca/publications/UAI05.pdf .
//
// Taken verbatim from the linked paper above: "In Leduc hold'em, the deck
// consists of two suits with three cards in each suit. There are two rounds.
// In the first round a single private card is dealt to each player. In the
// second round a single board card is revealed. There is a two-bet maximum,
// with raise amounts of 2 and 4 in the first and second round, respectively.
// Both players start the first round with 1 already in the pot.
//
// So the maximin sequence is of the form:
// private card player 0, private card player 1, [bets], public card, [bets]
//
// Parameters:
//     "players"           int    number of players          (default = 2)
//     "action_mapping"    bool   regard all actions as legal and internally
//                                map otherwise illegal actions to check/call
//                                                           (default = false)
//     "suit_isomorphism"  bool   player observations do not distinguish
//                                between cards of different suits with
//                                the same rank              (default = false)

#ifndef OPEN_SPIEL_GAMES_LEDUC_POKER_H_
#define OPEN_SPIEL_GAMES_LEDUC_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace leduc_poker {

// Default parameters.

inline constexpr int kInvalidCard = -10000;
inline constexpr int kDefaultPlayers = 2;
inline constexpr int kNumSuits = 2;
inline constexpr int kFirstRaiseAmount = 2;
inline constexpr int kSecondRaiseAmount = 4;
inline constexpr int kTotalRaisesPerRound = 2;
inline constexpr int kMaxRaises = 2;
inline constexpr int kStartingMoney = 100;

// Number of info states in the 2P game with default params.
inline constexpr int kNumInfoStates = 936;

class LeducGame;
class LeducObserver;

enum ActionType { kFold = 0, kCall = 1, kRaise = 2 };

class LeducState : public State {
 public:
  explicit LeducState(std::shared_ptr<const Game> game,
                      bool action_mapping, bool suit_isomorphism);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Additional methods
  int round() const { return round_; }
  int deck_size() const { return deck_size_; }
  int public_card() const { return public_card_; }
  int raises() const { return num_raises_; }
  int private_card(Player player) const { return private_cards_[player]; }
  std::vector<Action> LegalActions() const override;

  // Gets the private cards.
  std::vector<int> GetPrivateCards() const { return private_cards_; }

  // Sets the private cards to specific ones. Note that this function does not
  // change the history, so any functions relying on the history will not longer
  // work properly.
  void SetPrivateCards(const std::vector<int>& new_private_cards);

  // Returns a vector of MaxGameLength containing all of the betting actions
  // taken so far. If the round has ended, the actions are kInvalidAction.
  std::vector<int> padded_betting_sequence() const;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }

 protected:
  // The meaning of `action_id` varies:
  // - At decision nodes, one of ActionType::{kFold, kCall, kRaise}.
  // - At a chance node, indicates the card to be dealt to the player or
  // revealed publicly. The interpretation of each chance outcome depends on
  // the number of players, but always follows:
  //    lowest value of first suit,
  //    lowest value of second suit,
  //    next lowest value of first suit,
  //    next lowest value of second suit,
  //             .
  //             .
  //             .
  //    highest value of first suit,
  //    highest value of second suit.
  // So, e.g. in the two player case (6 cards): 0 = Jack1, 1 = Jack2,
  // 2 = Queen1, ... , 5 = King2.
  void DoApplyAction(Action move) override;

 private:
  friend class LeducObserver;

  int NextPlayer() const;
  void ResolveWinner();
  bool ReadyForNextRound() const;
  void NewRound();
  int RankHand(Player player) const;
  void SequenceAppendMove(int move);
  void Ante(Player player, int amount);
  void SetPrivate(Player player, Action move);
  int NumObservableCards() const;
  int MaxBetsPerRound() const;

  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  Player cur_player_;

  int num_calls_;    // Number of calls this round (total, not per player).
  int num_raises_;   // Number of raises made in the round (not per player).
  int round_;        // Round number (1 or 2).
  int stakes_;       // The current 'level' of the bet.
  int num_winners_;  // Number of winning players.
  int pot_;          // Number of chips in the pot.
  int public_card_;  // The public card revealed after round 1.
  int deck_size_;    // Number of cards remaining; not equal deck_.size()
  int private_cards_dealt_;  // How many private cards currently dealt.
  int remaining_players_;    // Num. players still in (not folded).

  // Is this player a winner? Indexed by pid.
  std::vector<bool> winner_;
  // Each player's single private card. Indexed by pid.
  std::vector<int> private_cards_;
  // Cards by value (0-6 for standard 2-player game, -1 if no longer in the
  // deck.)
  std::vector<int> deck_;
  // How much money each player has, indexed by pid.
  std::vector<double> money_;
  // How much each player has contributed to the pot, indexed by pid.
  std::vector<int> ante_;
  // Flag for whether the player has folded, indexed by pid.
  std::vector<bool> folded_;
  // Sequence of actions for each round. Needed to report information state.
  std::vector<int> round1_sequence_;
  std::vector<int> round2_sequence_;
  // Always regard all actions as legal, and internally map otherwise illegal
  // actions to check/call.
  bool action_mapping_;
  // Players cannot distinguish between cards of different suits with the same
  // rank.
  bool suit_isomorphism_;
};

class LeducGame : public Game {
 public:
  explicit LeducGame(const GameParameters& params);

  int NumDistinctActions() const override { return 3; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  constexpr int MaxBetsPerRound() const {
    // E.g. longest round for 4-player is 10 bets:
    //   check, check, check, bet, call, call, raise, call, call, call
    // = 1 bet + 1 raise + (num_players_-1)*2 calls + (num_players_-2) calls
    return 3 * num_players_ - 2;
  }
  int MaxGameLength() const override {
    // 2 rounds.
    return 2 * MaxBetsPerRound();
  }
  int MaxChanceNodesInHistory() const override { return 3; }
  int NumObservableCards() const {
    return suit_isomorphism_ ? total_cards_ / 2 : total_cards_;
  }

  std::string ActionToString(Player player, Action action) const override;
  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<LeducObserver> default_observer_;
  std::shared_ptr<LeducObserver> info_state_observer_;

 private:
  int num_players_;  // Number of players.
  int total_cards_;  // Number of cards total cards in the game.
  // Always regard all actions as legal, and internally map otherwise illegal
  // actions to check/call.
  bool action_mapping_;
  // Players cannot distinguish between cards of different suits with the same
  // rank.
  bool suit_isomorphism_;
};

// Returns policy that always folds.
TabularPolicy GetAlwaysFoldPolicy(const Game& game);

// Returns policy that always calls.
TabularPolicy GetAlwaysCallPolicy(const Game& game);

// Returns policy that always raises.
TabularPolicy GetAlwaysRaisePolicy(const Game& game);

}  // namespace leduc_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_LEDUC_POKER_H_
