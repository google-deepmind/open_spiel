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

#ifndef OPEN_SPIEL_GAMES_BELOTE_H_
#define OPEN_SPIEL_GAMES_BELOTE_H_

// C++ implementation of classic (non-contract) French Belote for 4 players
// in 2 fixed partnerships (players 0 & 2 vs players 1 & 3). This mirrors the
// rules implemented by the Python reference implementation in
// open_spiel/python/games/belote.py (registered there as "belote_python"):
// trump is chosen via the "prise" procedure -- 5 cards are dealt to each
// player, the next stock card is turned face up, and players in turn may
// take it (round 1) or, if everyone passes, choose one of the three other
// suits (round 2). If everyone passes twice, the deal is redealt with the
// next player as dealer. Card play follows standard suit- and
// trump-following obligations, and scoring uses the standard 162-point deck
// (152 card points + 10 for the last trick), with an all-or-nothing rule:
// the declaring team keeps its points only if it scores strictly more than
// 81; otherwise the defending team collects all 162 points.

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace belote {

inline constexpr int kNumPlayers = 4;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 8;
inline constexpr int kNumCards = kNumSuits * kNumRanks;
inline constexpr int kMaxScore = 162;

// Card actions are 0..31 (card = suit * kNumRanks + rank).
inline constexpr int kPassAction = kNumCards;               // 32
inline constexpr int kTakeAction = kNumCards + 1;            // 33
inline constexpr int kChooseSuitActionBase = kNumCards + 2;  // 34..37
inline constexpr int kNumDistinctActions = kNumCards + 2 + kNumSuits;  // 38

constexpr char kSuitChar[] = "CDHS";
// Ranks, low to high face value: 7 8 9 10 J Q K A.
constexpr const char* kRankNames[] = {"7", "8", "9", "10",
                                      "J", "Q", "K", "A"};

enum class Phase { kDeal, kBid1, kBid2, kPlay, kGameOver };

inline int CardSuit(int card) { return card / kNumRanks; }
inline int CardRank(int card) { return card % kNumRanks; }
std::string CardString(int card);
int CardPoints(int card, int trump_suit);
int CardStrength(int card, int trump_suit);
inline int TeamOf(Player player) { return player % 2; }
inline Player PartnerOf(Player player) { return (player + 2) % kNumPlayers; }

class BeloteState : public State {
 public:
  explicit BeloteState(std::shared_ptr<const Game> game, Player dealer);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new BeloteState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> LegalCardPlays(Player player) const;
  bool IsBetter(int card, int other, int led_suit) const;
  Player TrickWinner(const std::vector<std::pair<Player, int>>& trick) const;
  void EnterPlayPhase();
  void ApplyDealAction(int card);
  void StartCompletionDeal(std::vector<Player> schedule, Phase next_phase);
  void ApplyBid1Action(int action, Player player);
  void ApplyBid2Action(int action, Player player);
  void ApplyPlayAction(int card, Player player);
  void FinalizeScores();
  void WriteObservation(Player player, bool perfect_recall,
                        absl::Span<float> values) const;

  Player dealer_;
  std::array<std::vector<int>, kNumPlayers> hands_{};
  std::vector<int> deck_;
  int turned_card_ = kInvalidAction;

  Phase phase_ = Phase::kDeal;
  std::vector<Player> deal_schedule_;
  int deal_index_ = 0;
  Phase after_deal_phase_ = Phase::kBid1;

  std::vector<Player> bid_turn_order_;
  int bid_pointer_ = 0;

  Player taker_ = kInvalidPlayer;
  int trump_suit_ = -1;
  int declarer_team_ = -1;

  std::vector<std::pair<Player, int>> trick_;
  Player trick_leader_ = kInvalidPlayer;
  Player current_player_play_ = kInvalidPlayer;
  int tricks_played_ = 0;
  std::vector<int> played_cards_;
  std::vector<std::vector<int>> trick_history_;
  std::array<int, 2> team_points_ = {0, 0};
  std::vector<double> returns_ = std::vector<double>(kNumPlayers, 0.0);
};

class BeloteGame : public Game {
 public:
  explicit BeloteGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BeloteState(shared_from_this(),
                                                  dealer_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -kMaxScore; }
  double MaxUtility() const override { return kMaxScore; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // Dealing (~32 draws, more with redeals) + bidding (up to 8 calls) +
    // card play (32 plays). Redeals are rare but unbounded in principle;
    // this bound matches the Python reference implementation.
    return kNumCards + 8 + kNumCards;
  }

 private:
  const Player dealer_;
};

}  // namespace belote
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BELOTE_H_
