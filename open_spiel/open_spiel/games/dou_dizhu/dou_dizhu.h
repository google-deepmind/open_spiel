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

#ifndef OPEN_SPIEL_GAMES_DOU_DIZHU_H_
#define OPEN_SPIEL_GAMES_DOU_DIZHU_H_

// The game of dou dizhu (the three-player version).
// For a general description of the rules, see
// https://en.wikipedia.org/wiki/Dou_dizhu It uses a standard 54-card deck
// (including two Jokers). The game starts by randomly picking one card face up,
// which is then inserted into the shuffled deck. Then each player is dealt 17
// cards. Then the bidding phase starts. The player who got the face-up card
// becomes the first one to bid. Bidding round ends if (1) no one bids (2) two
// consecutive passes (3) maximum bid 3 was bidded. The one who wins the bidding
// phase becomes dizhu (landlord). Dizhu get the remaining 3 cards. The other
// players are called peasants. Starting with dizhu, the playing phase
// consisting of multiple tricks. The leader of a trick can play several
// allowable categories of hands. The players during a trick can only pass or
// play hands of the same pattern of higher rank. A player becomes the winner of
// a trick if the other two players passes. And then it becomes the leader of
// the next trick. In this game, suits DO NOT MATTER.
//
// The allowable categories of hands:
// Solo: a single card
// SoloChain: >=5 consecutive cards in rank, e.g., 34567
// Pair: a pair of card with the same rank
// PairChain: >= 3 consecutive pairs. e.g., 334455
// Trio: three of a rank. e.g., 444
// TrioWithSolo: a trio + a single hand. e.g., 3334
// Trio With Pair: a trio + a pair. e.g., 33344
// Airplane (TrioChain). >=2 consecutive trio. e.g., 333-444
// Airplane+solo. airplane where each trio carries a solo. e.g., 333-444-5-6
// Airplane+pair. airplane where each trio carries a pair. e.g., 333-444-55-66
// Bomb. Four of a rank. e.g., 4444
// Rocket. Two jokers
//
// Some other rules:
// The order for solo card is: ColoredJoker>BlackWhiteJoker>2>A>K>Q>....>4>3
// For combination hands, the primal part determines the order.
// e.g. the primal part of 333-444-5-6 is 333-444
// 2s and Jokers cannot be in a chain.
// Rocket dominates all other hands.
// A bomb dominates all other hands except rocket or bombs of higher rank.
// Bomb/rocket cannot appear in an airplane combination
// E.g., 333-444-555-666-7777 is prohibited.
// But in this implementation any pair and any trio can be kickers
// For more, see https://rezunli96.github.io/blog/doudizhu_count.html
//
// A game ends if a player has played all their cards.
// The winning bid determines the initial stake.
// Each bomb played doubles the stake.
// And if (1) both peasants do not play any cards
// (2) dizhu does not play any cards after its first hand, then it's called
// spring. And the stake is also doubled.

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/dou_dizhu/dou_dizhu_utils.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace dou_dizhu {

class Trick {
 public:
  Trick() : Trick(kInvalidPlayer, kInvalidAction) {}
  Trick(Player leader, int action);
  // winning_player_ is the current winner of the trick
  void Play(Player player, int action) {
    winning_player_ = player;
    winning_action_ = action;
  }
  int WinningAction() const { return winning_action_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }

 private:
  int winning_action_;
  const Player leader_;
  Player winning_player_;
};

class DouDizhuState : public State {
 public:
  DouDizhuState(std::shared_ptr<const Game> game);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return returns_; }
  std::string ObservationString(Player player) const override;
  void WriteObservationTensor(Player player, absl::Span<float> values) const;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return absl::make_unique<DouDizhuState>(*this);
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  // Current phase.
  int CurrentPhase() const { return static_cast<int>(phase_); }

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int action);
  void ApplyBiddingAction(int action);
  void ApplyPlayAction(int action);
  void ScoreUp();

  bool AfterPlayHand(int player, int action);
  Trick& CurrentTrick() { return tricks_[trick_played_]; }
  const Trick& CurrentTrick() const { return tricks_[trick_played_]; }
  // Recording each player got how many cards for each rank
  std::array<std::array<int, kNumRanks>, kNumPlayers> OriginalDeal() const;

  std::string FormatDeal() const;
  std::string FormatAuction() const;
  std::string FormatPlay() const;
  std::string FormatResult() const;
  // the ranks of the cards left over after dealing phase
  std::vector<int> cards_left_over_;

  int num_passes_ = 0;  // Number of consecutive passes since the last non-pass.
  int winning_bid_ = 0;
  int trick_played_ = 0;
  int num_played_ = 0;  // number of plays during playing phase
  int card_face_up_position_ = -1;
  int card_rank_face_up_ = kInvalidAction;
  bool new_trick_begin_ = false;
  Player current_player_ = kInvalidPlayer;
  Player first_player_ = kInvalidPlayer;
  Player dizhu_ = kInvalidPlayer;
  Player final_winner_ = kInvalidPlayer;
  Phase phase_ = Phase::kDeal;

  std::array<int, kNumCards> dealer_deck_{};
  std::array<int, kNumRanks> played_deck_{};
  std::vector<Trick> tricks_{};
  // for score computation
  int bombs_played_ = 0;
  std::array<int, kNumPlayers> players_hands_played{};

  std::vector<double> returns_ = std::vector<double>(kNumPlayers);
  // recording the current hands of players
  std::array<std::array<int, kNumRanks>, kNumPlayers> holds_{};
};

class DouDizhuGame : public Game {
 public:
  explicit DouDizhuGame(const GameParameters& params);
  int NumDistinctActions() const override { return kRocketActionBase + 1; }
  int MaxChanceOutcomes() const override {
    return kDealingActionBase + kNumCards;
  }
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<DouDizhuState>(shared_from_this());
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kMinUtility; }
  double MaxUtility() const override { return kMaxUtility; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  int MaxGameLength() const override {
    return kMaxAuctionLength + kNumCards * kNumPlayers;
  }
};
}  // namespace dou_dizhu
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOU_DIZHU_H_
