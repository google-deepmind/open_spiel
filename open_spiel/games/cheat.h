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

#ifndef OPEN_SPIEL_GAMES_CHEAT_H_
#define OPEN_SPIEL_GAMES_CHEAT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace cheat {

inline constexpr int kNumPlayers = 2;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumCards = 52;
inline constexpr int kFirstPlayer = 0;
inline constexpr int kSecondPlayer = 1;
// inline constexpr int kNumCardsInPass = 3;
// inline constexpr int kNumTricks = kNumCards / kNumPlayers;
// inline constexpr int kPointsForQS = 13;
inline constexpr int kMinScore = -1;
inline constexpr int kMaxScore = 1;
// inline constexpr int kTrickTensorSize = kNumCards * 7;  // N E S W N E S
inline constexpr int kActionSize = kNumCards * kNumCards + 1; // +1 for pass
inline constexpr int kMaxNumberOfTurns = kNumCards; // ! Check here
inline constexpr int kInformationStateTensorSize =
    kNumCards                           // Cards claimed
    + kNumCards                         // Cards seen
    + kNumCards                         // Current hand
    + kMaxNumberOfTurns * kActionSize;  // History of Actions

enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };
// Cards are represented as rank * kNumSuits + suit.
inline Suit CardSuit(int card) { return Suit(card % kNumSuits); }
inline int CardRank(int card) { return card / kNumSuits; }
inline int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}
constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";
inline std::string CardString(int card) {
  return {kRankChar[CardRank(card)],
          kSuitChar[static_cast<int>(CardSuit(card))]};
}

// State of a single trick.
class Trick {
 public:
  Trick() : Trick{kInvalidPlayer, 0, false} {}
  Trick(Player leader, int card, bool jd_bonus);
  void Play(Player player, int card);
  Suit LedSuit() const { return led_suit_; }
  Player Winner() const { return winning_player_; }
  Player Leader() const { return leader_; }
  int Points() const { return points_; }
  std::vector<int> Cards() const { return cards_; }

 private:
  bool jd_bonus_;
  int winning_rank_;
  int points_;
  Suit led_suit_;
  Player leader_;
  Player winning_player_;
  std::vector<int> cards_;
};

class CheatState : public State {
 public:
  CheatState(std::shared_ptr<const Game> game);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new CheatState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override; // ! Check here

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase { kDeal, kPass, kPlay, kGameOver };

  std::vector<Action> DealLegalActions() const;
  std::vector<Action> PassLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealAction(int card);
  void ApplyPassAction(int card);
  void ApplyPlayAction(int card);

  void ComputeScore();
  // int CurrentTrickIndex() const {
  //   return std::min(num_cards_played_ / kNumPlayers,
  //                   static_cast<int>(tricks_.size()));
  // }
  // Trick& CurrentTrick() { return tricks_[CurrentTrickIndex()]; }
  // const Trick& CurrentTrick() const { return tricks_[CurrentTrickIndex()]; }
  std::array<std::string, kNumSuits> FormatHand(int player,
                                                bool mark_voids) const;
  std::string FormatPlay() const;
  std::string FormatPass() const;
  std::string FormatPass(Player player) const;
  std::string FormatDeal() const;
  std::string FormatPoints() const;

  absl::optional<Player> Played(int card) const;
  bool KnowsLocation(Player player, int card) const;

  int num_cards_dealt_ = 0;
  int num_cards_played_ = 0;
  // bool hearts_broken_ = false;
  Player current_player_ = kChancePlayerId;
  Phase phase_ = Phase::kDeal;
  // std::array<Trick, kNumTricks> tricks_{};
  std::array<absl::optional<Player>, kNumCards> holder_{};
  std::array<absl::optional<Player>, kNumCards> initial_deal_{};
  std::vector<std::vector<int>> passed_cards_{kNumPlayers};
  std::vector<double> points_ = std::vector<double>(kNumPlayers, 0);
};

class CheatGame : public Game {
 public:
  explicit CheatGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCards; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new CheatState(
        shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kMinScore; }
  double MaxUtility() const override { return kMaxScore; }
  std::vector<int> InformationStateTensorShape() const override {
    return {kInformationStateTensorSize};
  }
  int MaxGameLength() const override {
    return 0; // ! Fix here
  }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
};

}  // namespace hearts
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HEARTS_H_
