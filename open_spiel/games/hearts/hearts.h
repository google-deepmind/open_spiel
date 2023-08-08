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

#ifndef OPEN_SPIEL_GAMES_HEARTS_H_
#define OPEN_SPIEL_GAMES_HEARTS_H_

// Full implementation of the classic trick taking game Hearts.
// https://www.pagat.com/reverse/hearts.html
//
// Some notes on this implementation:
//
// - Pass Direction
// The direction of the pass is decided by the first chance action. If the
// "pass_cards" game parameter is set to false, the "No Pass" action will be
// the only legal action at the first chance node.
//
// In standard play, the direction of the pass alternates in
// a fixed pattern. Here, however, state is not preserved between hands, so
// the game itself cannot enforce that pattern. By using the first chance
// action to set the pass direction, the game can be dropped in to pre-existing
// algorithms without requiring modifications to coordinate training.
//
// - Returns
// Hearts is a trick-avoidance game in which the goal is to accumulate the
// fewest number of points. Because RL algorithms are designed to maximize
// reward, returns are calculated by subtracting the in-game points from an
// upper bound.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace hearts {

inline constexpr int kNumPlayers = 4;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 13;
inline constexpr int kNumCards = 52;
inline constexpr int kNumCardsInPass = 3;
inline constexpr int kNumTricks = kNumCards / kNumPlayers;
inline constexpr int kPointsForHeart = 1;
inline constexpr int kPointsForQS = 13;
inline constexpr int kPointsForJD = -10;
inline constexpr int kTotalPositivePoints = 26;  // All hearts + QS
inline constexpr int kMinScore = 0;
inline constexpr int kMaxScore = kTotalPositivePoints - kPointsForJD;
inline constexpr int kAvoidAllTricksBonus = -5;
inline constexpr int kTrickTensorSize = kNumCards * 7;  // N E S W N E S
inline constexpr int kInformationStateTensorSize =
    kNumPlayers                       // Pass direction
    + kNumCards                       // Dealt hand
    + kNumCards                       // 3 passed cards
    + kNumCards                       // 3 received cards
    + kNumCards                       // Current hand
    + kMaxScore * kNumPlayers         // Current point totals
    + kNumTricks * kTrickTensorSize;  // History of tricks

enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };
enum class PassDir { kNoPass = 0, kLeft = 1, kAcross = 2, kRight = 3 };
enum Seat { kNorth, kEast, kSouth, kWest };
// Cards are represented as rank * kNumSuits + suit.
inline Suit CardSuit(int card) { return Suit(card % kNumSuits); }
inline int CardRank(int card) { return card / kNumSuits; }
inline int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}
inline int CardPoints(int card, bool jd_bonus) {
  if (CardSuit(card) == Suit::kHearts) {
    return kPointsForHeart;
  } else if (card == Card(Suit::kSpades, 10)) {
    return kPointsForQS;
  } else if (card == Card(Suit::kDiamonds, 9) && jd_bonus) {
    return kPointsForJD;
  } else {
    return 0;
  }
}
constexpr char kRankChar[] = "23456789TJQKA";
constexpr char kSuitChar[] = "CDHS";
constexpr char kDirChar[] = "NESW";
inline std::string DirString(int dir) { return {kDirChar[dir]}; }
inline std::string CardString(int card) {
  return {kRankChar[CardRank(card)],
          kSuitChar[static_cast<int>(CardSuit(card))]};
}
inline std::map<int, std::string> pass_dir_str = {
    {0, "No Pass"}, {1, "Left"}, {2, "Across"}, {3, "Right"}};

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

class HeartsState : public State {
 public:
  HeartsState(std::shared_ptr<const Game> game, bool pass_cards,
              bool no_pts_on_first_trick, bool can_lead_any_club, bool jd_bonus,
              bool avoid_all_tricks_bonus, bool must_break_hearts,
              bool qs_breaks_hearts, bool can_lead_hearts_instead_of_qs);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new HeartsState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

 protected:
  void DoApplyAction(Action action) override;

 private:
  enum class Phase { kPassDir, kDeal, kPass, kPlay, kGameOver };

  std::vector<Action> PassDirLegalActions() const;
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> PassLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyPassDirAction(int pass_dir);
  void ApplyDealAction(int card);
  void ApplyPassAction(int card);
  void ApplyPlayAction(int card);

  void ComputeScore();
  int CurrentTrickIndex() const {
    return std::min(num_cards_played_ / kNumPlayers,
                    static_cast<int>(tricks_.size()));
  }
  Trick& CurrentTrick() { return tricks_[CurrentTrickIndex()]; }
  const Trick& CurrentTrick() const { return tricks_[CurrentTrickIndex()]; }
  std::array<std::string, kNumSuits> FormatHand(int player,
                                                bool mark_voids) const;
  std::string FormatPlay() const;
  std::string FormatPass() const;
  std::string FormatPass(Player player) const;
  std::string FormatDeal() const;
  std::string FormatPoints() const;

  absl::optional<Player> Played(int card) const;
  bool KnowsLocation(Player player, int card) const;

  const bool pass_cards_;
  const bool no_pts_on_first_trick_;
  const bool can_lead_any_club_;
  const bool jd_bonus_;
  const bool avoid_all_tricks_bonus_;
  const bool qs_breaks_hearts_;
  const bool must_break_hearts_;
  const bool can_lead_hearts_instead_of_qs_;

  int num_cards_dealt_ = 0;
  int num_cards_played_ = 0;
  bool hearts_broken_ = false;
  Player current_player_ = kChancePlayerId;
  Phase phase_ = Phase::kPassDir;
  PassDir pass_dir_ = PassDir::kNoPass;
  std::array<Trick, kNumTricks> tricks_{};
  std::array<absl::optional<Player>, kNumCards> holder_{};
  std::array<absl::optional<Player>, kNumCards> initial_deal_{};
  std::vector<std::vector<int>> passed_cards_{kNumPlayers};
  std::vector<double> points_ = std::vector<double>(kNumPlayers, 0);
};

class HeartsGame : public Game {
 public:
  explicit HeartsGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumCards; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new HeartsState(
        shared_from_this(), /*pass_cards=*/pass_cards_,
        /*no_pts_on_first_trick=*/no_pts_on_first_trick_,
        /*can_lead_any_club=*/can_lead_any_club_,
        /*jd_bonus=*/jd_bonus_,
        /*avoid_all_tricks_bonus=*/avoid_all_tricks_bonus_,
        /*must_break_hearts=*/must_break_hearts_,
        /*qs_breaks_hearts=*/qs_breaks_hearts_,
        /*can_lead_hearts_instead_of_qs=*/can_lead_hearts_instead_of_qs_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kMinScore; }
  double MaxUtility() const override { return kMaxScore; }
  std::vector<int> InformationStateTensorShape() const override {
    return {kInformationStateTensorSize};
  }
  int MaxGameLength() const override {
    return (kNumCardsInPass * kNumPlayers) + kNumCards;
  }
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }

 private:
  const bool pass_cards_;
  const bool no_pts_on_first_trick_;
  const bool can_lead_any_club_;
  const bool jd_bonus_;
  const bool avoid_all_tricks_bonus_;
  const bool qs_breaks_hearts_;
  const bool must_break_hearts_;
  const bool can_lead_hearts_instead_of_qs_;
};

}  // namespace hearts
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_HEARTS_H_
