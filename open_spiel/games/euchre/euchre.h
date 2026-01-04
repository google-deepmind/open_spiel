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

#ifndef OPEN_SPIEL_GAMES_EUCHRE_H_
#define OPEN_SPIEL_GAMES_EUCHRE_H_

// Full implementation of the classic trick taking game Euchre.
//
// https://en.wikipedia.org/wiki/Euchre
// https://www.pagat.com/euchre/euchre.html
//
// This implementation uses standard North American rules with "super-Euchres",
// i.e. the makers lose 4 points if they fail to win a single trick. By default,
// only the declarer has the option of playing alone, but optionally the
// defenders can go alone as well. The popular variation "stick the dealer" is
// enabled by default as it has interesting strategic implications and increases
// playability by avoiding drawn hands.

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace euchre {

inline constexpr int kNumPlayers = 4;
inline constexpr int kJackRank = 2;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumCardsPerSuit = 6;
inline constexpr int kNumCards = 24;

inline constexpr int kPassAction = 24;
inline constexpr int kClubsTrumpAction = 25;
inline constexpr int kDiamondsTrumpAction = 26;
inline constexpr int kHeartsTrumpAction = 27;
inline constexpr int kSpadesTrumpAction = 28;
inline constexpr int kGoAloneAction = 29;
inline constexpr int kPlayWithPartnerAction = 30;
inline constexpr int kNumDistinctActions = 31;
// Dealer selection + deal + upcard
inline constexpr int kFirstBiddingActionInHistory = 22;

inline constexpr int kMaxBids = 8;
inline constexpr int kNumTricks = 5;
inline constexpr int kFullHandSize = 5;
inline constexpr int kMaxScore = 4;
inline constexpr int kMinScore = -4;
inline constexpr int kTrickTensorSize = kNumCards * 7;  // N E S W N E S
inline constexpr int kInformationStateTensorSize =
    kNumPlayers                       // Dealer
    + kNumCards                       // Upcard
    + (kNumSuits + 1) * kMaxBids      // Bidding
    + 3                               // Go alone (declarer, defender 1 & 2)
    + kNumCards                       // Current hand
    + kNumTricks * kTrickTensorSize;  // History of tricks

enum class Phase { kDealerSelection, kDeal, kBidding, kDiscard, kGoAlone, kPlay,
                   kGameOver };
enum class Suit { kInvalidSuit = -1, kClubs = 0, kDiamonds = 1,
                  kHearts = 2, kSpades = 3 };
enum Seat { kNorth, kEast, kSouth, kWest };
// Cards are represented as rank * kNumSuits + suit.
inline Suit CardSuit(int card) { return Suit(card % kNumSuits); }
Suit CardSuit(int card, Suit trump_suit);
inline int CardRank(int card) { return card / kNumSuits; }
int CardRank(int card, Suit trump_suit);
inline int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}
constexpr char kRankChar[] = "9TJQKA";
constexpr char kSuitChar[] = "CDHS";
constexpr char kDirChar[] = "NESW";
inline std::string DirString(int dir) {
  if (dir < 0)
    return "";
  else
    return {kDirChar[dir]};
}
inline std::string CardString(int card) {
  return {kSuitChar[static_cast<int>(CardSuit(card))],
          kRankChar[CardRank(card)]};
}


// State of a single trick.
class Trick {
 public:
  Trick() : Trick{kInvalidPlayer, Suit::kInvalidSuit, kInvalidAction} {}
  Trick(Player leader, Suit trump_suit, int card);
  void Play(Player player, int card);
  int WinningCard() const { return winning_card_; }
  Suit LedSuit() const { return led_suit_; }
  Suit TrumpSuit() const { return trump_suit_; }
  bool TrumpPlayed() const { return trump_played_; }
  Player Leader() const { return leader_; }
  Player Winner() const { return winning_player_; }
  std::vector<int> Cards() const { return cards_; }

 private:
  int winning_card_;
  Suit led_suit_;
  Suit trump_suit_;
  bool trump_played_;
  Player leader_;  // First player to throw.
  Player winning_player_;
  std::vector<int> cards_;
};

class EuchreState : public State {
 public:
  EuchreState(std::shared_ptr<const Game> game, bool allow_lone_defender,
              bool stick_the_dealer);
  Player CurrentPlayer() const override { return current_player_; }
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override { return points_; }
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new EuchreState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  int NumCardsDealt() const { return num_cards_dealt_; }
  int NumCardsPlayed() const { return num_cards_played_; }
  int NumPasses() const { return num_passes_; }
  int Upcard() const { return upcard_; }
  int Discard() const { return discard_; }
  int TrumpSuit() const { return static_cast<int>(trump_suit_); }
  int LeftBower() const { return left_bower_; }
  int RightBower() const { return right_bower_; }
  int Declarer() const { return declarer_; }
  int FirstDefender() const { return first_defender_; }
  int DeclarerPartner() const { return declarer_partner_; }
  int SecondDefender() const { return second_defender_; }
  absl::optional<bool> DeclarerGoAlone() const { return declarer_go_alone_; }
  Player LoneDefender() const { return lone_defender_; }
  std::vector<bool> ActivePlayers() const { return active_players_; }
  Player Dealer() const { return dealer_; }

  Phase CurrentPhase() const { return phase_; }

  int CurrentTrickIndex() const {
    return std::min(num_cards_played_ / num_active_players_,
                    static_cast<int>(tricks_.size()));
  }
  Trick& CurrentTrick() { return tricks_[CurrentTrickIndex()]; }
  const Trick& CurrentTrick() const { return tricks_[CurrentTrickIndex()]; }

  std::array<absl::optional<Player>, kNumCards> CardHolder() const {
    return holder_;
  }
  std::vector<Trick> Tricks() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> DealerSelectionLegalActions() const;
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BiddingLegalActions() const;
  std::vector<Action> DiscardLegalActions() const;
  std::vector<Action> GoAloneLegalActions() const;
  std::vector<Action> PlayLegalActions() const;
  void ApplyDealerSelectionAction(int selected_dealer);
  void ApplyDealAction(int card);
  void ApplyBiddingAction(int action);
  void ApplyDiscardAction(int card);
  void ApplyGoAloneAction(int action);
  void ApplyPlayAction(int card);

  void ComputeScore();

  std::array<std::string, kNumSuits> FormatHand(int player,
                                                bool mark_voids) const;
  std::string FormatBidding() const;
  std::string FormatDeal() const;
  std::string FormatPlay() const;
  std::string FormatPoints() const;

  const bool allow_lone_defender_;
  const bool stick_the_dealer_;

  int num_cards_dealt_ = 0;
  int num_cards_played_ = 0;
  int num_passes_ = 0;
  int upcard_ = kInvalidAction;
  int discard_ = kInvalidAction;
  Suit trump_suit_ = Suit::kInvalidSuit;
  int left_bower_ = kInvalidAction;
  int right_bower_ = kInvalidAction;
  Player declarer_ = kInvalidPlayer;
  Player declarer_partner_ = kInvalidPlayer;
  Player first_defender_ = kInvalidPlayer;
  Player second_defender_ = kInvalidPlayer;
  absl::optional<bool> declarer_go_alone_;
  Player lone_defender_ = kInvalidPlayer;
  std::vector<bool> active_players_ = std::vector<bool>(kNumPlayers, true);
  int num_active_players_ = kNumPlayers;
  Player current_player_ = kChancePlayerId;
  Player dealer_ = kChancePlayerId;
  Phase phase_ = Phase::kDealerSelection;
  std::array<Trick, kNumTricks> tricks_{};
  std::array<absl::optional<Player>, kNumCards> holder_{};
  std::array<absl::optional<Player>, kNumCards> initial_deal_{};
  std::vector<double> points_ = std::vector<double>(kNumPlayers, 0);
};

class EuchreGame : public Game {
 public:
  explicit EuchreGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new EuchreState(shared_from_this(),
        /*allow_lone_defender=*/allow_lone_defender_,
        /*stick_the_dealer=*/stick_the_dealer_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kMinScore; }
  double MaxUtility() const override { return kMaxScore; }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override {
    return {kInformationStateTensorSize};
  }
  int MaxGameLength() const override {
    return (2 * kNumPlayers) +        // Max 2 rounds of bidding
        1 +                           // Declarer go alone?
        (2 * allow_lone_defender_) +  // Defenders go alone? (optional)
        (kNumPlayers * kNumTricks);   // Play of hand
  }
  int MaxChanceNodesInHistory() const override {
    return 1 +                        // Dealer selection
        (kNumPlayers * kNumTricks) +  // Deal hands
        1;                            // Upcard
  }

 private:
  const bool allow_lone_defender_;
  const bool stick_the_dealer_;
};

}  // namespace euchre
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_EUCHRE_H_
