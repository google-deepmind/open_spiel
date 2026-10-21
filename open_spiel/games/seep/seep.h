// Copyright 2026 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_SEEP_SEEP_H_
#define OPEN_SPIEL_GAMES_SEEP_SEEP_H_

// Implementation of the 4-player partnership fishing and house-building card
// game Seep (Sweep).
//
// https://www.pagat.com/fishing/seep.html
//
// Rules summary:
// - 4 players in two fixed partnerships: Team 0 (Players 0 and 2) vs.
//   Team 1 (Players 1 and 3).
// - Standard 52-card deck. Card capture values are Ace=1, 2..10=pip value,
//   Jack=11, Queen=12, King=13.
// - Card scoring points (100 total across the deck):
//   * All Spades: capture value (1 + 2 + ... + 13 = 91 points).
//   * Non-Spade Aces (Clubs, Diamonds, Hearts): 1 point each (3 points).
//   * Ten of Diamonds (TD): 6 points.
//   * All other cards: 0 points.
// - Dealing & Bidding:
//   * 4 cards are dealt to Player 0 (the bidder).
//   * If Player 0's initial 4-card hand contains no card with capture value
//     >= 9, those 4 dealt cards are returned to the deck, reshuffled, and
//     dealt again.
//   * Player 0 must bid a house value in {9, 10, 11, 12, 13} matching the
//     capture value of at least one card held in their initial 4-card hand
//     (before any floor cards are dealt or revealed).
//   * After the bid is made, 4 cards are dealt face-up to the floor.
//   * Player 0 immediately makes their opening play (turn 0) from their initial
//     4 cards onto the 4 floor cards, satisfying the bid value.
//   * Remaining 44 cards are then dealt (8 more to Player 0 and 12 each to
//     Players 1, 2, and 3) before Player 1 takes turn 1.
// - Play Phase (48 turns, 12 per player):
//   * On turn 0, Player 0 must make a move from their initial 4 cards whose
//     target value equals the bid (build/cement a house of the bid value,
//     capture floor cards of the bid value, or throw a card of the bid value).
//   * On any turn, a player plays one card from hand with an intent:
//     - kThrow: place the card as a loose pile on the floor (illegal if the
//       card can capture any pile or combination of uncemented piles on the
//       floor).
//     - kCapture: capture all floor piles of the card's value and/or disjoint
//       groups of uncemented floor piles summing to the card's value.
//     - kBuild9..kBuild13: build, raise, or cement a house of value H in 9..13
//       (legal only if the player retains another card of value H in hand, or
//       if cementing/adding to a house of value H already owned by their
//       partner; a player cannot break their own uncemented house, and must
//       retain a card of value V while owning a house of value V on the floor).
// - Seep (Sweep) Bonus:
//   * Clearing all cards from the floor awards +50 points to the capturing
//     partnership, except on the final (48th) play of the deal.
// - End of Deal:
//   * Any cards left on the floor after all 48 cards are played are awarded to
//     the partnership of the player who made the last capture.

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/json/include/nlohmann/json.hpp"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace seep {

inline constexpr int kNumPlayers = 4;
inline constexpr int kNumTeams = 2;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 13;
inline constexpr int kNumCards = kNumSuits * kNumRanks;  // 52
inline constexpr int kInitialFloorCards = 4;
inline constexpr int kInitialBidderCards = 4;
inline constexpr int kInitialDealCards =
    kInitialFloorCards + kInitialBidderCards;  // 8
inline constexpr int kCardsPerPlayer = 12;
inline constexpr int kTotalPlayTurns = kNumPlayers * kCardsPerPlayer;  // 48

inline constexpr int kMinBidValue = 9;
inline constexpr int kMaxBidValue = 13;
inline constexpr int kNumBids = kMaxBidValue - kMinBidValue + 1;  // 5 (9..13)

// Action space layout:
// - Bidding actions: 0..4 (representing bid values 9..13).
// - Play actions: 5 + (card * kNumIntents + intent) * kMaxSubsetChoices +
//   choice_index.
//   `choice_index` (0..kMaxSubsetChoices-1) selects among mutually exclusive
//   maximal floor-pile combinations when more than one exists (e.g. choosing
//   between 8+A and 7+2 when playing Ace to build 10); choice_index = 0 is
//   always the highest-point combination.
inline constexpr int kNumIntents = 7;  // Throw, Capture, Build 9..13
inline constexpr int kMaxSubsetChoices = 8;
inline constexpr int kBidActionBase = 0;
inline constexpr int kPlayActionBase = kNumBids;  // 5
inline constexpr int kNumDistinctActions =
    kPlayActionBase + kNumCards * kNumIntents * kMaxSubsetChoices;  // 2917

inline constexpr int kDeckTotalPoints = 100;
inline constexpr int kSeepBonusPoints = 50;
inline constexpr double kMaxUtility = 300.0;
inline constexpr double kMinUtility = -300.0;
inline constexpr int kMaxGameLength = 1 + kTotalPlayTurns;  // 1 bid + 48 plays
inline constexpr int kMaxRedeals = 10;
inline constexpr int kMaxChanceNodes =
    kNumCards + kMaxRedeals * kInitialBidderCards;
inline constexpr int kNumPhases = 6;

// Observation tensor size:
//   4 (observing player one-hot)
// + 6 (phase one-hot)
// + 5 (bid value 9..13 one-hot)
// + 52 (observing player's private hand)
// + 52 (cards currently on the floor)
// + 5 * 3 (houses of values 9..13: [exists, is_cemented, normalized_points])
// + 2 * 52 (captured cards per team)
// + 2 * 2 (normalized card points and seep count per team)
// = 242 floats.
inline constexpr int kObservationTensorSize =
    kNumPlayers + kNumPhases + kNumBids + kNumCards + kNumCards +
    (kNumBids * 3) + (kNumTeams * kNumCards) + (kNumTeams * 2);

enum class Suit { kClubs = 0, kDiamonds = 1, kHearts = 2, kSpades = 3 };

enum class Phase {
  kDealInitial = 0,
  kBid = 1,
  kDealFloor = 2,
  kDealRemainder = 3,
  kPlay = 4,
  kGameOver = 5,
};

enum class Intent {
  kThrow = 0,
  kCapture = 1,
  kBuild9 = 2,
  kBuild10 = 3,
  kBuild11 = 4,
  kBuild12 = 5,
  kBuild13 = 6,
};

std::string PhaseToString(Phase phase);
std::string IntentToString(Intent intent);
std::ostream& operator<<(std::ostream& os, Suit suit);
std::ostream& operator<<(std::ostream& os, Phase phase);
std::ostream& operator<<(std::ostream& os, Intent intent);

// Card encoding and scoring utilities.
inline Suit CardSuit(int card) { return static_cast<Suit>(card % kNumSuits); }
inline int CardRank(int card) { return card / kNumSuits; }
inline int Card(Suit suit, int rank) {
  return rank * kNumSuits + static_cast<int>(suit);
}
// Capture value: Ace = 1, 2..10 = 2..10, Jack = 11, Queen = 12, King = 13.
inline int CardCaptureValue(int card) { return CardRank(card) + 1; }
int CardPointValue(int card);
std::string CardString(int card);
int CardFromString(absl::string_view str);

// Partnership team index (0 for Players 0 & 2, 1 for Players 1 & 3).
inline int PlayerTeam(Player player) { return player % kNumTeams; }

// Action encoding/decoding utilities.
inline Action MakeBidAction(int bid_value) {
  return kBidActionBase + (bid_value - kMinBidValue);
}
inline int BidValueFromAction(Action action) {
  return static_cast<int>(action - kBidActionBase) + kMinBidValue;
}
inline Action MakePlayAction(int card, Intent intent, int choice_index = 0) {
  return kPlayActionBase +
         (card * kNumIntents + static_cast<int>(intent)) * kMaxSubsetChoices +
         choice_index;
}
inline std::pair<int, Intent> DecodePlayAction(Action action) {
  int offset = (static_cast<int>(action) - kPlayActionBase) / kMaxSubsetChoices;
  return {offset / kNumIntents, static_cast<Intent>(offset % kNumIntents)};
}
inline int DecodePlayActionChoiceIndex(Action action) {
  return (static_cast<int>(action) - kPlayActionBase) % kMaxSubsetChoices;
}
inline int IntentHouseValue(Intent intent) {
  return kMinBidValue +
         (static_cast<int>(intent) - static_cast<int>(Intent::kBuild9));
}

// Represents a pile on the floor (either a single loose card or a multi-card
// house of capture value 9..13).
struct House {
  int value = 0;             // Capture value (1..13).
  bool is_cemented = false;  // True if cemented (pakka); cannot be raised.
  std::vector<int> cards;    // Sorted card IDs in this pile.
  int owners_mask = 0;       // Bitmask (1 << player) of house owner(s).

  int PointValue() const;
  std::string ToString() const;
  bool operator==(const House& other) const {
    return value == other.value && is_cemented == other.is_cemented &&
           cards == other.cards && owners_mask == other.owners_mask;
  }
};

struct SeepStructContents {
  std::string phase;
  int current_player = kInvalidPlayer;
  int bid_value = 0;
  std::vector<std::vector<std::string>> hands;
  std::vector<std::string> floor_piles;
  std::vector<int> team_card_points;
  std::vector<int> team_seeps;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(SeepStructContents, phase, current_player,
                                 bid_value, hands, floor_piles,
                                 team_card_points, team_seeps);
};

SPIEL_DEFINE_STRUCT(SeepStateStruct, StateStruct, SeepStructContents);

struct SeepObservationExtras {
  int observing_player = kInvalidPlayer;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(SeepObservationExtras,
                                              observing_player);
};

SPIEL_DEFINE_STRUCT_WITH_EXTRAS(SeepObservationStruct, ObservationStruct,
                                SeepStructContents, SeepObservationExtras);

class SeepGame;

class SeepState : public State {
 public:
  explicit SeepState(std::shared_ptr<const Game> game);
  SeepState(const SeepState&) = default;
  SeepState& operator=(const SeepState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new SeepState(*this));
  }
  std::vector<Action> LegalActions() const override;
  ActionsAndProbs ChanceOutcomes() const override;
  std::unique_ptr<StateStruct> ToStruct() const override;
  std::unique_ptr<ObservationStruct> ToObservationStruct(
      Player player) const override;

  // Game-specific state inspection and test setup helpers.
  Phase CurrentPhase() const { return phase_; }
  int BidValue() const { return bid_value_; }
  const std::vector<House>& FloorPiles() const { return floor_piles_; }
  std::vector<int> Hand(Player player) const;
  const std::array<std::vector<int>, kNumTeams>& TeamCapturedCards() const {
    return team_captured_cards_;
  }
  std::array<int, kNumTeams> TeamCardPoints() const;
  const std::array<int, kNumTeams>& TeamSeeps() const { return team_seeps_; }
  std::array<int, kNumTeams> TeamTotalScores() const;
  Player LastCapturePlayer() const { return last_capture_player_; }
  int NumCardsDealt() const { return num_cards_dealt_; }
  int NumCardsPlayed() const { return num_cards_played_; }
  int NumRedeals() const { return num_redeals_; }

  // Configures a custom play-phase state for deterministic unit testing.
  void SetCustomPlayState(
      Player current_player, int bid_value, int num_cards_played,
      const std::array<std::vector<int>, kNumPlayers>& hands,
      const std::vector<House>& floor_piles,
      const std::array<std::vector<int>, kNumTeams>& team_captured_cards = {},
      const std::array<int, kNumTeams>& team_seeps = {0, 0},
      Player last_capture_player = kInvalidPlayer);

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  std::vector<Action> DealLegalActions() const;
  std::vector<Action> BidLegalActions() const;
  std::vector<Action> PlayLegalActions() const;

  void ApplyDealAction(int card);
  void ApplyBidAction(int bid_value);
  void ApplyPlayAction(int card, Intent intent, int choice_index);

  bool CanThrowCard(int card) const;
  bool CanCaptureWithCard(int card) const;
  bool CanBuildHouseWithCard(Player player, int card, int house_value) const;
  bool PlayerHoldsAnotherCardOfValue(Player player, int excluded_card,
                                     int value) const;

  void SortFloorPiles();
  void SweepRemainingFloorCardsAtEnd();

  Phase phase_ = Phase::kDealInitial;
  Player current_player_ = kChancePlayerId;
  int bid_value_ = 0;
  int num_cards_dealt_ = 0;
  int num_cards_played_ = 0;
  int num_redeals_ = 0;
  Player last_capture_player_ = kInvalidPlayer;

  std::array<bool, kNumCards> card_dealt_{};
  std::array<std::vector<int>, kNumPlayers> hands_{};
  std::vector<House> floor_piles_;
  std::array<std::vector<int>, kNumTeams> team_captured_cards_{};
  std::array<int, kNumTeams> team_seeps_ = {0, 0};
};

class SeepGame : public Game {
 public:
  explicit SeepGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new SeepState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return kMinUtility; }
  double MaxUtility() const override { return kMaxUtility; }
  absl::optional<double> UtilitySum() const override { return 0.0; }
  std::vector<int> ObservationTensorShape() const override {
    return {kObservationTensorSize};
  }
  int MaxGameLength() const override { return kMaxGameLength; }
  int MaxChanceNodesInHistory() const override { return kMaxChanceNodes; }
};

}  // namespace seep
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SEEP_SEEP_H_
