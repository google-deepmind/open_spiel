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

#ifndef OPEN_SPIEL_GAMES_SOLITAIRE_H
#define OPEN_SPIEL_GAMES_SOLITAIRE_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// An implementation of klondike solitaire:
// https://en.wikipedia.org/wiki/Klondike_(solitaire) More specifically, it is
// K+ solitaire, which allows the player to play any card from the deck/waste
// that would normally become playable after some number of draws in standard
// klondike solitaire. For a more in-depth description of K+ solitaire, see
// http://web.engr.oregonstate.edu/~afern/papers/solitaire.pdf. This
// implementation also gives rewards at intermediate states like most electronic
// versions of solitaire do, rather than only at terminal states.

namespace open_spiel::solitaire {

// Default Game Parameters =====================================================

inline constexpr int kDefaultPlayers = 1;
inline constexpr int kDefaultDepthLimit = 150;
inline constexpr bool kDefaultIsColored = false;

// Constants ===================================================================

inline constexpr int kHiddenCard = 99;

// Enumerations ================================================================

enum class SuitType {
  kNone = 0,
  kSpades,
  kHearts,
  kClubs,
  kDiamonds,
  kHidden,
};

enum class RankType {
  kNone = 0,
  kA,
  k2,
  k3,
  k4,
  k5,
  k6,
  k7,
  k8,
  k9,
  kT,
  kJ,
  kQ,
  kK,
  kHidden,
};

enum class LocationType {
  kDeck = 0,
  kWaste = 1,
  kFoundation = 2,
  kTableau = 3,
  kMissing = 4,
};

enum class PileID {
  kWaste = 0,
  kSpades = 1,
  kHearts = 2,
  kClubs = 3,
  kDiamonds = 4,
  k1stTableau = 5,
  k2ndTableau = 6,
  k3rdTableau = 7,
  k4thTableau = 8,
  k5thTableau = 9,
  k6thTableau = 10,
  k7thTableau = 11,
  kMissing = 12
};

// Support Classes =============================================================

class Card {
 public:
  // Constructors
  explicit Card(bool hidden = false, SuitType suit = SuitType::kHidden,
                RankType rank = RankType::kHidden,
                LocationType location = LocationType::kMissing);
  explicit Card(int index, bool hidden = false,
                LocationType location = LocationType::kMissing);

  // Getters
  RankType GetRank() const;
  SuitType GetSuit() const;
  LocationType GetLocation() const;
  bool GetHidden() const;
  int GetIndex() const;

  // Setters
  void SetRank(RankType new_rank);
  void SetSuit(SuitType new_suit);
  void SetLocation(LocationType new_location);
  void SetHidden(bool new_hidden);

  // Operators
  bool operator==(const Card& other_card) const;
  bool operator<(const Card& other_card) const;

  // Other Methods
  std::string ToString(bool colored = true) const;
  std::vector<Card> LegalChildren() const;

 private:
  RankType rank_ = RankType::kHidden;  // Indicates the rank of the card
  SuitType suit_ = SuitType::kHidden;  // Indicates the suit of the card
  LocationType location_ =
      LocationType::kMissing;  // Indicates the type of pile the card is in
  bool hidden_ = false;        // Indicates whether the card is hidden or not
  int index_ = kHiddenCard;    // Identifies the card with an integer
};

class Pile {
 public:
  // Constructor
  Pile(LocationType type, PileID id, SuitType suit = SuitType::kNone);

  // Destructor
  virtual ~Pile() = default;

  // Getters/Setters
  bool GetIsEmpty() const;
  SuitType GetSuit() const;
  LocationType GetType() const;
  PileID GetID() const;
  Card GetFirstCard() const;
  Card GetLastCard() const;
  std::vector<Card> GetCards() const;
  void SetCards(std::vector<Card> new_cards);

  // Other Methods
  virtual std::vector<Card> Sources() const;
  virtual std::vector<Card> Targets() const;
  virtual std::vector<Card> Split(Card card);
  virtual void Reveal(Card card_to_reveal);
  void Extend(std::vector<Card> source_cards);
  std::string ToString(bool colored = true) const;

 protected:
  std::vector<Card> cards_;
  const LocationType type_;
  const SuitType suit_;
  const PileID id_;
  const int max_size_;
};

class Tableau : public Pile {
 public:
  // Constructor
  explicit Tableau(PileID id);

  // Other Methods
  std::vector<Card> Sources() const override;
  std::vector<Card> Targets() const override;
  std::vector<Card> Split(Card card) override;
  void Reveal(Card card_to_reveal) override;
};

class Foundation : public Pile {
 public:
  // Constructor
  Foundation(PileID id, SuitType suit);

  // Other Methods
  std::vector<Card> Sources() const override;
  std::vector<Card> Targets() const override;
  std::vector<Card> Split(Card card) override;
};

class Waste : public Pile {
 public:
  // Constructor
  Waste();

  // Other Methods
  std::vector<Card> Sources() const override;
  std::vector<Card> Targets() const override;
  std::vector<Card> Split(Card card) override;
  void Reveal(Card card_to_reveal) override;
};

class Move {
 public:
  // Constructors
  Move(Card target_card, Card source_card);
  Move(RankType target_rank, SuitType target_suit, RankType source_rank,
       SuitType source_suit);
  explicit Move(Action action);

  // Getters
  Card GetTarget() const;
  Card GetSource() const;

  // Other Methods
  // ===========================================================================
  std::string ToString(bool colored = true) const;
  bool operator<(const Move& other_move) const;
  Action ActionId() const;

 private:
  Card target_;
  Card source_;
};

class SolitaireGame : public Game {
 public:
  // Constructor
  explicit SolitaireGame(const GameParameters& params);

  // Overridden Methods
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  // TODO: verify whether this bound is tight and/or tighten it.
  int MaxChanceNodesInHistory() const override { return MaxGameLength(); }
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;

  std::vector<int> ObservationTensorShape() const override;
  std::unique_ptr<State> NewInitialState() const override;

 private:
  int num_players_;
  int depth_limit_;
  bool is_colored_;
};

class SolitaireState : public State {
 public:
  // Constructors
  explicit SolitaireState(std::shared_ptr<const Game> game);

  // Overridden Methods
  Player CurrentPlayer() const override;
  std::unique_ptr<State> Clone() const override;
  bool IsTerminal() const override;
  bool IsChanceNode() const override;
  std::string ToString() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  void DoApplyAction(Action action) override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Other Methods
  std::vector<Card> Targets(const absl::optional<LocationType>& location =
                                LocationType::kMissing) const;
  std::vector<Card> Sources(const absl::optional<LocationType>& location =
                                LocationType::kMissing) const;
  std::vector<Move> CandidateMoves() const;
  Pile* GetPile(const Card& card);
  const Pile* GetPile(const Card& card) const;
  void MoveCards(const Move& move);
  bool IsReversible(const Card& source, const Pile* source_pile) const;

 private:
  Waste waste_;
  std::vector<Foundation> foundations_;
  std::vector<Tableau> tableaus_;
  std::vector<Action> revealed_cards_;

  bool is_finished_ = false;
  bool is_reversible_ = false;
  int current_depth_ = 0;

  std::set<std::size_t> previous_states_ = {};
  std::map<Card, PileID> card_map_;

  double current_returns_ = 0.0;
  double current_rewards_ = 0.0;

  // Parameters
  int depth_limit_ = kDefaultDepthLimit;
  bool is_colored_ = kDefaultIsColored;
};

}  // namespace open_spiel::solitaire

#endif  // OPEN_SPIEL_GAMES_SOLITAIRE_H
