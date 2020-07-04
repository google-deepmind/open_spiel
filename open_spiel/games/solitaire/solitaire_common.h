#ifndef OPEN_SPIEL_GAMES_SOLITAIRE_SOLITAIRE_COMMON_H_
#define OPEN_SPIEL_GAMES_SOLITAIRE_SOLITAIRE_COMMON_H_

#include <string>
#include <vector>
#include <open_spiel/spiel_utils.h>

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
 private:
  RankType rank = RankType::kHidden;  // Indicates the rank of the card
  SuitType suit = SuitType::kHidden;  // Indicates the suit of the card
  LocationType location = LocationType::kMissing; // Indicates the type of pile the card is in
  bool hidden = false;      // Indicates whether the card is hidden or not
  int index = kHiddenCard;  // Identifies the card with an integer

 public:
  // Constructors
  explicit Card(bool hidden = false, SuitType suit = SuitType::kHidden,
                RankType rank = RankType::kHidden, LocationType location = LocationType::kMissing);
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
  bool operator==(const Card &other_card) const;
  bool operator<(const Card &other_card) const;

  // Other Methods
  std::string ToString(bool colored = true) const;
  std::vector<Card> LegalChildren() const;
};

class Pile {
 protected:
  std::vector<Card> cards;
  const LocationType type;
  const SuitType suit;
  const PileID id;
  const int max_size;

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
 private:
  Card target;
  Card source;

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
  bool operator<(const Move &other_move) const;
  Action ActionId() const;
};

}

#endif  // OPEN_SPIEL_GAMES_SOLITAIRE_SOLITAIRE_COMMON_H_