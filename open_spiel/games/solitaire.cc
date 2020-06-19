#include "solitaire.h"

namespace open_spiel::solitaire {

namespace {
const GameType kGameType{/*short_name=*/"solitaire",
                         /*long_name=*/"Klondike Solitaire",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kGeneralSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/1,
                         /*min_num_players=*/1,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/false,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)},
                          {"is_colored", GameParameter(kDefaultIsColored)},
                          {"depth_limit", GameParameter(kDefaultDepthLimit)}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new SolitaireGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory)
}  // namespace

namespace {
// ANSI color codes
inline constexpr const char *kReset = "\033[0m";
inline constexpr const char *kRed = "\033[31m";
inline constexpr const char *kBlack = "\033[37m";

// Unicode Glyphs
inline constexpr const char *kGlyphHidden = "\U0001F0A0";
inline constexpr const char *kGlyphEmpty = "\U0001F0BF";
inline constexpr const char *kGlyphSpades = "\U00002660";
inline constexpr const char *kGlyphHearts = "\U00002665";
inline constexpr const char *kGlyphClubs = "\U00002663";
inline constexpr const char *kGlyphDiamonds = "\U00002666";
inline constexpr const char *kGlyphArrow = "\U00002190";

// Constants ===================================================================
inline constexpr int kNumRanks = 13;

// Number of cards that can be in each pile type
inline constexpr int kMaxSizeWaste = 24;
inline constexpr int kMaxSizeFoundation = 13;
inline constexpr int kMaxSizeTableau = 19;

// Number of sources that can be in each pile type
inline constexpr int kMaxSourcesWaste = 8;
inline constexpr int kMaxSourcesFoundation = 1;
inline constexpr int kMaxSourcesTableau = 13;

// These divide up the action ids into sections. kEnd is a single action that is
// used to end the game when no other actions are available.
inline constexpr int kEnd = 0;

// kReveal actions are ones that can be taken at chance nodes; they change a
// hidden card to a card of the same index as the action id (e.g. 2 would reveal
// a 2 of spades)
inline constexpr int kRevealStart = 1;
inline constexpr int kRevealEnd = 52;

// kMove actions are ones that are taken at decision nodes; they involve moving
// a card to another cards location. It starts at 53 because there are 52 reveal
// actions before it. See `NumDistinctActions()` in solitaire.cc.
inline constexpr int kMoveStart = 53;
inline constexpr int kMoveEnd = 204;

// Indices for special cards
// inline constexpr int kHiddenCard = 99;
inline constexpr int kEmptySpadeCard = -5;
inline constexpr int kEmptyHeartCard = -4;
inline constexpr int kEmptyClubCard = -3;
inline constexpr int kEmptyDiamondCard = -2;
inline constexpr int kEmptyTableauCard = -1;

// 1 empty + 13 ranks
inline constexpr int kFoundationTensorLength = 14;

// 6 hidden cards + 1 empty tableau + 52 ordinary cards
inline constexpr int kTableauTensorLength = 59;

// 1 hidden card + 52 ordinary cards
inline constexpr int kWasteTensorLength = 53;

// Constant for how many hidden cards can show up in a tableau. As hidden cards
// can't be added, the max is the highest number in a tableau at the start of
// the game: 6
inline constexpr int kMaxHiddenCard = 6;

// Only used in one place and just for consistency (to match kChancePlayerId &
// kTerminalPlayerId)
inline constexpr int kPlayerId = 0;

// Indicates the last index before the first player action (the last kReveal
// action has an ID of 52)
inline constexpr int kActionOffset = 52;

// Order of suits
const std::vector<SuitType> kSuits = {
    kSuitSpades, kSuitHearts, kSuitClubs, kSuitDiamonds
};

// These correspond with their enums, not with the two vectors directly above
const std::vector<std::string> kSuitStrs = {
    "", kGlyphSpades, kGlyphHearts, kGlyphClubs, kGlyphDiamonds, ""};
const std::vector<std::string> kRankStrs = {
    "", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", ""};

const std::map<RankType, double> kFoundationPoints = {
    // region Maps a RankType to the reward for moving a card of that rank to
    // the foundation
    {kRankA, 100.0}, {kRank2, 90.0}, {kRank3, 80.0}, {kRank4, 70.0},
    {kRank5, 60.0},  {kRank6, 50.0}, {kRank7, 40.0}, {kRank8, 30.0},
    {kRank9, 20.0},  {kRankT, 10.0}, {kRankJ, 10.0}, {kRankQ, 10.0},
    {kRankK, 10.0}
    // endregion
};

const std::map<SuitType, PileID> kSuitToPile = {
    // region Maps a foundation suit to the ID of the foundation
    {kSuitSpades, kPileSpades},
    {kSuitHearts, kPileHearts},
    {kSuitClubs, kPileClubs},
    {kSuitDiamonds, kPileDiamonds}
    // endregion
};

const std::map<int, PileID> kIntToPile = {
    // region Maps an integer to a tableau pile ID (used when initializing
    // SolitaireState)
    {1, kPile1stTableau}, {2, kPile2ndTableau}, {3, kPile3rdTableau},
    {4, kPile4thTableau}, {5, kPile5thTableau}, {6, kPile6thTableau},
    {7, kPile7thTableau}
    // endregion
};

}  // namespace

// Miscellaneous ===============================================================

std::vector<SuitType> GetOppositeSuits(const SuitType &suit) {
  /* Just returns a vector of the suits of opposite color. For red suits
   * (kSuitHearts and kSuitDiamonds), this returns the black suits (kSuitSpades
   * and kSuitClubs). For a black suit, this returns the red suits. The last
   * `SuitType` would be `kSuitNone` which should only occur with empty tableau
   * cards or hidden cards. Empty tableau cards should accept any suit, but
   * hidden cards are the opposite; they shouldn't accept any. There isn't
   * really a use case for calling this function with the suit of a hidden card
   * though. */

  switch (suit) {
    case kSuitSpades: {
      return {kSuitHearts, kSuitDiamonds};
    }
    case kSuitHearts: {
      return {kSuitSpades, kSuitClubs};
    }
    case kSuitClubs: {
      return {kSuitHearts, kSuitDiamonds};
    }
    case kSuitDiamonds: {
      return {kSuitSpades, kSuitClubs};
    }
    case kSuitNone: {
      return {kSuitSpades, kSuitHearts, kSuitClubs, kSuitDiamonds};
    }
    default: {
      SpielFatalError("suit is not in (s, h, c, d)");
    }
  }
}

int GetCardIndex(RankType rank, SuitType suit) {
  /* Using a given rank and/or suit, gets an integer representing the index of
   * the card. */

  if (rank == kRankHidden || suit == kSuitHidden) {
    // Handles hidden cards
    return kHiddenCard;
  } else if (rank == kRankNone) {
    // Handles special cards
    if (suit == kSuitNone) {
      // Handles empty tableau cards
      return kEmptyTableauCard;
    } else {
      // Handles empty foundation cards
      switch (suit) {
        case kSuitSpades: {
          return kEmptySpadeCard;
        }
        case kSuitHearts: {
          return kEmptyHeartCard;
        }
        case kSuitClubs: {
          return kEmptyClubCard;
        }
        case kSuitDiamonds: {
          return kEmptyDiamondCard;
        }
        default: {
          SpielFatalError("Failed to get card index");
        }
      }
    }
  } else {
    // Handles ordinary cards (e.g. 0-13 -> spades, 14-26 -> hearts, etc.)
    return (suit - 1) * kNumRanks + rank;
  }
}

int GetMaxSize(LocationType location) {
  switch (location) {
    case kDeck ... kWaste: {
      // Cards can only be removed from the waste & there are 24 cards in it at
      // the start of the game
      return kMaxSizeWaste;
    }
    case kFoundation: {
      // There are 13 cards in a suit
      return kMaxSizeFoundation;
    }
    case kTableau: {
      // There are a maximum of 6 hidden cards and 13 non-hidden cards in a
      // tableau (1 for each rank)
      return kMaxSizeTableau;
    }
    default: {
      return 0;
    }
  }
}

std::hash<std::string> hasher;

// Card Methods ================================================================

Card::Card(bool hidden, SuitType suit, RankType rank, LocationType location)
    : hidden(hidden), suit(suit), rank(rank), location(location) {}

Card::Card(int index, bool hidden, LocationType location)
    : index(index), hidden(hidden), location(location) {
  if (!hidden) {
    switch (index) {
      case kHiddenCard: {
        rank = kRankHidden;
        suit = kSuitHidden;
        break;
      }
      case kEmptyTableauCard: {
        rank = kRankNone;
        suit = kSuitNone;
        break;
      }
      case kEmptySpadeCard: {
        rank = kRankNone;
        suit = kSuitSpades;
        break;
      }
      case kEmptyHeartCard: {
        rank = kRankNone;
        suit = kSuitHearts;
        break;
      }
      case kEmptyClubCard: {
        rank = kRankNone;
        suit = kSuitClubs;
        break;
      }
      case kEmptyDiamondCard: {
        rank = kRankNone;
        suit = kSuitDiamonds;
        break;
      }
      default: {
        // Converts an index back into a rank and suit for ordinary cards
        rank = static_cast<RankType>(1 + ((index - 1) % kNumRanks));
        suit = static_cast<SuitType>(
            static_cast<int>(1 + floor((index - 1) / 13.0)));
      }
    }
  }
}

// Getters

RankType Card::GetRank() const { return rank; }

SuitType Card::GetSuit() const { return suit; }

LocationType Card::GetLocation() const { return location; }

bool Card::GetHidden() const { return hidden; }

int Card::GetIndex() const {
  /* Basically it just calculates the index if it hasn't been calculated before,
   * otherwise it will just return a stored value. If `force` is true and the
   * card isn't hidden, then the index is calculated again. */
  return hidden ? kHiddenCard : GetCardIndex(rank, suit);
}

// Setters

void Card::SetRank(RankType new_rank) { rank = new_rank; }

void Card::SetSuit(SuitType new_suit) { suit = new_suit; }

void Card::SetLocation(LocationType new_location) { location = new_location; }

void Card::SetHidden(bool new_hidden) { hidden = new_hidden; }

// Other Methods

std::string Card::ToString(bool colored) const {
  std::string result;

  // Determine color of string
  if (colored && !hidden) {
    if (suit == kSuitSpades || suit == kSuitClubs) {
      absl::StrAppend(&result, kBlack);
    } else if (suit == kSuitHearts || suit == kSuitDiamonds) {
      absl::StrAppend(&result, kRed);
    }
  }

  // Determine contents of string
  if (rank == kRankHidden || suit == kSuitHidden) {
    absl::StrAppend(&result, kGlyphHidden, " ");
  } else if (rank == kRankNone && suit == kSuitNone) {
    absl::StrAppend(&result, kGlyphEmpty);
  } else {
    absl::StrAppend(&result, kRankStrs.at(rank));
    absl::StrAppend(&result, kSuitStrs.at(suit));
  }

  if (colored) {
    // Reset color if applicable
    absl::StrAppend(&result, kReset);
  }

  return result;
}

std::vector<Card> Card::LegalChildren() const {
  if (hidden) {
    return {};
  } else {
    RankType child_rank;
    std::vector<SuitType> child_suits;

    // A card can have a maximum of 4 children
    // (specifically, an empty tableau card can accept a king of any suit)
    child_suits.reserve(4);

    switch (location) {
      case kTableau: {
        switch (rank) {
          case kRankNone: {
            if (suit == kSuitNone) {
              // Empty tableaus can accept a king of any suit
              child_rank = kRankK;
              child_suits = kSuits;
              break;
            } else {
              return {};
            }
          }
          case kRank2 ... kRankK: {
            // Ordinary cards (except aces) can accept cards of an opposite suit
            // that is one rank lower
            child_rank = static_cast<RankType>(rank - 1);
            child_suits = GetOppositeSuits(suit);
            break;
          }
          default: {
            // This will catch kRankA and kRankHidden
            return {};
          }
        }
        break;
      }
      case kFoundation: {
        switch (rank) {
          case kRankNone: {
            if (suit != kSuitNone) {
              child_rank = static_cast<RankType>(rank + 1);
              child_suits = {suit};
              break;
            } else {
              return {};
            }
          }
          case kRankA ... kRankQ: {
            // Cards (except kings) can accept a card of the same suit that is
            // one rank higher
            child_rank = static_cast<RankType>(rank + 1);
            child_suits = {suit};
            break;
          }
          default: {
            // This could catch kRankK and kRankHidden
            return {};
          }
        }
        break;
      }
      default: {
        // This catches all cards that aren't located in a tableau or foundation
        return {};
      }
    }

    std::vector<Card> legal_children;
    legal_children.reserve(4);

    if (child_suits.empty()) {
      SpielFatalError("child_suits should not be empty");
    }

    for (const auto &child_suit : child_suits) {
      auto child = Card(false, child_suit, child_rank);
      legal_children.push_back(child);
    }

    return legal_children;
  }
}

bool Card::operator==(const Card &other_card) const {
  return rank == other_card.rank && suit == other_card.suit;
}

bool Card::operator<(const Card &other_card) const {
  if (suit != other_card.suit) {
    return suit < other_card.suit;
  } else if (rank != other_card.rank) {
    return rank < other_card.rank;
  } else {
    return false;
  }
}

// Pile Methods ================================================================

Pile::Pile(LocationType type, PileID id, SuitType suit)
    : type(type), id(id), suit(suit), max_size(GetMaxSize(type)) {
  cards.reserve(max_size);
}

// Getters/Setters

bool Pile::GetIsEmpty() const { return cards.empty(); }

Card Pile::GetFirstCard() const { return cards.front(); }

Card Pile::GetLastCard() const { return cards.back(); }

SuitType Pile::GetSuit() const { return suit; }

LocationType Pile::GetType() const { return type; }

PileID Pile::GetID() const { return id; }

std::vector<Card> Pile::GetCards() const { return cards; }

void Pile::SetCards(std::vector<Card> new_cards) {
  cards = std::move(new_cards);
}

// Other Methods

std::vector<Card> Pile::Targets() const {
  switch (type) {
    case kFoundation: {
      if (!cards.empty()) {
        return {cards.back()};
      } else {
        // Empty foundation card with the same suit as the pile
        return {Card(false, suit, kRankNone, kFoundation)};
      }
    }
    case kTableau: {
      if (!cards.empty()) {
        auto back_card = cards.back();
        if (!back_card.GetHidden()) {
          return {cards.back()};
        } else {
          return {};
        }
      } else {
        // Empty tableau card (no rank or suit)
        return {Card(false, kSuitNone, kRankNone, kTableau)};
      }
    }
    default: {
      SpielFatalError("Pile::Targets() called with unsupported type");
    }
  }
}

std::vector<Card> Pile::Sources() const {
  std::vector<Card> sources;
  // A pile can have a maximum of 13 cards as sources (1 for each rank)
  sources.reserve(kNumRanks);
  switch (type) {
    case kFoundation: {
      if (!cards.empty()) {
        return {cards.back()};
      } else {
        return {};
      }
    }
    case kTableau: {
      if (!cards.empty()) {
        for (const auto &card : cards) {
          if (!card.GetHidden()) {
            sources.push_back(card);
          }
        }
        return sources;
      } else {
        return {};
      }
    }
    case kWaste: {
      if (!cards.empty()) {
        int i = 0;
        for (const auto &card : cards) {
          if (!card.GetHidden()) {
            if (i % 3 == 0) {
              sources.push_back(card);
            }
            ++i;
          } else {
            break;
          }
        }
        return sources;
      } else {
        return {};
      }
    }
    default: {
      SpielFatalError("Pile::Sources() called with unsupported type");
    }
  }
}

std::vector<Card> Pile::Split(Card card) {
  std::vector<Card> split_cards;
  switch (type) {
    case kFoundation: {
      if (cards.back() == card) {
        split_cards = {cards.back()};
        cards.pop_back();
      }
      break;
    }
    case kTableau: {
      if (!cards.empty()) {
        bool split_flag = false;
        for (auto it = cards.begin(); it != cards.end();) {
          if (*it == card) {
            split_flag = true;
          }
          if (split_flag) {
            split_cards.push_back(*it);
            it = cards.erase(it);
          } else {
            ++it;
          }
        }
      }
      break;
    }
    case kWaste: {
      if (!cards.empty()) {
        for (auto it = cards.begin(); it != cards.end();) {
          if (*it == card) {
            split_cards.push_back(*it);
            it = cards.erase(it);
            break;
          } else {
            ++it;
          }
        }
      }
      break;
    }
    default: {
      return {};
    }
  }
  return split_cards;
}

void Pile::Reveal(Card card_to_reveal) {
  SpielFatalError("Pile::Reveal() is not implemented.");
}

void Pile::Extend(std::vector<Card> source_cards) {
  for (auto &card : source_cards) {
    card.SetLocation(type);
    cards.push_back(card);
  }
}

std::string Pile::ToString(bool colored) const {
  std::string result;
  for (const auto &card : cards) {
    absl::StrAppend(&result, card.ToString(colored), " ");
  }
  return result;
}

// Tableau Methods =============================================================

Tableau::Tableau(PileID id) : Pile(kTableau, id, kSuitNone) {}

std::vector<Card> Tableau::Targets() const {
  if (!cards.empty()) {
    auto back_card = cards.back();
    if (!back_card.GetHidden()) {
      return {cards.back()};
    } else {
      return {};
    }
  } else {
    // Empty tableau card (no rank or suit)
    return {Card(false, kSuitNone, kRankNone, kTableau)};
  }
}

std::vector<Card> Tableau::Sources() const {
  std::vector<Card> sources;
  sources.reserve(kMaxSourcesTableau);
  if (!cards.empty()) {
    for (const auto &card : cards) {
      if (!card.GetHidden()) {
        sources.push_back(card);
      }
    }
    return sources;
  } else {
    return {};
  }
}

std::vector<Card> Tableau::Split(Card card) {
  std::vector<Card> split_cards;
  if (!cards.empty()) {
    bool split_flag = false;
    for (auto it = cards.begin(); it != cards.end();) {
      if (*it == card) {
        split_flag = true;
      }
      if (split_flag) {
        split_cards.push_back(*it);
        it = cards.erase(it);
      } else {
        ++it;
      }
    }
  }
  return split_cards;
}

void Tableau::Reveal(Card card_to_reveal) {
  cards.back().SetRank(card_to_reveal.GetRank());
  cards.back().SetSuit(card_to_reveal.GetSuit());
  cards.back().SetHidden(false);
}

// Foundation Methods ==========================================================

Foundation::Foundation(PileID id, SuitType suit)
    : Pile(kFoundation, id, suit) {}

std::vector<Card> Foundation::Targets() const {
  if (!cards.empty()) {
    return {cards.back()};
  } else {
    // Empty foundation card with the same suit as the pile
    return {Card(false, suit, kRankNone, kFoundation)};
  }
}

std::vector<Card> Foundation::Sources() const {
  std::vector<Card> sources;
  sources.reserve(kMaxSourcesFoundation);
  if (!cards.empty()) {
    return {cards.back()};
  } else {
    return {};
  }
}

std::vector<Card> Foundation::Split(Card card) {
  std::vector<Card> split_cards;
  if (cards.back() == card) {
    split_cards = {cards.back()};
    cards.pop_back();
  }
  return split_cards;
}

// Waste Methods ===============================================================

Waste::Waste() : Pile(kWaste, kPileWaste, kSuitNone) {}

std::vector<Card> Waste::Targets() const { return {}; }

std::vector<Card> Waste::Sources() const {
  std::vector<Card> sources;
  sources.reserve(kMaxSourcesWaste);
  if (!cards.empty()) {
    int i = 0;
    for (const auto &card : cards) {
      if (!card.GetHidden()) {
        // Every 3rd card in the waste can be moved
        if (i % 3 == 0) {
          sources.push_back(card);
        }
        ++i;
      } else {
        break;
      }
    }
    return sources;
  } else {
    return {};
  }
}

std::vector<Card> Waste::Split(Card card) {
  std::vector<Card> split_cards;
  if (!cards.empty()) {
    for (auto it = cards.begin(); it != cards.end();) {
      if (*it == card) {
        split_cards.push_back(*it);
        it = cards.erase(it);
        break;
      } else {
        ++it;
      }
    }
  }
  return split_cards;
}

void Waste::Reveal(Card card_to_reveal) {
  for (auto &card : cards) {
    if (card.GetHidden()) {
      card.SetRank(card_to_reveal.GetRank());
      card.SetSuit(card_to_reveal.GetSuit());
      card.SetHidden(false);
      break;
    }
  }
}

// Move Methods ================================================================

Move::Move(Card target_card, Card source_card) {
  target = target_card;
  source = source_card;
}

Move::Move(RankType target_rank, SuitType target_suit, RankType source_rank,
           SuitType source_suit) {
  target = Card(false, target_suit, target_rank, kMissing);
  source = Card(false, source_suit, source_rank, kMissing);
}

Move::Move(Action action) {
  // `base` refers to the starting point that indices start from (e.g. if it's
  // 7, and there's 3 cards in its group, their action ids will be 8, 9, 10).
  // `residual` is just the difference between the id and the base.

  int residual;
  int target_rank;
  int source_rank;
  int target_suit;
  int source_suit;

  std::vector<SuitType> opposite_suits;
  action -= kActionOffset;

  // The numbers used in the cases below are just used to divide action ids into
  // groups (e.g. 1-132 are regular moves, 133-136 are the action ids of moves
  // that move an ace to an empty foundation, etc.)

  switch (action) {
    case 1 ... 132: {
      // Handles ordinary moves
      target_rank = ((action - 1) / 3) % 11 + 2;
      target_suit = ((action - 1) / 33) + 1;
      residual = ((action - 1) % 3);
      if (residual == 0) {
        source_rank = target_rank + 1;
        source_suit = target_suit;
      } else {
        opposite_suits = GetOppositeSuits(static_cast<SuitType>(target_suit));
        source_rank = target_rank - 1;
        source_suit = opposite_suits[residual - 1];
      }
      break;
    }
    case 133 ... 136: {
      // Handles ace to empty foundation moves
      target_rank = 0;
      target_suit = action - 132;
      source_rank = 1;
      source_suit = target_suit;
      break;
    }
    case 137 ... 140: {
      // Handles king to empty tableau moves
      target_rank = 0;
      target_suit = 0;
      source_rank = 13;
      source_suit = action - 136;
      break;
    }
    case 141 ... 144: {
      // Handles moves with ace targets
      target_rank = 1;
      target_suit = action - 140;
      source_rank = 2;
      source_suit = target_suit;
      break;
    }
    case 145 ... 152: {
      // Handles moves with king targets
      target_rank = 13;
      target_suit = (action - 143) / 2;

      residual = (action - 143) % 2;
      opposite_suits = GetOppositeSuits(static_cast<SuitType>(target_suit));

      source_rank = 12;
      source_suit = opposite_suits[residual];
      break;
    }
    default: {
      SpielFatalError("action provided does not correspond with a move");
    }
  }

  target = Card(false, static_cast<SuitType>(target_suit),
                static_cast<RankType>(target_rank));
  source = Card(false, static_cast<SuitType>(source_suit),
                static_cast<RankType>(source_rank));
}

// Getters

Card Move::GetTarget() const { return target; }

Card Move::GetSource() const { return source; }

// Other Methods

Action Move::ActionId() const {
  RankType target_rank = target.GetRank();
  RankType source_rank = source.GetRank();
  SuitType target_suit = target.GetSuit();
  SuitType source_suit = source.GetSuit();

  int base;
  int residual;

  // `base` refers to the starting point that indices start from (e.g. if it's
  // 7, and there's 3 cards in its group, their action ids will be 8, 9, 10).
  // `residual` is just the difference between the id and the base.

  switch (target_rank) {
    case kRankNone: {
      switch (source_rank) {
        case kRankA: {
          base = 132;
          break;
        }
        case kRankK: {
          base = 136;
          break;
        }
        default: {
          base = -999;
          break;
          // SpielFatalError("source.rank has an incorrect value");
        }
      }
      return base + source_suit + kActionOffset;
    }
    case kRankA: {
      base = 140;
      return base + source_suit + kActionOffset;
    }
    case kRankK: {
      base = 144;
      if (source_suit <= 2) {
        residual = -1;
      } else {
        residual = 0;
      }
      return base + (2 * target_suit) + residual + kActionOffset;
    }
    default: {
      base = (target_suit - 1) * 33 + (target_rank - 2) * 3;
      if (target_suit == source_suit) {
        residual = 1;
      } else if (source_suit <= 2) {
        residual = 2;
      } else {
        residual = 3;
      }
      return base + residual + kActionOffset;
    }
  }
}

std::string Move::ToString(bool colored) const {
  std::string result;
  absl::StrAppend(&result, target.ToString(colored), " ", kGlyphArrow, " ",
                  source.ToString(colored));
  return result;
}

bool Move::operator<(const Move &other_move) const {
  int index = target.GetIndex() * 100 + source.GetIndex();
  int other_index =
      other_move.target.GetIndex() * 100 + other_move.source.GetIndex();
  return index < other_index;
}

// SolitaireState Methods ======================================================

SolitaireState::SolitaireState(std::shared_ptr<const Game> game)
    : State(game), waste() {
  // Extract parameters from `game`
  auto parameters = game->GetParameters();
  is_colored = parameters.at("is_colored").bool_value();
  depth_limit = parameters.at("depth_limit").int_value();

  // Create foundations
  for (const auto &suit : kSuits) {
    foundations.emplace_back(kSuitToPile.at(suit), suit);
  }

  // Create tableaus
  for (int i = 1; i <= 7; i++) {
    // Create `i` hidden cards
    std::vector<Card> cards_to_add;
    for (int j = 1; j <= i; j++) {
      cards_to_add.emplace_back(true, kSuitHidden, kRankHidden, kTableau);
    }

    // Create a new tableau and add cards
    auto tableau = Tableau(kIntToPile.at(i));
    tableau.SetCards(cards_to_add);

    // Add resulting tableau to tableaus
    tableaus.push_back(tableau);
  }

  // Create waste
  for (int i = 1; i <= 24; i++) {
    auto new_card = Card(true, kSuitHidden, kRankHidden, kWaste);
    waste.Extend({new_card});
  }
}

Player SolitaireState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else if (IsChanceNode()) {
    return kChancePlayerId;
  } else {
    return kPlayerId;
  }
}

std::unique_ptr<State> SolitaireState::Clone() const {
  return std::unique_ptr<State>(new SolitaireState(*this));
}

bool SolitaireState::IsTerminal() const { return is_finished; }

bool SolitaireState::IsChanceNode() const {
  for (const auto &tableau : tableaus) {
    if (!tableau.GetIsEmpty() && tableau.GetLastCard().GetHidden()) {
      return true;
    }
  }

  if (!waste.GetIsEmpty()) {
    for (const auto &card : waste.GetCards()) {
      if (card.GetHidden()) {
        return true;
      }
    }
  }

  return false;
}

std::string SolitaireState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "WASTE       : ", waste.ToString(is_colored));

  absl::StrAppend(&result, "\nFOUNDATIONS : ");
  for (const auto &foundation : foundations) {
    absl::StrAppend(&result, foundation.Targets()[0].ToString(is_colored), " ");
  }

  absl::StrAppend(&result, "\nTABLEAUS    : ");
  for (const auto &tableau : tableaus) {
    if (!tableau.GetIsEmpty()) {
      absl::StrAppend(&result, "\n", tableau.ToString(is_colored));
    }
  }

  absl::StrAppend(&result, "\nTARGETS : ");
  for (const auto &card : Targets()) {
    absl::StrAppend(&result, card.ToString(is_colored), " ");
  }

  absl::StrAppend(&result, "\nSOURCES : ");
  for (const auto &card : Sources()) {
    absl::StrAppend(&result, card.ToString(is_colored), " ");
  }

  return result;
}

std::string SolitaireState::ActionToString(Player player,
                                           Action action_id) const {
  switch (action_id) {
    case kEnd: {
      return "kEnd";
    }
    case kRevealStart ... kRevealEnd: {
      auto revealed_card = Card(static_cast<int>(action_id));
      std::string result;
      absl::StrAppend(&result, "kReveal", revealed_card.ToString(is_colored));
      return result;
    }
    case kMoveStart ... kMoveEnd: {
      auto move = Move(action_id);
      return move.ToString(is_colored);
    }
    default: {
      return "Missing Action";
    }
  }
}

std::string SolitaireState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string SolitaireState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void SolitaireState::ObservationTensor(Player player,
                                       std::vector<double> *values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  values->resize(game_->ObservationTensorSize());
  std::fill(values->begin(), values->end(), 0.0);
  auto ptr = values->begin();

  for (const auto &foundation : foundations) {
    if (foundation.GetIsEmpty()) {
      ptr[0] = 1;
    } else {
      auto last_rank = foundation.GetLastCard().GetRank();
      if (last_rank >= kRankA && last_rank <= kRankK) {
        ptr[last_rank] = 1;
      }
    }
    ptr += kFoundationTensorLength;
  }

  for (const auto &tableau : tableaus) {
    if (tableau.GetIsEmpty()) {
      ptr[7] = 1.0;
      continue;
    } else {
      int num_hidden_cards = 0;
      for (const auto &card : tableau.GetCards()) {
        if (card.GetHidden() && num_hidden_cards <= kMaxHiddenCard) {
          ptr[num_hidden_cards] = 1.0;
          ++num_hidden_cards;
        } else {
          auto tensor_index = card.GetIndex() + kMaxHiddenCard;
          ptr[tensor_index] = 1.0;
        }
      }
    }
    ptr += kTableauTensorLength;
  }

  if (waste.GetIsEmpty()) {
    return;
  } else {
    for (auto &card : waste.GetCards()) {
      if (card.GetHidden()) {
        ptr[0] = 1.0;
      } else {
        auto tensor_index = card.GetIndex();
        ptr[tensor_index] = 1.0;
      }
      ptr += kWasteTensorLength;
    }
  }
}

void SolitaireState::DoApplyAction(Action action) {
  switch (action) {
    case kEnd: {
      is_finished = true;
      current_rewards = 0;
      break;
    }
    case kRevealStart ... kRevealEnd: {
      auto revealed_card = Card(static_cast<int>(action));
      bool found_card = false;

      for (auto &tableau : tableaus) {
        if (!tableau.GetIsEmpty() && tableau.GetLastCard().GetHidden()) {
          tableau.Reveal(revealed_card);
          card_map.insert_or_assign(tableau.GetLastCard(), tableau.GetID());
          found_card = true;
          break;
        }
      }
      if (!found_card && !waste.GetIsEmpty()) {
        waste.Reveal(revealed_card);
        card_map.insert_or_assign(revealed_card, waste.GetID());
      }
      revealed_cards.push_back(action);
      break;
    }
    case kMoveStart ... kMoveEnd: {
      Move selected_move = Move(action);
      is_reversible = IsReversible(selected_move.GetSource(),
                                   GetPile(selected_move.GetSource()));

      if (is_reversible) {
        std::string current_observation = ObservationString(0);
        previous_states.insert(hasher(current_observation));
      } else {
        previous_states.clear();
      }

      MoveCards(selected_move);
      current_returns += current_rewards;
      break;
    }
    default: {
    }
  }

  ++current_depth;
  if (current_depth >= depth_limit) {
    is_finished = true;
  }
}

std::vector<double> SolitaireState::Returns() const {
  // Returns the sum of rewards up to and including the most recent state
  // transition.
  return {current_returns};
}

std::vector<double> SolitaireState::Rewards() const {
  // Should be the reward for the action that created this state, not the action
  // applied to this state
  return {current_rewards};
}

std::vector<Action> SolitaireState::LegalActions() const {
  if (IsTerminal()) {
    return {};
  } else if (IsChanceNode()) {
    return LegalChanceOutcomes();
  } else {
    std::vector<Action> legal_actions;

    if (is_reversible) {
      // If the state is reversible, we need to check each move to see if it is
      // too.
      for (const auto &move : CandidateMoves()) {
        if (IsReversible(move.GetSource(), GetPile(move.GetSource()))) {
          auto action_id = move.ActionId();
          auto child = Child(action_id);

          if (child->CurrentPlayer() == kChancePlayerId) {
            legal_actions.push_back(action_id);
          } else {
            auto child_hash = hasher(child->ObservationString());
            if (previous_states.count(child_hash) == 0) {
              legal_actions.push_back(action_id);
            }
          }
        } else {
          legal_actions.push_back(move.ActionId());
        }
      }
    } else {
      // If the state isn't reversible, all candidate moves are legal
      for (const auto &move : CandidateMoves()) {
        legal_actions.push_back(move.ActionId());
      }
    }

    if (!legal_actions.empty()) {
      std::sort(legal_actions.begin(), legal_actions.end());
    } else {
      legal_actions.push_back(kEnd);
    }

    return legal_actions;
  }
}

std::vector<std::pair<Action, double>> SolitaireState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  const double p = 1.0 / (52 - revealed_cards.size());

  for (int i = 1; i <= 52; i++) {
    if (std::find(revealed_cards.begin(), revealed_cards.end(), i) ==
        revealed_cards.end()) {
      outcomes.emplace_back(i, p);
    }
  }

  return outcomes;
}

// Other Methods

std::vector<Card> SolitaireState::Targets(
    const absl::optional<LocationType> &location) const {
  LocationType loc = location.value_or(kMissing);
  std::vector<Card> targets;

  if (loc == kTableau || loc == kMissing) {
    for (const auto &tableau : tableaus) {
      std::vector<Card> current_targets = tableau.Targets();
      targets.insert(targets.end(), current_targets.begin(),
                     current_targets.end());
    }
  }

  if (loc == kFoundation || loc == kMissing) {
    for (const auto &foundation : foundations) {
      std::vector<Card> current_targets = foundation.Targets();
      targets.insert(targets.end(), current_targets.begin(),
                     current_targets.end());
    }
  }

  return targets;
}

std::vector<Card> SolitaireState::Sources(
    const absl::optional<LocationType> &location) const {
  LocationType loc = location.value_or(kMissing);
  std::vector<Card> sources;

  if (loc == kTableau || loc == kMissing) {
    for (const auto &tableau : tableaus) {
      std::vector<Card> current_sources = tableau.Sources();
      sources.insert(sources.end(), current_sources.begin(),
                     current_sources.end());
    }
  }

  if (loc == kFoundation || loc == kMissing) {
    for (const auto &foundation : foundations) {
      std::vector<Card> current_sources = foundation.Sources();
      sources.insert(sources.end(), current_sources.begin(),
                     current_sources.end());
    }
  }

  if (loc == kWaste || loc == kMissing) {
    std::vector<Card> current_sources = waste.Sources();
    sources.insert(sources.end(), current_sources.begin(),
                   current_sources.end());
  }

  return sources;
}

Pile *SolitaireState::GetPile(const Card &card) const {
  PileID pile_id = kPileMissing;

  if (card.GetRank() == kRankNone) {
    if (card.GetSuit() == kSuitNone) {
      for (const auto &tableau : tableaus) {
        if (tableau.GetIsEmpty()) {
          return (Pile *)&tableau;
        }
      }
    } else if (card.GetSuit() != kSuitHidden) {
      for (const auto &foundation : foundations) {
        if (foundation.GetSuit() == card.GetSuit()) {
          return (Pile *)&foundation;
        }
      }
    } else {
      SpielFatalError("The pile containing the card wasn't found");
    }
  } else {
    pile_id = card_map.at(card);
  }

  switch (pile_id) {
    case kPileWaste: {
      return (Pile *)&waste;
    }
    case kPileSpades ... kPileDiamonds: {
      return (Pile *)&foundations.at(pile_id - 1);
    }
    case kPile1stTableau ... kPile7thTableau: {
      return (Pile *)&tableaus.at(pile_id - 5);
    }
    default: {
      SpielFatalError("The pile containing the card wasn't found");
    }
  }
}

std::vector<Move> SolitaireState::CandidateMoves() const {
  std::vector<Move> candidate_moves;
  std::vector<Card> targets = Targets();
  std::vector<Card> sources = Sources();
  bool found_empty_tableau = false;

  for (auto &target : targets) {
    if (target.GetSuit() == kSuitNone && target.GetRank() == kRankNone) {
      if (found_empty_tableau) {
        continue;
      } else {
        found_empty_tableau = true;
      }
    }
    for (auto &source : target.LegalChildren()) {
      if (std::find(sources.begin(), sources.end(), source) != sources.end()) {
        auto *source_pile = GetPile(source);
        if (target.GetLocation() == kFoundation &&
            source_pile->GetType() == kTableau) {
          if (source_pile->GetLastCard() == source) {
            candidate_moves.emplace_back(target, source);
          }
        } else if (source.GetRank() == kRankK &&
                   target.GetSuit() == kSuitNone &&
                   target.GetRank() == kRankNone) {
          // Check is source is not a bottom
          if (source_pile->GetType() == kWaste ||
              (source_pile->GetType() == kTableau &&
               !(source_pile->GetFirstCard() == source))) {
            candidate_moves.emplace_back(target, source);
          }
        } else {
          candidate_moves.emplace_back(target, source);
        }
      } else {
        continue;
      }
    }
  }

  return candidate_moves;
}

void SolitaireState::MoveCards(const Move &move) {
  Card target = move.GetTarget();
  Card source = move.GetSource();

  auto *target_pile = GetPile(target);
  auto *source_pile = GetPile(source);

  std::vector<Card> split_cards = source_pile->Split(source);
  for (auto &card : split_cards) {
    card_map.insert_or_assign(card, target_pile->GetID());
    target_pile->Extend({card});
  }

  // Calculate rewards/returns for this move in the current state
  double move_reward = 0.0;

  // Reward for moving a card to or from a foundation
  if (target_pile->GetType() == kFoundation) {
    // Adds points for moving TO a foundation
    move_reward += kFoundationPoints.at(source.GetRank());
  } else if (source_pile->GetType() == kFoundation) {
    // Subtracts points for moving AWAY from a foundation
    move_reward -= kFoundationPoints.at(source.GetRank());
  }

  // Reward for revealing a hidden card
  if (source_pile->GetType() == kTableau && !source_pile->GetIsEmpty() &&
      source_pile->GetLastCard().GetHidden()) {
    move_reward += 20.0;
  }

  // Reward for moving a card from the waste
  if (source_pile->GetType() == kWaste) {
    move_reward += 20.0;
  }

  // Add current rewards to current returns
  current_rewards = move_reward;
}

bool SolitaireState::IsReversible(const Card &source, Pile *source_pile) const {
  switch (source.GetLocation()) {
    case kWaste: {
      return false;
    }
    case kFoundation: {
      return true;
    }
    case kTableau: {
      // Move is irreversible if its source is a bottom card or over a hidden
      // card. Basically if it's the first non-hidden card in the pile/tableau.
      auto it = std::find_if(source_pile->GetCards().begin(),
                             source_pile->GetCards().end(),
                             [](const Card &card) { return card.GetHidden(); });

      return !(*it == source);
    }
    default: {
      // Returns false if the source card is not in the waste, foundations, or
      // tableaus
      return false;
    }
  }
}

// SolitaireGame Methods =======================================================

SolitaireGame::SolitaireGame(const GameParameters &params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      depth_limit_(ParameterValue<int>("depth_limit")),
      is_colored_(ParameterValue<bool>("is_colored")) {}

int SolitaireGame::NumDistinctActions() const {
  /* 52 kReveal Moves (one for each ordinary card)
   * 52 Foundation Moves (one for every ordinary card)
   * 96 Tableau Moves (two for every ordinary card except aces)
   *  4 King to Empty Tableau (one for every king)
   *  1 End Game Move */
  return 205;
}

int SolitaireGame::MaxGameLength() const { return depth_limit_; }

int SolitaireGame::NumPlayers() const { return 1; }

double SolitaireGame::MinUtility() const {
  /* Returns start at zero and the only negative rewards come from undoing an
   * action. Undoing an action just takes away the reward that was gained from
   * the action, so utility can never go below 0. */
  return 0.0;
}

double SolitaireGame::MaxUtility() const {
  /* Waste (24 * 20 = 480)
      24 cards are in the waste initially. 20 points are rewarded for every one
     that is moved from the waste. Tableau (21 * 20 = 420) 21 cards are hidden
     in the tableaus initially. 20 points are rewarded for every one that is
     revealed. Foundation (4 * (100 + 90 + 80 + 70 + 60 + 50 + 40 + 30 + 20 + 10
     + 10 + 10 + 10) = 4 * 580 = 2,320) 0 cards are in the foundations
     initially. A varying number of points, based on the cards rank, are awarded
       when the card is moved to the foundation. Each complete suit in the
     foundation is worth 580 points. `kFoundationPoints` in `solitaire.h`
     outlines how much each rank is worth. */
  return 3220.0;
}

std::vector<int> SolitaireGame::ObservationTensorShape() const {
  /* Waste (24 * 53 = 1,272)
       24 locations and each location is a 53 element vector (52 normal cards +
    1 hidden) Tableau (7 * 59 = 413) Each tableau is represented as a 59 element
    vector (6 hidden cards + 1 empty tableau + 52 normal cards) Foundation (4 *
    14 = 56) Each foundation is represented as a 14 element vector (13 ranks + 1
    empty foundation) Total Length = 1,272 + 413 + 56 = 1,741 */
  return {1741};
}

std::unique_ptr<State> SolitaireGame::NewInitialState() const {
  return std::unique_ptr<State>(new SolitaireState(shared_from_this()));
}

std::shared_ptr<const Game> SolitaireGame::Clone() const {
  return std::shared_ptr<const Game>(new SolitaireGame(*this));
}

}  // namespace open_spiel::solitaire
