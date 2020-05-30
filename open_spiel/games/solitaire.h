#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_SOLITAIRE_H
#define THIRD_PARTY_OPEN_SPIEL_GAMES_SOLITAIRE_H

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <variant>
#include <any>
#include <unordered_map>
#include <set>
#include <optional>
#include <tuple>

#include "open_spiel/spiel.h"

// TODO: Please run `clang-format -style=google` on your code. There are also some `clang-tidy` issues.

// An implementation of klondike solitaire: https://en.wikipedia.org/wiki/Klondike_(solitaire)
// More specifically, it is K+ solitaire, which allows the player to play any card from the deck/waste that would
// normally become playable after some number of draws in standard klondike solitaire. For a more in-depth
// description of K+ solitaire, see http://web.engr.oregonstate.edu/~afern/papers/solitaire.pdf. This implementation
// also gives rewards at intermediate states like most electronic versions of solitaire do, rather than only at
// terminal states.

// ANSI color codes
inline constexpr const char* kReset = "\033[0m";
inline constexpr const char* kRed   = "\033[31m";
inline constexpr const char* kBlack = "\033[37m";

// Glyphs & Strings
inline constexpr const char* kGlyphHidden   = "\U0001F0A0";
inline constexpr const char* kGlyphEmpty    = "\U0001F0BF";
inline constexpr const char* kGlyphSpades   = "\U00002660";
inline constexpr const char* kGlyphHearts   = "\U00002665";
inline constexpr const char* kGlyphClubs    = "\U00002663";
inline constexpr const char* kGlyphDiamonds = "\U00002666";
inline constexpr const char* kGlyphArrow    = "\U00002190";

namespace open_spiel::solitaire {

    // Default Game Parameters =========================================================================================

    inline constexpr int    kDefaultPlayers      = 1;
    inline constexpr int    kDefaultDepthLimit   = 150;
    inline constexpr bool   kDefaultIsColored    = true;

    // Enumerations ====================================================================================================

    enum SuitType     {
        kSuitNone = 0,
        kSuitSpades,
        kSuitHearts,
        kSuitClubs,
        kSuitDiamonds,
        kSuitHidden,
    };
    enum RankType     {
        kRankNone = 0,
        kRankA,
        kRank2,
        kRank3,
        kRank4,
        kRank5,
        kRank6,
        kRank7,
        kRank8,
        kRank9,
        kRankT,
        kRankJ,
        kRankQ,
        kRankK,
        kRankHidden,
    };
    enum LocationType {
        kDeck       = 0,
        kWaste      = 1,
        kFoundation = 2,
        kTableau    = 3,
        kMissing    = 4,
    };
    enum PileID       {
        kPileWaste      = 0,
        kPileSpades     = 1,
        kPileHearts     = 2,
        kPileClubs      = 3,
        kPileDiamonds   = 4,
        kPile1stTableau = 5,
        kPile2ndTableau = 6,
        kPile3rdTableau = 7,
        kPile4thTableau = 8,
        kPile5thTableau = 9,
        kPile6thTableau = 10,
        kPile7thTableau = 11,
        kNoPile         = 12
    };

    // TODO: This is hard to read and check. Is it really necessary or could you just define an int representation of actions and a function to map from the int to a set of fields?

    enum ActionType   {

        // End Action (1) =============================================================================================
        kEnd = 0,

        // Reveal Actions (52) =========================================================================================
        // Spades ------------------------------------------------------------------------------------------------------
        kRevealAs = 1,
        kReveal2s = 2,
        kReveal3s = 3,
        kReveal4s = 4,
        kReveal5s = 5,
        kReveal6s = 6,
        kReveal7s = 7,
        kReveal8s = 8,
        kReveal9s = 9,
        kRevealTs = 10,
        kRevealJs = 11,
        kRevealQs = 12,
        kRevealKs = 13,

        // Hearts ------------------------------------------------------------------------------------------------------
        kRevealAh = 14,
        kReveal2h = 15,
        kReveal3h = 16,
        kReveal4h = 17,
        kReveal5h = 18,
        kReveal6h = 19,
        kReveal7h = 20,
        kReveal8h = 21,
        kReveal9h = 22,
        kRevealTh = 23,
        kRevealJh = 24,
        kRevealQh = 25,
        kRevealKh = 26,

        // Clubs -------------------------------------------------------------------------------------------------------
        kRevealAc = 27,
        kReveal2c = 28,
        kReveal3c = 29,
        kReveal4c = 30,
        kReveal5c = 31,
        kReveal6c = 32,
        kReveal7c = 33,
        kReveal8c = 34,
        kReveal9c = 35,
        kRevealTc = 36,
        kRevealJc = 37,
        kRevealQc = 38,
        kRevealKc = 39,

        // Diamonds ----------------------------------------------------------------------------------------------------
        kRevealAd = 40,
        kReveal2d = 41,
        kReveal3d = 42,
        kReveal4d = 43,
        kReveal5d = 44,
        kReveal6d = 45,
        kReveal7d = 46,
        kReveal8d = 47,
        kReveal9d = 48,
        kRevealTd = 49,
        kRevealJd = 50,
        kRevealQd = 51,
        kRevealKd = 52,

        // Special Moves (8) ===========================================================================================
        // To Empty Tableau --------------------------------------------------------------------------------------------
        kMove__Ks,
        kMove__Kh,
        kMove__Kc,
        kMove__Kd,

        // To Empty Foundation -----------------------------------------------------------------------------------------
        kMove__Ah,
        kMove__As,
        kMove__Ac,
        kMove__Ad,

        // Foundation Moves (48) =======================================================================================
        // To Spades ---------------------------------------------------------------------------------------------------
        kMoveAs2s,
        kMove2s3s,
        kMove3s4s,
        kMove4s5s,
        kMove5s6s,
        kMove6s7s,
        kMove7s8s,
        kMove8s9s,
        kMove9sTs,
        kMoveTsJs,
        kMoveJsQs,
        kMoveQsKs,

        // To Hearts ---------------------------------------------------------------------------------------------------
        kMoveAh2h,
        kMove2h3h,
        kMove3h4h,
        kMove4h5h,
        kMove5h6h,
        kMove6h7h,
        kMove7h8h,
        kMove8h9h,
        kMove9hTh,
        kMoveThJh,
        kMoveJhQh,
        kMoveQhKh,

        // To Clubs ----------------------------------------------------------------------------------------------------
        kMoveAc2c,
        kMove2c3c,
        kMove3c4c,
        kMove4c5c,
        kMove5c6c,
        kMove6c7c,
        kMove7c8c,
        kMove8c9c,
        kMove9cTc,
        kMoveTcJc,
        kMoveJcQc,
        kMoveQcKc,

        // To Diamonds -------------------------------------------------------------------------------------------------
        kMoveAd2d,
        kMove2d3d,
        kMove3d4d,
        kMove4d5d,
        kMove5d6d,
        kMove6d7d,
        kMove7d8d,
        kMove8d9d,
        kMove9dTd,
        kMoveTdJd,
        kMoveJdQd,
        kMoveQdKd,

        // Tableau Moves (96) ==========================================================================================
        // To Spades ---------------------------------------------------------------------------------------------------
        kMove2sAh,
        kMove3s2h,
        kMove4s3h,
        kMove5s4h,
        kMove6s5h,
        kMove7s6h,
        kMove8s7h,
        kMove9s8h,
        kMoveTs9h,
        kMoveJsTh,
        kMoveQsJh,
        kMoveKsQh,
        kMove2sAd,
        kMove3s2d,
        kMove4s3d,
        kMove5s4d,
        kMove6s5d,
        kMove7s6d,
        kMove8s7d,
        kMove9s8d,
        kMoveTs9d,
        kMoveJsTd,
        kMoveQsJd,
        kMoveKsQd,

        // To Hearts ---------------------------------------------------------------------------------------------------
        kMove2hAs,
        kMove3h2s,
        kMove4h3s,
        kMove5h4s,
        kMove6h5s,
        kMove7h6s,
        kMove8h7s,
        kMove9h8s,
        kMoveTh9s,
        kMoveJhTs,
        kMoveQhJs,
        kMoveKhQs,
        kMove2hAc,
        kMove3h2c,
        kMove4h3c,
        kMove5h4c,
        kMove6h5c,
        kMove7h6c,
        kMove8h7c,
        kMove9h8c,
        kMoveTh9c,
        kMoveJhTc,
        kMoveQhJc,
        kMoveKhQc,

        // To Clubs ----------------------------------------------------------------------------------------------------
        kMove2cAh,
        kMove3c2h,
        kMove4c3h,
        kMove5c4h,
        kMove6c5h,
        kMove7c6h,
        kMove8c7h,
        kMove9c8h,
        kMoveTc9h,
        kMoveJcTh,
        kMoveQcJh,
        kMoveKcQh,
        kMove2cAd,
        kMove3c2d,
        kMove4c3d,
        kMove5c4d,
        kMove6c5d,
        kMove7c6d,
        kMove8c7d,
        kMove9c8d,
        kMoveTc9d,
        kMoveJcTd,
        kMoveQcJd,
        kMoveKcQd,

        // To Diamonds -------------------------------------------------------------------------------------------------
        kMove2dAs,
        kMove3d2s,
        kMove4d3s,
        kMove5d4s,
        kMove6d5s,
        kMove7d6s,
        kMove8d7s,
        kMove9d8s,
        kMoveTd9s,
        kMoveJdTs,
        kMoveQdJs,
        kMoveKdQs,
        kMove2dAc,
        kMove3d2c,
        kMove4d3c,
        kMove5d4c,
        kMove6d5c,
        kMove7d6c,
        kMove8d7c,
        kMove9d8c,
        kMoveTd9c,
        kMoveJdTc,
        kMoveQdJc,
        kMoveKdQc,
    };

    // Constants =======================================================================================================

    // Indices for special cards
    inline constexpr int kHiddenCard       = 99;
    inline constexpr int kNoCard           =  0;
    inline constexpr int kEmptyTableauCard = -1;
    inline constexpr int kEmptySpadeCard   = -2;
    inline constexpr int kEmptyHeartCard   = -3;
    inline constexpr int kEmptyClubCard    = -4;
    inline constexpr int kEmptyDiamondCard = -5;

    // 1 empty + 13 ranks
    inline constexpr int kFoundationTensorLength = 14;

    // 6 hidden cards + 1 empty tableau + 52 ordinary cards
    inline constexpr int kTableauTensorLength = 59;

    // 1 hidden card + 52 ordinary cards
    inline constexpr int kWasteTensorLength = 53;

    // Constant for how many hidden cards can show up in a tableau
    inline constexpr int kMaxHiddenCard = 6;

    // Only used in one place and just for consistency (to match kChancePlayerId & kTerminalPlayerId)
    inline constexpr int kPlayerId = 0;

    // Type aliases
    using Ranksuit = std::pair<RankType, SuitType>;

    // Other lists and maps
    const std::vector<SuitType> kSuits = {kSuitSpades, kSuitHearts, kSuitClubs, kSuitDiamonds};
    const std::vector<RankType> kRanks = {kRankA, kRank2, kRank3, kRank4, kRank5, kRank6, kRank7, kRank8, kRank9, kRankT, kRankJ, kRankQ, kRankK};

    // These correspond with their enums, not with the two vectors directly above
    const std::vector<std::string> kSuitStrs = {"", kGlyphSpades, kGlyphHearts, kGlyphClubs, kGlyphDiamonds, ""};
    const std::vector<std::string> kRankStrs = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", ""};

    const std::map<RankType, double> kFoundationPoints = {
            //region Maps a RankType to a double that represents the reward for moving a card of that rank to the foundation
            {kRankA, 100.0},
            {kRank2, 90.0},
            {kRank3, 80.0},
            {kRank4, 70.0},
            {kRank5, 60.0},
            {kRank6, 50.0},
            {kRank7, 40.0},
            {kRank8, 30.0},
            {kRank9, 20.0},
            {kRankT, 10.0},
            {kRankJ, 10.0},
            {kRankQ, 10.0},
            {kRankK, 10.0}
            //endregion
    };

    const std::map<SuitType, PileID> kSuitToPile = {
            // region Maps a foundation suit to the ID of the foundation
            {kSuitSpades, kPileSpades},
            {kSuitHearts, kPileHearts},
            {kSuitClubs,  kPileClubs},
            {kSuitDiamonds, kPileDiamonds}
            // endregion
    };

    const std::map<int, PileID> kIntToPile = {
            // region Maps an integer to a tableau pile ID (used when initializing SolitaireState)
            {1, kPile1stTableau},
            {2, kPile2ndTableau},
            {3, kPile3rdTableau},
            {4, kPile4thTableau},
            {5, kPile5thTableau},
            {6, kPile6thTableau},
            {7, kPile7thTableau}
            // endregion
    };

    // Support Classes =================================================================================================

    class Card {
    private:
        RankType      rank     = kRankHidden;       // Indicates the rank of the card
        SuitType      suit     = kSuitHidden;       // Indicates the suit of the card
        LocationType  location = kMissing;          // Indicates the type of pile the card is in
        bool          hidden   = false;             // Indicates whether the card is hidden or not
        int           index    = kHiddenCard;       // Identifies the card with an integer

    public:
        // Constructors ================================================================================================
        Card(bool hidden = false, SuitType suit = kSuitHidden, RankType rank = kRankHidden, LocationType location = kMissing);
        Card(int index, bool hidden = false, LocationType location = kMissing);

        // Getters =====================================================================================================
        RankType GetRank() const;
        SuitType GetSuit() const;
        LocationType GetLocation() const;
        bool GetHidden() const;
        int GetIndex() const;

        // Setters =====================================================================================================
        void SetRank(RankType new_rank);
        void SetSuit(SuitType new_suit);
        void SetLocation(LocationType new_location);
        void SetHidden(bool new_hidden);

        // Operators ===================================================================================================
        bool operator==(Card & other_card) const;
        bool operator==(const Card & other_card) const;
        bool operator<(const Card & other_card) const;

        // Other Methods ===============================================================================================
        std::string ToString(bool colored = true) const;
        std::vector<Card> LegalChildren() const;

    };

    class Pile {

    protected:
        std::vector<Card> cards;
        const LocationType type;
        const SuitType     suit;
        const PileID       id;
        const int          max_size;

    public:

        // Constructor =================================================================================================
        Pile(LocationType type, PileID id, SuitType suit = kSuitNone);

        // Destructor ==================================================================================================
        virtual ~Pile();

        // Getters/Setters =============================================================================================
        bool              GetIsEmpty() const;
        SuitType          GetSuit() const;
        LocationType      GetType() const;
        PileID            GetID() const;
        Card              GetFirstCard() const;
        Card              GetLastCard() const;
        std::vector<Card> GetCards() const;
        void              SetCards(std::vector<Card> new_cards);

        // Other Methods ===============================================================================================
        virtual std::vector<Card> Sources() const;
        virtual std::vector<Card> Targets() const;
        virtual std::vector<Card> Split(Card card);
        virtual void              Reveal(Card card_to_reveal);
        void                      Extend(std::vector<Card> source_cards);
        std::string               ToString(bool colored = true) const;

    };

    class Tableau : public Pile {
    public:
        // Constructor =================================================================================================
        Tableau(PileID id);

        // Other Methods ===============================================================================================
        std::vector<Card> Sources() const override;
        std::vector<Card> Targets() const override;
        std::vector<Card> Split(Card card) override;
        void              Reveal(Card card_to_reveal) override;
    };

    class Foundation : public Pile {
    public:
        // Constructor =================================================================================================
        Foundation(PileID id, SuitType suit);

        // Other Methods ===============================================================================================
        std::vector<Card> Sources() const override;
        std::vector<Card> Targets() const override;
        std::vector<Card> Split(Card card) override;
    };

    class Waste : public Pile {
    public:
        // Constructor =================================================================================================
        Waste();

        // Other Methods ===============================================================================================
        std::vector<Card> Sources() const override;
        std::vector<Card> Targets() const override;
        std::vector<Card> Split(Card card) override;
        void              Reveal(Card card_to_reveal) override;
    };

    class Move {
    private:
        Card target;
        Card source;

    public:
        // Constructors ================================================================================================
        Move(Card target_card, Card source_card);
        Move(RankType target_rank, SuitType target_suit, RankType source_rank, SuitType source_suit);
        explicit Move(Action action);

        // Getters =====================================================================================================
        Card GetTarget() const;
        Card GetSource() const;

        // Other Methods ===============================================================================================
        std::string ToString(bool colored = true) const;
        bool operator<(const Move & other_move) const;
        Action ActionId() const;

    };

    // More Constants ==================================================================================================

    // TODO: Please compute this on-the-fly instead of having this long mapping
    // TODO: Make this a method of `Move`
    const std::map<Move, Action> kMoveToAction = {
            // region Mapping of a move to an action declared in ActionType;
            
            // region Moves To Empty Tableau
            {Move(kRankNone, kSuitNone, kRankK, kSuitSpades),       kMove__Ks},
            {Move(kRankNone, kSuitNone, kRankK, kSuitHearts),       kMove__Kh},
            {Move(kRankNone, kSuitNone, kRankK, kSuitClubs),        kMove__Kc},
            {Move(kRankNone, kSuitNone, kRankK, kSuitDiamonds),     kMove__Kd},
            // endregion

            // region Moves To Empty Foundation
            {Move(kRankNone, kSuitSpades, kRankA, kSuitSpades),     kMove__As},
            {Move(kRankNone, kSuitHearts, kRankA, kSuitHearts),     kMove__Ah},
            {Move(kRankNone, kSuitClubs, kRankA, kSuitClubs),       kMove__Ac},
            {Move(kRankNone, kSuitDiamonds, kRankA, kSuitDiamonds), kMove__Ad},
            // endregion

            // region Moves to Foundation (To Spades)
            {Move(kRankA, kSuitSpades, kRank2, kSuitSpades),     kMoveAs2s},
            {Move(kRank2, kSuitSpades, kRank3, kSuitSpades),        kMove2s3s},
            {Move(kRank3, kSuitSpades, kRank4, kSuitSpades),        kMove3s4s},
            {Move(kRank4, kSuitSpades, kRank5, kSuitSpades),        kMove4s5s},
            {Move(kRank5, kSuitSpades, kRank6, kSuitSpades),        kMove5s6s},
            {Move(kRank6, kSuitSpades, kRank7, kSuitSpades),        kMove6s7s},
            {Move(kRank7, kSuitSpades, kRank8, kSuitSpades),        kMove7s8s},
            {Move(kRank8, kSuitSpades, kRank9, kSuitSpades),        kMove8s9s},
            {Move(kRank9, kSuitSpades, kRankT, kSuitSpades),        kMove9sTs},
            {Move(kRankT, kSuitSpades, kRankJ, kSuitSpades),        kMoveTsJs},
            {Move(kRankJ, kSuitSpades, kRankQ, kSuitSpades),        kMoveJsQs},
            {Move(kRankQ, kSuitSpades, kRankK, kSuitSpades),        kMoveQsKs},
            // endregion

            // region Moves to Foundation (To Hearts)
            {Move(kRankA, kSuitHearts, kRank2, kSuitHearts),        kMoveAh2h},
            {Move(kRank2, kSuitHearts, kRank3, kSuitHearts),        kMove2h3h},
            {Move(kRank3, kSuitHearts, kRank4, kSuitHearts),        kMove3h4h},
            {Move(kRank4, kSuitHearts, kRank5, kSuitHearts),        kMove4h5h},
            {Move(kRank5, kSuitHearts, kRank6, kSuitHearts),        kMove5h6h},
            {Move(kRank6, kSuitHearts, kRank7, kSuitHearts),        kMove6h7h},
            {Move(kRank7, kSuitHearts, kRank8, kSuitHearts),        kMove7h8h},
            {Move(kRank8, kSuitHearts, kRank9, kSuitHearts),        kMove8h9h},
            {Move(kRank9, kSuitHearts, kRankT, kSuitHearts),        kMove9hTh},
            {Move(kRankT, kSuitHearts, kRankJ, kSuitHearts),        kMoveThJh},
            {Move(kRankJ, kSuitHearts, kRankQ, kSuitHearts),        kMoveJhQh},
            {Move(kRankQ, kSuitHearts, kRankK, kSuitHearts),        kMoveQhKh},
            // endregion

            // region Moves to Foundation (To Clubs)
            {Move(kRankA, kSuitClubs, kRank2, kSuitClubs),          kMoveAc2c},
            {Move(kRank2, kSuitClubs, kRank3, kSuitClubs),          kMove2c3c},
            {Move(kRank3, kSuitClubs, kRank4, kSuitClubs),          kMove3c4c},
            {Move(kRank4, kSuitClubs, kRank5, kSuitClubs),          kMove4c5c},
            {Move(kRank5, kSuitClubs, kRank6, kSuitClubs),          kMove5c6c},
            {Move(kRank6, kSuitClubs, kRank7, kSuitClubs),          kMove6c7c},
            {Move(kRank7, kSuitClubs, kRank8, kSuitClubs),          kMove7c8c},
            {Move(kRank8, kSuitClubs, kRank9, kSuitClubs),          kMove8c9c},
            {Move(kRank9, kSuitClubs, kRankT, kSuitClubs),          kMove9cTc},
            {Move(kRankT, kSuitClubs, kRankJ, kSuitClubs),          kMoveTcJc},
            {Move(kRankJ, kSuitClubs, kRankQ, kSuitClubs),          kMoveJcQc},
            {Move(kRankQ, kSuitClubs, kRankK, kSuitClubs),          kMoveQcKc},
            // endregion

            // region Moves to Foundation (To Diamonds)
            {Move(kRankA, kSuitDiamonds, kRank2, kSuitDiamonds),    kMoveAd2d},
            {Move(kRank2, kSuitDiamonds, kRank3, kSuitDiamonds),    kMove2d3d},
            {Move(kRank3, kSuitDiamonds, kRank4, kSuitDiamonds),    kMove3d4d},
            {Move(kRank4, kSuitDiamonds, kRank5, kSuitDiamonds),    kMove4d5d},
            {Move(kRank5, kSuitDiamonds, kRank6, kSuitDiamonds),    kMove5d6d},
            {Move(kRank6, kSuitDiamonds, kRank7, kSuitDiamonds),    kMove6d7d},
            {Move(kRank7, kSuitDiamonds, kRank8, kSuitDiamonds),    kMove7d8d},
            {Move(kRank8, kSuitDiamonds, kRank9, kSuitDiamonds),    kMove8d9d},
            {Move(kRank9, kSuitDiamonds, kRankT, kSuitDiamonds),    kMove9dTd},
            {Move(kRankT, kSuitDiamonds, kRankJ, kSuitDiamonds),    kMoveTdJd},
            {Move(kRankJ, kSuitDiamonds, kRankQ, kSuitDiamonds),    kMoveJdQd},
            {Move(kRankQ, kSuitDiamonds, kRankK, kSuitDiamonds),    kMoveQdKd},
            // endregion

            // Spades --------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Spades <- Hearts)
            {Move(kRank2, kSuitSpades, kRankA, kSuitHearts),        kMove2sAh},
            {Move(kRank3, kSuitSpades, kRank2, kSuitHearts),        kMove3s2h},
            {Move(kRank4, kSuitSpades, kRank3, kSuitHearts),        kMove4s3h},
            {Move(kRank5, kSuitSpades, kRank4, kSuitHearts),        kMove5s4h},
            {Move(kRank6, kSuitSpades, kRank5, kSuitHearts),        kMove6s5h},
            {Move(kRank7, kSuitSpades, kRank6, kSuitHearts),        kMove7s6h},
            {Move(kRank8, kSuitSpades, kRank7, kSuitHearts),        kMove8s7h},
            {Move(kRank9, kSuitSpades, kRank8, kSuitHearts),        kMove9s8h},
            {Move(kRankT, kSuitSpades, kRank9, kSuitHearts),        kMoveTs9h},
            {Move(kRankJ, kSuitSpades, kRankT, kSuitHearts),        kMoveJsTh},
            {Move(kRankQ, kSuitSpades, kRankJ, kSuitHearts),        kMoveQsJh},
            {Move(kRankK, kSuitSpades, kRankQ, kSuitHearts),        kMoveKsQh},
            // endregion

            // region Moves to Tableau (Spades <- Diamonds)
            {Move(kRank2, kSuitSpades, kRankA, kSuitDiamonds),      kMove2sAd},
            {Move(kRank3, kSuitSpades, kRank2, kSuitDiamonds),      kMove3s2d},
            {Move(kRank4, kSuitSpades, kRank3, kSuitDiamonds),      kMove4s3d},
            {Move(kRank5, kSuitSpades, kRank4, kSuitDiamonds),      kMove5s4d},
            {Move(kRank6, kSuitSpades, kRank5, kSuitDiamonds),      kMove6s5d},
            {Move(kRank7, kSuitSpades, kRank6, kSuitDiamonds),      kMove7s6d},
            {Move(kRank8, kSuitSpades, kRank7, kSuitDiamonds),      kMove8s7d},
            {Move(kRank9, kSuitSpades, kRank8, kSuitDiamonds),      kMove9s8d},
            {Move(kRankT, kSuitSpades, kRank9, kSuitDiamonds),      kMoveTs9d},
            {Move(kRankJ, kSuitSpades, kRankT, kSuitDiamonds),      kMoveJsTd},
            {Move(kRankQ, kSuitSpades, kRankJ, kSuitDiamonds),      kMoveQsJd},
            {Move(kRankK, kSuitSpades, kRankQ, kSuitDiamonds),      kMoveKsQd},
            // endregion
            
            // Hearts --------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Hearts <- Spades)
            {Move(kRank2, kSuitHearts, kRankA, kSuitSpades),        kMove2hAs},
            {Move(kRank3, kSuitHearts, kRank2, kSuitSpades),        kMove3h2s},
            {Move(kRank4, kSuitHearts, kRank3, kSuitSpades),        kMove4h3s},
            {Move(kRank5, kSuitHearts, kRank4, kSuitSpades),        kMove5h4s},
            {Move(kRank6, kSuitHearts, kRank5, kSuitSpades),        kMove6h5s},
            {Move(kRank7, kSuitHearts, kRank6, kSuitSpades),        kMove7h6s},
            {Move(kRank8, kSuitHearts, kRank7, kSuitSpades),        kMove8h7s},
            {Move(kRank9, kSuitHearts, kRank8, kSuitSpades),        kMove9h8s},
            {Move(kRankT, kSuitHearts, kRank9, kSuitSpades),        kMoveTh9s},
            {Move(kRankJ, kSuitHearts, kRankT, kSuitSpades),        kMoveJhTs},
            {Move(kRankQ, kSuitHearts, kRankJ, kSuitSpades),        kMoveQhJs},
            {Move(kRankK, kSuitHearts, kRankQ, kSuitSpades),        kMoveKhQs},
            // endregion

            // region Moves to Tableau (Hearts <- Clubs)
            {Move(kRank2, kSuitHearts, kRankA, kSuitClubs),         kMove2hAc},
            {Move(kRank3, kSuitHearts, kRank2, kSuitClubs),         kMove3h2c},
            {Move(kRank4, kSuitHearts, kRank3, kSuitClubs),         kMove4h3c},
            {Move(kRank5, kSuitHearts, kRank4, kSuitClubs),         kMove5h4c},
            {Move(kRank6, kSuitHearts, kRank5, kSuitClubs),         kMove6h5c},
            {Move(kRank7, kSuitHearts, kRank6, kSuitClubs),         kMove7h6c},
            {Move(kRank8, kSuitHearts, kRank7, kSuitClubs),         kMove8h7c},
            {Move(kRank9, kSuitHearts, kRank8, kSuitClubs),         kMove9h8c},
            {Move(kRankT, kSuitHearts, kRank9, kSuitClubs),         kMoveTh9c},
            {Move(kRankJ, kSuitHearts, kRankT, kSuitClubs),         kMoveJhTc},
            {Move(kRankQ, kSuitHearts, kRankJ, kSuitClubs),         kMoveQhJc},
            {Move(kRankK, kSuitHearts, kRankQ, kSuitClubs),         kMoveKhQc},
            // endregion
            
            // Clubs ---------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Clubs <- Hearts)
            {Move(kRank2, kSuitClubs, kRankA, kSuitHearts),         kMove2cAh},
            {Move(kRank3, kSuitClubs, kRank2, kSuitHearts),         kMove3c2h},
            {Move(kRank4, kSuitClubs, kRank3, kSuitHearts),         kMove4c3h},
            {Move(kRank5, kSuitClubs, kRank4, kSuitHearts),         kMove5c4h},
            {Move(kRank6, kSuitClubs, kRank5, kSuitHearts),         kMove6c5h},
            {Move(kRank7, kSuitClubs, kRank6, kSuitHearts),         kMove7c6h},
            {Move(kRank8, kSuitClubs, kRank7, kSuitHearts),         kMove8c7h},
            {Move(kRank9, kSuitClubs, kRank8, kSuitHearts),         kMove9c8h},
            {Move(kRankT, kSuitClubs, kRank9, kSuitHearts),         kMoveTc9h},
            {Move(kRankJ, kSuitClubs, kRankT, kSuitHearts),         kMoveJcTh},
            {Move(kRankQ, kSuitClubs, kRankJ, kSuitHearts),         kMoveQcJh},
            {Move(kRankK, kSuitClubs, kRankQ, kSuitHearts),         kMoveKcQh},
            // endregion

            // region Moves to Tableau (Clubs <- Diamonds)
            {Move(kRank2, kSuitClubs, kRankA, kSuitDiamonds),       kMove2cAd},
            {Move(kRank3, kSuitClubs, kRank2, kSuitDiamonds),       kMove3c2d},
            {Move(kRank4, kSuitClubs, kRank3, kSuitDiamonds),       kMove4c3d},
            {Move(kRank5, kSuitClubs, kRank4, kSuitDiamonds),       kMove5c4d},
            {Move(kRank6, kSuitClubs, kRank5, kSuitDiamonds),       kMove6c5d},
            {Move(kRank7, kSuitClubs, kRank6, kSuitDiamonds),       kMove7c6d},
            {Move(kRank8, kSuitClubs, kRank7, kSuitDiamonds),       kMove8c7d},
            {Move(kRank9, kSuitClubs, kRank8, kSuitDiamonds),       kMove9c8d},
            {Move(kRankT, kSuitClubs, kRank9, kSuitDiamonds),       kMoveTc9d},
            {Move(kRankJ, kSuitClubs, kRankT, kSuitDiamonds),       kMoveJcTd},
            {Move(kRankQ, kSuitClubs, kRankJ, kSuitDiamonds),       kMoveQcJd},
            {Move(kRankK, kSuitClubs, kRankQ, kSuitDiamonds),       kMoveKcQd},
            // endregion

            // Diamonds ------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Diamonds <- Spades)
            {Move(kRank2, kSuitDiamonds, kRankA, kSuitSpades),      kMove2dAs},
            {Move(kRank3, kSuitDiamonds, kRank2, kSuitSpades),      kMove3d2s},
            {Move(kRank4, kSuitDiamonds, kRank3, kSuitSpades),      kMove4d3s},
            {Move(kRank5, kSuitDiamonds, kRank4, kSuitSpades),      kMove5d4s},
            {Move(kRank6, kSuitDiamonds, kRank5, kSuitSpades),      kMove6d5s},
            {Move(kRank7, kSuitDiamonds, kRank6, kSuitSpades),      kMove7d6s},
            {Move(kRank8, kSuitDiamonds, kRank7, kSuitSpades),      kMove8d7s},
            {Move(kRank9, kSuitDiamonds, kRank8, kSuitSpades),      kMove9d8s},
            {Move(kRankT, kSuitDiamonds, kRank9, kSuitSpades),      kMoveTd9s},
            {Move(kRankJ, kSuitDiamonds, kRankT, kSuitSpades),      kMoveJdTs},
            {Move(kRankQ, kSuitDiamonds, kRankJ, kSuitSpades),      kMoveQdJs},
            {Move(kRankK, kSuitDiamonds, kRankQ, kSuitSpades),      kMoveKdQs},
            // endregion

            // region Moves to Tableau (Diamonds <- Clubs)
            {Move(kRank2, kSuitDiamonds, kRankA, kSuitClubs),       kMove2dAc},
            {Move(kRank3, kSuitDiamonds, kRank2, kSuitClubs),       kMove3d2c},
            {Move(kRank4, kSuitDiamonds, kRank3, kSuitClubs),       kMove4d3c},
            {Move(kRank5, kSuitDiamonds, kRank4, kSuitClubs),       kMove5d4c},
            {Move(kRank6, kSuitDiamonds, kRank5, kSuitClubs),       kMove6d5c},
            {Move(kRank7, kSuitDiamonds, kRank6, kSuitClubs),       kMove7d6c},
            {Move(kRank8, kSuitDiamonds, kRank7, kSuitClubs),       kMove8d7c},
            {Move(kRank9, kSuitDiamonds, kRank8, kSuitClubs),       kMove9d8c},
            {Move(kRankT, kSuitDiamonds, kRank9, kSuitClubs),       kMoveTd9c},
            {Move(kRankJ, kSuitDiamonds, kRankT, kSuitClubs),       kMoveJdTc},
            {Move(kRankQ, kSuitDiamonds, kRankJ, kSuitClubs),       kMoveQdJc},
            {Move(kRankK, kSuitDiamonds, kRankQ, kSuitClubs),       kMoveKdQc},
            // endregion
            
            // endregion
    };

    // TODO: Please compute this on-the-fly instead of having this long mapping
    const std::map<Action, Move> kActionToMove = {
            // region Mapping of an action declared in ActionType to a move;

            // region Moves To Empty Tableau
            {kMove__Ks, Move(kRankNone, kSuitNone, kRankK, kSuitSpades)},
            {kMove__Kh, Move(kRankNone, kSuitNone, kRankK, kSuitHearts)},
            {kMove__Kc, Move(kRankNone, kSuitNone, kRankK, kSuitClubs)},
            {kMove__Kd, Move(kRankNone, kSuitNone, kRankK, kSuitDiamonds)},
            // endregion

            // region Moves To Empty Foundation
            {kMove__As, Move(kRankNone, kSuitSpades, kRankA, kSuitSpades)},
            {kMove__Ah, Move(kRankNone, kSuitHearts, kRankA, kSuitHearts)},
            {kMove__Ac, Move(kRankNone, kSuitClubs, kRankA, kSuitClubs)},
            {kMove__Ad, Move(kRankNone, kSuitDiamonds, kRankA, kSuitDiamonds)},
            // endregion

            // region Moves to Foundation (To Spades)
            {kMoveAs2s, Move(kRankA, kSuitSpades, kRank2, kSuitSpades)},
            {kMove2s3s, Move(kRank2, kSuitSpades, kRank3, kSuitSpades)},
            {kMove3s4s, Move(kRank3, kSuitSpades, kRank4, kSuitSpades)},
            {kMove4s5s, Move(kRank4, kSuitSpades, kRank5, kSuitSpades)},
            {kMove5s6s, Move(kRank5, kSuitSpades, kRank6, kSuitSpades)},
            {kMove6s7s, Move(kRank6, kSuitSpades, kRank7, kSuitSpades)},
            {kMove7s8s, Move(kRank7, kSuitSpades, kRank8, kSuitSpades)},
            {kMove8s9s, Move(kRank8, kSuitSpades, kRank9, kSuitSpades)},
            {kMove9sTs, Move(kRank9, kSuitSpades, kRankT, kSuitSpades)},
            {kMoveTsJs, Move(kRankT, kSuitSpades, kRankJ, kSuitSpades)},
            {kMoveJsQs, Move(kRankJ, kSuitSpades, kRankQ, kSuitSpades)},
            {kMoveQsKs, Move(kRankQ, kSuitSpades, kRankK, kSuitSpades)},
            // endregion

            // region Moves to Foundation (To Hearts)
            {kMoveAh2h, Move(kRankA, kSuitHearts, kRank2, kSuitHearts)},
            {kMove2h3h, Move(kRank2, kSuitHearts, kRank3, kSuitHearts)},
            {kMove3h4h, Move(kRank3, kSuitHearts, kRank4, kSuitHearts)},
            {kMove4h5h, Move(kRank4, kSuitHearts, kRank5, kSuitHearts)},
            {kMove5h6h, Move(kRank5, kSuitHearts, kRank6, kSuitHearts)},
            {kMove6h7h, Move(kRank6, kSuitHearts, kRank7, kSuitHearts)},
            {kMove7h8h, Move(kRank7, kSuitHearts, kRank8, kSuitHearts)},
            {kMove8h9h, Move(kRank8, kSuitHearts, kRank9, kSuitHearts)},
            {kMove9hTh, Move(kRank9, kSuitHearts, kRankT, kSuitHearts)},
            {kMoveThJh, Move(kRankT, kSuitHearts, kRankJ, kSuitHearts)},
            {kMoveJhQh, Move(kRankJ, kSuitHearts, kRankQ, kSuitHearts)},
            {kMoveQhKh, Move(kRankQ, kSuitHearts, kRankK, kSuitHearts)},
            // endregion

            // region Moves to Foundation (To Clubs)
            {kMoveAc2c, Move(kRankA, kSuitClubs, kRank2, kSuitClubs)},
            {kMove2c3c, Move(kRank2, kSuitClubs, kRank3, kSuitClubs)},
            {kMove3c4c, Move(kRank3, kSuitClubs, kRank4, kSuitClubs)},
            {kMove4c5c, Move(kRank4, kSuitClubs, kRank5, kSuitClubs)},
            {kMove5c6c, Move(kRank5, kSuitClubs, kRank6, kSuitClubs)},
            {kMove6c7c, Move(kRank6, kSuitClubs, kRank7, kSuitClubs)},
            {kMove7c8c, Move(kRank7, kSuitClubs, kRank8, kSuitClubs)},
            {kMove8c9c, Move(kRank8, kSuitClubs, kRank9, kSuitClubs)},
            {kMove9cTc, Move(kRank9, kSuitClubs, kRankT, kSuitClubs)},
            {kMoveTcJc, Move(kRankT, kSuitClubs, kRankJ, kSuitClubs)},
            {kMoveJcQc, Move(kRankJ, kSuitClubs, kRankQ, kSuitClubs)},
            {kMoveQcKc, Move(kRankQ, kSuitClubs, kRankK, kSuitClubs)},
            // endregion

            // region Moves to Foundation (To Diamonds)
            {kMoveAd2d, Move(kRankA, kSuitDiamonds, kRank2, kSuitDiamonds)},
            {kMove2d3d, Move(kRank2, kSuitDiamonds, kRank3, kSuitDiamonds)},
            {kMove3d4d, Move(kRank3, kSuitDiamonds, kRank4, kSuitDiamonds)},
            {kMove4d5d, Move(kRank4, kSuitDiamonds, kRank5, kSuitDiamonds)},
            {kMove5d6d, Move(kRank5, kSuitDiamonds, kRank6, kSuitDiamonds)},
            {kMove6d7d, Move(kRank6, kSuitDiamonds, kRank7, kSuitDiamonds)},
            {kMove7d8d, Move(kRank7, kSuitDiamonds, kRank8, kSuitDiamonds)},
            {kMove8d9d, Move(kRank8, kSuitDiamonds, kRank9, kSuitDiamonds)},
            {kMove9dTd, Move(kRank9, kSuitDiamonds, kRankT, kSuitDiamonds)},
            {kMoveTdJd, Move(kRankT, kSuitDiamonds, kRankJ, kSuitDiamonds)},
            {kMoveJdQd, Move(kRankJ, kSuitDiamonds, kRankQ, kSuitDiamonds)},
            {kMoveQdKd, Move(kRankQ, kSuitDiamonds, kRankK, kSuitDiamonds)},
            // endregion

            // Spades --------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Spades <- Hearts)
            {kMove2sAh, Move(kRank2, kSuitSpades, kRankA, kSuitHearts)},
            {kMove3s2h, Move(kRank3, kSuitSpades, kRank2, kSuitHearts)},
            {kMove4s3h, Move(kRank4, kSuitSpades, kRank3, kSuitHearts)},
            {kMove5s4h, Move(kRank5, kSuitSpades, kRank4, kSuitHearts)},
            {kMove6s5h, Move(kRank6, kSuitSpades, kRank5, kSuitHearts)},
            {kMove7s6h, Move(kRank7, kSuitSpades, kRank6, kSuitHearts)},
            {kMove8s7h, Move(kRank8, kSuitSpades, kRank7, kSuitHearts)},
            {kMove9s8h, Move(kRank9, kSuitSpades, kRank8, kSuitHearts)},
            {kMoveTs9h, Move(kRankT, kSuitSpades, kRank9, kSuitHearts)},
            {kMoveJsTh, Move(kRankJ, kSuitSpades, kRankT, kSuitHearts)},
            {kMoveQsJh, Move(kRankQ, kSuitSpades, kRankJ, kSuitHearts)},
            {kMoveKsQh, Move(kRankK, kSuitSpades, kRankQ, kSuitHearts)},
            // endregion

            // region Moves to Tableau (Spades <- Diamonds)
            {kMove2sAd, Move(kRank2, kSuitSpades, kRankA, kSuitDiamonds)},
            {kMove3s2d, Move(kRank3, kSuitSpades, kRank2, kSuitDiamonds)},
            {kMove4s3d, Move(kRank4, kSuitSpades, kRank3, kSuitDiamonds)},
            {kMove5s4d, Move(kRank5, kSuitSpades, kRank4, kSuitDiamonds)},
            {kMove6s5d, Move(kRank6, kSuitSpades, kRank5, kSuitDiamonds)},
            {kMove7s6d, Move(kRank7, kSuitSpades, kRank6, kSuitDiamonds)},
            {kMove8s7d, Move(kRank8, kSuitSpades, kRank7, kSuitDiamonds)},
            {kMove9s8d, Move(kRank9, kSuitSpades, kRank8, kSuitDiamonds)},
            {kMoveTs9d, Move(kRankT, kSuitSpades, kRank9, kSuitDiamonds)},
            {kMoveJsTd, Move(kRankJ, kSuitSpades, kRankT, kSuitDiamonds)},
            {kMoveQsJd, Move(kRankQ, kSuitSpades, kRankJ, kSuitDiamonds)},
            {kMoveKsQd, Move(kRankK, kSuitSpades, kRankQ, kSuitDiamonds)},
            // endregion

            // Hearts --------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Hearts <- Spades)
            {kMove2hAs, Move(kRank2, kSuitHearts, kRankA, kSuitSpades)},
            {kMove3h2s, Move(kRank3, kSuitHearts, kRank2, kSuitSpades)},
            {kMove4h3s, Move(kRank4, kSuitHearts, kRank3, kSuitSpades)},
            {kMove5h4s, Move(kRank5, kSuitHearts, kRank4, kSuitSpades)},
            {kMove6h5s, Move(kRank6, kSuitHearts, kRank5, kSuitSpades)},
            {kMove7h6s, Move(kRank7, kSuitHearts, kRank6, kSuitSpades)},
            {kMove8h7s, Move(kRank8, kSuitHearts, kRank7, kSuitSpades)},
            {kMove9h8s, Move(kRank9, kSuitHearts, kRank8, kSuitSpades)},
            {kMoveTh9s, Move(kRankT, kSuitHearts, kRank9, kSuitSpades)},
            {kMoveJhTs, Move(kRankJ, kSuitHearts, kRankT, kSuitSpades)},
            {kMoveQhJs, Move(kRankQ, kSuitHearts, kRankJ, kSuitSpades)},
            {kMoveKhQs, Move(kRankK, kSuitHearts, kRankQ, kSuitSpades)},
            // endregion

            // region Moves to Tableau (Hearts <- Clubs)
            {kMove2hAc, Move(kRank2, kSuitHearts, kRankA, kSuitClubs)},
            {kMove3h2c, Move(kRank3, kSuitHearts, kRank2, kSuitClubs)},
            {kMove4h3c, Move(kRank4, kSuitHearts, kRank3, kSuitClubs)},
            {kMove5h4c, Move(kRank5, kSuitHearts, kRank4, kSuitClubs)},
            {kMove6h5c, Move(kRank6, kSuitHearts, kRank5, kSuitClubs)},
            {kMove7h6c, Move(kRank7, kSuitHearts, kRank6, kSuitClubs)},
            {kMove8h7c, Move(kRank8, kSuitHearts, kRank7, kSuitClubs)},
            {kMove9h8c, Move(kRank9, kSuitHearts, kRank8, kSuitClubs)},
            {kMoveTh9c, Move(kRankT, kSuitHearts, kRank9, kSuitClubs)},
            {kMoveJhTc, Move(kRankJ, kSuitHearts, kRankT, kSuitClubs)},
            {kMoveQhJc, Move(kRankQ, kSuitHearts, kRankJ, kSuitClubs)},
            {kMoveKhQc, Move(kRankK, kSuitHearts, kRankQ, kSuitClubs)},
            // endregion

            // Clubs ---------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Clubs <- Hearts)
            {kMove2cAh, Move(kRank2, kSuitClubs, kRankA, kSuitHearts)},
            {kMove3c2h, Move(kRank3, kSuitClubs, kRank2, kSuitHearts)},
            {kMove4c3h, Move(kRank4, kSuitClubs, kRank3, kSuitHearts)},
            {kMove5c4h, Move(kRank5, kSuitClubs, kRank4, kSuitHearts)},
            {kMove6c5h, Move(kRank6, kSuitClubs, kRank5, kSuitHearts)},
            {kMove7c6h, Move(kRank7, kSuitClubs, kRank6, kSuitHearts)},
            {kMove8c7h, Move(kRank8, kSuitClubs, kRank7, kSuitHearts)},
            {kMove9c8h, Move(kRank9, kSuitClubs, kRank8, kSuitHearts)},
            {kMoveTc9h, Move(kRankT, kSuitClubs, kRank9, kSuitHearts)},
            {kMoveJcTh, Move(kRankJ, kSuitClubs, kRankT, kSuitHearts)},
            {kMoveQcJh, Move(kRankQ, kSuitClubs, kRankJ, kSuitHearts)},
            {kMoveKcQh, Move(kRankK, kSuitClubs, kRankQ, kSuitHearts)},
            // endregion

            // region Moves to Tableau (Clubs <- Diamonds)
            {kMove2cAd, Move(kRank2, kSuitClubs, kRankA, kSuitDiamonds)},
            {kMove3c2d, Move(kRank3, kSuitClubs, kRank2, kSuitDiamonds)},
            {kMove4c3d, Move(kRank4, kSuitClubs, kRank3, kSuitDiamonds)},
            {kMove5c4d, Move(kRank5, kSuitClubs, kRank4, kSuitDiamonds)},
            {kMove6c5d, Move(kRank6, kSuitClubs, kRank5, kSuitDiamonds)},
            {kMove7c6d, Move(kRank7, kSuitClubs, kRank6, kSuitDiamonds)},
            {kMove8c7d, Move(kRank8, kSuitClubs, kRank7, kSuitDiamonds)},
            {kMove9c8d, Move(kRank9, kSuitClubs, kRank8, kSuitDiamonds)},
            {kMoveTc9d, Move(kRankT, kSuitClubs, kRank9, kSuitDiamonds)},
            {kMoveJcTd, Move(kRankJ, kSuitClubs, kRankT, kSuitDiamonds)},
            {kMoveQcJd, Move(kRankQ, kSuitClubs, kRankJ, kSuitDiamonds)},
            {kMoveKcQd, Move(kRankK, kSuitClubs, kRankQ, kSuitDiamonds)},
            // endregion

            // Diamonds ------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Diamonds <- Spades)
            {kMove2dAs, Move(kRank2, kSuitDiamonds, kRankA, kSuitSpades)},
            {kMove3d2s, Move(kRank3, kSuitDiamonds, kRank2, kSuitSpades)},
            {kMove4d3s, Move(kRank4, kSuitDiamonds, kRank3, kSuitSpades)},
            {kMove5d4s, Move(kRank5, kSuitDiamonds, kRank4, kSuitSpades)},
            {kMove6d5s, Move(kRank6, kSuitDiamonds, kRank5, kSuitSpades)},
            {kMove7d6s, Move(kRank7, kSuitDiamonds, kRank6, kSuitSpades)},
            {kMove8d7s, Move(kRank8, kSuitDiamonds, kRank7, kSuitSpades)},
            {kMove9d8s, Move(kRank9, kSuitDiamonds, kRank8, kSuitSpades)},
            {kMoveTd9s, Move(kRankT, kSuitDiamonds, kRank9, kSuitSpades)},
            {kMoveJdTs, Move(kRankJ, kSuitDiamonds, kRankT, kSuitSpades)},
            {kMoveQdJs, Move(kRankQ, kSuitDiamonds, kRankJ, kSuitSpades)},
            {kMoveKdQs, Move(kRankK, kSuitDiamonds, kRankQ, kSuitSpades)},
            // endregion

            // region Moves to Tableau (Diamonds <- Clubs)
            {kMove2dAc, Move(kRank2, kSuitDiamonds, kRankA, kSuitClubs)},
            {kMove3d2c, Move(kRank3, kSuitDiamonds, kRank2, kSuitClubs)},
            {kMove4d3c, Move(kRank4, kSuitDiamonds, kRank3, kSuitClubs)},
            {kMove5d4c, Move(kRank5, kSuitDiamonds, kRank4, kSuitClubs)},
            {kMove6d5c, Move(kRank6, kSuitDiamonds, kRank5, kSuitClubs)},
            {kMove7d6c, Move(kRank7, kSuitDiamonds, kRank6, kSuitClubs)},
            {kMove8d7c, Move(kRank8, kSuitDiamonds, kRank7, kSuitClubs)},
            {kMove9d8c, Move(kRank9, kSuitDiamonds, kRank8, kSuitClubs)},
            {kMoveTd9c, Move(kRankT, kSuitDiamonds, kRank9, kSuitClubs)},
            {kMoveJdTc, Move(kRankJ, kSuitDiamonds, kRankT, kSuitClubs)},
            {kMoveQdJc, Move(kRankQ, kSuitDiamonds, kRankJ, kSuitClubs)},
            {kMoveKdQc, Move(kRankK, kSuitDiamonds, kRankQ, kSuitClubs)},
            // endregion

            // endregion
    };

    // OpenSpiel Classes ===============================================================================================

    class SolitaireGame;

    class SolitaireState : public State {
    public:
        
        // Constructors ================================================================================================

        explicit SolitaireState(std::shared_ptr<const Game> game);

        // Overridden Methods ==========================================================================================

        Player                  CurrentPlayer() const override;
        std::unique_ptr<State>  Clone() const override;
        bool                    IsTerminal() const override;
        bool                    IsChanceNode() const override;
        std::string             ToString() const override;
        std::string             ActionToString(Player player, Action action_id) const override;
        std::string             InformationStateString(Player player) const override;
        std::string             ObservationString(Player player) const override;
        void                    ObservationTensor(Player player, std::vector<double> * values) const override;
        void                    DoApplyAction(Action move) override;
        std::vector<double>     Returns() const override;
        std::vector<double>     Rewards() const override;
        std::vector<Action>     LegalActions() const override;
        std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

        // Other Methods ===============================================================================================

        std::vector<Card>       Targets(const std::optional<LocationType> & location = kMissing) const;
        std::vector<Card>       Sources(const std::optional<LocationType> & location = kMissing) const;
        std::vector<Move>       CandidateMoves() const;
        Pile *                  GetPile(const Card & card) const;
        Pile *                  GetPile(const PileID & pile_id) const;
        void                    MoveCards(const Move & move);
        bool                    IsReversible(const Card & source, Pile * source_pile) const;

    private:
        Waste                    waste;
        std::vector<Foundation>  foundations;
        std::vector<Tableau>     tableaus;
        std::vector<Action>      revealed_cards;

        bool   is_finished    = false;
        bool   is_reversible  = false;
        int    current_depth  = 0;

        std::set<std::size_t>  previous_states = {};
        std::map<Card, PileID> card_map;

        double current_returns  = 0.0;
        double current_rewards  = 0.0;

        // Parameters
        int  depth_limit = kDefaultDepthLimit;
        bool is_colored  = kDefaultIsColored;

    };

    class SolitaireGame : public Game {
    public:

        // Constructor =================================================================================================

        explicit    SolitaireGame(const GameParameters & params);

        // Overridden Methods ==========================================================================================

        int         NumDistinctActions() const override;
        int         MaxGameLength() const override;
        int         NumPlayers() const override;
        double      MinUtility() const override;
        double      MaxUtility() const override;

        std::vector<int> ObservationTensorShape() const override;
        std::unique_ptr<State> NewInitialState() const override;
        std::shared_ptr<const Game> Clone() const override;

    private:
        int  num_players_;
        int  depth_limit_;
        bool is_colored_;

    };

} // namespace open_spiel::solitaire

#endif // THIRD_PARTY_OPEN_SPIEL_GAMES_SOLITAIRE_H
