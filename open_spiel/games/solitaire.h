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

// ANSI color codes
#define RESET "\033[0m"
#define RED   "\033[31m"
#define BLACK "\033[37m"
#define BLUE  "\033[34m"

// TODO: Clion automatically inverts colors based on light/dark theme. So even though "\033[30m" is black,
//       it shows up as white with a dark theme. Colab doesn't do this, so using a dark theme, it shows black
//       on a dark theme.

// Glyphs & Strings

#define GLYPH_HIDDEN   "\U0001F0A0"
#define GLYPH_EMPTY    "\U0001F0BF"
#define GLYPH_SPADES   "\U00002660"
#define GLYPH_HEARTS   "\U00002665"
#define GLYPH_CLUBS    "\U00002663"
#define GLYPH_DIAMONDS "\U00002666"
#define GLYPH_ARROW    "\U00002190"


namespace open_spiel::solitaire {

    // Default Game Parameters =========================================================================================

    inline constexpr int    kDefaultPlayers    = 1;
    inline constexpr int    kPlayerId          = 0;
    inline constexpr int    kDepthLimit        = 500;
    inline constexpr bool   kDefaultColored    = true;
    inline constexpr bool   kDefaultThoughtful = false;

    // Enumerations ====================================================================================================

    enum SuitType     {
        kNoSuit = 0,
        kS,
        kH,
        kC,
        kD,
        kHiddenSuit,
    };
    enum RankType     {
        kNoRank = 0,
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
        kHiddenRank,
    };
    enum LocationType {
        kDeck       = 0,
        kWaste      = 1,
        kFoundation = 2,
        kTableau    = 3,
        kMissing    = 4,
    };
    enum ActionType   {

        // Draw Action (1) =============================================================================================
        kDraw = 0,

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

    // Constants =======================================================================================================

    //region Indices for special cards
    inline constexpr int HIDDEN_CARD          = 99;
    inline constexpr int NO_CARD              =  0;
    inline constexpr int EMPTY_TABLEAU_CARD   = -1;
    inline constexpr int EMPTY_SPADE_CARD     = -2;
    inline constexpr int EMPTY_HEART_CARD     = -3;
    inline constexpr int EMPTY_CLUB_CARD      = -4;
    inline constexpr int EMPTY_DIAMOND_CARD   = -5;

    // Type aliases
    using Ranksuit = std::pair<RankType, SuitType>;

    // Other lists and maps
    const std::vector<SuitType> SUITS = {kS, kH, kC, kD};
    const std::vector<RankType> RANKS = {kA, k2, k3, k4, k5, k6, k7, k8, k9, kT, kJ, kQ, kK};

    // These correspond with their enums, not with the two vectors directly above
    const std::vector<std::string> SUIT_STRS = {"", GLYPH_SPADES, GLYPH_HEARTS, GLYPH_CLUBS, GLYPH_DIAMONDS, ""};
    const std::vector<std::string> RANK_STRS = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", ""};

    const std::map<RankType, double> FOUNDATION_POINTS = {
            //region Maps a RankType to a double that represents the reward for moving a card of that rank to the foundation
            {kA, 100.0},
            {k2, 90.0},
            {k3, 80.0},
            {k4, 70.0},
            {k5, 60.0},
            {k6, 50.0},
            {k7, 40.0},
            {k8, 30.0},
            {k9, 20.0},
            {kT, 10.0},
            {kJ, 10.0},
            {kQ, 10.0},
            {kK, 10.0}
            //endregion
    };

    const std::map<SuitType, PileID> SUIT_TO_PILE = {
            // region Maps a foundation suit to the ID of the foundation
            {kS, kPileSpades},
            {kH, kPileHearts},
            {kC, kPileClubs},
            {kD, kPileDiamonds}
            // endregion
    };

    const std::map<int, PileID> INT_TO_PILE = {
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

    // Miscellaneous Functions =========================================================================================

    std::vector<SuitType> GetOppositeSuits(const SuitType & suit);

    int GetCardIndex(RankType rank, SuitType suit);

    int GetMaxSize(LocationType location);

    // Support Classes =================================================================================================

    class Card {
    public:

        // Attributes ==================================================================================================
        RankType      rank     = kHiddenRank;       // Indicates the rank of the card
        SuitType      suit     = kHiddenSuit;       // Indicates the suit of the card
        LocationType  location = kMissing;          // Indicates the type of pile the card is in
        bool          hidden   = false;             // Indicates whether the card is hidden or not
        int           index    = HIDDEN_CARD;       // Identifies the card with an integer

        // Constructors ================================================================================================
        Card(bool hidden = false, SuitType suit = kHiddenSuit, RankType rank = kHiddenRank, LocationType location = kMissing);
        Card(int index, bool hidden = false, LocationType location = kMissing);

        // Other Methods ===============================================================================================
        int GetIndex() const ;
        std::string ToString(bool colored = true) const;
        std::vector<Card> LegalChildren() const;
        bool operator==(Card & other_card) const;
        bool operator==(const Card & other_card) const;
        bool operator<(const Card & other_card) const;

    };

    class Pile {
    public:

        // Attributes ==================================================================================================
        std::vector<Card>  cards;
        const LocationType type;
        const SuitType     suit;
        const PileID       id;
        const int          max_size;

        // Constructors ================================================================================================
        Pile(LocationType type, PileID id, SuitType suit = kNoSuit);

        // Other Methods ===============================================================================================
        std::vector<Card>   Sources() const;
        std::vector<Card>   Targets() const;
        std::vector<Card>   Split(Card card);
        void                Extend(std::vector<Card> source_cards);

        std::vector<double> Tensor() const;
        std::string         ToString(bool colored = true) const;

    };

    class Move {
    public:

        // Attributes ==================================================================================================
        Card target;
        Card source;

        // Constructors ================================================================================================
        Move(Card target_card, Card source_card);
        Move(RankType target_rank, SuitType target_suit, RankType source_rank, SuitType source_suit);
        explicit Move(Action action);

        // Other Methods ===============================================================================================
        std::string ToString(bool colored = true) const;
        bool operator<(const Move & other_move) const;
        Action ActionId() const;

    };

    // More Constants ==================================================================================================

    const std::map<Move, Action> MOVE_TO_ACTION = {
            // region Mapping of a move to an action declared in ActionType;
            
            // region Moves To Empty Tableau
            {Move(kNoRank, kNoSuit, kK, kS), kMove__Ks},
            {Move(kNoRank, kNoSuit, kK, kH), kMove__Kh},
            {Move(kNoRank, kNoSuit, kK, kC), kMove__Kc},
            {Move(kNoRank, kNoSuit, kK, kD), kMove__Kd},
            // endregion

            // region Moves To Empty Foundation
            {Move(kNoRank, kS, kA, kS),      kMove__As},
            {Move(kNoRank, kH, kA, kH),      kMove__Ah},
            {Move(kNoRank, kC, kA, kC),      kMove__Ac},
            {Move(kNoRank, kD, kA, kD),      kMove__Ad},
            // endregion

            // region Moves to Foundation (To Spades)
            {Move(kA, kS, k2, kS),           kMoveAs2s},
            {Move(k2, kS, k3, kS),           kMove2s3s},
            {Move(k3, kS, k4, kS),           kMove3s4s},
            {Move(k4, kS, k5, kS),           kMove4s5s},
            {Move(k5, kS, k6, kS),           kMove5s6s},
            {Move(k6, kS, k7, kS),           kMove6s7s},
            {Move(k7, kS, k8, kS),           kMove7s8s},
            {Move(k8, kS, k9, kS),           kMove8s9s},
            {Move(k9, kS, kT, kS),           kMove9sTs},
            {Move(kT, kS, kJ, kS),           kMoveTsJs},
            {Move(kJ, kS, kQ, kS),           kMoveJsQs},
            {Move(kQ, kS, kK, kS),           kMoveQsKs},
            // endregion

            // region Moves to Foundation (To Hearts)
            {Move(kA, kH, k2, kH),           kMoveAh2h},
            {Move(k2, kH, k3, kH),           kMove2h3h},
            {Move(k3, kH, k4, kH),           kMove3h4h},
            {Move(k4, kH, k5, kH),           kMove4h5h},
            {Move(k5, kH, k6, kH),           kMove5h6h},
            {Move(k6, kH, k7, kH),           kMove6h7h},
            {Move(k7, kH, k8, kH),           kMove7h8h},
            {Move(k8, kH, k9, kH),           kMove8h9h},
            {Move(k9, kH, kT, kH),           kMove9hTh},
            {Move(kT, kH, kJ, kH),           kMoveThJh},
            {Move(kJ, kH, kQ, kH),           kMoveJhQh},
            {Move(kQ, kH, kK, kH),           kMoveQhKh},
            // endregion

            // region Moves to Foundation (To Clubs)
            {Move(kA, kC, k2, kC),           kMoveAc2c},
            {Move(k2, kC, k3, kC),           kMove2c3c},
            {Move(k3, kC, k4, kC),           kMove3c4c},
            {Move(k4, kC, k5, kC),           kMove4c5c},
            {Move(k5, kC, k6, kC),           kMove5c6c},
            {Move(k6, kC, k7, kC),           kMove6c7c},
            {Move(k7, kC, k8, kC),           kMove7c8c},
            {Move(k8, kC, k9, kC),           kMove8c9c},
            {Move(k9, kC, kT, kC),           kMove9cTc},
            {Move(kT, kC, kJ, kC),           kMoveTcJc},
            {Move(kJ, kC, kQ, kC),           kMoveJcQc},
            {Move(kQ, kC, kK, kC),           kMoveQcKc},
            // endregion

            // region Moves to Foundation (To Diamonds)
            {Move(kA, kD, k2, kD),           kMoveAd2d},
            {Move(k2, kD, k3, kD),           kMove2d3d},
            {Move(k3, kD, k4, kD),           kMove3d4d},
            {Move(k4, kD, k5, kD),           kMove4d5d},
            {Move(k5, kD, k6, kD),           kMove5d6d},
            {Move(k6, kD, k7, kD),           kMove6d7d},
            {Move(k7, kD, k8, kD),           kMove7d8d},
            {Move(k8, kD, k9, kD),           kMove8d9d},
            {Move(k9, kD, kT, kD),           kMove9dTd},
            {Move(kT, kD, kJ, kD),           kMoveTdJd},
            {Move(kJ, kD, kQ, kD),           kMoveJdQd},
            {Move(kQ, kD, kK, kD),           kMoveQdKd},
            // endregion

            // Spades --------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Spades <- Hearts)
            {Move(k2, kS, kA, kH), kMove2sAh},
            {Move(k3, kS, k2, kH), kMove3s2h},
            {Move(k4, kS, k3, kH), kMove4s3h},
            {Move(k5, kS, k4, kH), kMove5s4h},
            {Move(k6, kS, k5, kH), kMove6s5h},
            {Move(k7, kS, k6, kH), kMove7s6h},
            {Move(k8, kS, k7, kH), kMove8s7h},
            {Move(k9, kS, k8, kH), kMove9s8h},
            {Move(kT, kS, k9, kH), kMoveTs9h},
            {Move(kJ, kS, kT, kH), kMoveJsTh},
            {Move(kQ, kS, kJ, kH), kMoveQsJh},
            {Move(kK, kS, kQ, kH), kMoveKsQh},
            // endregion

            // region Moves to Tableau (Spades <- Diamonds)
            {Move(k2, kS, kA, kD), kMove2sAd},
            {Move(k3, kS, k2, kD), kMove3s2d},
            {Move(k4, kS, k3, kD), kMove4s3d},
            {Move(k5, kS, k4, kD), kMove5s4d},
            {Move(k6, kS, k5, kD), kMove6s5d},
            {Move(k7, kS, k6, kD), kMove7s6d},
            {Move(k8, kS, k7, kD), kMove8s7d},
            {Move(k9, kS, k8, kD), kMove9s8d},
            {Move(kT, kS, k9, kD), kMoveTs9d},
            {Move(kJ, kS, kT, kD), kMoveJsTd},
            {Move(kQ, kS, kJ, kD), kMoveQsJd},
            {Move(kK, kS, kQ, kD), kMoveKsQd},
            // endregion
            
            // Hearts --------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Hearts <- Spades)
            {Move(k2, kH, kA, kS), kMove2hAs},
            {Move(k3, kH, k2, kS), kMove3h2s},
            {Move(k4, kH, k3, kS), kMove4h3s},
            {Move(k5, kH, k4, kS), kMove5h4s},
            {Move(k6, kH, k5, kS), kMove6h5s},
            {Move(k7, kH, k6, kS), kMove7h6s},
            {Move(k8, kH, k7, kS), kMove8h7s},
            {Move(k9, kH, k8, kS), kMove9h8s},
            {Move(kT, kH, k9, kS), kMoveTh9s},
            {Move(kJ, kH, kT, kS), kMoveJhTs},
            {Move(kQ, kH, kJ, kS), kMoveQhJs},
            {Move(kK, kH, kQ, kS), kMoveKhQs},
            // endregion

            // region Moves to Tableau (Hearts <- Clubs)
            {Move(k2, kH, kA, kC), kMove2hAc},
            {Move(k3, kH, k2, kC), kMove3h2c},
            {Move(k4, kH, k3, kC), kMove4h3c},
            {Move(k5, kH, k4, kC), kMove5h4c},
            {Move(k6, kH, k5, kC), kMove6h5c},
            {Move(k7, kH, k6, kC), kMove7h6c},
            {Move(k8, kH, k7, kC), kMove8h7c},
            {Move(k9, kH, k8, kC), kMove9h8c},
            {Move(kT, kH, k9, kC), kMoveTh9c},
            {Move(kJ, kH, kT, kC), kMoveJhTc},
            {Move(kQ, kH, kJ, kC), kMoveQhJc},
            {Move(kK, kH, kQ, kC), kMoveKhQc},
            // endregion
            
            // Clubs ---------------------------------------------------------------------------------------------------
            
            // region Moves to Tableau (Clubs <- Hearts)
            {Move(k2, kC, kA, kH), kMove2cAh},
            {Move(k3, kC, k2, kH), kMove3c2h},
            {Move(k4, kC, k3, kH), kMove4c3h},
            {Move(k5, kC, k4, kH), kMove5c4h},
            {Move(k6, kC, k5, kH), kMove6c5h},
            {Move(k7, kC, k6, kH), kMove7c6h},
            {Move(k8, kC, k7, kH), kMove8c7h},
            {Move(k9, kC, k8, kH), kMove9c8h},
            {Move(kT, kC, k9, kH), kMoveTc9h},
            {Move(kJ, kC, kT, kH), kMoveJcTh},
            {Move(kQ, kC, kJ, kH), kMoveQcJh},
            {Move(kK, kC, kQ, kH), kMoveKcQh},
            // endregion

            // region Moves to Tableau (Clubs <- Diamonds)
            {Move(k2, kC, kA, kD), kMove2cAd},
            {Move(k3, kC, k2, kD), kMove3c2d},
            {Move(k4, kC, k3, kD), kMove4c3d},
            {Move(k5, kC, k4, kD), kMove5c4d},
            {Move(k6, kC, k5, kD), kMove6c5d},
            {Move(k7, kC, k6, kD), kMove7c6d},
            {Move(k8, kC, k7, kD), kMove8c7d},
            {Move(k9, kC, k8, kD), kMove9c8d},
            {Move(kT, kC, k9, kD), kMoveTc9d},
            {Move(kJ, kC, kT, kD), kMoveJcTd},
            {Move(kQ, kC, kJ, kD), kMoveQcJd},
            {Move(kK, kC, kQ, kD), kMoveKcQd},
            // endregion

            // Diamonds ------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Diamonds <- Spades)
            {Move(k2, kD, kA, kS), kMove2dAs},
            {Move(k3, kD, k2, kS), kMove3d2s},
            {Move(k4, kD, k3, kS), kMove4d3s},
            {Move(k5, kD, k4, kS), kMove5d4s},
            {Move(k6, kD, k5, kS), kMove6d5s},
            {Move(k7, kD, k6, kS), kMove7d6s},
            {Move(k8, kD, k7, kS), kMove8d7s},
            {Move(k9, kD, k8, kS), kMove9d8s},
            {Move(kT, kD, k9, kS), kMoveTd9s},
            {Move(kJ, kD, kT, kS), kMoveJdTs},
            {Move(kQ, kD, kJ, kS), kMoveQdJs},
            {Move(kK, kD, kQ, kS), kMoveKdQs},
            // endregion

            // region Moves to Tableau (Diamonds <- Clubs)
            {Move(k2, kD, kA, kC), kMove2dAc},
            {Move(k3, kD, k2, kC), kMove3d2c},
            {Move(k4, kD, k3, kC), kMove4d3c},
            {Move(k5, kD, k4, kC), kMove5d4c},
            {Move(k6, kD, k5, kC), kMove6d5c},
            {Move(k7, kD, k6, kC), kMove7d6c},
            {Move(k8, kD, k7, kC), kMove8d7c},
            {Move(k9, kD, k8, kC), kMove9d8c},
            {Move(kT, kD, k9, kC), kMoveTd9c},
            {Move(kJ, kD, kT, kC), kMoveJdTc},
            {Move(kQ, kD, kJ, kC), kMoveQdJc},
            {Move(kK, kD, kQ, kC), kMoveKdQc},
            // endregion
            
            // endregion
    };

    const std::map<Action, Move> ACTION_TO_MOVE = {
            // region Mapping of an action declared in ActionType to a move;

            // region Moves To Empty Tableau
            {kMove__Ks, Move(kNoRank, kNoSuit, kK, kS)},
            {kMove__Kh, Move(kNoRank, kNoSuit, kK, kH)},
            {kMove__Kc, Move(kNoRank, kNoSuit, kK, kC)},
            {kMove__Kd, Move(kNoRank, kNoSuit, kK, kD)},
            // endregion

            // region Moves To Empty Foundation
            {kMove__As, Move(kNoRank, kS, kA, kS)},
            {kMove__Ah, Move(kNoRank, kH, kA, kH)},
            {kMove__Ac, Move(kNoRank, kC, kA, kC)},
            {kMove__Ad, Move(kNoRank, kD, kA, kD)},
            // endregion

            // region Moves to Foundation (To Spades)
            {kMoveAs2s, Move(kA, kS, k2, kS)},
            {kMove2s3s, Move(k2, kS, k3, kS)},
            {kMove3s4s, Move(k3, kS, k4, kS)},
            {kMove4s5s, Move(k4, kS, k5, kS)},
            {kMove5s6s, Move(k5, kS, k6, kS)},
            {kMove6s7s, Move(k6, kS, k7, kS)},
            {kMove7s8s, Move(k7, kS, k8, kS)},
            {kMove8s9s, Move(k8, kS, k9, kS)},
            {kMove9sTs, Move(k9, kS, kT, kS)},
            {kMoveTsJs, Move(kT, kS, kJ, kS)},
            {kMoveJsQs, Move(kJ, kS, kQ, kS)},
            {kMoveQsKs, Move(kQ, kS, kK, kS)},
            // endregion

            // region Moves to Foundation (To Hearts)
            {kMoveAh2h, Move(kA, kH, k2, kH)},
            {kMove2h3h, Move(k2, kH, k3, kH)},
            {kMove3h4h, Move(k3, kH, k4, kH)},
            {kMove4h5h, Move(k4, kH, k5, kH)},
            {kMove5h6h, Move(k5, kH, k6, kH)},
            {kMove6h7h, Move(k6, kH, k7, kH)},
            {kMove7h8h, Move(k7, kH, k8, kH)},
            {kMove8h9h, Move(k8, kH, k9, kH)},
            {kMove9hTh, Move(k9, kH, kT, kH)},
            {kMoveThJh, Move(kT, kH, kJ, kH)},
            {kMoveJhQh, Move(kJ, kH, kQ, kH)},
            {kMoveQhKh, Move(kQ, kH, kK, kH)},
            // endregion

            // region Moves to Foundation (To Clubs)
            {kMoveAc2c, Move(kA, kC, k2, kC)},
            {kMove2c3c, Move(k2, kC, k3, kC)},
            {kMove3c4c, Move(k3, kC, k4, kC)},
            {kMove4c5c, Move(k4, kC, k5, kC)},
            {kMove5c6c, Move(k5, kC, k6, kC)},
            {kMove6c7c, Move(k6, kC, k7, kC)},
            {kMove7c8c, Move(k7, kC, k8, kC)},
            {kMove8c9c, Move(k8, kC, k9, kC)},
            {kMove9cTc, Move(k9, kC, kT, kC)},
            {kMoveTcJc, Move(kT, kC, kJ, kC)},
            {kMoveJcQc, Move(kJ, kC, kQ, kC)},
            {kMoveQcKc, Move(kQ, kC, kK, kC)},
            // endregion

            // region Moves to Foundation (To Diamonds)
            {kMoveAd2d, Move(kA, kD, k2, kD)},
            {kMove2d3d, Move(k2, kD, k3, kD)},
            {kMove3d4d, Move(k3, kD, k4, kD)},
            {kMove4d5d, Move(k4, kD, k5, kD)},
            {kMove5d6d, Move(k5, kD, k6, kD)},
            {kMove6d7d, Move(k6, kD, k7, kD)},
            {kMove7d8d, Move(k7, kD, k8, kD)},
            {kMove8d9d, Move(k8, kD, k9, kD)},
            {kMove9dTd, Move(k9, kD, kT, kD)},
            {kMoveTdJd, Move(kT, kD, kJ, kD)},
            {kMoveJdQd, Move(kJ, kD, kQ, kD)},
            {kMoveQdKd, Move(kQ, kD, kK, kD)},
            // endregion

            // Spades --------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Spades <- Hearts)
            {kMove2sAh, Move(k2, kS, kA, kH)},
            {kMove3s2h, Move(k3, kS, k2, kH)},
            {kMove4s3h, Move(k4, kS, k3, kH)},
            {kMove5s4h, Move(k5, kS, k4, kH)},
            {kMove6s5h, Move(k6, kS, k5, kH)},
            {kMove7s6h, Move(k7, kS, k6, kH)},
            {kMove8s7h, Move(k8, kS, k7, kH)},
            {kMove9s8h, Move(k9, kS, k8, kH)},
            {kMoveTs9h, Move(kT, kS, k9, kH)},
            {kMoveJsTh, Move(kJ, kS, kT, kH)},
            {kMoveQsJh, Move(kQ, kS, kJ, kH)},
            {kMoveKsQh, Move(kK, kS, kQ, kH)},
            // endregion

            // region Moves to Tableau (Spades <- Diamonds)
            {kMove2sAd, Move(k2, kS, kA, kD)},
            {kMove3s2d, Move(k3, kS, k2, kD)},
            {kMove4s3d, Move(k4, kS, k3, kD)},
            {kMove5s4d, Move(k5, kS, k4, kD)},
            {kMove6s5d, Move(k6, kS, k5, kD)},
            {kMove7s6d, Move(k7, kS, k6, kD)},
            {kMove8s7d, Move(k8, kS, k7, kD)},
            {kMove9s8d, Move(k9, kS, k8, kD)},
            {kMoveTs9d, Move(kT, kS, k9, kD)},
            {kMoveJsTd, Move(kJ, kS, kT, kD)},
            {kMoveQsJd, Move(kQ, kS, kJ, kD)},
            {kMoveKsQd, Move(kK, kS, kQ, kD)},
            // endregion

            // Hearts --------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Hearts <- Spades)
            {kMove2hAs, Move(k2, kH, kA, kS)},
            {kMove3h2s, Move(k3, kH, k2, kS)},
            {kMove4h3s, Move(k4, kH, k3, kS)},
            {kMove5h4s, Move(k5, kH, k4, kS)},
            {kMove6h5s, Move(k6, kH, k5, kS)},
            {kMove7h6s, Move(k7, kH, k6, kS)},
            {kMove8h7s, Move(k8, kH, k7, kS)},
            {kMove9h8s, Move(k9, kH, k8, kS)},
            {kMoveTh9s, Move(kT, kH, k9, kS)},
            {kMoveJhTs, Move(kJ, kH, kT, kS)},
            {kMoveQhJs, Move(kQ, kH, kJ, kS)},
            {kMoveKhQs, Move(kK, kH, kQ, kS)},
            // endregion

            // region Moves to Tableau (Hearts <- Clubs)
            {kMove2hAc, Move(k2, kH, kA, kC)},
            {kMove3h2c, Move(k3, kH, k2, kC)},
            {kMove4h3c, Move(k4, kH, k3, kC)},
            {kMove5h4c, Move(k5, kH, k4, kC)},
            {kMove6h5c, Move(k6, kH, k5, kC)},
            {kMove7h6c, Move(k7, kH, k6, kC)},
            {kMove8h7c, Move(k8, kH, k7, kC)},
            {kMove9h8c, Move(k9, kH, k8, kC)},
            {kMoveTh9c, Move(kT, kH, k9, kC)},
            {kMoveJhTc, Move(kJ, kH, kT, kC)},
            {kMoveQhJc, Move(kQ, kH, kJ, kC)},
            {kMoveKhQc, Move(kK, kH, kQ, kC)},
            // endregion

            // Clubs ---------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Clubs <- Hearts)
            {kMove2cAh, Move(k2, kC, kA, kH)},
            {kMove3c2h, Move(k3, kC, k2, kH)},
            {kMove4c3h, Move(k4, kC, k3, kH)},
            {kMove5c4h, Move(k5, kC, k4, kH)},
            {kMove6c5h, Move(k6, kC, k5, kH)},
            {kMove7c6h, Move(k7, kC, k6, kH)},
            {kMove8c7h, Move(k8, kC, k7, kH)},
            {kMove9c8h, Move(k9, kC, k8, kH)},
            {kMoveTc9h, Move(kT, kC, k9, kH)},
            {kMoveJcTh, Move(kJ, kC, kT, kH)},
            {kMoveQcJh, Move(kQ, kC, kJ, kH)},
            {kMoveKcQh, Move(kK, kC, kQ, kH)},
            // endregion

            // region Moves to Tableau (Clubs <- Diamonds)
            {kMove2cAd, Move(k2, kC, kA, kD)},
            {kMove3c2d, Move(k3, kC, k2, kD)},
            {kMove4c3d, Move(k4, kC, k3, kD)},
            {kMove5c4d, Move(k5, kC, k4, kD)},
            {kMove6c5d, Move(k6, kC, k5, kD)},
            {kMove7c6d, Move(k7, kC, k6, kD)},
            {kMove8c7d, Move(k8, kC, k7, kD)},
            {kMove9c8d, Move(k9, kC, k8, kD)},
            {kMoveTc9d, Move(kT, kC, k9, kD)},
            {kMoveJcTd, Move(kJ, kC, kT, kD)},
            {kMoveQcJd, Move(kQ, kC, kJ, kD)},
            {kMoveKcQd, Move(kK, kC, kQ, kD)},
            // endregion

            // Diamonds ------------------------------------------------------------------------------------------------

            // region Moves to Tableau (Diamonds <- Spades)
            {kMove2dAs, Move(k2, kD, kA, kS)},
            {kMove3d2s, Move(k3, kD, k2, kS)},
            {kMove4d3s, Move(k4, kD, k3, kS)},
            {kMove5d4s, Move(k5, kD, k4, kS)},
            {kMove6d5s, Move(k6, kD, k5, kS)},
            {kMove7d6s, Move(k7, kD, k6, kS)},
            {kMove8d7s, Move(k8, kD, k7, kS)},
            {kMove9d8s, Move(k9, kD, k8, kS)},
            {kMoveTd9s, Move(kT, kD, k9, kS)},
            {kMoveJdTs, Move(kJ, kD, kT, kS)},
            {kMoveQdJs, Move(kQ, kD, kJ, kS)},
            {kMoveKdQs, Move(kK, kD, kQ, kS)},
            // endregion

            // region Moves to Tableau (Diamonds <- Clubs)
            {kMove2dAc, Move(k2, kD, kA, kC)},
            {kMove3d2c, Move(k3, kD, k2, kC)},
            {kMove4d3c, Move(k4, kD, k3, kC)},
            {kMove5d4c, Move(k5, kD, k4, kC)},
            {kMove6d5c, Move(k6, kD, k5, kC)},
            {kMove7d6c, Move(k7, kD, k6, kC)},
            {kMove8d7c, Move(k8, kD, k7, kC)},
            {kMove9d8c, Move(k9, kD, k8, kC)},
            {kMoveTd9c, Move(kT, kD, k9, kC)},
            {kMoveJdTc, Move(kJ, kD, kT, kC)},
            {kMoveQdJc, Move(kQ, kD, kJ, kC)},
            {kMoveKdQc, Move(kK, kD, kQ, kC)},
            // endregion

            // endregion
    };

    // OpenSpiel Classes ===============================================================================================

    class SolitaireGame;

    class SolitaireState : public State {
    public:

        // Attributes ==================================================================================================

        Pile                waste;
        std::vector<Pile>   foundations;
        std::vector<Pile>   tableaus;
        std::vector<Action> revealed_cards;

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
        void                    InformationStateTensor(Player player, std::vector<double> * values) const override;
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
        LocationType            FindLocation(const Card & card) const;
        Pile *                  GetPile(const Card &card) const;
        void                    MoveCards(const Move & move);

    private:
        bool   is_finished    = false;
        bool   is_reversible  = false;
        int    current_depth  = 0;
        double previous_score = 0.0;
        std::set<std::size_t> previous_states = {};
        std::map<Card, PileID> card_map;

        double current_returns  = 0.0;
        double current_rewards  = 0.0;
        double previous_rewards = 0.0;
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

        std::vector<int> InformationStateTensorShape() const override;
        std::vector<int> ObservationTensorShape() const override;

        std::unique_ptr<State> NewInitialState() const override;
        std::shared_ptr<const Game> Clone() const override;

    private:
        int num_players_;
        int depth_limit_;
        bool is_colored_;
    };

} // namespace open_spiel::solitaire

#endif // THIRD_PARTY_OPEN_SPIEL_GAMES_SOLITAIRE_H
