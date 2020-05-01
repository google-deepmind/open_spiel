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
#include "open_spiel/spiel.h"

#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

namespace open_spiel::solitaire {

    // Sets default number of players
    inline constexpr int    kDefaultPlayers = 1;

    // Special "card" indices used in ObservationTensor
    inline constexpr double HIDDEN_CARD = 98.0;
    inline constexpr double NO_CARD     = 99.0;

    template <typename Container, typename Element>
    int GetIndex (Container container, Element element) {
        return std::distance(std::begin(container), std::find(container.begin(), container.end(), element));
    }

    const std::vector<std::string> SUITS = {"s", "h", "c", "d"};
    const std::vector<std::string> RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"};
    const std::map<std::string, double> FOUNDATION_POINTS = {
            //region Reward for move to the foundation with a source card of this rank
            {"A", 100.0},
            {"2", 90.0},
            {"3", 80.0},
            {"4", 70.0},
            {"5", 60.0},
            {"6", 50.0},
            {"7", 40.0},
            {"8", 30.0},
            {"9", 20.0},
            {"T", 10.0},
            {"J", 10.0},
            {"Q", 10.0},
            {"K", 10.0}
            //endregion
    };

    // Enumerations ====================================================================================================

    enum ActionType {

        // Setup Action (1) ============================================================================================
        kSetup = 0,
        
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
        
        // Draw Action (1) =============================================================================================
        kDraw = 53,
        
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

    enum Location {
        kDeck       = 0,
        kWaste      = 1,
        kFoundation = 2,
        kTableau    = 3,
        kMissing    = 4,
    };

    // Support Classes =================================================================================================

    class Card {
    public:

        // Attributes ==================================================================================================

        std::string rank;       // Indicates the rank of the card, cannot be changed once set
        std::string suit;       // Indicates the suit of the card, cannot be changed once set
        bool        hidden;     // Indicates whether the card is hidden or not
        Location    location;   // Indicates the type of pile the card is in

        // Constructors ================================================================================================

        Card();                                     // Create an empty card, default constructor
        Card(std::string rank, std::string suit);   // Create a card from rank, suit, and hidden
        explicit Card(int index);                   // Create a card from its index (e.g. 0 -> As)

        // Type Casting ================================================================================================

        explicit operator int() const;                    // Represent a card as its integer index (e.g. As -> 0)

        // Operators ===================================================================================================

        bool operator==(Card & other_card) const;          // Compare two cards for equality
        bool operator==(const Card & other_card) const;    // Compare two cards for equality

        // Other Methods ===============================================================================================

        std::vector<Card> LegalChildren() const;    // Get legal children of the card depending on its location
        std::string ToString() const;               // Gets human-readable representation of the card as a string

    };

    class Deck {
    public:

        // Attributes ==================================================================================================

        std::deque<Card> cards;             // Holds the card current in the deck
        std::deque<Card> waste;             // Holds the waste cards, the top of which can be played
        std::deque<Card> initial_order;     // Holds the initial order of the deck, so that it can be rebuilt
        int times_rebuilt = 0;              // Number of times Rebuild() is called, used for score or terminality

        // Constructors ================================================================================================

        Deck();     // Default constructor

        // Other Methods ===============================================================================================

        std::vector<Card> Sources() const;      // Returns vector containing the top card of the waste pile
        std::vector<Card> Split(Card card);     // Returns split card from waste
        void draw(unsigned long num_cards);     // Moves cards to the waste
        void rebuild();                         // Repopulates the deck in the order cards were originally drawn

    };

    class Foundation {
    public:

        // Attributes ==================================================================================================

        const std::string suit;                                // Indicates the suit of cards that can be added
        std::deque<Card>  cards;                               // Contains the cards inside the foundation

        // Constructors ================================================================================================

        Foundation();                                          // Default constructor
        explicit Foundation(std::string suit);                 // Construct an empty foundation of a given suit

        // Other Methods ===============================================================================================

        std::vector<Card> Sources() const;                     // Cards in the foundation that can be moved
        std::vector<Card> Targets() const;                     // A card in the foundation that can have cards moved to it

        std::vector<Card> Split(Card card);                    // Splits on given card and returns it and all cards beneath it
        void Extend(const std::vector<Card>& source_cards);    // Adds cards to the foundation

    };

    class Tableau {
    public:

        // Attributes ==================================================================================================

        std::deque<Card> cards;                               // Contains the cards inside the foundation

        // Constructors ================================================================================================

        Tableau();                                            // Default constructor
        explicit Tableau(int num_cards);                      // Construct a tableau with the given cards

        // Other Methods ===============================================================================================

        std::vector<Card> Sources() const;                    // Cards in the foundation that can be moved
        std::vector<Card> Targets() const;                    // A card in the foundation that can have cards moved to it

        std::vector<Card> Split(Card card);                   // Splits on given card and returns it and all cards beneath it
        void Extend(const std::vector<Card>& source_cards);   // Adds cards to the foundation
    };

    class Move {
    public:

        // Attributes ==================================================================================================

        Card target;    // The card that the source will be moved to
        Card source;    // The card that will be moved to the target

        // Constructors ================================================================================================

        Move(Card target_card, Card source_card);   // Creates Move object from target and source cards
        explicit Move(Action action_id);            // Creates Move object from Action kMove... (54 -> 206)

        // Other Methods ===============================================================================================

        std::string ToString() const;       // Gets human-readable representation of the move as a string
        Action      ActionId() const;       // Gets Action kMove... from the Move object


    };

    const std::map<std::pair<std::string, std::string>, int> RANKSUIT_TO_INDEX = {
            //region Mapping of a pair of rank and suit to a card index

            // Special Cards
            {std::pair<std::string, std::string>("", "s"), -1},
            {std::pair<std::string, std::string>("", "h"), -2},
            {std::pair<std::string, std::string>("", "c"), -3},
            {std::pair<std::string, std::string>("", "d"), -4},
            {std::pair<std::string, std::string>("", ""),  -5},

            // Spades
            {std::pair<std::string, std::string>("A", "s"), 0},
            {std::pair<std::string, std::string>("2", "s"), 1},
            {std::pair<std::string, std::string>("3", "s"), 2},
            {std::pair<std::string, std::string>("4", "s"), 3},
            {std::pair<std::string, std::string>("5", "s"), 4},
            {std::pair<std::string, std::string>("6", "s"), 5},
            {std::pair<std::string, std::string>("7", "s"), 6},
            {std::pair<std::string, std::string>("8", "s"), 7},
            {std::pair<std::string, std::string>("9", "s"), 8},
            {std::pair<std::string, std::string>("T", "s"), 9},
            {std::pair<std::string, std::string>("J", "s"), 10},
            {std::pair<std::string, std::string>("Q", "s"), 11},
            {std::pair<std::string, std::string>("K", "s"), 12},

            // Hearts
            {std::pair<std::string, std::string>("A", "h"), 13},
            {std::pair<std::string, std::string>("2", "h"), 14},
            {std::pair<std::string, std::string>("3", "h"), 15},
            {std::pair<std::string, std::string>("4", "h"), 16},
            {std::pair<std::string, std::string>("5", "h"), 17},
            {std::pair<std::string, std::string>("6", "h"), 18},
            {std::pair<std::string, std::string>("7", "h"), 19},
            {std::pair<std::string, std::string>("8", "h"), 20},
            {std::pair<std::string, std::string>("9", "h"), 21},
            {std::pair<std::string, std::string>("T", "h"), 22},
            {std::pair<std::string, std::string>("J", "h"), 23},
            {std::pair<std::string, std::string>("Q", "h"), 24},
            {std::pair<std::string, std::string>("K", "h"), 25},

            // Clubs
            {std::pair<std::string, std::string>("A", "c"), 26},
            {std::pair<std::string, std::string>("2", "c"), 27},
            {std::pair<std::string, std::string>("3", "c"), 28},
            {std::pair<std::string, std::string>("4", "c"), 29},
            {std::pair<std::string, std::string>("5", "c"), 30},
            {std::pair<std::string, std::string>("6", "c"), 31},
            {std::pair<std::string, std::string>("7", "c"), 32},
            {std::pair<std::string, std::string>("8", "c"), 33},
            {std::pair<std::string, std::string>("9", "c"), 34},
            {std::pair<std::string, std::string>("T", "c"), 35},
            {std::pair<std::string, std::string>("J", "c"), 36},
            {std::pair<std::string, std::string>("Q", "c"), 37},
            {std::pair<std::string, std::string>("K", "c"), 38},

            // Diamonds
            {std::pair<std::string, std::string>("A", "d"), 39},
            {std::pair<std::string, std::string>("2", "d"), 40},
            {std::pair<std::string, std::string>("3", "d"), 41},
            {std::pair<std::string, std::string>("4", "d"), 42},
            {std::pair<std::string, std::string>("5", "d"), 43},
            {std::pair<std::string, std::string>("6", "d"), 44},
            {std::pair<std::string, std::string>("7", "d"), 45},
            {std::pair<std::string, std::string>("8", "d"), 46},
            {std::pair<std::string, std::string>("9", "d"), 47},
            {std::pair<std::string, std::string>("T", "d"), 48},
            {std::pair<std::string, std::string>("J", "d"), 49},
            {std::pair<std::string, std::string>("Q", "d"), 50},
            {std::pair<std::string, std::string>("K", "d"), 51},
            //endregion
    };

    const std::map<std::pair<int, int>, Action> MOVE_TO_ACTION = {
            //region Mapping of std::pair<int, int> where ints are card indices to an action declared in ActionType

            // Special Moves ===========================================================================================
            // To Empty Tableau ----------------------------------------------------------------------------------------
            {{(int) Card("", ""), (int) Card("K", "s")}, kMove__Ks},
            {{(int) Card("", ""), (int) Card("K", "h")}, kMove__Kh},
            {{(int) Card("", ""), (int) Card("K", "c")}, kMove__Kc},
            {{(int) Card("", ""), (int) Card("K", "d")}, kMove__Kd},

            // To Empty Foundation -------------------------------------------------------------------------------------
            {{(int) Card("", "h"), (int) Card("A", "h")}, kMove__Ah},
            {{(int) Card("", "s"), (int) Card("A", "s")}, kMove__As},
            {{(int) Card("", "c"), (int) Card("A", "c")}, kMove__Ac},
            {{(int) Card("", "d"), (int) Card("A", "d")}, kMove__Ad},

            // Foundation Moves ========================================================================================
            // To Spades -----------------------------------------------------------------------------------------------
            {{(int) Card("A", "s"), (int) Card("2", "s")}, kMoveAs2s},
            {{(int) Card("2", "s"), (int) Card("3", "s")}, kMove2s3s},
            {{(int) Card("3", "s"), (int) Card("4", "s")}, kMove3s4s},
            {{(int) Card("4", "s"), (int) Card("5", "s")}, kMove4s5s},
            {{(int) Card("5", "s"), (int) Card("6", "s")}, kMove5s6s},
            {{(int) Card("6", "s"), (int) Card("7", "s")}, kMove6s7s},
            {{(int) Card("7", "s"), (int) Card("8", "s")}, kMove7s8s},
            {{(int) Card("8", "s"), (int) Card("9", "s")}, kMove8s9s},
            {{(int) Card("9", "s"), (int) Card("T", "s")}, kMove9sTs},
            {{(int) Card("T", "s"), (int) Card("J", "s")}, kMoveTsJs},
            {{(int) Card("J", "s"), (int) Card("Q", "s")}, kMoveJsQs},
            {{(int) Card("Q", "s"), (int) Card("K", "s")}, kMoveQsKs},

            // To Hearts -----------------------------------------------------------------------------------------------
            {{(int) Card("A", "h"), (int) Card("2", "h")}, kMoveAh2h},
            {{(int) Card("2", "h"), (int) Card("3", "h")}, kMove2h3h},
            {{(int) Card("3", "h"), (int) Card("4", "h")}, kMove3h4h},
            {{(int) Card("4", "h"), (int) Card("5", "h")}, kMove4h5h},
            {{(int) Card("5", "h"), (int) Card("6", "h")}, kMove5h6h},
            {{(int) Card("6", "h"), (int) Card("7", "h")}, kMove6h7h},
            {{(int) Card("7", "h"), (int) Card("8", "h")}, kMove7h8h},
            {{(int) Card("8", "h"), (int) Card("9", "h")}, kMove8h9h},
            {{(int) Card("9", "h"), (int) Card("T", "h")}, kMove9hTh},
            {{(int) Card("T", "h"), (int) Card("J", "h")}, kMoveThJh},
            {{(int) Card("J", "h"), (int) Card("Q", "h")}, kMoveJhQh},
            {{(int) Card("Q", "h"), (int) Card("K", "h")}, kMoveQhKh},

            // To Clubs ------------------------------------------------------------------------------------------------
            {{(int) Card("A", "c"), (int) Card("2", "c")}, kMoveAc2c},
            {{(int) Card("2", "c"), (int) Card("3", "c")}, kMove2c3c},
            {{(int) Card("3", "c"), (int) Card("4", "c")}, kMove3c4c},
            {{(int) Card("4", "c"), (int) Card("5", "c")}, kMove4c5c},
            {{(int) Card("5", "c"), (int) Card("6", "c")}, kMove5c6c},
            {{(int) Card("6", "c"), (int) Card("7", "c")}, kMove6c7c},
            {{(int) Card("7", "c"), (int) Card("8", "c")}, kMove7c8c},
            {{(int) Card("8", "c"), (int) Card("9", "c")}, kMove8c9c},
            {{(int) Card("9", "c"), (int) Card("T", "c")}, kMove9cTc},
            {{(int) Card("T", "c"), (int) Card("J", "c")}, kMoveTcJc},
            {{(int) Card("J", "c"), (int) Card("Q", "c")}, kMoveJcQc},
            {{(int) Card("Q", "c"), (int) Card("K", "c")}, kMoveQcKc},

            // To Diamonds ---------------------------------------------------------------------------------------------
            {{(int) Card("A", "d"), (int) Card("2", "d")}, kMoveAd2d},
            {{(int) Card("2", "d"), (int) Card("3", "d")}, kMove2d3d},
            {{(int) Card("3", "d"), (int) Card("4", "d")}, kMove3d4d},
            {{(int) Card("4", "d"), (int) Card("5", "d")}, kMove4d5d},
            {{(int) Card("5", "d"), (int) Card("6", "d")}, kMove5d6d},
            {{(int) Card("6", "d"), (int) Card("7", "d")}, kMove6d7d},
            {{(int) Card("7", "d"), (int) Card("8", "d")}, kMove7d8d},
            {{(int) Card("8", "d"), (int) Card("9", "d")}, kMove8d9d},
            {{(int) Card("9", "d"), (int) Card("T", "d")}, kMove9dTd},
            {{(int) Card("T", "d"), (int) Card("J", "d")}, kMoveTdJd},
            {{(int) Card("J", "d"), (int) Card("Q", "d")}, kMoveJdQd},
            {{(int) Card("Q", "d"), (int) Card("K", "d")}, kMoveQdKd},

            // Tableau Moves ===========================================================================================
            // To Spades -----------------------------------------------------------------------------------------------
            {{(int) Card("2", "s"), (int) Card("A", "h")}, kMove2sAh},
            {{(int) Card("3", "s"), (int) Card("2", "h")}, kMove3s2h},
            {{(int) Card("4", "s"), (int) Card("3", "h")}, kMove4s3h},
            {{(int) Card("5", "s"), (int) Card("4", "h")}, kMove5s4h},
            {{(int) Card("6", "s"), (int) Card("5", "h")}, kMove6s5h},
            {{(int) Card("7", "s"), (int) Card("6", "h")}, kMove7s6h},
            {{(int) Card("8", "s"), (int) Card("7", "h")}, kMove8s7h},
            {{(int) Card("9", "s"), (int) Card("8", "h")}, kMove9s8h},
            {{(int) Card("T", "s"), (int) Card("9", "h")}, kMoveTs9h},
            {{(int) Card("J", "s"), (int) Card("T", "h")}, kMoveJsTh},
            {{(int) Card("Q", "s"), (int) Card("J", "h")}, kMoveQsJh},
            {{(int) Card("K", "s"), (int) Card("Q", "h")}, kMoveKsQh},
            {{(int) Card("2", "s"), (int) Card("A", "d")}, kMove2sAd},
            {{(int) Card("3", "s"), (int) Card("2", "d")}, kMove3s2d},
            {{(int) Card("4", "s"), (int) Card("3", "d")}, kMove4s3d},
            {{(int) Card("5", "s"), (int) Card("4", "d")}, kMove5s4d},
            {{(int) Card("6", "s"), (int) Card("5", "d")}, kMove6s5d},
            {{(int) Card("7", "s"), (int) Card("6", "d")}, kMove7s6d},
            {{(int) Card("8", "s"), (int) Card("7", "d")}, kMove8s7d},
            {{(int) Card("9", "s"), (int) Card("8", "d")}, kMove9s8d},
            {{(int) Card("T", "s"), (int) Card("9", "d")}, kMoveTs9d},
            {{(int) Card("J", "s"), (int) Card("T", "d")}, kMoveJsTd},
            {{(int) Card("Q", "s"), (int) Card("J", "d")}, kMoveQsJd},
            {{(int) Card("K", "s"), (int) Card("Q", "d")}, kMoveKsQd},

            // To Hearts -----------------------------------------------------------------------------------------------
            {{(int) Card("2", "h"), (int) Card("A", "s")}, kMove2hAs},
            {{(int) Card("3", "h"), (int) Card("2", "s")}, kMove3h2s},
            {{(int) Card("4", "h"), (int) Card("3", "s")}, kMove4h3s},
            {{(int) Card("5", "h"), (int) Card("4", "s")}, kMove5h4s},
            {{(int) Card("6", "h"), (int) Card("5", "s")}, kMove6h5s},
            {{(int) Card("7", "h"), (int) Card("6", "s")}, kMove7h6s},
            {{(int) Card("8", "h"), (int) Card("7", "s")}, kMove8h7s},
            {{(int) Card("9", "h"), (int) Card("8", "s")}, kMove9h8s},
            {{(int) Card("T", "h"), (int) Card("9", "s")}, kMoveTh9s},
            {{(int) Card("J", "h"), (int) Card("T", "s")}, kMoveJhTs},
            {{(int) Card("Q", "h"), (int) Card("J", "s")}, kMoveQhJs},
            {{(int) Card("K", "h"), (int) Card("Q", "s")}, kMoveKhQs},
            {{(int) Card("2", "h"), (int) Card("A", "c")}, kMove2hAc},
            {{(int) Card("3", "h"), (int) Card("2", "c")}, kMove3h2c},
            {{(int) Card("4", "h"), (int) Card("3", "c")}, kMove4h3c},
            {{(int) Card("5", "h"), (int) Card("4", "c")}, kMove5h4c},
            {{(int) Card("6", "h"), (int) Card("5", "c")}, kMove6h5c},
            {{(int) Card("7", "h"), (int) Card("6", "c")}, kMove7h6c},
            {{(int) Card("8", "h"), (int) Card("7", "c")}, kMove8h7c},
            {{(int) Card("9", "h"), (int) Card("8", "c")}, kMove9h8c},
            {{(int) Card("T", "h"), (int) Card("9", "c")}, kMoveTh9c},
            {{(int) Card("J", "h"), (int) Card("T", "c")}, kMoveJhTc},
            {{(int) Card("Q", "h"), (int) Card("J", "c")}, kMoveQhJc},
            {{(int) Card("K", "h"), (int) Card("Q", "c")}, kMoveKhQc},

            // To Clubs ------------------------------------------------------------------------------------------------
            {{(int) Card("2", "c"), (int) Card("A", "h")}, kMove2cAh},
            {{(int) Card("3", "c"), (int) Card("2", "h")}, kMove3c2h},
            {{(int) Card("4", "c"), (int) Card("3", "h")}, kMove4c3h},
            {{(int) Card("5", "c"), (int) Card("4", "h")}, kMove5c4h},
            {{(int) Card("6", "c"), (int) Card("5", "h")}, kMove6c5h},
            {{(int) Card("7", "c"), (int) Card("6", "h")}, kMove7c6h},
            {{(int) Card("8", "c"), (int) Card("7", "h")}, kMove8c7h},
            {{(int) Card("9", "c"), (int) Card("8", "h")}, kMove9c8h},
            {{(int) Card("T", "c"), (int) Card("9", "h")}, kMoveTc9h},
            {{(int) Card("J", "c"), (int) Card("T", "h")}, kMoveJcTh},
            {{(int) Card("Q", "c"), (int) Card("J", "h")}, kMoveQcJh},
            {{(int) Card("K", "c"), (int) Card("Q", "h")}, kMoveKcQh},
            {{(int) Card("2", "c"), (int) Card("A", "d")}, kMove2cAd},
            {{(int) Card("3", "c"), (int) Card("2", "d")}, kMove3c2d},
            {{(int) Card("4", "c"), (int) Card("3", "d")}, kMove4c3d},
            {{(int) Card("5", "c"), (int) Card("4", "d")}, kMove5c4d},
            {{(int) Card("6", "c"), (int) Card("5", "d")}, kMove6c5d},
            {{(int) Card("7", "c"), (int) Card("6", "d")}, kMove7c6d},
            {{(int) Card("8", "c"), (int) Card("7", "d")}, kMove8c7d},
            {{(int) Card("9", "c"), (int) Card("8", "d")}, kMove9c8d},
            {{(int) Card("T", "c"), (int) Card("9", "d")}, kMoveTc9d},
            {{(int) Card("J", "c"), (int) Card("T", "d")}, kMoveJcTd},
            {{(int) Card("Q", "c"), (int) Card("J", "d")}, kMoveQcJd},
            {{(int) Card("K", "c"), (int) Card("Q", "d")}, kMoveKcQd},

            // To Diamonds ---------------------------------------------------------------------------------------------
            {{(int) Card("2", "d"), (int) Card("A", "s")}, kMove2dAs},
            {{(int) Card("3", "d"), (int) Card("2", "s")}, kMove3d2s},
            {{(int) Card("4", "d"), (int) Card("3", "s")}, kMove4d3s},
            {{(int) Card("5", "d"), (int) Card("4", "s")}, kMove5d4s},
            {{(int) Card("6", "d"), (int) Card("5", "s")}, kMove6d5s},
            {{(int) Card("7", "d"), (int) Card("6", "s")}, kMove7d6s},
            {{(int) Card("8", "d"), (int) Card("7", "s")}, kMove8d7s},
            {{(int) Card("9", "d"), (int) Card("8", "s")}, kMove9d8s},
            {{(int) Card("T", "d"), (int) Card("9", "s")}, kMoveTd9s},
            {{(int) Card("J", "d"), (int) Card("T", "s")}, kMoveJdTs},
            {{(int) Card("Q", "d"), (int) Card("J", "s")}, kMoveQdJs},
            {{(int) Card("K", "d"), (int) Card("Q", "s")}, kMoveKdQs},
            {{(int) Card("2", "d"), (int) Card("A", "c")}, kMove2dAc},
            {{(int) Card("3", "d"), (int) Card("2", "c")}, kMove3d2c},
            {{(int) Card("4", "d"), (int) Card("3", "c")}, kMove4d3c},
            {{(int) Card("5", "d"), (int) Card("4", "c")}, kMove5d4c},
            {{(int) Card("6", "d"), (int) Card("5", "c")}, kMove6d5c},
            {{(int) Card("7", "d"), (int) Card("6", "c")}, kMove7d6c},
            {{(int) Card("8", "d"), (int) Card("7", "c")}, kMove8d7c},
            {{(int) Card("9", "d"), (int) Card("8", "c")}, kMove9d8c},
            {{(int) Card("T", "d"), (int) Card("9", "c")}, kMoveTd9c},
            {{(int) Card("J", "d"), (int) Card("T", "c")}, kMoveJdTc},
            {{(int) Card("Q", "d"), (int) Card("J", "c")}, kMoveQdJc},
            {{(int) Card("K", "d"), (int) Card("Q", "c")}, kMoveKdQc},
            //endregion
    };

    const std::map<Action, std::pair<int, int>> ACTION_TO_MOVE = {
            //region Mapping of Action to a std::pair<int, int>, representing target & source card indices

            // Special Moves ===============================================================================================
            // To Empty Tableau --------------------------------------------------------------------------------------------
            {kMove__Ks, {(int) Card("", ""), (int) Card("K", "s")}},
            {kMove__Kh, {(int) Card("", ""), (int) Card("K", "h")}},
            {kMove__Kc, {(int) Card("", ""), (int) Card("K", "c")}},
            {kMove__Kd, {(int) Card("", ""), (int) Card("K", "d")}},

            // To Empty Foundation -------------------------------------------------------------------------------------
            {kMove__Ah, {(int) Card("", "h"), (int) Card("A", "h")}},
            {kMove__As, {(int) Card("", "s"), (int) Card("A", "s")}},
            {kMove__Ac, {(int) Card("", "c"), (int) Card("A", "c")}}, // <-----
            {kMove__Ad, {(int) Card("", "d"), (int) Card("A", "d")}},

            // Foundation Moves ========================================================================================
            // To Spades -----------------------------------------------------------------------------------------------
            {kMoveAs2s, {(int) Card("A", "s"), (int) Card("2", "s")}},
            {kMove2s3s, {(int) Card("2", "s"), (int) Card("3", "s")}},
            {kMove3s4s, {(int) Card("3", "s"), (int) Card("4", "s")}},
            {kMove4s5s, {(int) Card("4", "s"), (int) Card("5", "s")}},
            {kMove5s6s, {(int) Card("5", "s"), (int) Card("6", "s")}},
            {kMove6s7s, {(int) Card("6", "s"), (int) Card("7", "s")}},
            {kMove7s8s, {(int) Card("7", "s"), (int) Card("8", "s")}},
            {kMove8s9s, {(int) Card("8", "s"), (int) Card("9", "s")}},
            {kMove9sTs, {(int) Card("9", "s"), (int) Card("T", "s")}},
            {kMoveTsJs, {(int) Card("T", "s"), (int) Card("J", "s")}},
            {kMoveJsQs, {(int) Card("J", "s"), (int) Card("Q", "s")}},
            {kMoveQsKs, {(int) Card("Q", "s"), (int) Card("K", "s")}},

            // To Hearts -----------------------------------------------------------------------------------------------
            {kMoveAh2h, {(int) Card("A", "h"), (int) Card("2", "h")}},
            {kMove2h3h, {(int) Card("2", "h"), (int) Card("3", "h")}},
            {kMove3h4h, {(int) Card("3", "h"), (int) Card("4", "h")}},
            {kMove4h5h, {(int) Card("4", "h"), (int) Card("5", "h")}},
            {kMove5h6h, {(int) Card("5", "h"), (int) Card("6", "h")}},
            {kMove6h7h, {(int) Card("6", "h"), (int) Card("7", "h")}},
            {kMove7h8h, {(int) Card("7", "h"), (int) Card("8", "h")}},
            {kMove8h9h, {(int) Card("8", "h"), (int) Card("9", "h")}},
            {kMove9hTh, {(int) Card("9", "h"), (int) Card("T", "h")}},
            {kMoveThJh, {(int) Card("T", "h"), (int) Card("J", "h")}},
            {kMoveJhQh, {(int) Card("J", "h"), (int) Card("Q", "h")}},
            {kMoveQhKh, {(int) Card("Q", "h"), (int) Card("K", "h")}},

            // To Clubs ------------------------------------------------------------------------------------------------
            {kMoveAc2c, {(int) Card("A", "c"), (int) Card("2", "c")}},
            {kMove2c3c, {(int) Card("2", "c"), (int) Card("3", "c")}},
            {kMove3c4c, {(int) Card("3", "c"), (int) Card("4", "c")}},
            {kMove4c5c, {(int) Card("4", "c"), (int) Card("5", "c")}},
            {kMove5c6c, {(int) Card("5", "c"), (int) Card("6", "c")}},
            {kMove6c7c, {(int) Card("6", "c"), (int) Card("7", "c")}},
            {kMove7c8c, {(int) Card("7", "c"), (int) Card("8", "c")}},
            {kMove8c9c, {(int) Card("8", "c"), (int) Card("9", "c")}},
            {kMove9cTc, {(int) Card("9", "c"), (int) Card("T", "c")}},
            {kMoveTcJc, {(int) Card("T", "c"), (int) Card("J", "c")}},
            {kMoveJcQc, {(int) Card("J", "c"), (int) Card("Q", "c")}},
            {kMoveQcKc, {(int) Card("Q", "c"), (int) Card("K", "c")}},

            // To Diamonds ---------------------------------------------------------------------------------------------
            {kMoveAd2d, {(int) Card("A", "d"), (int) Card("2", "d")}},
            {kMove2d3d, {(int) Card("2", "d"), (int) Card("3", "d")}},
            {kMove3d4d, {(int) Card("3", "d"), (int) Card("4", "d")}},
            {kMove4d5d, {(int) Card("4", "d"), (int) Card("5", "d")}},
            {kMove5d6d, {(int) Card("5", "d"), (int) Card("6", "d")}},
            {kMove6d7d, {(int) Card("6", "d"), (int) Card("7", "d")}},
            {kMove7d8d, {(int) Card("7", "d"), (int) Card("8", "d")}},
            {kMove8d9d, {(int) Card("8", "d"), (int) Card("9", "d")}},
            {kMove9dTd, {(int) Card("9", "d"), (int) Card("T", "d")}},
            {kMoveTdJd, {(int) Card("T", "d"), (int) Card("J", "d")}},
            {kMoveJdQd, {(int) Card("J", "d"), (int) Card("Q", "d")}},
            {kMoveQdKd, {(int) Card("Q", "d"), (int) Card("K", "d")}},

            // Tableau Moves ===========================================================================================
            // To Spades -----------------------------------------------------------------------------------------------
            {kMove2sAh, {(int) Card("2", "s"), (int) Card("A", "h")}},
            {kMove3s2h, {(int) Card("3", "s"), (int) Card("2", "h")}},
            {kMove4s3h, {(int) Card("4", "s"), (int) Card("3", "h")}},
            {kMove5s4h, {(int) Card("5", "s"), (int) Card("4", "h")}},
            {kMove6s5h, {(int) Card("6", "s"), (int) Card("5", "h")}},
            {kMove7s6h, {(int) Card("7", "s"), (int) Card("6", "h")}},
            {kMove8s7h, {(int) Card("8", "s"), (int) Card("7", "h")}},
            {kMove9s8h, {(int) Card("9", "s"), (int) Card("8", "h")}},
            {kMoveTs9h, {(int) Card("T", "s"), (int) Card("9", "h")}},
            {kMoveJsTh, {(int) Card("J", "s"), (int) Card("T", "h")}},
            {kMoveQsJh, {(int) Card("Q", "s"), (int) Card("J", "h")}},
            {kMoveKsQh, {(int) Card("K", "s"), (int) Card("Q", "h")}},
            {kMove2sAd, {(int) Card("2", "s"), (int) Card("A", "d")}},
            {kMove3s2d, {(int) Card("3", "s"), (int) Card("2", "d")}},
            {kMove4s3d, {(int) Card("4", "s"), (int) Card("3", "d")}},
            {kMove5s4d, {(int) Card("5", "s"), (int) Card("4", "d")}},
            {kMove6s5d, {(int) Card("6", "s"), (int) Card("5", "d")}},
            {kMove7s6d, {(int) Card("7", "s"), (int) Card("6", "d")}},
            {kMove8s7d, {(int) Card("8", "s"), (int) Card("7", "d")}},
            {kMove9s8d, {(int) Card("9", "s"), (int) Card("8", "d")}},
            {kMoveTs9d, {(int) Card("T", "s"), (int) Card("9", "d")}},
            {kMoveJsTd, {(int) Card("J", "s"), (int) Card("T", "d")}},
            {kMoveQsJd, {(int) Card("Q", "s"), (int) Card("J", "d")}},
            {kMoveKsQd, {(int) Card("K", "s"), (int) Card("Q", "d")}},

            // To Hearts -----------------------------------------------------------------------------------------------
            {kMove2hAs, {(int) Card("2", "h"), (int) Card("A", "s")}},
            {kMove3h2s, {(int) Card("3", "h"), (int) Card("2", "s")}},
            {kMove4h3s, {(int) Card("4", "h"), (int) Card("3", "s")}},
            {kMove5h4s, {(int) Card("5", "h"), (int) Card("4", "s")}},
            {kMove6h5s, {(int) Card("6", "h"), (int) Card("5", "s")}},
            {kMove7h6s, {(int) Card("7", "h"), (int) Card("6", "s")}},
            {kMove8h7s, {(int) Card("8", "h"), (int) Card("7", "s")}},
            {kMove9h8s, {(int) Card("9", "h"), (int) Card("8", "s")}},
            {kMoveTh9s, {(int) Card("T", "h"), (int) Card("9", "s")}},
            {kMoveJhTs, {(int) Card("J", "h"), (int) Card("T", "s")}},
            {kMoveQhJs, {(int) Card("Q", "h"), (int) Card("J", "s")}},
            {kMoveKhQs, {(int) Card("K", "h"), (int) Card("Q", "s")}},
            {kMove2hAc, {(int) Card("2", "h"), (int) Card("A", "c")}},
            {kMove3h2c, {(int) Card("3", "h"), (int) Card("2", "c")}},
            {kMove4h3c, {(int) Card("4", "h"), (int) Card("3", "c")}},
            {kMove5h4c, {(int) Card("5", "h"), (int) Card("4", "c")}},
            {kMove6h5c, {(int) Card("6", "h"), (int) Card("5", "c")}},
            {kMove7h6c, {(int) Card("7", "h"), (int) Card("6", "c")}},
            {kMove8h7c, {(int) Card("8", "h"), (int) Card("7", "c")}},
            {kMove9h8c, {(int) Card("9", "h"), (int) Card("8", "c")}},
            {kMoveTh9c, {(int) Card("T", "h"), (int) Card("9", "c")}},
            {kMoveJhTc, {(int) Card("J", "h"), (int) Card("T", "c")}},
            {kMoveQhJc, {(int) Card("Q", "h"), (int) Card("J", "c")}},
            {kMoveKhQc, {(int) Card("K", "h"), (int) Card("Q", "c")}},

            // To Clubs ------------------------------------------------------------------------------------------------
            {kMove2cAh, {(int) Card("2", "c"), (int) Card("A", "h")}},
            {kMove3c2h, {(int) Card("3", "c"), (int) Card("2", "h")}},
            {kMove4c3h, {(int) Card("4", "c"), (int) Card("3", "h")}},
            {kMove5c4h, {(int) Card("5", "c"), (int) Card("4", "h")}},
            {kMove6c5h, {(int) Card("6", "c"), (int) Card("5", "h")}},
            {kMove7c6h, {(int) Card("7", "c"), (int) Card("6", "h")}},
            {kMove8c7h, {(int) Card("8", "c"), (int) Card("7", "h")}},
            {kMove9c8h, {(int) Card("9", "c"), (int) Card("8", "h")}},
            {kMoveTc9h, {(int) Card("T", "c"), (int) Card("9", "h")}},
            {kMoveJcTh, {(int) Card("J", "c"), (int) Card("T", "h")}},
            {kMoveQcJh, {(int) Card("Q", "c"), (int) Card("J", "h")}},
            {kMoveKcQh, {(int) Card("K", "c"), (int) Card("Q", "h")}},
            {kMove2cAd, {(int) Card("2", "c"), (int) Card("A", "d")}},
            {kMove3c2d, {(int) Card("3", "c"), (int) Card("2", "d")}},
            {kMove4c3d, {(int) Card("4", "c"), (int) Card("3", "d")}},
            {kMove5c4d, {(int) Card("5", "c"), (int) Card("4", "d")}},
            {kMove6c5d, {(int) Card("6", "c"), (int) Card("5", "d")}},
            {kMove7c6d, {(int) Card("7", "c"), (int) Card("6", "d")}},
            {kMove8c7d, {(int) Card("8", "c"), (int) Card("7", "d")}},
            {kMove9c8d, {(int) Card("9", "c"), (int) Card("8", "d")}},
            {kMoveTc9d, {(int) Card("T", "c"), (int) Card("9", "d")}},
            {kMoveJcTd, {(int) Card("J", "c"), (int) Card("T", "d")}},
            {kMoveQcJd, {(int) Card("Q", "c"), (int) Card("J", "d")}},
            {kMoveKcQd, {(int) Card("K", "c"), (int) Card("Q", "d")}},

            // To Diamonds ---------------------------------------------------------------------------------------------
            {kMove2dAs, {(int) Card("2", "d"), (int) Card("A", "s")}},
            {kMove3d2s, {(int) Card("3", "d"), (int) Card("2", "s")}},
            {kMove4d3s, {(int) Card("4", "d"), (int) Card("3", "s")}},
            {kMove5d4s, {(int) Card("5", "d"), (int) Card("4", "s")}},
            {kMove6d5s, {(int) Card("6", "d"), (int) Card("5", "s")}},
            {kMove7d6s, {(int) Card("7", "d"), (int) Card("6", "s")}},
            {kMove8d7s, {(int) Card("8", "d"), (int) Card("7", "s")}},
            {kMove9d8s, {(int) Card("9", "d"), (int) Card("8", "s")}},
            {kMoveTd9s, {(int) Card("T", "d"), (int) Card("9", "s")}},
            {kMoveJdTs, {(int) Card("J", "d"), (int) Card("T", "s")}},
            {kMoveQdJs, {(int) Card("Q", "d"), (int) Card("J", "s")}},
            {kMoveKdQs, {(int) Card("K", "d"), (int) Card("Q", "s")}},
            {kMove2dAc, {(int) Card("2", "d"), (int) Card("A", "c")}},
            {kMove3d2c, {(int) Card("3", "d"), (int) Card("2", "c")}},
            {kMove4d3c, {(int) Card("4", "d"), (int) Card("3", "c")}},
            {kMove5d4c, {(int) Card("5", "d"), (int) Card("4", "c")}},
            {kMove6d5c, {(int) Card("6", "d"), (int) Card("5", "c")}},
            {kMove7d6c, {(int) Card("7", "d"), (int) Card("6", "c")}},
            {kMove8d7c, {(int) Card("8", "d"), (int) Card("7", "c")}},
            {kMove9d8c, {(int) Card("9", "d"), (int) Card("8", "c")}},
            {kMoveTd9c, {(int) Card("T", "d"), (int) Card("9", "c")}},
            {kMoveJdTc, {(int) Card("J", "d"), (int) Card("T", "c")}},
            {kMoveQdJc, {(int) Card("Q", "d"), (int) Card("J", "c")}},
            {kMoveKdQc, {(int) Card("K", "d"), (int) Card("Q", "c")}},
            //endregion
    };

    // OpenSpiel Classes ===============================================================================================

    class SolitaireGame;

    class SolitaireState : public State {
    public:

        // Attributes ==================================================================================================

        Deck                    deck;
        std::vector<Foundation> foundations;
        std::vector<Tableau>    tableaus;
        std::vector<Action>     revealed_cards;

        // Constructors ================================================================================================

        explicit SolitaireState(std::shared_ptr<const Game> game);

        // Overriden Methods ===========================================================================================

        Player                 CurrentPlayer() const override;
        std::unique_ptr<State> Clone() const override;
        bool                   IsTerminal() const override;
        bool                   IsChanceNode() const override;
        std::string            ToString() const override;
        std::string            ActionToString(Player player, Action action_id) const override;
        std::string            InformationStateString(Player player) const override;
        std::string            ObservationString(Player player) const override;
        void                   InformationStateTensor(Player player, std::vector<double> * values) const override;
        void                   ObservationTensor(Player player, std::vector<double> * values) const override;
        void                   DoApplyAction(Action move) override;
        std::vector<double>    Returns() const override;
        std::vector<double>    Rewards() const override;
        std::vector<Action>    LegalActions() const override;
        std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

        // Other Methods ===============================================================================================

        std::vector<Card>      Targets(const std::optional<std::string> & location = {}) const;
        std::vector<Card>      Sources(const std::optional<std::string> & location = {}) const;
        std::vector<Move>      CandidateMoves() const;
        Tableau *              FindTableau(const Card & card) const;
        Foundation *           FindFoundation(const Card & card) const;
        Location               FindLocation(const Card & card) const;
        void                   MoveCards(const Move & move);
        bool                   IsOverHidden(const Card & card) const;
        bool                   IsReversible(const Move & move) const;
        bool                   IsBottomCard(Card card) const;
        bool                   IsTopCard(const Card & card) const;
        bool                   IsSolvable() const;

    private:
        bool   is_setup;
        bool   is_started = false;
        bool   is_finished = false;
        bool   is_reversible = false;
        int    draw_counter = 0;
        int    current_depth = 0;
        double previous_score;
        std::set<std::size_t> previous_states = {};

    };

    class SolitaireGame : public Game {
    public:

        // Constructor =================================================================================================

        explicit SolitaireGame(const GameParameters & params);

        // Overriden Methods ===========================================================================================

        int     NumDistinctActions() const override;
        int     MaxGameLength() const override;
        int     NumPlayers() const override;
        double  MinUtility() const override;
        double  MaxUtility() const override;

        std::vector<int> InformationStateTensorShape() const override;
        std::vector<int> ObservationTensorShape() const override;

        std::unique_ptr<State>       NewInitialState() const override;
        std::shared_ptr<const Game>  Clone() const override;
        
    private:
        int num_players_;

    };

} // namespace open_spiel::solitaire

#endif // THIRD_PARTY_OPEN_SPIEL_GAMES_SOLITAIRE_H
