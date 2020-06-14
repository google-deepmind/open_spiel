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

// Unicode Glyphs
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
    };

    // Constants =======================================================================================================

    // These divide up the action ids into sections. kEnd is a single action that is used to end the game when no
    // other actions are available.
    inline constexpr int kEnd = 0;

    // kReveal actions are ones that can be taken at chance nodes; they change a hidden card to a card of the same
    // index as the action id (e.g. 2 would reveal a 2 of spades)
    inline constexpr int kRevealStart = 1;
    inline constexpr int kRevealEnd   = 52;

    // kMove actions are ones that are taken at decision nodes; they involve moving a card to another cards location.
    // It starts at 53 because there are 52 reveal actions before it. See `NumDistinctActions()` in solitaire.cc.
    inline constexpr int kMoveStart = 53;
    inline constexpr int kMoveEnd   = 204;

    // Indices for special cards
    inline constexpr int kHiddenCard       = 99;
    inline constexpr int kEmptySpadeCard   = -5;
    inline constexpr int kEmptyHeartCard   = -4;
    inline constexpr int kEmptyClubCard    = -3;
    inline constexpr int kEmptyDiamondCard = -2;
    inline constexpr int kEmptyTableauCard = -1;

    // 1 empty + 13 ranks
    inline constexpr int kFoundationTensorLength = 14;

    // 6 hidden cards + 1 empty tableau + 52 ordinary cards
    inline constexpr int kTableauTensorLength = 59;

    // 1 hidden card + 52 ordinary cards
    inline constexpr int kWasteTensorLength = 53;

    // Constant for how many hidden cards can show up in a tableau. As hidden cards can't be added, the max is the
    // highest number in a tableau at the start of the game: 6
    inline constexpr int kMaxHiddenCard = 6;

    // Only used in one place and just for consistency (to match kChancePlayerId & kTerminalPlayerId)
    inline constexpr int kPlayerId = 0;

    // Indicates the last index before the first player action (the last kReveal action has an ID of 52)
    inline constexpr int kActionOffset = 52;

    // Order of suits
    const std::vector<SuitType> kSuits = {kSuitSpades, kSuitHearts, kSuitClubs, kSuitDiamonds};

    // These correspond with their enums, not with the two vectors directly above
    const std::vector<std::string> kSuitStrs = {"", kGlyphSpades, kGlyphHearts, kGlyphClubs, kGlyphDiamonds, ""};
    const std::vector<std::string> kRankStrs = {"", "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", ""};

    const std::map<RankType, double> kFoundationPoints = {
            //region Maps a RankType to the reward for moving a card of that rank to the foundation
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
        explicit Card(bool hidden = false, SuitType suit = kSuitHidden, RankType rank = kRankHidden, LocationType location = kMissing);
        explicit Card(int index, bool hidden = false, LocationType location = kMissing);

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
        explicit Tableau(PileID id);

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