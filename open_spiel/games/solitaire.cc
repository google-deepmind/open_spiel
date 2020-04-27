#include "solitaire.h"

#include <deque>
#include <random>
#include <algorithm>
#include <optional>
#include <utility>
#include <math.h>

#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define YELLOW  "\033[33m"
#define WHITE   "\033[37m"

/* GAME INFO
   An implementation of Klondike Solitaire with 3 cards drawn at a time & infinite redeals. By default, `COLOR_FLAG` is
   set to true and will color cards when they are converted to strings. If you see anything like "\033[...m" instead of
   colored cards, then you can turn off this functionality by compiling again with `COLOR_FLAG` set to false.

   LOOP PREVENTION
   Part of the difficulty in implementing this game is that no matter how many cards are drawn at a time or how many
   redeals are allowed, certain situations lead to an infinite loop (e.g. moving a queen of hearts back and forth
   between a king of spades and king of clubs).

   In this implementation, we prevent these loops by recognizing when a "reversible move" is made; one that has the
   potential to be returned to at a later state. Starting from this first reversible move, the output of
   `ObservationString` is hashed and stored in `previous_states` in its descendants. `CandidateMoves` then provides a
   list of moves that are technically legal according to the rules of the game and all reversible moves in this list are
   checked to ensure they don't form a back edge to any member of `previous_states`. The moves that satisfy this
   condition are passed along as `LegalActions`.

   If an irreversible move is made, `previous_states` is cleared and all candidate moves will be passed along as
   `LegalActions` without having to check them for back edges. This continues until a reversible move is made, at which
   point the above paragraph apply again. */

/* TERMINOLOGY
    Pile:
        An abstract term referring to an ordered set of cards. In this game, it can refer to
        a tableau, foundation, deck, or waste.

    Tableau:
        A pile that can have 0 to 6 hidden cards at the beginning. The last card is never hidden
        at decision nodes. The last/top-most card is called the target of the tableau. A source
        card (and potentially cards underneath it) can be moved to the target if the source is
        of an opposite color and one rank lower than the target. An empty tableau has a target
        represented by the special card, Card("", ""); in this case, a king of any suit can be
        moved to it.

    Foundation:
        A pile that has no hidden cards. The last/top-most card is called the target, the same
        as a tableau. A single source card can be moved from the waste or tableau onto the target
        if it's they are the same suit and the source card is one rank higher. Foundations have a
        suit, and an empty one is represented by a special card, Card("", "x") where "x" is the suit
        of the foundation.

    Deck:
        Purely a container of cards that can be drawn in groups of 3 into the waste. When it is empty,
        the deck is rebuilt by moving all remaining cards in the waste back into the deck in the
        order they were originally revealed in the waste. It has no targets or sources. The deck and waste
        are both implemented in the `Deck` class as some of its attributes.

    Waste:
        Another ordered set of cards. It has no targets and has only one source card at most. The source
        can be moved to a tableau or foundation, in which case the next card in the waste becomes the
        source. If a card is hidden in the waste, a chance node is forced and chooses what to rank and suit
        to reveal the card as.

    Source Card:
        The card being moved to a target. In the case of a pile being moved, it is the first card
        in the pile, e.g. in the move (Ks <- Qh) where the cards being moved are {Qh, Js, 10h},
        Qh would be the source card. Sources can be in any pile except the deck.

    Target Card:
        The single card that a source is being moved to. In the move (Ks <- Qh), the target would be
        Ks. Uses its LegalChildren method to determine what source cards can be moved to it. At any given
        state, there are a maximum of 11 target cards in the game (7 tableaus, 4 foundations).

    Opposite Suits:
        The suits in solitaire are spades, hearts, clubs, diamonds. Spades and clubs are black; hearts and
        diamonds are red. Opposite suits are suits of the opposite color, e.g. the opposite suits of spades
        would be hearts and diamonds.

    Ordinary Cards:
        These are cards that are either hidden or have a rank and suit that is not an empty string "".
        They're what you would normally consider cards in a game of solitaire.

    Special Cards:
        These represent an empty tableau or foundation. They're represented as cards so that Move(target, source)
        doesn't have to be overloaded. Another reason for doing so is that the idea of LegalChildren still makes
        sense for them. An empty tableau, Card("", "") has legal children Ks, Kh, Kc, Kd. An empty foundation
        could be Card("", "s") which has a legal child Card("A", "s"). Another reason for representing them as cards
        is because the Targets() and Sources() methods of various classes return vectors of cards. One thing to note,
        special cards do not appear in the `cards` vector of any pile. They are only returned from Targets() when
        `cards` is empty and they can never be returned from Sources().

    LegalChildren:
        A Card method that determines what source cards can be moved to the card. It's a function of the
        cards rank, suit, and location. In the case of a foundation, they can have at most one legal child.
        If the card is a king in a foundation, it has no legal children. If it's an empty foundation like
        Card("", "h"), then it has one child, Card("A", "h"). Otherwise, they all have one child of the same
        suit and one rank higher. In the case of a tableau, they can have 0, 2, or 4 children. An ace in the tableau
        has 0 legal children, while an empty tableau represented by a special card will have four (kings of the four
        suits). Otherwise, the card will have two children that are one rank lower; one child for each opposite
        suit.

    Move:
        A type of Action, it involves moving a card or a pile of cards to another card. Represented as a
        string by (target <- source) where source is in target.LegalChildren(), e.g. (Ks <- Qh). In order to make
        sure the game terminates, we have to introduce the idea of reversible and irreversible moves.

    Reversible/Irreversible Moves:
        There are two ways a move can be irreversible: the source of the move is in the waste, or the source
        is directly on top of a hidden card, in which case a chance node will reveal it in the next state. As you
        can't hide a card again once it's been revealed or move a card to the waste, these moves have no way of
        reaching a previous state. Reversible moves are just defined as moves that aren't irreversible.
*/

namespace open_spiel::solitaire {

    namespace {
        const GameType kGameType {
                "solitaire",
                "Solitaire",
                GameType::Dynamics::kSequential,
                GameType::ChanceMode::kExplicitStochastic,
                GameType::Information::kImperfectInformation,
                GameType::Utility::kGeneralSum,
                GameType::RewardModel::kRewards,
                1,
                1,
                true,
                true,
                true,
                true,
                {{"players", GameParameter(kDefaultPlayers)}}
        };

        std::shared_ptr<const Game> Factory(const GameParameters & params) {
            return std::shared_ptr<const Game>(new SolitaireGame(params));
        }

        REGISTER_SPIEL_GAME(kGameType, Factory)
    }

    // Miscellaneous ===================================================================================================

    /* Used to hash the std::string of the current state. This is only used when a reversible move is made and
     * we need to check if candidate moves would form a back edge to this state. Instead of comparing strings, we
     * compare the hash of the strings created with this object. */
    std::hash<std::string> hasher;

    /* Whether to format strings with ANSI colors or not.
     * Only used by the function below, `color()`. */
    bool COLOR_FLAG = true;

    std::string color(std::string color) {
        /* Returns an ANSI color provided by the argument (e.g. RED, YELLOW, WHITE, RESET) if `COLOR_FLAG` is true.
         * Otherwise it returns an empty string. This is used so we can quickly turn color on and off. It's usually
         * useful to have it on, as we can quickly see valid moves more quickly. In some cases, the terminal doesn't
         * support colors and rather than print a bunch of gibberish, we can turn it off by setting `COLOR_FLAG`
         * to false. */
        return COLOR_FLAG ? std::move(color) : "";
    }

    std::vector<std::string> GetOppositeSuits(const std::string & suit) {
        /* Just returns a vector of the suits of opposite color. For red suits ("h" and "d"), this returns the
         * black suits ("s", "c"). For a black suit, this returns the red suits. */

        if (suit == "s" or suit == "c") {
            return {"h", "d"};
        } else if (suit == "h" or suit == "d") {
            return {"s", "c"};
        } else {
            std::cout << YELLOW << "WARNING: `suit` is not in (s, h, c, d)" << RESET << std::endl;
        }

    }

    std::string LocationString(Location location) {
        /* `Location` is an enum, meaning that at runtime we can only see the value its tied to. This function lets
         * us get a std::string of that value during runtime (e.g. kDeck is 0, LocationString(0) -> "kDeck") */

        switch (location) {
            case kDeck : {
                return "kDeck";
            }
            case kWaste : {
                return "kWaste";
            }
            case kFoundation : {
                return "kFoundation";
            }
            case kTableau : {
                return "kTableau";
            }
            case kMissing : {
                return "kMissing";
            }
            default : {
                return "n/a";
            }
        }
    }

    std::vector<double> ToCardIndices(const std::deque<Card> & pile, int length) {

        /* Turns a pile of cards into a pile of card indices that is resized to a given length. If it's resized
         * to be longer than the pile of cards, it is filled with `NO_CARD` (99.0) at the end. */

        std::vector<double> index_vector;
        if (not pile.empty()) {
            for (auto & card : pile) {
                if (card.hidden) {
                    index_vector.push_back(HIDDEN_CARD);
                } else {
                    index_vector.push_back((int) card);
                }
            }
        }

        index_vector.resize(length, NO_CARD);
        
        return index_vector;

    }

    // Card Methods ====================================================================================================

    Card::Card(std::string rank, std::string suit) :
        rank(std::move(rank)),
        suit(std::move(suit)),
        hidden(true),
        location(kMissing) {
        /* Constructs a card with a given rank and suit. Must be unhidden and have it location set outside
         * of this constructor */
    }

    Card::Card() : rank(""), suit(""), hidden(true), location(kMissing) {
        /* Default constructor, just creates a hidden card with no rank, suit, or location. */
    }

    Card::Card(int index) : hidden(false), location(kMissing) {

        /* Constructs an unhidden card from its index; `location` is set to kMissing here and must be set outside of
         * this constructor to be used by methods that depend on it */

        // Handles special cards
        if (index < 0) {
            rank = "";
            switch (index) {
                case -1 : { suit = "s"; break; }
                case -2 : { suit = "h"; break; }
                case -3 : { suit = "c"; break; }
                case -4 : { suit = "d"; break; }
                case -5 : { suit =  ""; break; }
                default : {
                    std::cout << YELLOW << "WARNING: Incorrect index for special card";
                    break;
                }
            }
        }
        // Handles ordinary cards
        else {
            int rank_value = index % 13;
            int suit_value = floor(index / 13);
            rank = RANKS.at(rank_value);
            suit = SUITS.at(suit_value);
        }
        
    }

    Card::operator int() const {
        /* Looks up the card index with the map `RANKSUIT_TO_INDEX`. Depends only on the rank and suit of the card,
         * so it could be called on a hidden card. Special cards (those with an empty rank "") are supported too. */

        std::pair<std::string, std::string> ranksuit = {rank, suit};
        return RANKSUIT_TO_INDEX.at(ranksuit);
    }

    bool Card::operator==(Card & other_card) const {
        return rank == other_card.rank and suit == other_card.suit;
    }

    bool Card::operator==(const Card & other_card) const {
        return rank == other_card.rank and suit == other_card.suit;
    }

    std::vector<Card> Card::LegalChildren() const {

        /* Returns a vector of the legal children a card can have. Depends on rank, suit, location, and if the card is
         * hidden. Hidden cards can have no legal children. Cards with no rank or suit have four legal children, kings
         * of all four suits. Cards with no rank and a suit have one child, an ace of the same suit. Otherwise, for
         * cards with both a rank and a suit, their legal children depend on their location. In a foundation, they have
         * one legal child, a card of the same suit but one rank higher (except for kings which have none). In a
         * tableau, they have two legal children that are one rank lower, one for each opposite color suit (except for
         * aces which have no legal children) */

        std::vector<Card>        legal_children = {};
        std::string              child_rank;
        std::vector<std::string> child_suits;

        // A hidden card has no legal children
        if (hidden) {
            // log_exit("Exiting Card::operator==");
            return legal_children;
        }

        switch (location) {
            case kTableau:
                // Handles empty tableau cards (children are kings of all suits)
                if (rank.empty()) {
                    child_rank  = "K";
                    child_suits = SUITS;
                }
                // Handles regular cards (except aces)
                else if (rank != "A") {
                    child_rank  = RANKS.at(GetIndex(RANKS, rank) - 1);
                    child_suits = GetOppositeSuits(suit);
                }
                break;

            case kFoundation:
                // Handles empty foundation cards (children are aces of same suit)
                if (rank.empty()) {
                    child_rank  = "A";
                    child_suits = {suit};
                }
                // Handles regular cards (except kings)
                else if (rank != "K") {
                    child_rank  = RANKS.at(GetIndex(RANKS, rank) + 1);
                    child_suits = {suit};
                }
                break;

            default:
                return legal_children;
        }

        // TODO: Child suits could technically be empty if OppositeSuits() returns {}
        for (const auto & child_suit : child_suits) {
            auto child   = Card(child_rank, child_suit);
            child.hidden = false;
            legal_children.push_back(child);
        }

        return legal_children;
    }

    std::string Card::ToString() const {

        /* Represents the card as a string. Hidden and special cards have a single unicode glyph. Ordinary cards are
         * represented as a concatenation of their rank and suit, optionally colored based on their suit if `COLOR_FLAG`
         * is true (e.g. Card("Q", "h') would be "Qh") */

        std::string result;

        if (hidden) {
            // Representation of a hidden card
            absl::StrAppend(&result, "\U0001F0A0", " ");
        }
        else {
            // Suit Color
            if (suit == "s" or suit == "c") {
                absl::StrAppend(&result, color(WHITE));
            } else if (suit == "h" or suit == "d") {
                absl::StrAppend(&result, color(RED));
            }

            // Special Cards
            if (rank.empty()) {
                // Handles special tableau cards which have no rank or suit
                if (suit.empty()) {
                    absl::StrAppend(&result, "\U0001F0BF");
                }
                // Handles special foundation cards which have a suit but not a rank
                else {
                    if (suit == "s") {
                        absl::StrAppend(&result, "\U00002660");
                    } else if (suit == "h") {
                        absl::StrAppend(&result, "\U00002665");
                    } else if (suit == "c") {
                        absl::StrAppend(&result, "\U00002663");
                    } else if (suit == "d") {
                        absl::StrAppend(&result, "\U00002666");
                    }
                }
            }

            // Ordinary Cards
            else {
                absl::StrAppend(&result, rank, suit);
            }



        }

        absl::StrAppend(&result, color(RESET));
        return result;
    }

    // Deck Methods ====================================================================================================

    Deck::Deck() {

        /* Default constructor, fills its attribute `cards` with 24 hidden cards and sets their location to kDeck.
         * These cards have no rank or suit initially, those are given to them when revealed by a kReveal move */

        for (int i = 1; i <= 24; i++) {
            cards.emplace_back();
        }
        for (auto & card : cards) {
            card.location = kDeck;
        }
    }

    std::vector<Card> Deck::Sources() const {

        /* Despite being a vector, Sources() can only return one card at most. This is always the card that is at the
         * front of the waste unless it is hidden. If the waste is empty or the front card is hidden, this returns
         * an empty vector */

        // If the waste is not empty, sources is just a vector of the top card of the waste
        if (not waste.empty()) {
            if (waste.front().hidden) {
                return {};
            } else {
                return {waste.front()};
            }
        }
        // If it is empty, sources is just an empty vector
        else {
            return {};
        }

    }

    std::vector<Card> Deck::Split(Card card) {

        /* Returns the front card of the waste if it matches the `card` argument provided. This card is removed from
         * the waste during the method. Behavior for what happens when the card doesn't match or there is no front
         * card to check is undefined */

        std::vector<Card> split_cards;
        if (waste.front() == card) {
            split_cards = {waste.front()};
            waste.pop_front();
            return split_cards;
        }
    }

    void Deck::draw(unsigned long num_cards) {

        /* Pops and moves up to `num_cards` cards from the deck into the waste. Their location is changed from
         * kDeck to kWaste when this happens. */

        std::deque<Card> drawn_cards;
        num_cards = std::min(num_cards, cards.size());

        int i = 1;
        while (i <= num_cards) {
            auto card = cards.front();
            card.location = kWaste;
            drawn_cards.push_back(card);
            cards.pop_front();
            i++;
        }

        waste.insert(waste.begin(), drawn_cards.begin(), drawn_cards.end());
    }

    void Deck::rebuild() {

        /* If `deck.cards` is empty, this method repopulates it with cards from `deck.waste` in the order given by
         * `initial_order`. This allows us to keep drawing from the deck in the exact same order every time, only
         * changing when a card is split from the waste and moved somewhere else */

        // TODO: Make sure cards and initial_order are never both empty at the same time.
        if (cards.empty()) {
            for (Card & card : initial_order) {
                if (std::find(waste.begin(), waste.end(), card) != waste.end()) {
                    card.location = kDeck;
                    cards.push_back(card);
                }
            }
            waste.clear();
            times_rebuilt += 1;
        } else {
            std::cout << YELLOW << "WARNING: Cannot rebuild a non-empty deck" << RESET << std::endl;
        }
    }

    // Foundation Methods ==============================================================================================

    Foundation::Foundation() {
        cards = {};
    }

    Foundation::Foundation(std::string suit) : suit(std::move(suit)) {
        cards = {};
    }

    std::vector<Card> Foundation::Sources() const {
        // If the foundation is not empty, sources is just a vector of the top card of the foundation
        if (not cards.empty()) {
            return {cards.back()};
        }
        // If it is empty, then sources is just an empty vector
        else {
            return {};
        }
    }

    std::vector<Card> Foundation::Targets() const {

        // If the foundation is not empty, targets is just the top card of the foundation
        if (not cards.empty()) {
            return {cards.back()};
        }
        // If it is empty, then targets is just a special card with no rank and a suit matching this foundation
        else {
            auto card     = Card("", suit);
            card.hidden   = false;
            card.location = kFoundation;
            return {card};
        }
    }

    std::vector<Card> Foundation::Split(Card card) {
        std::vector<Card> split_cards;
        if (cards.back() == card) {
            split_cards = {cards.back()};
            cards.pop_back();
            return split_cards;
        }
    }

    void Foundation::Extend(const std::vector<Card> & source_cards) {
        for (auto card : source_cards) {
            card.location = kFoundation;
            cards.push_back(card);
        }
    }

    // Tableau Methods =================================================================================================

    Tableau::Tableau() = default;

    Tableau::Tableau(int num_cards) {
        for (int i = 1; i <= num_cards; i++) {
            cards.emplace_back();
        }
        for (auto & card : cards) {
            card.location = kTableau;
        }
    }

    std::vector<Card> Tableau::Sources() const {
        // If the tableau is not empty, sources is just a vector of all cards that are not hidden
        if (not cards.empty()) {
            std::vector<Card> sources;
            for (auto & card : cards) {
                if (not card.hidden) {
                    sources.push_back(card);
                }
            }
            return sources;
        }
        // If it is empty, then sources is just an empty vector
        else {
            return {};
        }
    }

    std::vector<Card> Tableau::Targets() const {
        // If the tableau is not empty, targets is just a vector of the top card of the tableau
        if (not cards.empty()) {
            if (cards.back().hidden) {
                return {};
            } else {
                return {cards.back()};
            }
        }
        // If it is empty, then targets is just a special card with no rank or suit
        else {
            auto card     = Card();
            card.hidden   = false;
            card.location = kTableau;
            return {card};
        }
    }

    std::vector<Card> Tableau::Split(Card card) {
        std::vector<Card> split_cards;
        if (not cards.empty()) {
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
        } else {
            std::cout << YELLOW << "WARNING: Cannot split an empty tableau" << RESET << std::endl;
        }

        return split_cards;
    }

    void Tableau::Extend(const std::vector<Card> & source_cards) {
        for (auto card : source_cards) {
            card.location = kTableau;
            cards.push_back(card);
        }
    }

    // Move Methods ====================================================================================================

    Move::Move(Card target_card, Card source_card) {
        /* A move is essentially just a pair of cards where `target` is in `SolitaireState::Targets()`, `source` is
         * in `SolitaireState::Sources()`, and `source` is in `target.LegalChildren()`. Each possible combination
         * in a game has a corresponding `Action` in `enum ActionType` in solitaire.h */

        target = std::move(target_card);
        source = std::move(source_card);
    }

    Move::Move(Action action_id) {
        /* Creates a move from an `Action` (e.g. Move(kMove7d6s) would have a `target` 7d and a `source` 6s).
         * Used in `DoApplyAction` when given a kMove action to convert the action to a move that can be moved
         * with `SolitaireState::MoveCards()` */

        auto card_pair = ACTION_TO_MOVE.at(action_id);
        target = Card(card_pair.first);
        source = Card(card_pair.second);
    }

    std::string Move::ToString() const {
        /* Represents the target and source in the form (target ← source). The target and source are converted to
         * strings via the `Card::ToString()` method, which can optionally be formatted with colors. */

        std::string result;
        absl::StrAppend(&result, target.ToString(), " ", "\U00002190", " ",source.ToString());
        return result;
    }

    Action Move::ActionId() const {
        /* The inverse of `Move(Action)`. Instead of making a move from an action, it takes an existing move
         * and converts it back into its action. Used in `LegalActions()` to convert the moves returned from
         * `CandidateMoves()` into actions. */

        return MOVE_TO_ACTION.at(std::make_pair((int) target, (int) source));
    }

    // SolitaireState Methods ==========================================================================================

    SolitaireState::SolitaireState(std::shared_ptr<const Game> game) :
        State(game),
        deck(),
        foundations(),
        tableaus() {
            is_started = false;
            is_setup = false;
            previous_score = 0.0;
        }

    // Overriden Methods -----------------------------------------------------------------------------------------------

    Player                  SolitaireState::CurrentPlayer() const {

        /* As there are only two actors in the game, chance and the player, we can determine who the current player
         * is based on if the state is a chance node or not. At terminal nodes, the player is always kTerminalPlayer.
         * If it's not terminal, then if it's a chance node, the player is kChancePlayerId. Otherwise, it's always
         * the player, which has index 0.
         */

        if (IsTerminal()) {
            // Index of the chance player
            return kTerminalPlayerId;
        }
        else if (IsChanceNode()) {
            // Index of the terminal player
            return kChancePlayerId;
        }
        else {
            // Index of the player
            return 0;
        }
    }

    std::unique_ptr<State>  SolitaireState::Clone() const {

        // tracer trace("SolitaireState::Clone()");
        return std::unique_ptr<State>(new SolitaireState(*this));

    }

    bool                    SolitaireState::IsTerminal() const {

        /* Decides if the game is over, `is_finished` is an attribute of the state that allows us to make the next
         * state terminal no matter what. The state is also terminal if the last 8 actions have been kDraw (Drawing
         * through the entire deck without moving anything from it is equivalent to never having drawn through it at
         * all). Finally, if LegalActions() is empty, then the state is also terminal as it can't be progressed
         * any further */

        if (is_finished or draw_counter >= 9) {
            return true;
        }

        else if (History().size() >= 8) {

            std::vector<Action> history = History();
            std::vector<Action> recent_history(history.end() - 8, history.end());

            for (auto action : recent_history) {
                if (action != kDraw) {
                    return false;
                }
            }

            // If all 8 recent actions are kDraw, then the state is terminal
            return true;

        }

        else if (LegalActions().empty()) {
            return true;
        }

        else {
            return false;
        }

    }

    bool                    SolitaireState::IsChanceNode() const {

        /* The first state in the game is always a chance node, so this returns true if `is_setup` is false. For
         * subsequent states, if there is a hidden card in the waste or at the top card of a tableau, then this
         * is also true so that a chance outcome can be chosen to reveal what rank and suit it has. */

        if (not is_setup) {
            // If setup is not started, this is a chance node
            return true;
        }

        else {

            // If there is a hidden card on the top of a tableau, this is a chance node
            for (auto & tableau : tableaus) {
                if (tableau.cards.empty()) {
                    continue;
                }
                else if (tableau.cards.back().hidden) {
                    return true;
                }
            }

            // If any card in the waste is hidden, this is a chance node
            if (not deck.waste.empty()) {
                for (auto & card : deck.waste) {
                    if (card.hidden) {
                        return true;
                    }
                }
            }

            // Otherwise, this is not a chance node; it's a decision node
            return false;

        }
    }

    std::string             SolitaireState::ToString() const {

        /* Only used for debugging, this method provides a representation of the current state. Other sections can
         * be uncommented for more information or to debug but they will significantly impact performance as most of
         * them don't reference a stored value and must be calculated again. */

        std::string result;

        /*
        absl::StrAppend(&result, "DRAW COUNTER    : ", draw_counter);

        absl::StrAppend(&result, "\nIS REVERSIBLE   : ", is_reversible);
        */

        absl::StrAppend(&result, "\nCURRENT_DEPTH   : ", current_depth);

        absl::StrAppend(&result, "\n\nDECK        : ");
        for (const Card & card : deck.cards) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\nWASTE       : ");
        for (const Card & card : deck.waste) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\nORDER       : ");
        for (const Card & card : deck.initial_order) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\nFOUNDATIONS : ");
        for (const Foundation & foundation : foundations) {
            if (foundation.cards.empty()) {
                Card foundation_base = Card("", foundation.suit);
                foundation_base.hidden = false;
                absl::StrAppend(&result, foundation_base.ToString(), " ");
            } else {
                absl::StrAppend(&result, foundation.cards.back().ToString(), " ");
            }
        }

        absl::StrAppend(&result, "\nTABLEAUS    : ");
        for (const Tableau & tableau : tableaus) {
            if (not tableau.cards.empty()) {
                absl::StrAppend(&result, "\n");
                for (const Card & card : tableau.cards) {
                    absl::StrAppend(&result, card.ToString(), " ");
                }
            }
        }

        /*
        absl::StrAppend(&result, "\n\nTARGETS : ");
        for (const Card & card : Targets()) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\nSOURCES : ");
        for (const Card & card : Sources()) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\n\nCANDIDATE MOVES : ");
        for (const Move & move : CandidateMoves()) {
            absl::StrAppend(&result, "\n", move.ToString(), " : ", move.ActionId());
            absl::StrAppend(&result, ", ", IsReversible(move));
        }
        */

        return result;
    }

    std::string             SolitaireState::ActionToString(Player player, Action action_id) const {
        /* Just converts an `Action` back to the name it was defined with (e.g. The `Action` kSetup maps to 0, this
         * method converts 0 into a string "kSetup"). For a list of actions, see `enum ActionType` in solitaire.h */

        switch (action_id) {
            case kSetup : {
                return "kSetup";
            }
            case kRevealAs ... kRevealKd : {
                // Reveal starts at 1 while card indices start at 0, so we subtract one here
                Card revealed_card = Card(action_id - 1);

                std::string result;
                absl::StrAppend(&result, "kReveal", revealed_card.rank, revealed_card.suit);

                return result;
            }
            case kDraw : {
                return "kDraw";
            }
            case kMove__Ks ... kMoveKdQc : {
                Move move = Move(action_id);
                std::string result;

                absl::StrAppend(&result, "kMove");
                if (move.target.rank.empty()) { absl::StrAppend(&result, "__"); }
                else { absl::StrAppend(&result, move.target.rank, move.target.suit); }
                absl::StrAppend(&result, move.source.rank, move.source.suit);

                return result;
            }
            default : {
                return "kMissingAction";
            }
        }
    }

    std::string             SolitaireState::InformationStateString(Player player) const {
        /* Same thing as `HistoryString()`. It's just a list of actions taken since the beginning of the game.
         * Can be useful for some debugging although I've found that because of it's large length it's usually better
         * to split `InformationStateTensor` at the first occurrence of `kInvalidAction` and then reverse whats left
         * in order to see the most recent actions taken */

        return HistoryString();
    }

    std::string             SolitaireState::ObservationString(Player player) const {

        /* This method is actually very important in loop prevention as its value is hashed when a reversible move is
         * done. It is basically just a subset of the information contained in `SolitaireState::ToString()` */

        // TODO: Rely on a different method to get hashed state

        std::string result;

        absl::StrAppend(&result, "\nFOUNDATIONS : ");
        for (const Foundation & foundation : foundations) {
            if (foundation.cards.empty()) {
                Card foundation_base = Card("", foundation.suit);
                foundation_base.hidden = false;
                absl::StrAppend(&result, foundation_base.ToString());
            } else {
                absl::StrAppend(&result, foundation.cards.back().ToString(), " ");
            }
        }

        absl::StrAppend(&result, "\nTABLEAUS    : ");
        for (const Tableau & tableau : tableaus) {
            if (not tableau.cards.empty()) {
                absl::StrAppend(&result, "\n");
                for (const Card & card : tableau.cards) {
                    absl::StrAppend(&result, card.ToString(), " ");
                }
            }
        }

        return result;

    }

    void                    SolitaireState::InformationStateTensor(Player player, std::vector<double> *values) const {

        /* As there are multiple ways to reach the same state in solitaire, we use a history of actions taken.
         * This method just basically just pads `History()` with `kInvalidAction` up to length of
         * `InformationStateTensorShape()`. */

        values->resize(game_->InformationStateTensorShape()[0]);
        std::fill(values->begin(), values->end(), kInvalidAction);

        int i = 0;
        for (auto & action : History()) {
            (*values)[i] = action;
            ++i;
        }

    }

    void                    SolitaireState::ObservationTensor(Player player, std::vector<double> *values) const {

        /* A view of the current state as a vector of doubles. Each pile in the state is represented as a list of
         * card indices, hidden cards use `HIDDEN_CARD` (98.0) and non-existing cards use `NO_CARD` (99.0). Each pile
         * is padded to its maximum length with `NO_CARD` if applicable. They are returned as a single vector
         * concatenated together with size 233. */

        // TODO: Not sure if any pile being empty would have an effect on the final result

        for (const auto & tableau : tableaus) {
            std::vector<double> tableau_obs = ToCardIndices(tableau.cards, 19);
            values->insert(values->end(), tableau_obs.begin(), tableau_obs.end());
        }

        for (const auto & foundation : foundations) {
            std::vector<double> foundation_obs = ToCardIndices(foundation.cards, 13);
            values->insert(values->end(), foundation_obs.begin(), foundation_obs.end());
        }

        std::vector<double> waste_obs = ToCardIndices(deck.waste, 24);
        values->insert(values->end(), waste_obs.begin(), waste_obs.end());

        std::vector<double> deck_obs = ToCardIndices(deck.cards, 24);
        values->insert(values->end(), deck_obs.begin(), deck_obs.end());

    }

    void                    SolitaireState::DoApplyAction(Action move) {

        /* Responsible for executing any legal actions or chance outcome. Sets `previous_score` in non-chance nodes.
         * Also finishes solving the game if `IsSolvable` is true. */

        if (not IsChanceNode()) {
            previous_score = Returns().front();
        }

        // Action Handling =============================================================================================

        // Handles kSetup
        if (move == kSetup) {

            // Creates tableaus
            for (int i = 1; i <= 7; i++) {
                tableaus.emplace_back(i);
            }

            // Creates foundations
            for (const auto & suit : SUITS) {
                foundations.emplace_back(suit);
            }

            is_setup       = true;
            is_started     = false;
            is_finished     = false;
            is_reversible  = false;
            draw_counter   = 0;
            current_depth  = 0;
            previous_score = 0.0;

        }

        // Handles kReveal
        else if (1 <= move and move <= 52) {

            // Cards start at 0 instead of 1 which is why we subtract 1 to move here.
            Card revealed_card = Card(move - 1);
            bool found_hidden_card = false;

            // For tableau in tableaus ...
            for (auto & tableau : tableaus) {

                // If it isn't empty ...
                if (not tableau.cards.empty()) {

                    // If the last card is hidden ...
                    if (tableau.cards.back().hidden) {

                        // Then reveal it
                        tableau.cards.back().rank = revealed_card.rank;
                        tableau.cards.back().suit = revealed_card.suit;
                        tableau.cards.back().hidden = false;

                        // And indicate that we found a hidden card so we don't have to search for one in the waste
                        found_hidden_card = true;

                        // Breaks out and goes to check the waste, if applicable
                        break;
                    }
                }
            }

            // If we didn't find a hidden card in the tableau and the waste isn't empty ...
            if ((not found_hidden_card) and not deck.waste.empty()) {

                // Then for card in the waste ...
                for (auto & card : deck.waste) {

                    // If the card is hidden ...
                    if (card.hidden) {

                        // Reveal it by setting its rank and suit
                        card.rank = revealed_card.rank;
                        card.suit = revealed_card.suit;
                        card.hidden = false;

                        // Add the revealed card to the initial order
                        deck.initial_order.push_back(card);
                        break;

                    }
                }
            }

            // Add move to revealed cards so we don't try to reveal it again
            revealed_cards.push_back(move);


            // TODO: There shouldn't ever be a time before `is_started` where a tableau is empty
            // If the game hasn't been started ...
            if (not is_started) {
                // For every tableau in tableaus ...
                for (auto & tableau : tableaus) {
                    // If the last card is hidden ...
                    if (tableau.cards.back().hidden) {
                        // Then we are not ready to start the game.
                        // Return with is_started still false;
                        return;
                    }
                    // If the last card is not hidden, continue the loop and check the next tableau
                    else {
                        continue;
                    }
                }

                // This is only reached if all cards at the back of the tableaus are not hidden.
                is_started = true;
                previous_score = 0.0;
            }

        }

        // Handles kDraw
        else if (move == kDraw) {
            
            // kDraw is not reversible (well, you'd have to go through the deck again)
            // is_reversible = false;
            if (deck.cards.empty()) {
                deck.rebuild();
            }
            deck.draw(3);

            // Loop Detection
            std::vector<Action> legal_actions = LegalActions();

            // We check here if there are any other legal actions besides kDraw
            if (legal_actions.size() == 1) {
                draw_counter += 1;
            }

            if (draw_counter > 8) {
                is_finished = true;
            }
        }

        // Handles kMove
        else {

            // Create a move from the action id provided by 'move'
            Move selected_move = Move(move);

            // If the move we are about to execute is reversible, set to true, else set to false
            is_reversible = IsReversible(selected_move);

            if (is_reversible) {
                // If a move being executed is reversible, we need to hash its state and store it
                std::string current_observation = ObservationString(CurrentPlayer());
                previous_states.insert(hasher(current_observation));
            } else {
                // If it's not hashable, then we need to clear all the hashes in previous states
                previous_states.clear();
            }

            // Execute the selected move
            MoveCards(selected_move);

            // Reset the draw_counter if it's not below 8
            if (draw_counter <= 8) {
                draw_counter = 0;
            }

        }

        // Increase Current Depth ======================================================================================

        ++current_depth;
        if (current_depth >= game_->MaxGameLength()) {
            is_finished = true;
        }

        // Finish Game =================================================================================================

        if (IsSolvable()) {
            // Clear Tableaus
            for (auto & tableau : tableaus) {
                tableau.cards.clear();
            }

            // Clear Foundations & Repopulate
            for (auto & foundation : foundations) {
                foundation.cards.clear();
                for (const auto & rank : RANKS) {
                    Card card = Card(rank, foundation.suit);
                    card.hidden = false;
                    card.location = kFoundation;
                    foundation.cards.push_back(card);
                }
            }

            // Set Game to Finished
            is_finished = true;
        }

    }

    std::vector<double>     SolitaireState::Returns() const {

        /* Returns are calculated through 3 things: the cards on the foundations (points for each is depends on their
         * rank and is defined in `FOUNDATION_POINTS`), the number of hidden cards revealed in the tableau, and the
         * number of cards moved from the waste. */

        if (is_started) {

            double returns;

            // Foundation Score
            double foundation_score = 0.0;
            for (auto & foundation : foundations) {
                for (auto & card : foundation.cards) {
                    foundation_score += FOUNDATION_POINTS.at(card.rank);
                }
            }

            // Tableau Score
            double tableau_score = 0.0;
            int num_hidden_cards = 0;
            for (auto & tableau : tableaus) {
                if (not tableau.cards.empty()) {
                    for (auto & card : tableau.cards) {
                        // Cards that will be revealed by a chance node next turn are not counted
                        if (card.hidden) {
                            num_hidden_cards += 1;
                        }
                    }
                    if (tableau.cards.back().hidden) {
                        num_hidden_cards += -1;
                    }
                }
            }
            tableau_score = (21 - num_hidden_cards) * 20;

            // Waste Score
            double waste_score = 0.0;
            int waste_cards_remaining;
            waste_cards_remaining = deck.cards.size() + deck.waste.size();
            waste_score = (24 - waste_cards_remaining) * 20;

            // Total Score
            returns = foundation_score + tableau_score + waste_score;
            return {returns};
        }

        else {
            return {0.0};
        }

    }

    std::vector<double>     SolitaireState::Rewards() const {

        /* Rewards are calculated by finding the difference between the current returns and `previous_score`.
         * The attribute `previous_score` is set by `DoApplyAction` on non-chance nodes.
         *
         * Highest possible reward per action is 120.0 (e.g. ♠ ← As where As is on a hidden card)
         * Lowest possible reward per action is -100.0 (e.g. 2h ← As where As is in foundation initially) */

        // TODO: Should not be called on chance nodes (undefined and crashes)

        if (is_started) {
            std::vector<double> current_returns = Returns();
            double current_score = current_returns.front();
            return {current_score - previous_score};
        } else {
            return {0.0};
        }

    }

    std::vector<Action>     SolitaireState::LegalActions() const {

        /* Takes the output of CandidateMoves and filters it down and converts each of them to an `Action`. If there
         * are cards left in the deck or waste, it also adds kDraw. If there are no legal moves, kDraw is added anyway
         * just to make sure LegalActions isn't empty (TODO: Find a better method later)
         *
         * We filter candidate moves that would form cycles in the graph by comparing the result of reversible moves to
         * `previous_states`. The one exception to this is moves that move a card to the foundations. This seems to be
         * necessary to get the game to solve quickly in the late game, when almost all moves are reversible. If the
         * state isn't reversible, then all candidate moves are legal moves, as they can't form cycles. */

        std::vector<Action> legal_actions;

        // Adds all candidate_moves to legal_actions that don't form a back edge to a previous state.
        for (const auto & move : CandidateMoves()) {

            // We allow all moves to the foundation
            if (move.target.location == kFoundation) {
                legal_actions.push_back(move.ActionId());
            }

            // Else if the state is reversible ...
            else if (is_reversible) {

                // And the candidate move is reversible ...
                if (IsReversible(move)) {

                    // Then get the resulting state
                    auto child = Child(move.ActionId());

                    // Then get the hash of the child state
                    auto child_hash = hasher(child->ObservationString());

                    // If the child state is in previous_states, then it forms a loop
                    if (previous_states.count(child_hash) > 0) {
                        continue;
                    }

                    // Otherwise, it doesn't form a loop and we can add it to legal actions
                    else {
                        legal_actions.push_back(move.ActionId());
                    }

                }

                // And the candidate move is not reversible ...
                else {
                    legal_actions.push_back(move.ActionId());
                }
            }

            // If the state isn't reversible, then all candidate_moves are legal actions
            else {
                legal_actions.push_back(move.ActionId());
            }

        }

        // kDraw is added if there are cards to draw from or all candidate moves were not legal actions.
        if (deck.cards.size() + deck.waste.size() > 0 or legal_actions.empty()) {
            legal_actions.push_back(kDraw);
        }

        // Sorts the actions as required by tests
        std::sort(legal_actions.begin(), legal_actions.end());

        return legal_actions;

    }

    std::vector<std::pair<Action, double>> SolitaireState::ChanceOutcomes() const {

        /* At the beginning of the game, `kSetup` is always called. When a hidden card is about to be revealed, this
         * method provides kReveal moves corresponding to the cards that haven't been revealed yet. The probability
         * of choosing a particular card to reveal is uniformly distributed (1 / number of unrevealed cards). */

        if (!is_setup) {
            return {{kSetup, 1.0}};
        } else {
            std::vector<std::pair<Action, double>> outcomes;
            const double p = 1.0 / (52 - revealed_cards.size());

            for (int i = 1; i <= 52; i++) {
                if (std::find(revealed_cards.begin(), revealed_cards.end(), i) != revealed_cards.end()) {
                    continue;
                } else {
                    outcomes.emplace_back(i, p);
                }
            }
            return outcomes;
        }
    }

    // Other Methods ---------------------------------------------------------------------------------------------------

    std::vector<Card>       SolitaireState::Targets(const std::optional<std::string> & location) const {

        /* A target is a card that another card can be moved to. They only exist in tableaus and foundations, and in
         * both, they are the top card (equivalently, the one at `cards.back()`). If `location` is not given, this
         * method returns all targets in all tableaus and foundations. If it is "tableau" or "foundation", then it
         * returns all targets in those locations instead.
         *
         * If a tableau or foundation is empty, a special card is a target instead (i.e. Card("", "") for tableaus,
         * Card("", x) for foundations where x is the foundation suit). These special cards don't actually exist
         * in their containers `cards` attribute. */

        std::string loc = location.value_or("all");
        std::vector<Card> targets;

        // Gets targets from tableaus
        if (loc == "tableau" or loc == "all") {
            for (const Tableau & tableau : tableaus) {
                std::vector<Card> current_targets = tableau.Targets();
                targets.insert(targets.end(), current_targets.begin(), current_targets.end());
            }
        }

        // Gets targets from foundations
        if (loc == "foundation" or loc == "all") {
            for (const Foundation & foundation : foundations) {
                std::vector<Card> current_targets = foundation.Targets();
                targets.insert(targets.end(), current_targets.begin(), current_targets.end());
            }
        }

        // Returns targets as a vector of cards in all piles specified by "location"
        return targets;

    }

    std::vector<Card>       SolitaireState::Sources(const std::optional<std::string> & location) const {

        /* A source is a card that can be moved, along with all other cards below it, to a target card. They exist
         * in tableaus, foundations, and the waste. In the waste, only the front card can be a source. In foundations,
         * only the back card can be one. And in tableaus, all unhidden cards are sources. If `location` isn't provided
         * to this method, then sources from all of these piles are returned. If it is (e.g. `location` is "tableau",
         * "foundation", or "waste") then only sources in those kinds of piles are returned.
         *
         * Because special cards can't be moved and don't actually exist in the `cards` container, they will never
         * show up in sources. This behavior is mostly defined in the pile classes, this method just concatenates
         * all of their sources together. */

        std::string loc = location.value_or("all");
        std::vector<Card> sources;

        // Gets sources from tableaus
        if (loc == "tableau" or loc == "all") {
            for (const Tableau & tableau : tableaus) {
                std::vector<Card> current_sources = tableau.Sources();
                sources.insert(sources.end(), current_sources.begin(), current_sources.end());
            }
        }

        // Gets sources from foundations
        if (loc == "foundation" or loc == "all") {
            for (const Foundation & foundation : foundations) {
                std::vector<Card> current_sources = foundation.Sources();
                sources.insert(sources.end(), current_sources.begin(), current_sources.end());
            }
        }

        // Gets sources from waste
        if (loc == "waste" or loc == "all") {
            std::vector<Card> current_sources = deck.Sources();
            sources.insert(sources.end(), current_sources.begin(), current_sources.end());
        }

        // Returns sources as a vector of cards in all piles specified by "location"
        return sources;
    }

    std::vector<Move>       SolitaireState::CandidateMoves() const {

        /* Returns a list of moves that are technically legal given the rules of solitaire. They are filtered
         * down inside of `LegalActions()` in order to make sure the game terminates. We don't allow a king to be moved
         * to any empty tableau, only the first occurrence of one. Furthermore, we only allow this to happen if the king
         * is not the top card of the pile. Filtering these actions has no effect on the completeness of the game tree
         * because the order of tableaus has no effect on the game. */

        std::vector<Move> candidate_moves;
        std::vector<Card> targets = Targets();
        std::vector<Card> sources = Sources();

        /* Used to indicate that we have already seen a Card("", "") as a target once before
         * Essentially we are making sure Targets() has no duplicates, but only Card("", "") can be duplicated,
         * which is why we only check for that instead of checking for every target in Targets(); */

        bool found_empty_target = false;

        for (const auto & target : targets) {

            // Here we make sure only the first empty tableau can have a king move to it
            // If target is Card("", "") ...
            if (target.rank.empty() and target.suit.empty()) {

                // If we have already processed a Card("", "") target ...
                if (found_empty_target) {
                    // Then skip the duplicate and move on to the next target
                    continue;
                }

                // If we haven't processed a Card("", "") before ...
                else {
                    // Then set the flag to true so we do not process further duplicates, if any
                    found_empty_target = true;
                }

            }

            // Get the legal children of the target (depends on target.location)
            std::vector<Card> legal_children = target.LegalChildren();

            // Iterate over legal children and make sure they are sources in the current state
            for (auto source : legal_children) {

                // We don't need to find target.location here because it's already set when called from Targets()
                // But we do need to find the location of the source because LegalChildren() doesn't set that
                source.location = FindLocation(source);

                // Here we check that the legal child is a source in the current state
                if (std::find(sources.begin(), sources.end(), source) != sources.end()) {

                    // We check that if we're moving from tableau to foundation, that the source is the top of the pile
                    if (target.location == kFoundation and source.location == kTableau) {
                        if (IsTopCard(source)) {
                            candidate_moves.emplace_back(target, source);
                        }
                    }

                    // We prevent moves that shuffle a pile beginning with a king between empty tableaus
                    else if (target == Card("", "") and source.rank == "K") {
                        if (not IsBottomCard(source)) {
                            candidate_moves.emplace_back(target, source);
                        }
                    }

                    // By default, we add all other cases to candidate moves
                    else {
                        candidate_moves.emplace_back(target, source);
                    }

                }

                // If the legal child is not in the sources of this state, do nothing and continue to next legal child
                else {
                    continue;
                }

            }
        }

        return candidate_moves;

    }

    Tableau *               SolitaireState::FindTableau(const Card & card) const {

        /* Returns a pointer to a tableau that contains the card argument. It doesn't rely on the cards `location`
         * attribute at all, only its `rank` and `suit`. Don't call this with a card that isn't in any tableau. */

        // This branch finds the first Card("", "") in the tableaus
        if (card.rank.empty() and card.suit.empty()) {
            for (auto & tableau : tableaus) {
                if (tableau.cards.empty()) {
                    return const_cast<Tableau *>(& tableau);
                }
            }
        }

        // This branch handles finding any ordinary card in the tableaus
        else {
            for (auto & tableau : tableaus) {
                if (not tableau.cards.empty()) {
                    if (std::find(tableau.cards.begin(), tableau.cards.end(), card) != tableau.cards.end()) {
                        return const_cast<Tableau *>(& tableau);
                    }
                }
            }
        }

    }

    Foundation *            SolitaireState::FindFoundation(const Card & card) const {

        /* Returns a pointer to a foundations that contains the card argument. It doesn't rely on the cards `location`
         * attribute at all, only its `rank` and `suit`. Don't call this with a card that isn't in any foundation. */

        // This branch handles special foundation cards, Card("", x) where x is in `SUITS`.
        if (card.rank.empty()) {
            for (auto & foundation : foundations) {
                if (foundation.cards.empty() and foundation.suit == card.suit) {
                    return const_cast<Foundation *>(& foundation);
                }
            }
        }

        // This branch handles finding any ordinary card in the foundation
        else {
            for (auto & foundation : foundations) {
                if (not foundation.cards.empty() and foundation.suit == card.suit) {
                    if (std::find(foundation.cards.begin(), foundation.cards.end(), card) != foundation.cards.end()) {
                        return const_cast<Foundation *>(& foundation);
                    }
                }
            }
        }

    }

    Location                SolitaireState::FindLocation(const Card & card) const {

        /* Attempts to find `card` in the current state. It searches in the order: tableaus, foundations, waste, and
         * then finally, deck. If `card` isn't found, it returns `kMissing`. Otherwise it returns the `Location`
         * corresponding to where it was found (i.e. `kTableau`, `kFoundation`, `kWaste`, `kDeck`) */

        // OPTIMIZE: Shouldn't have to iterate through all piles just to find a card

        // Handles special cards
        if (card.rank.empty()) {
            if (card.suit.empty()) {
                return kTableau;
            } else {
                return kFoundation;
            }
        }

        // Attempts to find the card in a tableau
        for (auto & tableau : tableaus) {
            if (std::find(tableau.cards.begin(), tableau.cards.end(), card) != tableau.cards.end()) {
                return kTableau;
            }
        }

        // Attempts to find the card in a foundation
        for (auto & foundation : foundations) {
            if (std::find(foundation.cards.begin(), foundation.cards.end(), card) != foundation.cards.end()) {
                return kFoundation;
            }
        }

        // Attempts to find the card in the waste
        if (std::find(deck.waste.begin(), deck.waste.end(), card) != deck.waste.end()) {
            return kWaste;
        }

        // Attempts to find the card in the deck
        if (std::find(deck.cards.begin(), deck.cards.end(), card) != deck.cards.end()) {
            return kDeck;
        }

        // Default value is returned if the card isn't found
        return kMissing;

    }

    void                    SolitaireState::MoveCards(const Move & move) {

        /* This method is called by DoApplyAction when the action is kMove. It is the method that actually edits the
         * state to move cards between piles */

        // Unpack target and source from move
        Card target = move.target;
        Card source = move.source;

        // Find their locations in this state
        target.location = FindLocation(target);
        source.location = FindLocation(source);

        std::vector<Card> split_cards;

        switch (source.location) {
            case kTableau : {
                split_cards = FindTableau(source)->Split(source);
                break;
            }
            case kFoundation : {
                split_cards = FindFoundation(source)->Split(source);
                break;
            }
            case kWaste : {
                split_cards = deck.Split(source);
                break;
            }
            default : {
                std::cout << YELLOW << "WARNING: 'source' is not in a tableau, foundation, or waste" << RESET << std::endl;
                std::cout << YELLOW << "WARNING: 'source' = " << source.ToString() << RESET << std::endl;
            }
        }

        switch (target.location) {
            case kTableau : {
                auto target_container = FindTableau(target);
                target_container->Extend(split_cards);
                break;
            }
            case kFoundation : {
                auto target_container = FindFoundation(target);
                target_container->Extend(split_cards);
                break;
            }
            default : {
                std::cout << YELLOW << "WARNING: 'target' is not in a tableau or foundation" << RESET << std::endl;
                std::cout << YELLOW << "WARNING: 'target' = " << target.ToString() << RESET << std::endl;
            }
        }

    }

    bool                    SolitaireState::IsOverHidden(const Card & card) const {

        /* Returns true if the provided card is directly on top of a hidden card. This only makes sense in the
         * context of a tableau, so if the card isn't there, it returns false. Used to determine reversibility of
         * moves. Moving a card that is over a hidden one would reveal it, making the move irreversible. */

        if (card.location == kTableau) {
            auto container = FindTableau(card);
            auto p = std::find(container->cards.begin(), container->cards.end(), card);
            auto previous_card = std::prev(p);
            return previous_card->hidden;
        }

        return false;
    }

    bool                    SolitaireState::IsReversible(const Move & move) const {

        /* Determines if a move is reversible. There are three ways that it could be true: if the source is in the
         * waste (because you can't move a card back to the waste), if the source is on top of a hidden card
         * (which would be revealed after the move and couldn't be hidden again) or if the card is the top card of
         * a pile, in which case it couldn't be moved back */

        Card target = move.target;
        Card source = move.source;

        // target.location = FindLocation(target);
        source.location = FindLocation(source);

        switch (source.location) {

            // Cards cannot be moved back to the waste, therefore this is not reversible
            case kWaste : {
                return false;
            }
            // Cards can always be moved back from the foundation on the next state
            case kFoundation : {
                return true;
            }
            // Cards can be moved back if they don't reveal a hidden card upon being moved
            case kTableau : {
                if (IsBottomCard(source) or IsOverHidden(source)) {
                    return false;
                } else {
                    return true;
                }
            }
            // Cards can't be moved at all if they are in kDeck or kMissing
            default : {
                std::cout << YELLOW << "WARNING: 'source' is not in a tableau, foundation, or waste" << RESET << std::endl;
                // I guess we return false here since it's not even a valid move?
                return false;
            }

        }
    }

    bool                    SolitaireState::IsBottomCard(Card card) const {

        /* Returns true if the card is at the front of the `cards` vector that it's contained in.
         * Used to help determine reversibility as you can't move a pile back to an empty tableau (except for
         * ones starting with a king in some circumstances) */

        // Only implemented for cards in a tableau at the moment.
        if (card.location == kTableau) {
            auto container = FindTableau(card);
            // This line assumes three things:
            //  - That `FindTableau()` actually found a tableau (TODO: add exception to it later)
            //  - That the tableau found is not empty, otherwise we could not call `cards.front()`
            //  - That `card` is an ordinary card (e.g. its suit and rank are defined)
            return container->cards.front() == card;
        } else {
            // While it a card could be the bottom one in different locations, there isn't much use
            return false;
        }
    }

    bool                    SolitaireState::IsTopCard(const Card & card) const {

        /* Similar to `IsBottomCard` except that it checks if the card is at the back of the vector.
         * Assumes that card is found in a container, meaning container.back() will be defined (this isn't true for
         * special cards (e.g. Card("", "") or Card("", "s") etc.) */

        std::deque<Card> container;
        switch (card.location) {
            case kTableau : {
                container = FindTableau(card)->cards;
                return card == container.back();
            }
            case kFoundation : {
                container = FindFoundation(card)->cards;
                return card == container.back();
            }
            case kWaste : {
                container = deck.waste;
                return card == container.front();
            }
            default : {
                return false;
            }
        }
    }

    bool                    SolitaireState::IsSolvable() const {

        /* When there are no cards left in the deck or waste and no hidden cards left in the tableau, the state
         * is guaranteed to be solvable. Used in DoApplyAction to complete the game in one step so that we don't
         * waste time moving cards to the foundation. It's not uncommon for implementations of solitaire to finish
         * the game automatically for you when you reach this state */

        if (deck.cards.empty() and deck.waste.empty()) {
            for (auto & tableau : tableaus) {
                if (not tableau.cards.empty()) {
                    for (auto & card : tableau.cards) {
                        // Returns false if at least one tableau card is hidden
                        if (card.hidden) {
                            return false;
                        }
                    }
                } else {
                    continue;
                }
            }
            // Only returns true if all cards are revealed and there are no cards in deck or waste
            return true;
        }
        else {
            // Returned if there are cards in deck or waste
            return false;
        }

    }

    // SolitaireGame Methods ===========================================================================================

    SolitaireGame::SolitaireGame(const GameParameters & params) :
        Game(kGameType, params),
        num_players_(ParameterValue<int>("players")) {

    }

    int     SolitaireGame::NumDistinctActions() const {
        /* This is just the number of elements in `enum ActionType` in solitaire.h. If actions are added or removed
         * there, this number will have to be changed accordingly */
        return 206;
    }

    int     SolitaireGame::MaxGameLength() const {
        /* The maximum number of actions that can be taken in a game. Because the information state is represented
         * as a vector of actions taken so far, they both must be the same. See `InformationStateTensorShape()` for
         * more information on choosing this number. */

        return 500;
    }

    int     SolitaireGame::NumPlayers() const {
        /* Solitaire is a single player game, any different number of players might break this implementation. */
        return 1;
    }

    double  SolitaireGame::MinUtility() const {
        /* There are negative rewards in this game, but only for moving a card from the foundation to a tableau.
         * The returns for a move and its inverse are zero sum though, so we can never actually go below 0.0 */
        return 0.0;
    }

    double  SolitaireGame::MaxUtility() const {
        /* If a game is won, that means there are no cards left in the waste, all cards are revealed, and every
         * card is on its foundation. The way returns are setup currently, the maximum return is 3,220.0
         *   21 hidden cards * 20 points   = 420      (Revealing hidden cards)
         *   24 cards in waste * 20 points = 480      (Moving cards from the waste)
         *   4 suits * 580 points          = 2,320    (Having cards in the foundation)
         *   TOTAL                         = 3,320 */
        return 3220.0;
    }

    std::vector<int> SolitaireGame::InformationStateTensorShape() const {
        /* Basically the same thing as `MaxGameLength()`, although this particular method seems to have more methods
         * that depend on it than `MaxGameLength()`. In a game without loop prevention, solitaire be played infinitely.
         * With restrictions, it's hard to say what the maximum length would be. If this number is set too low, the
         * game will crash if a game goes beyond that length. If it's too high, then we are wasting time and memory. */

        return {500};
    }

    std::vector<int> SolitaireGame::ObservationTensorShape() const {
        /* Observation tensors are always padded with the maximum length of each pile. The deck starts with 24 cards
         * and no cards can be added to it, the waste can be at most the length of the deck so its maximum length is 24
         * as well. A tableau can have at most 13 cards from a king to an ace and 6 from hidden cards underneath it, so
         * its maximum length is 19. Foundations can only have one suit in them, and there are 13 cards in a suit.
         *   Deck       = 24
         *   Waste      = 24
         *   Tableau    = 19 * 7 = 133
         *   Foundation = 13 * 4 = 52
         *   TOTAL      = 233 */
        return {233};
    }

    std::unique_ptr<State> SolitaireGame::NewInitialState() const {
        return std::unique_ptr<State>(new SolitaireState(shared_from_this()));
    }

    std::shared_ptr<const Game> SolitaireGame::Clone() const {
        return std::shared_ptr<const Game>(new SolitaireGame(*this));
    }

} // namespace open_spiel::solitaire


