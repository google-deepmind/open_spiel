#include "solitaire.h"

#include <deque>
#include <random>
#include <algorithm>
#include <optional>
#include <utility>
#include <math.h>
#include <unordered_map>
#include <assert.h>

#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

#define RESET   "\033[0m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"


/* =====================================================================================================================

    An implementation of Klondike Solitaire with 3 cards drawn at a time & infinite redeals. Part of the difficulty in
    implementing this game is that no matter how many cards are drawn at a time or how many redeals are allowed,
    certain situations lead to an infinite loop (e.g. moving a queen of hearts back and forth between a king of spades
    and king of clubs). In this implementation, we prevent these loops by recognizing when a "reversible move" is made;
    one that has the potential to be returned to at a later state. Starting from this first reversible move, the output
    of `ObservationString` is hashed and stored in `previous_states` in its descendants. `CandidateMoves` then provides
    a list of moves that are technically legal according to the rules of the game and all reversible moves in this list
    are checked to ensure they don't form a back edge to any member of `previous_states`. The moves that satisfy this
    condition are passed along as `LegalActions`.

===================================================================================================================== */

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
        order they were originally revealed in the waste. It has no targets or sources.

    Waste:
        Another ordered set of cards. It has no targets and has only one source card at most. The source
        can be moved to a tableau or foundation, in which case the next card in the waste becomes the
        source. If a card is hidden in the waste, a chance node is forced and chooses what to rank and suit
        to reveal the card as.

    Source Card:
        The card being moved to a target. In the case of a pile being moved, it is the first card
        in the pile, e.g. in the move (Ks <- Qh) where the cards being moved are {Qh, Js, 10h},
        Qh would be the source card. Can be in any pile except the deck.

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

/* IDEAS
   - One way to test if the game tree is complete (kind of) is to search the game with no
     restrictions to a large depth. Record the maximum return in that simulation. Then search
     the tree with restrictions (so it terminates), ensuring that with restrictions doesn't
     limit the maximum return.
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

    std::hash<std::string> hasher;

    bool COLOR_FLAG = true;

    std::string color(std::string color) {
        return COLOR_FLAG ? std::move(color) : "";
    }

    std::vector<std::string> GetOppositeSuits(const std::string & suit) {

        // tracer trace("GetOppositeSuits()");

        if (suit == "s" or suit == "c") {
            return {"h", "d"};
        } else if (suit == "h" or suit == "d") {
            return {"s", "c"};
        } else {
            std::cout << YELLOW << "WARNING: `suit` is not in (s, h, c, d)" << RESET << std::endl;
        }

    }

    std::string LocationString(Location location) {

        // tracer trace("LocationString()");

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
        // TODO: Handle empty piles

        // tracer trace("ToCardIndices()");

        std::vector<double> index_vector;
        for (auto & card : pile) {
            if (card.hidden) {
                index_vector.push_back(HIDDEN_CARD);
            } else {
                index_vector.push_back((int) card);
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
        // tracer trace("Card(rank, suit)");
    }

    Card::Card() : rank(""), suit(""), hidden(true), location(kMissing) {
        // tracer trace("Card()");
    }

    Card::Card(int index) : hidden(false), location(kMissing) {

        // tracer trace("Card(int index)");

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
        // OPTIMIZE: This shouldn't have to recalculate the index after it's been set once before.

        // tracer trace("Card::operator int()");

        // NEW WAY OF GETTING INDEX
        std::pair<std::string, std::string> ranksuit = {rank, suit};
        return RANKSUIT_TO_INDEX.at(ranksuit);

        /* OLD WAY OF GETTING INDEX
        int old_index;
        if (rank.empty()) {
            if      (suit == "s")  { old_index = -1; }
            else if (suit == "h")  { old_index = -2; }
            else if (suit == "c")  { old_index = -3; }
            else if (suit == "d")  { old_index = -4; }
            else if (suit.empty()) { old_index = -5; }
        }
        else {
            int rank_value = GetIndex(RANKS, rank);
            int suit_value = GetIndex(SUITS, suit);
            old_index = 13 * suit_value + rank_value;
        }

        std::cout << RED << "Comparing old and new methods of getting card index" << RESET << std::endl;
        std::cout << "Card we are converting = " << this->ToString() << std::endl;
        std::cout << "Old index = " << old_index << std::endl;
        std::cout << "New index = " << index << std::endl;
        */

        /*
        if (hidden) {
            return (int) HIDDEN_CARD;
        } else {
            std::pair<std::string, std::string> ranksuit = {rank, suit};
            try {
                return RANKSUIT_TO_INDEX.at(ranksuit);
            } catch (std::out_of_range()) {
                std::cout << "ERROR: rank = " << ranksuit.first << "; suit = " << ranksuit.second << std::endl;
            }
        }
        */
    }

    bool Card::operator==(Card & other_card) const {
        // // tracer trace("Card::operator==");
        return rank == other_card.rank and suit == other_card.suit;
    }

    bool Card::operator==(const Card & other_card) const {
        // // tracer trace("Card::operator==");
        return rank == other_card.rank and suit == other_card.suit;
    }

    std::vector<Card> Card::LegalChildren() const {

        // tracer trace("LegalChildren()");

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
        // TODO: Don't include formatting whitespace in this method

        // tracer trace("Card::ToString()");

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

        // tracer trace("Deck()");

        for (int i = 1; i <= 24; i++) {
            cards.emplace_back();
        }
        for (auto & card : cards) {
            card.location = kDeck;
        }
    }

    std::vector<Card> Deck::Sources() const {

        // tracer trace("Deck::Sources()");

        // TODO: Can simplify this if statement
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

        // tracer trace("Deck::Split()");

        std::vector<Card> split_cards;
        if (waste.front() == card) {
            split_cards = {waste.front()};
            waste.pop_front();
            return split_cards;
        }
    }

    void Deck::draw(unsigned long num_cards) {

        // tracer trace("Deck::draw()");

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

        // tracer trace("Deck::rebuild()");

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
        // tracer trace("Foundation()");
        cards = {};
    }

    Foundation::Foundation(std::string suit) : suit(std::move(suit)) {
        // tracer trace("Foundation(suit)");
        cards = {};
    }

    std::vector<Card> Foundation::Sources() const {

        // tracer trace("Foundation::Sources()");

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

        // tracer trace("Foundation::Targets()");

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

        // tracer trace("Foundation::Split()");

        std::vector<Card> split_cards;
        if (cards.back() == card) {
            split_cards = {cards.back()};
            cards.pop_back();
            return split_cards;
        }
    }

    void Foundation::Extend(const std::vector<Card> & source_cards) {

        // tracer trace("Foundation::Extend()");

        // TODO: Can only extend with ordinary cards
        // TODO: Probably no use for setting hidden to false.

        for (auto card : source_cards) {
            card.location = kFoundation;
            cards.push_back(card);
        }
    }

    // Tableau Methods =================================================================================================

    Tableau::Tableau() {
        // tracer trace("Tableau()");
    }

    Tableau::Tableau(int num_cards) {

        // tracer trace("Tableau(num_cards)");

        for (int i = 1; i <= num_cards; i++) {
            cards.emplace_back();
        }
        for (auto & card : cards) {
            card.location = kTableau;
        }
    }

    std::vector<Card> Tableau::Sources() const {

        // tracer trace("Tableau::Sources()");

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
        // DECISION: Should targets return a vector, even though it will only return one card?
        // If the tableau is not empty, targets is just a vector of the top card of the tableau

        // tracer trace("Tableau::Targets()");

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

        // TODO: How to handle a split when card isn't in this tableau?

        // tracer trace("Tableau::Split()");

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

        // tracer trace("Tableau::Extend()");

        for (auto card : source_cards) {
            card.location = kTableau;
            cards.push_back(card);
        }
    }

    // Move Methods ====================================================================================================

    Move::Move(Card target_card, Card source_card) {
        // tracer trace("Move(target_card, source_card");
        target = std::move(target_card);
        source = std::move(source_card);
    }

    Move::Move(Action action_id) {
        // tracer trace("Move(action_id)");
        auto card_pair = ACTION_TO_MOVE.at(action_id);
        target = Card(card_pair.first);
        source = Card(card_pair.second);
    }

    std::string Move::ToString() const {
        // tracer trace("Move::ToString()");
        std::string result;
        absl::StrAppend(&result, target.ToString(), " ", "\U00002190", " ",source.ToString());
        return result;
    }

    Action Move::ActionId() const {
        // tracer trace("Move::ActionId()");
        return MOVE_TO_ACTION.at(std::make_pair((int) target, (int) source));
    }

    // SolitaireState Methods ==========================================================================================

    SolitaireState::SolitaireState(std::shared_ptr<const Game> game) :
        State(game),
        deck(),
        foundations(),
        tableaus() {
            // tracer trace("SolitaireState(game)");
            is_started = false;
            is_setup = false;
            previous_score = 0.0;
        }

    // Overriden Methods -----------------------------------------------------------------------------------------------

    Player                  SolitaireState::CurrentPlayer() const {

        // tracer trace("SolitaireState::CurrentPlayer()");

        // There are only two players in this game: chance and player 1.
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

        // tracer trace("SolitaireState::IsTerminal()");

        // Even if a player can't make any more moves this game, we still want to reveal
        /*
        int i = 0;
        for (auto & card : deck.waste) {
            if (card.hidden) { return false; }
            else if (i >= 2) {
                break;
            }
            else {
                ++i;
            }
        }
        */

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

        // tracer trace("SolitaireState::IsChanceNode()");

        if (not is_setup) {
            // If setup is not started, this is a chance node
            return true;
        }

        else {

            // If there is a hidden card on the top of a tableau, this is a chance ndoe
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

        // tracer trace("SolitaireState::ToString()");

        std::string result;

        absl::StrAppend(&result, "DRAW COUNTER    : ", draw_counter);

        absl::StrAppend(&result, "\nIS REVERSIBLE   : ", is_reversible);

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

        return result;
    }

    std::string             SolitaireState::ActionToString(Player player, Action action_id) const {
        // TODO: Probably use the enum names instead of the values the represent

        // tracer trace("SolitaireState::ActionToString()");

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
        // tracer trace("SolitaireState::InformationStateString()");
        return HistoryString();
    }

    std::string             SolitaireState::ObservationString(Player player) const {

        // tracer trace("SolitaireState::ObservationString()");

        std::string result;

        /*
        absl::StrAppend(&result, "\n\nDECK        : ");
        for (const Card & card : deck.cards) {
            absl::StrAppend(&result, card.ToString(), " ");
        }

        absl::StrAppend(&result, "\nWASTE       : ");
        for (const Card & card : deck.waste) {
            absl::StrAppend(&result, card.ToString(), " ");
        }
        */

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

        // tracer trace("SolitaireState::InformationStateTensor()");

        values->resize(game_->InformationStateTensorShape()[0]);
        std::fill(values->begin(), values->end(), kInvalidAction);

        int i = 0;
        for (auto & action : History()) {
            (*values)[i] = action;
            ++i;
        }

    }

    void                    SolitaireState::ObservationTensor(Player player, std::vector<double> *values) const {

        // TODO: Not sure if any pile being empty would have an effect on the final result

        // tracer trace("SolitaireState::ObservationTensor()");

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

        // Set previous_score to be equal to the returns from this state

        // tracer trace("SolitaireState::DoApplyAction()");

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
        // Equal to the sum of all rewards up to the current state

        // tracer trace("SolitaireState::Returns()");

        if (is_started) {

            double returns = 0.0;

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
            int waste_cards_remaning;
            waste_cards_remaning = deck.cards.size() + deck.waste.size();
            waste_score = (24 - waste_cards_remaning) * 20;

            // Total Score
            returns = foundation_score + tableau_score + waste_score;
            return {returns};
        }

        else {
            return {0.0};
        }

    }

    std::vector<double>     SolitaireState::Rewards() const {
        // TODO: Should not be called on chance nodes (undefined and crashes)
        // Highest possible reward per action is 120.0 (e.g. ♠ ← As where As is on a hidden card)
        // Lowest possible reward per action is -100.0 (e.g. 2h ← As where As is in foundation initially) */

        // tracer trace("SolitaireState::Rewards()");

        if (is_started) {
            std::vector<double> current_returns = Returns();
            double current_score = current_returns.front();
            return {current_score - previous_score};
        } else {
            return {0.0};
        }

    }

    std::vector<Action>     SolitaireState::LegalActions() const {

        // tracer trace("SolitaireState::LegalActions()");

        std::vector<Action> legal_actions;

        // Adds all candidate_moves to legal_actions that don't form a back edge to a previous state.
        for (const auto & move : CandidateMoves()) {

            // TODO: Check if this helps solve the endgame or not

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

        // tracer trace("SolitaireState::ChanceOutcomes()");

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

        // tracer trace("SolitaireState::Targets()");

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

        // tracer trace("SolitaireState::Sources()");

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

        // tracer trace("SolitaireState::CandidateMoves()");

        std::vector<Move> candidate_moves;
        std::vector<Card> targets = Targets();
        std::vector<Card> sources = Sources();

        for (const auto & target : targets) {
            std::vector<Card> legal_children = target.LegalChildren();

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

        // tracer trace("SolitaireState::FindTableau()");

        if (card.rank.empty() and card.suit.empty()) {
            for (auto & tableau : tableaus) {
                if (tableau.cards.empty()) {
                    return const_cast<Tableau *>(& tableau);
                }
            }
        }

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

        // tracer trace("SolitaireState::FindFoundation()");

        if (card.rank.empty()) {
            for (auto & foundation : foundations) {
                if (foundation.cards.empty() and foundation.suit == card.suit) {
                    return const_cast<Foundation *>(& foundation);
                }
            }
        }
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
        // OPTIMIZE: Shouldn't have to iterate through all piles just to find a card

        // tracer trace("SolitaireState::FindLocation()");

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

        // tracer trace("SolitaireState::MoveCards()");

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

        // tracer trace("SolitaireState::IsOverHidden()");

        if (card.location == kTableau) {
            auto container = FindTableau(card);
            auto p = std::find(container->cards.begin(), container->cards.end(), card);
            auto previous_card = std::prev(p);
            return previous_card->hidden;
        }

        return false;
    }

    bool                    SolitaireState::IsReversible(const Move & move) const {

        // tracer trace("SolitaireState::IsReversible()");

        Card target = move.target;
        Card source = move.source;

        target.location = FindLocation(target);
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

        // tracer trace("SolitaireState::IsBottomCard()");

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
        // Assumes that card is found in a container, meaning container.back() will be defined
        // This isn't true for special cards (e.g. Card("", "") or Card("", "s") etc.)

        // tracer trace("SolitaireState::IsTopCard()");

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

        // tracer trace("SolitaireState::IsSolvable()");

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
        return 206;
    }

    int     SolitaireGame::MaxGameLength() const {
        return 3000;
    }

    int     SolitaireGame::NumPlayers() const {
        return 1;
    }

    double  SolitaireGame::MinUtility() const {
        return 0.0;
    }

    double  SolitaireGame::MaxUtility() const {
        return 3220.0;
    }

    std::vector<int> SolitaireGame::InformationStateTensorShape() const {
        return {3000};
    }

    std::vector<int> SolitaireGame::ObservationTensorShape() const {
        return {233};
    }

    std::unique_ptr<State> SolitaireGame::NewInitialState() const {
        return std::unique_ptr<State>(new SolitaireState(shared_from_this()));
    }

    std::shared_ptr<const Game> SolitaireGame::Clone() const {
        return std::shared_ptr<const Game>(new SolitaireGame(*this));
    }

} // namespace open_spiel::solitaire


