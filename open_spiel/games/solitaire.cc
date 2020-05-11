#include "solitaire.h"

#include <optional>
#include <exception>

#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::solitaire {

    namespace {
        const GameType kGameType {
            "solitaire",
            "Klondike Solitaire",
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
            {{"players", GameParameter(kDefaultPlayers)},
             {"colored", GameParameter(kDefaultColored)},
             {"depth_limit", GameParameter(kDepthLimit)}}
        };

        std::shared_ptr<const Game> Factory(const GameParameters & params) {
            return std::shared_ptr<const Game>(new SolitaireGame(params));
        }

        REGISTER_SPIEL_GAME(kGameType, Factory);
    }

    // region Miscellaneous Functions =========================================================================================

    std::vector<SuitType> GetOppositeSuits(const SuitType & suit) {
        /* Just returns a vector of the suits of opposite color. For red suits (kH and kD), this returns the
           black suits (kS and kC). For a black suit, this returns the red suits. The last `SuitType` would be
          `kNoSuit` which should only occur with empty tableau cards or hidden cards. Empty tableau cards should
           accept any suit, but hidden cards are the opposite; they shouldn't accept any. There isn't really a use
           case for calling this function with the suit of a hidden card though.

        WARNING: Don't use this to find opposite suits of a hidden card, the result is meaningless */

        if (suit == kS || suit == kC) {
            return {kH, kD};
        } else if (suit == kH || suit == kD) {
            return {kS, kC};
        } else if (suit == kNoSuit) {
            return {kS, kH, kC, kD};
        } else {
            std::cerr << "WARNING: `suit` is not in (s, h, c, d)" << std::endl;
            return {};
        }
    }

    int GetCardIndex(RankType rank, SuitType suit) {
        /* Using a given rank and/or suit, gets an integer representing the index of the card. */

        if (rank == kHiddenRank or suit == kHiddenSuit) {
            // Handles hidden cards
            return HIDDEN_CARD;
        } else if (rank == kNoRank) {
            // Handles special cards
            if (suit == kNoSuit) {
                // Handles empty tableau cards
                return EMPTY_TABLEAU_CARD;
            } else {
                // Handles empty foundation cards
                switch (suit) {
                    case kS : {
                        return EMPTY_SPADE_CARD;
                    }
                    case kH : {
                        return EMPTY_HEART_CARD;
                    }
                    case kC : {
                        return EMPTY_CLUB_CARD;
                    }
                    case kD : {
                        return EMPTY_DIAMOND_CARD;
                    }
                    default : {
                        std::cerr << "WARNING: Something went wrong in GetCardIndex()" << std::endl;
                    }
                }
            }
        } else {
            // Handles ordinary cards
            return (suit - 1) * 13 + rank;
        }
    }

    int GetMaxSize(LocationType location) {
        switch (location) {
            case kDeck ... kWaste : {
                return 24;
            }
            case kFoundation : {
                return 13;
            }
            case kTableau : {
                return 19;
            }
            default : {
                return 0;
            }
        }
    }

    // endregion

    // region Card Methods ====================================================================================================

    Card::Card(bool hidden, SuitType suit, RankType rank, LocationType location) :
        hidden(hidden), suit(suit), rank(rank), location(location) {
        hidden   = hidden;
        suit     = suit;
        rank     = rank;
        location = location;
    }

    Card::Card(int index, bool hidden, LocationType location) :
        index(index), hidden(hidden), location(location) {
        // index == HIDDEN_CARD ? hidden = true : hidden = false;

        if (!hidden) {
            switch (index) {
                case HIDDEN_CARD : {
                    rank = kHiddenRank;
                    suit = kHiddenSuit;
                    break;
                }
                case EMPTY_TABLEAU_CARD : {
                    rank = kNoRank;
                    suit = kNoSuit;
                    break;
                }
                case EMPTY_SPADE_CARD : {
                    rank = kNoRank;
                    suit = kS;
                    break;
                }
                case EMPTY_HEART_CARD : {
                    rank = kNoRank;
                    suit = kH;
                    break;
                }
                case EMPTY_CLUB_CARD : {
                    rank = kNoRank;
                    suit = kC;
                    break;
                }
                case EMPTY_DIAMOND_CARD : {
                    rank = kNoRank;
                    suit = kD;
                    break;
                }
                default : {
                    // TODO: Find a more elegant way of doing while keeping kNoX inside enums for LegalChildren()
                    rank = static_cast<RankType>(1 + ((index - 1) % 13));
                    suit = static_cast<SuitType>(1 + floor((index - 1) / 13));
                }
            }
        }
    }

    int Card::GetIndex() const {
        /* Basically it just calculates the index if it hasn't been calculated before, otherwise
         * it will just return a stored value. If `force` is true and the card isn't hidden, then
         * the index is calculated again. */
        return hidden ? HIDDEN_CARD : GetCardIndex(rank, suit);
    }

    std::string Card::ToString(bool colored) const {
        std::string result;

        // Determine color of string
        if (colored && !hidden) {
            if (suit == kS || suit == kC) {
                absl::StrAppend(&result, BLACK);
            } else if (suit == kH || suit == kD) {
                absl::StrAppend(&result, RED);
            }
        }

        // Determine contents of string
        if (rank == kHiddenRank or suit == kHiddenSuit) {
            absl::StrAppend(&result, GLYPH_HIDDEN, " ");
        } else if (rank == kNoRank and suit == kNoSuit) {
            absl::StrAppend(&result, GLYPH_EMPTY);
        } else {
            absl::StrAppend(&result, RANK_STRS.at(rank));
            absl::StrAppend(&result, SUIT_STRS.at(suit));
        }

        if (colored) {
            // Reset color if applicable
            absl::StrAppend(&result, RESET);
        }

        return result;
    }

    std::vector<Card> Card::LegalChildren() const {

        if (hidden) {
            return {};
        } else {
            RankType child_rank;
            std::vector<SuitType> child_suits;
            child_suits.reserve(4);

            switch (location) {
                case kTableau : {
                    switch (rank) {
                        case kNoRank : {
                            if (suit == kNoSuit) {
                                // Empty tableaus can accept a king of any suit
                                child_rank  = kK;
                                child_suits = SUITS;
                                break;
                            } else {
                                return {};
                            }
                        }
                        case k2 ... kK : {
                            // Ordinary cards (except aces) can accept cards of an opposite suit that is one rank lower
                            child_rank  = static_cast<RankType>(rank - 1);
                            child_suits = GetOppositeSuits(suit);
                            break;
                        }
                        default : {
                            // This will catch kA and kHiddenRank
                            return {};
                        }
                    }
                    break;
                }
                case kFoundation : {
                    switch (rank) {
                        case kNoRank : {
                            if (suit != kNoSuit) {
                                child_rank  = static_cast<RankType>(rank + 1);
                                child_suits = {suit};
                                break;
                            } else {
                                return {};
                            }
                        }
                        case kA ... kQ : {
                            // Cards (except kings) can accept a card of the same suit that is one rank higher
                            child_rank  = static_cast<RankType>(rank + 1);
                            child_suits = {suit};
                            break;
                        }
                        default : {
                            // This could catch kK and kHiddenRank
                            return {};
                        }
                    }
                    break;
                }
                default : {
                    // This catches all cards that aren't located in a tableau or foundation
                    return {};
                }
            }

            std::vector<Card> legal_children;
            legal_children.reserve(4);

            // TODO: `child_suits` should never be empty at this line
            if (child_suits.empty()) {
                std::cerr << "WARNING: `child_suits` should never be empty" << std::endl;
                return {};
            }

            for (const auto & child_suit : child_suits) {
                auto child = Card(false, child_suit, child_rank);
                legal_children.push_back(child);
            }

            return legal_children;
        }
    }

    bool Card::operator==(Card & other_card) const {
        return rank == other_card.rank && suit == other_card.suit;
    }

    bool Card::operator==(const Card & other_card) const {
        return rank == other_card.rank && suit == other_card.suit;
    }

    bool Card::operator<(const Card & other_card) const {
        if (suit != other_card.suit) {
            return suit < other_card.suit;
        } else if (rank != other_card.rank) {
            return rank < other_card.rank;
        } else {
            return false;
        }
    }

    // endregion

    // region Pile Methods ====================================================================================================

    Pile::Pile(LocationType type, PileID id, SuitType suit) :
        type(type), id(id), suit(suit), max_size(GetMaxSize(type)) {
        cards.reserve(max_size);
    }

    std::vector<Card> Pile::Targets() const {
        switch (type) {
            case kFoundation : {
                if (!cards.empty()) {
                    return {cards.back()};
                } else {
                    // Empty foundation card with the same suit as the pile
                    return {Card(false, suit, kNoRank, kFoundation)};
                }
            }
            case kTableau : {
                if (!cards.empty()) {
                    auto back_card = cards.back();
                    if (!back_card.hidden) {
                        return {cards.back()};
                    } else {
                        return {};
                    }
                } else {
                    // Empty tableau card (no rank or suit)
                    return {Card(false, kNoSuit, kNoRank, kTableau)};
                }
            }
            default : {
                std::cout << "WARNING: Pile::Targets() called with unsupported type: " << type << std::endl;
                return {};
            }
        }
    }

    std::vector<Card> Pile::Sources() const {
        std::vector<Card> sources;
        sources.reserve(13);
        switch (type) {
            case kFoundation : {
                if (!cards.empty()) {
                    return {cards.back()};
                } else {
                    return {};
                }
            }
            case kTableau : {
                if (!cards.empty()) {
                    for (auto & card : cards) {
                        if (!card.hidden) {
                            sources.push_back(card);
                        }
                    }
                    return sources;
                } else {
                    return {};
                }

            }
            case kWaste : {
                if (!cards.empty()) {
                    int i = 0;
                    for (auto & card : cards) {
                        if (!card.hidden) {
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
            default : {
                std::cout << "WARNING: Pile::Sources() called with unsupported type: " << type << std::endl;
                return {};
            }
        }
    }

    std::vector<Card> Pile::Split(Card card) {
        // TODO: Refactor this method

        std::vector<Card> split_cards;
        switch (type) {
            case kFoundation : {
                if (cards.back() == card) {
                    split_cards = {cards.back()};
                    cards.pop_back();
                }
                break;
            }
            case kTableau : {
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
            case kWaste : {
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
            default : {
                return {};
            }
        }

        return split_cards;
    }

    void Pile::Extend(std::vector<Card> source_cards) {
        for (auto & card : source_cards) {
            // TODO: FIX THIS IMMEDIATELY
            card.location = type;
            cards.push_back(card);
        }
    }

    std::vector<double> Pile::Tensor() const {
        std::vector<double> dv;
        dv.reserve(max_size);
        if (!cards.empty()) {
            for (auto & card : cards) {
                card.hidden ? dv.push_back(HIDDEN_CARD) : dv.push_back(card.GetIndex());
            }
        }
        dv.resize(max_size, NO_CARD);
        return dv;
    }

    std::string Pile::ToString(bool colored) const {
        std::string result;
        for (const auto & card : cards) {
            absl::StrAppend(&result, card.ToString(colored), " ");
        }
        return result;
    }

    // endregion

    // region Move Methods ====================================================================================================

    Move::Move(Card target_card, Card source_card) {
        target = target_card;
        source = source_card;
    }

    Move::Move(RankType target_rank, SuitType target_suit, RankType source_rank, SuitType source_suit) {
        target = Card(false, target_suit, target_rank, kMissing);
        source = Card(false, source_suit, source_rank, kMissing);
    }

    Move::Move(Action action) {
        auto value = ACTION_TO_MOVE.at(action);
        target = value.target;
        source = value.source;
    }

    Action Move::ActionId() const {
        return MOVE_TO_ACTION.at(*this);
    }

    std::string Move::ToString(bool colored) const {
        std::string result;
        absl::StrAppend(&result, target.ToString(colored), " ", GLYPH_ARROW, " ", source.ToString(colored));
        return result;
    }

    bool Move::operator<(const Move & other_move) const {
        int index = target.GetIndex() * 100 + source.GetIndex();
        int other_index = other_move.target.GetIndex() * 100 + other_move.source.GetIndex();
        return index < other_index;
    }

    // endregion

    // region SolitaireState Methods ==========================================================================================

    SolitaireState::SolitaireState(std::shared_ptr<const Game> game) :
        State(game), waste(kWaste, kPileWaste) {

        // Create foundations
        for (const auto & suit : SUITS) {
            foundations.emplace_back(kFoundation, SUIT_TO_PILE.at(suit), suit);
        }

        // Create tableaus
        for (int i = 1; i <= 7; i++) {

            // Create `i` hidden cards
            std::vector<Card> cards_to_add;
            for (int j = 1; j <= i; j++) {
                cards_to_add.emplace_back(true, kHiddenSuit, kHiddenRank, kTableau);
            }

            // Create a new tableau and add cards
            auto tableau  = Pile(kTableau, INT_TO_PILE.at(i));
            tableau.cards = cards_to_add;

            // Add resulting tableau to tableaus
            tableaus.push_back(tableau);
        }

        // Create waste
        for (int i = 1; i <= 24; i++) {
            waste.cards.emplace_back(true, kHiddenSuit, kHiddenRank, kWaste);
        }
    }

    Player                  SolitaireState::CurrentPlayer() const {
        if (IsTerminal()) {
            return kTerminalPlayerId;
        } else if (IsChanceNode()) {
            return kChancePlayerId;
        } else {
            return kPlayerId;
        }
    }

    std::unique_ptr<State>  SolitaireState::Clone() const {
        return std::unique_ptr<State>(new SolitaireState(*this));
    }

    bool                    SolitaireState::IsTerminal() const {
        return is_finished;
    }

    bool                    SolitaireState::IsChanceNode() const {

        for (auto & tableau : tableaus) {
            if (!tableau.cards.empty() && tableau.cards.back().hidden) {
                return true;
            }
        }

        if (!waste.cards.empty()) {
            for (const auto & card : waste.cards) {
                if (card.hidden) {
                    return true;
                }
            }
        }

        return false;

    }

    std::string             SolitaireState::ToString() const {

        std::string result;

        absl::StrAppend(&result, "WASTE       : ", waste.ToString());

        absl::StrAppend(&result, "\nFOUNDATIONS : ");
        for (const auto & foundation : foundations) {
            absl::StrAppend(&result, foundation.Targets()[0].ToString(true), " ");
        }

        absl::StrAppend(&result, "\nTABLEAUS    : ");
        for (const auto & tableau : tableaus) {
            if (!tableau.cards.empty()) {
                absl::StrAppend(&result, "\n", tableau.ToString(true));
            }
        }

        absl::StrAppend(&result, "\nTARGETS : ");
        for (const auto & card : Targets()) {
            absl::StrAppend(&result, card.ToString(true), " ");
        }

        absl::StrAppend(&result, "\nSOURCES : ");
        for (const auto & card : Sources()) {
            absl::StrAppend(&result, card.ToString(true), " ");
        }

        /*
        if (!IsChanceNode()) {
            absl::StrAppend(&result, "\nCANDIDATE MOVES : ");
            for (auto & move : CandidateMoves()) {
                absl::StrAppend(&result, "\n", move.ToString(true));
            }
        }
        */
        /*
        absl::StrAppend(&result, "\nCARD MAP : ");
        for (auto & pair : card_map) {
            absl::StrAppend(&result, "\n",
                    pair.first.ToString(true),
                    " ", GLYPH_ARROW, " ",
                    GetPile(pair.first)->ToString(true)
                    );
        }
        */

        return result;
    }

    std::string             SolitaireState::ActionToString(Player player, Action action_id) const {
        switch (action_id) {
            case kDraw : {
                return "kDraw";
            }
            case kRevealAs ... kRevealKd : {
                auto revealed_card = Card((int) action_id);
                std::string result;
                absl::StrAppend(&result, "kReveal", revealed_card.ToString(false));
                return result;
            }
            case kMove__Ks ... kMoveKdQc : {
                auto move = Move(action_id);
                return move.ToString(true);
            }
            default : {
                return "Missing Action";
            }
        }
    }

    std::string             SolitaireState::InformationStateString(Player player) const {
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, num_players_);
        return HistoryString();
    }

    std::string             SolitaireState::ObservationString(Player player) const {
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, num_players_);
        return ToString();
    }

    void                    SolitaireState::InformationStateTensor(Player player, std::vector<double> *values) const {
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, num_players_);

        values->resize(game_->InformationStateTensorSize());
        std::fill(values->begin(), values->end(), kInvalidAction);

        int i = 0;
        for (auto & action : History()) {
            (*values)[i] = action;
            ++i;
        }
    }

    void                    SolitaireState::ObservationTensor(Player player, std::vector<double> *values) const {
        SPIEL_CHECK_GE(player, 0);
        SPIEL_CHECK_LT(player, num_players_);

        for (const auto & tableau : tableaus) {
            std::vector<double> dv = tableau.Tensor();
            values->insert(values->end(), dv.begin(), dv.end());
        }

        for (const auto & foundation : foundations) {
            std::vector<double> dv = foundation.Tensor();
            values->insert(values->end(), dv.begin(), dv.end());
        }

        std::vector<double> dv = waste.Tensor();
        values->insert(values->end(), dv.begin(), dv.end());
    }

    void                    SolitaireState::DoApplyAction(Action action) {

        if (!IsChanceNode()) {
            previous_score = Returns().front();
        }

        switch (action) {
            case kRevealAs ... kRevealKd : {
                auto revealed_card = Card((int) action);
                bool found_card = false;

                for (auto & tableau : tableaus) {
                    if (!tableau.cards.empty() && tableau.cards.back().hidden) {
                        tableau.cards.back().rank = revealed_card.rank;
                        tableau.cards.back().suit = revealed_card.suit;
                        tableau.cards.back().hidden = false;

                        tableau.cards.back().GetIndex();
                        card_map.insert_or_assign(tableau.cards.back(), tableau.id);
                        found_card = true;
                        break;
                    }
                }
                if (!found_card && !waste.cards.empty()) {
                    for (auto & card : waste.cards) {
                        if (card.hidden) {
                            card.rank = revealed_card.rank;
                            card.suit = revealed_card.suit;
                            card.hidden = false;

                            card.GetIndex();
                            card_map.insert_or_assign(card, waste.id);
                            break;
                        }
                    }
                }
                revealed_cards.push_back(action);

                break;
            }
            case kMove__Ks ... kMoveKdQc : {
                /* TODO: Figure out if move is reversible and add to previous states if so
                 */
                Move selected_move = Move(action);
                MoveCards(selected_move);
                current_returns += current_rewards;
                break;
            }
            default : {
                // Do nothing
            }
        }


        ++current_depth;
        if (current_depth >= 300) {
            is_finished = true;
        }
    }

    std::vector<double>     SolitaireState::Returns() const {
        // Returns the sum of rewards up to and including the most recent state transition.

        /*
        double old_returns;
        double f_score = 0.0;
        double t_score = 0.0;
        double w_score = 0.0;

        for (auto & foundation : foundations) {
            for (auto & card : foundation.cards) {
                f_score += FOUNDATION_POINTS.at(card.rank);
            }
        }

        int num_hidden = 0;
        for (auto & tableau : tableaus) {
            if (!tableau.cards.empty()) {
                for (auto & card : tableau.cards) {
                    if (card.hidden) {
                        num_hidden += 1;
                    }
                }
                if (tableau.cards.back().hidden) {
                    num_hidden -= 1;
                }
            }
        }
        t_score = (21 - num_hidden) * 20;

        w_score = (24 - waste.cards.size()) * 20;

        old_returns = f_score + t_score + w_score;

        if (old_returns != current_returns) {
            std::cout << RED << "Discrepancy in Returns()" << RESET << std::endl;
            std::cout << RED << "Old Returns = " << old_returns << RESET << std::endl;
            std::cout << RED << "New Returns = " << current_returns << RESET << std::endl;
        }
        */

        return {current_returns};
    }

    std::vector<double>     SolitaireState::Rewards() const {
        // Should be the reward for the action that created this state, not the action
        // taken at this state that produces a new one.
        return {current_rewards};
    }

    std::vector<Action>     SolitaireState::LegalActions() const {
        if (IsTerminal()) {
            return {};
        } else if (IsChanceNode()) {
            return LegalChanceOutcomes();
        } else {
            // TODO: Add loop detection later
            std::vector<Action> legal_actions;
            for (const auto & move : CandidateMoves()) {
                legal_actions.push_back(move.ActionId());
            }

            if (!legal_actions.empty()) {
                std::sort(legal_actions.begin(), legal_actions.end());
            } else {
                legal_actions.push_back(kDraw);
            }

            return legal_actions;
        }
    }

    std::vector<std::pair<Action, double>> SolitaireState::ChanceOutcomes() const {
        std::vector<std::pair<Action, double>> outcomes;
        const double p = 1.0 / (52 - revealed_cards.size());

        for (int i = 1; i <= 52; i++) {
            if (std::find(revealed_cards.begin(), revealed_cards.end(), i) == revealed_cards.end()) {
                outcomes.emplace_back(i, p);
            }
        }

        return outcomes;
    }

    // Other Methods ---------------------------------------------------------------------------------------------------

    std::vector<Card>       SolitaireState::Targets(const std::optional<LocationType> & location) const {
        // OPTIMIZE: Store values, calculate changes
        LocationType loc = location.value_or(kMissing);
        std::vector<Card> targets;

        if (loc == kTableau || loc == kMissing) {
            for (const auto & tableau : tableaus) {
                std::vector<Card> current_targets = tableau.Targets();
                targets.insert(targets.end(), current_targets.begin(), current_targets.end());
            }
        }

        if (loc == kFoundation || loc == kMissing) {
            for (const auto & foundation : foundations) {
                std::vector<Card> current_targets = foundation.Targets();
                targets.insert(targets.end(), current_targets.begin(), current_targets.end());
            }
        }

        return targets;
    }

    std::vector<Card>       SolitaireState::Sources(const std::optional<LocationType> & location) const {
        // OPTIMIZE: Store values, calculate changes
        LocationType loc = location.value_or(kMissing);
        std::vector<Card> sources;

        if (loc == kTableau || loc == kMissing) {
            for (const auto & tableau : tableaus) {
                std::vector<Card> current_sources = tableau.Sources();
                sources.insert(sources.end(), current_sources.begin(), current_sources.end());
            }
        }

        if (loc == kFoundation || loc == kMissing) {
            for (const auto & foundation : foundations) {
                std::vector<Card> current_sources = foundation.Sources();
                sources.insert(sources.end(), current_sources.begin(), current_sources.end());
            }
        }

        if (loc == kWaste || loc == kMissing) {
            std::vector<Card> current_sources = waste.Sources();
            sources.insert(sources.end(), current_sources.begin(), current_sources.end());
        }

        return sources;
    }

    Pile *                  SolitaireState::GetPile(const Card & card) const {
        PileID pile_id;

        if (card.rank == kNoRank) {
            if (card.suit == kNoSuit) {
                for (auto & tableau : tableaus) {
                    if (tableau.cards.empty()) {
                        return const_cast<Pile *>(& tableau);
                    }
                }
            } else if (card.suit != kHiddenSuit){
                for (auto & foundation : foundations) {
                    if (foundation.suit == card.suit) {
                        return const_cast<Pile *>(& foundation);
                    }
                }
            } else {
                std::cout << "Some error" << std::endl;
            }
        } else {
            pile_id = card_map.at(card);
        }

        /*
        try {
            pile_id = card_map.at(card);
        } catch (std::out_of_range) {
            if (card.rank == kNoRank) {
                if (card.suit == kNoSuit) {
                    // Handle finding an empty tableau pile
                    for (auto & tableau : tableaus) {
                        if (tableau.cards.empty()) {
                            return const_cast<Pile *>(& tableau);
                        }
                    }
                } else {
                    // Handle finding an empty foundation pile
                    for (auto & foundation : foundations) {
                        if (foundation.suit == card.suit) {
                            return const_cast<Pile *>(& foundation);
                        }
                    }
                }
            } else {
                // Shouldn't ever reach this point
                pile_id = kNoPile;
            }
        }
        */

        switch (pile_id) {
            case kPileWaste : {
                return const_cast<Pile *>(& waste);
            }
            case kPileSpades ... kPileDiamonds : {
                return const_cast<Pile *>(& foundations.at(pile_id - 1));
            }
            case kPile1stTableau ... kPile7thTableau : {
                return const_cast<Pile *>(& tableaus.at(pile_id - 5));
            }
            default : {
                std::cout << "The pile containing the card wasn't found" << std::endl;
            }
        }
    }

    std::vector<Move>       SolitaireState::CandidateMoves() const {
        std::vector<Move> candidate_moves;
        std::vector<Card> targets = Targets();
        std::vector<Card> sources = Sources();
        bool found_empty_tableau  = false;

        for (auto & target : targets) {
            // std::cout << "Target = " << target.ToString() << std::endl; //

            if (target.suit == kNoSuit && target.rank == kNoRank) {
                if (found_empty_tableau) {
                    continue;
                } else {
                    found_empty_tableau = true;
                }
            }
            for (auto & source : target.LegalChildren()) {
                // std::cout << "Source = " << source.ToString() << std::endl; //

                if (std::find(sources.begin(), sources.end(), source) != sources.end()) {
                    // std::cout << source.ToString() << " is in sources" << std::endl;
                    auto source_pile = GetPile(source);

                    if (target.location == kFoundation && source_pile->type == kTableau) {
                        // Check if source is a top card
                        if (source_pile->cards.back() == source) {
                            candidate_moves.emplace_back(target, source);
                        }
                    } else if (source.rank == kK && target.suit == kNoSuit && target.rank == kNoRank) {
                        // Check is source is not a bottom
                        if (source_pile->type == kTableau && !(source_pile->cards.front() == source)) {
                            candidate_moves.emplace_back(target, source);
                        } else if (source_pile->type == kWaste) {
                            candidate_moves.emplace_back(target, source);
                        }
                    } else {
                        auto move = Move(target, source);
                        candidate_moves.emplace_back(target, source);
                    }
                } else {
                    continue;
                }
            }
        }

        /*
        std::cout << "\n";
        for (auto & move : candidate_moves) {
            std::cout << "Move = " << move.ToString() << std::endl;
        }
        */

        return candidate_moves;
    }

    void                    SolitaireState::MoveCards(const Move & move) {
        Card target = move.target;
        Card source = move.source;

        auto target_pile = GetPile(target);
        auto source_pile = GetPile(source);

        std::vector<Card> split_cards = source_pile->Split(source);
        for (auto & card : split_cards) {
            card_map.insert_or_assign(card, target_pile->id);
            target_pile->Extend({card});
        }

        // Calculate rewards/returns for this move in the current state
        // Reset current reward to 0
        double move_reward = 0.0;

        // Reward for moving a card to or from a foundation
        if (target_pile->type == kFoundation) {
            // Adds points for moving TO a foundation
            move_reward += FOUNDATION_POINTS.at(source.rank);
        } else if (source_pile->type == kFoundation) {
            // Subtracts points for moving AWAY from a foundation
            move_reward -= FOUNDATION_POINTS.at(source.rank);
        }

        // Reward for revealing a hidden card
        if (source_pile->type == kTableau && !source_pile->cards.empty() && source_pile->cards.back().hidden) {
            move_reward += 20.0;    // TODO: Don't hardcode this
        }

        // Reward for moving a card from the waste
        if (source_pile->type == kWaste) {
            move_reward += 20.0;    // TODO: Don't hardcode this
        }

        // Add current rewards to current returns
        current_rewards = move_reward;
    }

    // endregion

    // region SolitaireGame Methods ===========================================================================================

    SolitaireGame::SolitaireGame(const GameParameters & params) :
        Game(kGameType, params),
        num_players_(ParameterValue<int>("players")),
        depth_limit_(ParameterValue<int>("depth_limit")),
        is_colored_(ParameterValue<bool>("colored")) {
        // Nothing here yet
    }

    int SolitaireGame::NumDistinctActions() const {
        return 205;
    }

    int SolitaireGame::MaxGameLength() const {
        return depth_limit_;
    }

    int SolitaireGame::NumPlayers() const {
        return 1;
    }

    double SolitaireGame::MinUtility() const {
        return 0.0;
    }

    double SolitaireGame::MaxUtility() const {
        return 3220.0;
    }

    std::vector<int> SolitaireGame::InformationStateTensorShape() const {
        return {depth_limit_};
    }

    std::vector<int> SolitaireGame::ObservationTensorShape() const {
        // TODO: Fix this later
        return {209};
    }

    std::unique_ptr<State> SolitaireGame::NewInitialState() const {
        return std::unique_ptr<State>(new SolitaireState(shared_from_this()));
    }

    std::shared_ptr<const Game> SolitaireGame::Clone() const {
        return std::shared_ptr<const Game>(new SolitaireGame(*this));
    }

    // endregion

} // namespace open_spiel::solitaire


