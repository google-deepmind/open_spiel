#include "open_spiel/games/universal_poker/card_abstraction/isomorphic.h"

extern "C" {
#include "open_spiel/games/universal_poker/hand-isomorphism/src/hand_index.h"
}

namespace open_spiel::universal_poker::card_abstraction {

IsomorphicCardAbstraction::IsomorphicCardAbstraction(
    std::vector<int> cards_per_round) {

    for(int r=0; r < cards_per_round.size(); r++){
        cards_per_round_.push_back(cards_per_round[r]);
        // create indexer for this round
        hand_indexer_t indexer;
        uint8_t array_slice[r+1];
	    std::copy(cards_per_round.begin(), cards_per_round.begin() + r + 1, array_slice);
        hand_indexer_init(r + 1, array_slice, &indexer);
        indexers_.push_back(indexer);
    }
}

uint8_t to_iso_card(uint8_t cs_card) {
    return deck_make_card(cs_card % 4, cs_card / 4);
}

uint8_t from_iso_card(uint8_t iso_card) {
    return deck_get_suit(iso_card) + 4 * deck_get_rank(iso_card);
}

std::pair<logic::CardSet, logic::CardSet>
IsomorphicCardAbstraction::abstract(logic::CardSet hole_cards,
                                    logic::CardSet board_cards) const {
    auto h_arr = hole_cards.ToCardArray();
    auto b_arr = board_cards.ToCardArray();
    h_arr.insert(h_arr.end(), b_arr.begin(), b_arr.end());

    int total_cards = h_arr.size();
    int round = -1;
    uint8_t cards[total_cards];

    for(int i = 0; i < total_cards; i++) {
        cards[i] = to_iso_card(h_arr[i]);
    }

    while (total_cards > 0) {
        total_cards -= cards_per_round_[++round];
    }

    auto indexer = indexers_[round];
    auto index = hand_index_last(&indexer, cards);
    hand_unindex(&indexer, round, index, cards);

    logic::CardSet r_hole_cards;
    logic::CardSet r_board_cards;

    for(int i = 0; i < hole_cards.NumCards(); i++) {
        r_hole_cards.AddCard(from_iso_card(cards[i]));
    }

    for(int i = hole_cards.NumCards();
        i < hole_cards.NumCards() + board_cards.NumCards();
        i++) {
        r_board_cards.AddCard(from_iso_card(cards[i]));
    }

    return {r_hole_cards, r_board_cards};
}

}
