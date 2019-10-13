//
// Created by Dennis Jöst on 01.05.18.
//

#include "CardSetIndex.h"
extern "C" {
#include "ACPC/game.h"
}

#include <algorithm>

CardSetIndex::CardSetIndex(int numSuits, int numRanks, uint32_t nbOfCards)
:deck(), nbOfCards(nbOfCards)
{
    calculateDeck(numSuits, numRanks);
    calculateCombinations();
}

void CardSetIndex::calculateDeck(int numSuits, int numRanks) {
    std::vector<int> cards;

    for( int r = MAX_RANKS-1; r >= MAX_RANKS-numRanks; r--) {
        for (int s = MAX_SUITS - 1; s >= MAX_SUITS - numSuits; s--) {

            cards.push_back(makeCard(r, s));
        }
    }

    deck = CardSet(cards);
}

void CardSetIndex::calculateCombinations() {
    auto cards = deck.toCardArray();
    size_t K = nbOfCards;
    size_t N = cards.size();

    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's
    // print integers and permute bitmask
    do {
        CardSet combination;
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]) {
                combination.addCard(cards[i]);
            }
        }

        combinations.emplace_back(combination);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

}
