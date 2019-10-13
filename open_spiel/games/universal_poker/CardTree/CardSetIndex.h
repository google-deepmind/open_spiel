//
// Created by Dennis Jöst on 01.05.18.
//

#ifndef DEEPSTACK_CPP_CARDSETINDEX_H
#define DEEPSTACK_CPP_CARDSETINDEX_H


#include "CardSet.h"

class CardSetIndex {

public:
    CardSet deck;
    uint32_t nbOfCards;
    std::vector<CardSet> combinations;

    CardSetIndex(int numSuits, int numRanks, uint32_t nbOfCards);
    void calculateDeck(int numSuits, int numRanks);
    void calculateCombinations();
};


#endif //DEEPSTACK_CPP_CARDSETINDEX_H
