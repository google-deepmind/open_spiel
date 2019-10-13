//
// Created by Dennis Jöst on 30.04.18.
//

#ifndef DEEPSTACK_CPP_CARDSET_H
#define DEEPSTACK_CPP_CARDSET_H

#include <inttypes.h>
#include <string>
#include <vector>

extern "C"
{
    #include "open_spiel/games/universal_poker/ACPC/game.h"
};

class CardSet {
public:
    union CardSetUnion {
        CardSetUnion() : cards(0) {}
        uint16_t bySuit[MAX_SUITS];
        uint64_t cards;
    } cs;


public:
    CardSet();
    CardSet(std::string cardString);
    CardSet(std::vector<int> cards);
    CardSet(uint8_t cards[], int size);

    std::string toString();
    std::vector<int> toCardArray();

    void addCard(uint8_t card);
    void removeCard(uint8_t card);

    uint32_t countCards();
    int rank();
    bool isBlocking(CardSet other);
};


#endif //DEEPSTACK_CPP_CARDSET_H
