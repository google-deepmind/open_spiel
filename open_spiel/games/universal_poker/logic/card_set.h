//
// Created by dj on 10/19/19.
//

#ifndef OPEN_SPIEL_CARD_SET_H
#define OPEN_SPIEL_CARD_SET_H

#include <cstdint>
#include <vector>
#include <string>

namespace open_spiel::universal_poker::logic {
    constexpr int MAX_SUITS = 4;

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
        CardSet(uint16_t numSuits, uint16_t numRanks);

        std::string ToString();
        std::vector<uint8_t> ToCardArray();

        void AddCard(uint8_t card);
        void RemoveCard(uint8_t card);

        uint32_t CountCards();
        int RankCards();
        bool IsBlocking(CardSet other);

        std::vector<CardSet> SampleCards(int nbCards);
    };


}


#endif //OPEN_SPIEL_CARD_SET_H
