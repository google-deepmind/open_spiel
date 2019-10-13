//
// Created by Dennis Jöst on 30.04.18.
//

#include "CardSet.h"
#include <string>
#include <sstream>

std::string suitChars = "cdhs";
std::string rankChars = "23456789TJQKA";
extern "C" {
#include "open_spiel/games/universal_poker/ACPC/game.h"
#include "open_spiel/games/universal_poker/ACPC/evalHandTables"
}
CardSet::CardSet()
    :cs()
{

}

CardSet::CardSet(std::string cardString)
 :cs()
{

    for( int i = 0; i < cardString.size(); i+=2 ){
        char rankChr = cardString[i];
        char suitChr = cardString[i+1];

        uint8_t rank = (uint8_t)rankChars.find(rankChr);
        uint8_t suit = (uint8_t)suitChars.find(suitChr);

        cs.bySuit[suit] |= ((uint16_t)1 << rank);
    }
}


CardSet::CardSet(std::vector<int> cards)
 :cs()
{
    for( int i = 0; i < cards.size(); i++ ){
        int rank = rankOfCard(cards[i]);
        int suit = suitOfCard(cards[i]);

        cs.bySuit[suit] |= ((uint16_t)1 << rank);
    }
}


CardSet::CardSet(uint8_t *cards, int size)
        :cs()
{
    for( int i = 0; i < size; i++ ){
        int rank = rankOfCard(cards[i]);
        int suit = suitOfCard(cards[i]);

        cs.bySuit[suit] |= ((uint16_t)1 << rank);
    }
}


std::string CardSet::toString() {

    std::ostringstream result;
    for( int r = MAX_RANKS-1; r >= 0; r-- ){
        for( int s = MAX_SUITS-1; s >= 0; s-- )
        {
            uint32_t mask = (uint32_t)1 << r;
            if( cs.bySuit[s] & mask ){

                result << rankChars[r] << suitChars[s];
            }

        }
    }

    return result.str();
}

std::vector<int> CardSet::toCardArray() {
    std::vector<int> result;

    for (int r = MAX_RANKS - 1; r >= 0; r--) {
        for( int s = MAX_SUITS-1; s >= 0; s-- ) {
            uint32_t mask = (uint32_t)1 << r;
            if( cs.bySuit[s] & mask ){
                result.push_back(makeCard(r, s));
            }
        }
    }
    return result;
}

void CardSet::addCard(uint8_t card) {

    int rank = rankOfCard(card);
    int suit = suitOfCard(card);

    cs.bySuit[suit] |= ((uint16_t)1 << rank);
}

void CardSet::removeCard(uint8_t card) {
    int rank = rankOfCard(card);
    int suit = suitOfCard(card);

    cs.bySuit[suit] ^= ((uint16_t)1 << rank);
}

uint32_t CardSet::countCards() {
    uint32_t rslt=0;
    for (int r = MAX_RANKS - 1; r >= 0; r--) {
        for( int s = MAX_SUITS-1; s >= 0; s-- ) {
            uint32_t mask = (uint32_t)1 << r;
            if( cs.bySuit[s] & mask ){
                rslt+=1;
            }
        }
    }
    return rslt;
}

int CardSet::rank() {
    Cardset csNative;
    csNative.cards = cs.cards;
    return rankCardset(csNative);
}

bool CardSet::isBlocking(CardSet other) {
    return (cs.cards & other.cs.cards) > 0;
}




