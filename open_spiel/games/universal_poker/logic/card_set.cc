#include "card_set.h"
#include <string>
#include <sstream>
#include <bitset>

std::string suitChars = "cdhs";
std::string rankChars = "23456789TJQKA";

extern "C" {
#include "open_spiel/games/universal_poker/acpc/game.h"
#include "open_spiel/games/universal_poker/acpc/evalHandTables"
}

namespace open_spiel::universal_poker::logic {
    using I = uint64_t ;

    auto dump(I v) { return std::bitset<sizeof(I) * __CHAR_BIT__>(v); }
    I bit_twiddle_permute(I v) {
        I t = v | (v - 1);
        I w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzl(v) + 1));

        return w;
    }


    CardSet::CardSet()
            : cs() {
    }

    CardSet::CardSet(std::string cardString)
            : cs() {

        for (int i = 0; i < cardString.size(); i += 2) {
            char rankChr = cardString[i];
            char suitChr = cardString[i + 1];

            uint8_t rank = (uint8_t) rankChars.find(rankChr);
            uint8_t suit = (uint8_t) suitChars.find(suitChr);

            cs.bySuit[suit] |= ((uint16_t) 1 << rank);
        }
    }


    CardSet::CardSet(std::vector<int> cards)
            : cs() {
        for (int i = 0; i < cards.size(); i++) {
            int rank = rankOfCard(cards[i]);
            int suit = suitOfCard(cards[i]);

            cs.bySuit[suit] |= ((uint16_t) 1 << rank);
        }
    }


    CardSet::CardSet(uint8_t *cards, int size)
            : cs() {
        for (int i = 0; i < size; i++) {
            int rank = rankOfCard(cards[i]);
            int suit = suitOfCard(cards[i]);

            cs.bySuit[suit] |= ((uint16_t) 1 << rank);
        }
    }

    CardSet::CardSet(uint16_t numSuits, uint16_t numRanks) : cs() {
        for (uint16_t r = 0; r < numRanks; r++) {
            for (uint16_t s = 0; s < numSuits; s++) {
                cs.bySuit[s] |= ((uint16_t) 1 << r);
            }
        }
    }


    std::string CardSet::ToString() const {

        std::ostringstream result;
        for (int r = MAX_RANKS - 1; r >= 0; r--) {
            for (int s = MAX_SUITS - 1; s >= 0; s--) {
                uint32_t mask = (uint32_t) 1 << r;
                if (cs.bySuit[s] & mask) {

                    result << rankChars[r] << suitChars[s];
                }

            }
        }

        return result.str();
    }

    std::vector<uint8_t> CardSet::ToCardArray() const {
        std::vector<uint8_t> result;

        for (int r = MAX_RANKS - 1; r >= 0; r--) {
            for (int s = MAX_SUITS - 1; s >= 0; s--) {
                uint32_t mask = (uint32_t) 1 << r;
                if (cs.bySuit[s] & mask) {
                    result.push_back(makeCard(r, s));
                }
            }
        }
        return result;
    }

    void CardSet::AddCard(uint8_t card) {

        int rank = rankOfCard(card);
        int suit = suitOfCard(card);

        cs.bySuit[suit] |= ((uint16_t) 1 << rank);
    }

    void CardSet::RemoveCard(uint8_t card) {
        int rank = rankOfCard(card);
        int suit = suitOfCard(card);

        cs.bySuit[suit] ^= ((uint16_t) 1 << rank);
    }

    uint32_t CardSet::CountCards() {
        return __builtin_popcountl(cs.cards);
    }

    int CardSet::RankCards() {
        Cardset csNative;
        csNative.cards = cs.cards;
        return rankCardset(csNative);
    }

    bool CardSet::IsBlocking(CardSet other) {
        return (cs.cards & other.cs.cards) > 0;
    }

    std::vector<CardSet> CardSet::SampleCards(int nbCards) {
            std::vector<CardSet> combinations;

            uint64_t p = 0;
            for (int i = 0; i < nbCards; i++)
            {
                p += (1<<i);
            }

            for (I n = bit_twiddle_permute(p); n>p; p = n, n = bit_twiddle_permute(p)) {
                uint64_t combo = n & cs.cards;
                if( __builtin_popcountl(combo) == nbCards ){
                    CardSet c;
                    c.cs.cards = combo;
                    combinations.emplace_back(c);
                }
            }

            //std::cout << "combinations.size() " << combinations.size() << std::endl;
            return combinations;

        }

    bool CardSet::ContainsCards(const uint8_t &card) {
        int rank = rankOfCard(card);
        int suit = suitOfCard(card);
        return (cs.bySuit[suit] & ((uint16_t) 1 << rank)) > 0;
    }


}



