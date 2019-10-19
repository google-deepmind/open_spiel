#include <iostream>
#include "card_set.h"


namespace open_spiel::universal_poker::logic {

void BasicCardSetTests() {

    CardSet cs("AhKaQhJhTh");

    std::cout << "CardSet: " << cs.ToString() <<std::endl;
    for( auto card : cs.ToCardArray() ){
        std::cout << "Card: " << card<<std::endl;
    }
    std::cout << "Rank: " << cs.RankCards() <<std::endl;
    std::cout << "Count Cards: " << cs.CountCards() <<std::endl;


    CardSet deck(4, 13);
    std::cout << "CardSet: " << deck.ToString() <<std::endl;
    std::cout << "Rank: " << deck.RankCards() <<std::endl;
    std::cout << "Count Cards: " << deck.CountCards() <<std::endl;


    for( auto combo : deck.SampleCards(3)) {
        std::cout << "CardSet: " << combo.ToString() <<std::endl;

    }

    for( auto combo : deck.SampleCards(1)) {
        std::cout << "CardSet: " << combo.ToString() <<std::endl;

    }


}

}  // namespace

int main(int argc, char **argv) {

    open_spiel::universal_poker::logic::BasicCardSetTests();

}
