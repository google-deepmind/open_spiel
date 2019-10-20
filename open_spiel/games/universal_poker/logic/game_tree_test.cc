#include <iostream>
#include "card_set.h"
#include "game_tree.h"
#include <cstdlib>
#include <ctime>

namespace open_spiel::universal_poker::logic {

void BasicGameTreeTests() {
    const std::string gameDesc("GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = 3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

    GameTree tree(gameDesc);

    std::srand(std::time(nullptr));


    for( int i = 0; i < 100; i++) {
        GameTree::GameNode node(tree);
        std::cout << node.ToString() << std::endl;
        while (!node.IsFinished()) {

            uint32_t actions = node.GetActionCount();
            uint32_t action = std::rand() % actions;

            std::cout << "Choose Action: " << action <<std::endl;
            node.ApplyAction(action);
            std::cout << node.ToString() << std::endl;
        }
    }











}
}  // namespace

int main(int argc, char **argv) {

    open_spiel::universal_poker::logic::BasicGameTreeTests();

}
