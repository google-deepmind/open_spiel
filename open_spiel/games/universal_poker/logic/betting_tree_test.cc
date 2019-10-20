#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/betting_tree.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

namespace open_spiel::universal_poker::logic {

void BasicBettingTreeTests() {
    const std::string gameDesc("GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = 3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

    BettingTree bettingTree(gameDesc);

    std::srand(std::time(nullptr));

    for( int i = 0; i < 100; i++) {
        BettingTree::BettingNode bettingNode(bettingTree);
        std::cout << "INIT" << std::endl;
        std::cout << bettingNode.ToString() << std::endl;

        while (!bettingNode.IsFinished()) {
            if (bettingNode.GetNodeType() == BettingTree::BettingNode::NODE_TYPE_CHANCE) {
                bettingNode.ApplyDealCards();
            } else {
                uint32_t actionIdx = std::rand() % bettingNode.GetPossibleActions().size();
                std::cout << "Selected Action: " << actionIdx << std::endl;

                bettingNode.ApplyChoiceAction(actionIdx);
            }

            std::cout << bettingNode.ToString() << std::endl;
        }
    }


}

}  // namespace

int main(int argc, char **argv) {

    open_spiel::universal_poker::logic::BasicBettingTreeTests();

}
