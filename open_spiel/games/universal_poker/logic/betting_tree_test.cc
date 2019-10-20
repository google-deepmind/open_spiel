#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/betting_tree.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

namespace open_spiel::universal_poker::logic {

void BasicBettingTreeTests(const std::string gameDesc) {
    BettingTree bettingTree(gameDesc);

    std::srand(std::time(nullptr));

    for( int i = 0; i < 100; i++) {
        BettingTree::BettingNode bettingNode(&bettingTree);
        std::cout << "INIT. DEPTH: " << bettingNode.GetDepth() << std::endl;
        std::cout << bettingNode.ToString() << std::endl;

        while (!bettingNode.IsFinished()) {
            if (bettingNode.GetNodeType() == BettingTree::BettingNode::NODE_TYPE_CHANCE) {
                bettingNode.ApplyDealCards();
            } else {
                uint32_t actionIdx = std::rand() % bettingNode.GetPossibleActionCount();
                uint32_t idx = 0;
                for(auto action : BettingTree::ALL_ACTIONS){
                    if(action & bettingNode.GetPossibleActionsMask()){
                        if(idx==actionIdx){
                            bettingNode.ApplyChoiceAction(action);
                            break;
                        }
                        idx++;
                    }
                }

            }

            std::cout << bettingNode.ToString() << std::endl;
        }
    }


}

}  // namespace

int main(int argc, char **argv) {
    const std::string gameDesc("GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = 3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

    //open_spiel::universal_poker::logic::BasicBettingTreeTests(gameDesc);


    const std::string gameDescHoldem("GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 4\nstack = 20000 20000\nblind = 100 50\nfirstPlayer = 2 1 1 1\nnumSuits = 4\nnumRanks = 13\nnumHoleCards = 2\nnumBoardCards = 0 3 1 1\nEND GAMEDEF");

    open_spiel::universal_poker::logic::BasicBettingTreeTests(gameDescHoldem);

}
