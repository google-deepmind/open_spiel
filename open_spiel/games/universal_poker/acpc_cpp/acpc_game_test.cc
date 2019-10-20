#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include <iostream>

namespace open_spiel::universal_poker::acpc_cpp {

void BasicACPCTests() {
    const std::string gameDesc("GAMEDEF\nnolimit\nnumPlayers = 2\nnumRounds = 2\nstack = 1200 1200\nblind = 100 100\nfirstPlayer = 1 1\nnumSuits = 2\nnumRanks = 3\nnumHoleCards = 1\nnumBoardCards = 0 1\nEND GAMEDEF");

    ACPCGame game(gameDesc);
    ACPCGame::ACPCState state(game);

    std::cout << game.ToString() << std::endl;
    std::cout << state.ToString() << std::endl;

    while( !state.IsFinished() ){
        int32_t minRaise = 0, maxRaise = 0;
        if( state.RaiseIsValid(&minRaise, &maxRaise) ){
            minRaise = state.MaxSpend() > minRaise ? state.MaxSpend() : minRaise;
        }

        const ACPCGame::ACPCState::ACPCActionType available_actions[] = {
                ACPCGame::ACPCState::ACPC_CALL,
                ACPCGame::ACPCState::ACPC_FOLD,
                //ACPCGame::ACPCState::ACPC_RAISE
        };

        for( const auto &action: available_actions ){
            if( state.IsValidAction(action, 0) ) {
                state.DoAction(action, 0);
                std::cout << state.ToString() << std::endl;
            }
        }
    }

    std::cout << state.ValueOfState(0) << std::endl;
    std::cout << state.ValueOfState(1) << std::endl;


}

}  // namespace

int main(int argc, char **argv) {

    open_spiel::universal_poker::acpc_cpp::BasicACPCTests();

}
