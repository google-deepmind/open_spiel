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
        state.RaiseIsValid(&minRaise, &maxRaise);

        const ACPCGame::ACPCState::ACPCAction available_actions[]{
                ACPCGame::ACPCState::ACPCAction(&game, ACPCGame::ACPCState::ACPCAction::ACTION_RAISE, minRaise),
                ACPCGame::ACPCState::ACPCAction(&game, ACPCGame::ACPCState::ACPCAction::ACTION_RAISE, minRaise*2),
                ACPCGame::ACPCState::ACPCAction(&game, ACPCGame::ACPCState::ACPCAction::ACTION_CALL, 0),
                ACPCGame::ACPCState::ACPCAction(&game, ACPCGame::ACPCState::ACPCAction::ACTION_FOLD, 0)
        };

        for( const auto &action: available_actions ){
            if( state.IsValidAction(false, action) ) {
                std::cout << action.ToString() << std::endl;
                state.DoAction(action);
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
