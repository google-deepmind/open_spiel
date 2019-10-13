//
// Created by dennis on 01.09.19.
//

#include <iostream>
#include <assert.h>
#include "PokerGame.h"
#include <regex>

PokerGame::PokerGame(Game *g, std::string gameName)
        : game(*g),
          gameName(std::move(gameName))
{


}

PokerGameState PokerGame::initialState() {

    return PokerGameState(&game, nullptr, PokerGameState::GameAction());


}

PokerGameState PokerGame::updateState(PokerGameState state, uint32_t actionIdx) {


    const std::vector<PokerGameState::GameAction> &actionsAllowed = state.getActionsAllowed();
    assert( actionIdx >= 0 && actionIdx < actionsAllowed.size());
    return PokerGameState(&game, &state, actionsAllowed[actionIdx]);

}

std::unique_ptr<PokerGame> PokerGame::createFromGamedef(const char *fileName) {
        FILE* f = fopen(fileName, "r");
        assert(f != NULL);

        std::string s (fileName);
        std::stringstream out;
        std::smatch m;
        std::regex e ("[0-9a-zA-Z_\\.-]+.game$");

        if(std::regex_search (s,m,e)) {
            out << m[0] ;
        }

        Game* game = readGame( f );
        std::unique_ptr<PokerGame> result = std::make_unique<PokerGame>(game, out.str());
        free(game);
        fclose(f);
        return result;

}
