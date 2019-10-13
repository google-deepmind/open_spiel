//
// Created by dennis on 01.09.19.
//

#include <iostream>
#include <assert.h>
#include "PokerGame.h"
#include <regex>

PokerGame::PokerGame(Game *g, std::string gameName)
        : game(*g),
          gameName(std::move(gameName)) {


}

PokerGameState PokerGame::initialState() {

    return PokerGameState(&game, nullptr, PokerGameState::GameAction());


}

PokerGameState PokerGame::updateState(PokerGameState state, uint32_t actionIdx) const {


    const std::vector<PokerGameState::GameAction> &actionsAllowed = state.getActionsAllowed();
    assert(actionIdx >= 0 && actionIdx < actionsAllowed.size());
    return PokerGameState(&game, &state, actionsAllowed[actionIdx]);

}


PokerGame PokerGame::createFromGamedef(const std::string& gamedef) {
    std::FILE *tmpf = std::tmpfile();
    std::fputs(gamedef.c_str(), tmpf);
    std::rewind(tmpf);

    Game *game = readGame(tmpf);
    PokerGame result = PokerGame(game, "universal_poker");
    free(game);
    return result;
}

const Game* PokerGame::getGame() const {
    return &game;
}

int PokerGame::getGameLength() {
    return getGameLength(initialState());
}

int PokerGame::getGameLength(PokerGameState state) {
    if(state.getType() == BettingNode::TERMINAL_FOLD_NODE || state.getType() == BettingNode::TERMINAL_FOLD_NODE){
        return 1;
    } else if( state.getType() == BettingNode::CHANCE_NODE) {
        return 1 + getGameLength( updateState(state, 0)); //Doesnt matter all choices do not affect game length
    }
    else {
        int length = 1;
        for( int a = 0; a < state.getActionsAllowed().size(); a++ ){
            int l = 1 + getGameLength( updateState(state, a));
            if( l > length) {
                length = l;
            }
        }
        return length;
    }
}

