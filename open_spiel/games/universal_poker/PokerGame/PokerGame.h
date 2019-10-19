//
// Created by dennis on 01.09.19.
//

#ifndef DEEPSTACK_CPP_POKERGAME_H
#define DEEPSTACK_CPP_POKERGAME_H

#include "PokerGameState.h"
#include <mutex>
#include <memory>
#include <random>

extern "C" {
    #include "open_spiel/games/universal_poker/acpc/game.h"
}



class PokerGame {
private:
    Game game;
public:
    const Game* getGame() const ;

private:
    std::string gameName;

public:
    PokerGame(Game* game, std::string gameName);

    PokerGameState initialState();
    PokerGameState updateState(PokerGameState state, uint32_t actionIdx) const;

    int getGameLength();

    static PokerGame createFromGamedef(const std::string &gamedef);

private:
    int getGameLength(PokerGameState state);
};


#endif //DEEPSTACK_CPP_POKERGAME_H
