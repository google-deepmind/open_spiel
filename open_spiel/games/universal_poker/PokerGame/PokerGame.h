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
    #include <ACPC/game.h>
}



class PokerGame {
private:
    Game game;
    std::string gameName;

public:
    PokerGame(Game* game, std::string gameName);

    PokerGameState initialState();
    PokerGameState updateState(PokerGameState state, uint32_t actionIdx);


    static std::unique_ptr<PokerGame> createFromGamedef(const char* fileName);


};


#endif //DEEPSTACK_CPP_POKERGAME_H
