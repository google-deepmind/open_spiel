//
// Created by Dennis Jöst on 03.05.18.
//

#ifndef DEEPSTACK_CPP_BETTINGNODE_H
#define DEEPSTACK_CPP_BETTINGNODE_H


#include <cstdint>
#include <string>
extern "C"
{
    #include <ACPC/game.h>
};

#define ACTION_FOLD 0
#define ACTION_CALL 1
#define ACTION_BET_POT 2
#define ACTION_ALL_IN 3
#define ACTION_DEAL 4

#define PLAYER_DEALER 255

#define MAX_BETTING_NODE_CHILDREN 4


class BettingNode {

public:
    enum BettingNodeType{ EMPTY, CHANCE_NODE, CHOICE_NODE, TERMINAL_SHOWDOWN_NODE, TERMINAL_FOLD_NODE };

private:
    BettingNodeType type;
    uint8_t player;
    uint8_t round;
    float potSize;
    float bet0;
    float bet1;
    uint32_t index;
    std::string sequence;

public:
    BettingNode* children[MAX_BETTING_NODE_CHILDREN];
    uint8_t childCount;

public:
    BettingNode();
    void initialize(BettingNodeType type, uint8_t player, uint8_t round, float potSize, float bet0, float bet1, std::string sequence, uint32_t index);

    BettingNodeType getType() const;
    uint8_t getPlayer() const;
    uint8_t getRound() const;
    float getPotSize() const;

    float getBet0() const;
    float getBet1() const;

    uint32_t getIndex() const;

    std::string getSequence() const;

};


#endif //DEEPSTACK_CPP_BETTINGNODE_H
