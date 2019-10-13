//
// Created by Dennis Jöst on 03.05.18.
//

#ifndef DEEPSTACK_CPP_GAMENODE_H
#define DEEPSTACK_CPP_GAMENODE_H


#include "open_spiel/games/universal_poker/BettingTree/BettingNode.h"
#include "open_spiel/games/universal_poker/CardTree/CardNode.h"

class GameNode {
public:
    BettingNode* bettingNode;
    CardNode* cardNode;
    Game* game;

    std::string name;
    std::vector<GameNode*> children;

    uint32_t depth;

public:
    GameNode();
    GameNode(std::string name, BettingNode* bettingNode, CardNode* cardNode, Game* game);

    BettingNode::BettingNodeType getType() const;
    uint8_t getCurrentPlayer() const;
    uint8_t getRound() const;
    float getPotSize() const;
    size_t getHandCount();
    float getBet0();
    float getBet1();

    float getBet0Normalized();
    float getBet1Normalized();

    int getChildCount();
    GameNode* getChild(int n);

    uint32_t getDepth() const;

    CardSet getBoardCards();
    Eigen::MatrixXf getFoldMatrix();
    Eigen::MatrixXf getCallMatrix();
    Eigen::MatrixXf getConstantStrategy();
    Eigen::VectorXf getHandMask();
};


#endif //DEEPSTACK_CPP_GAMENODE_H
