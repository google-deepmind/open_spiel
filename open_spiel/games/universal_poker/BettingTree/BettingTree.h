//
// Created by Dennis Jöst on 03.05.18.
//
#ifndef DEEPSTACK_CPP_BETTINGTREE_H
#define DEEPSTACK_CPP_BETTINGTREE_H


#define MAX_BETTING_NODE_COUNT (1024 * 1024)


#include <unordered_map>
#include "BettingNode.h"

class BettingTree {

private:
    Game* game;
    BettingNode* rootNode;
    BettingNode bettingNodes[MAX_BETTING_NODE_COUNT];
    uint32_t bettingNodeCount;
    std::unordered_map<std::string, BettingNode*> bettingNodeIndex;
public:
    uint32_t getBettingNodeCount() const;

public:
    explicit BettingTree(Game* game);

    BettingNode* createBettingNode(State currentState, BettingNode* parentNode, char action);
    void setChoiceNodeChildren(BettingNode *node, State state);

    BettingNode* getBettingNode(std::string bettingSequence);
    BettingNode* getBettingNodes();

    void printTree();
private:
    void printNodeRecursive(BettingNode* node, int depth);

};


#endif //DEEPSTACK_CPP_BETTINGTREE_H
