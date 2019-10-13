//
// Created by Dennis Jöst on 03.05.18.
//

#include <cassert>
extern "C"
{
#include <ACPC/game.h>
};

#include <cmath>
#include <ACPC/GameDefinitions.h>
#include <ACPC/game.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include "BettingTree.h"

BettingTree::BettingTree(Game *game)
:game(game), bettingNodeCount(0)
{
    State state;
    initState(game, 0, &state);
    rootNode = createBettingNode(state, nullptr, 'R');
}


BettingNode* BettingTree::createBettingNode(State state, BettingNode* parentNode, char action)
{
    assert(bettingNodeCount < MAX_BETTING_NODE_COUNT);
    bettingNodes[bettingNodeCount] = BettingNode();
    BettingNode* result = &bettingNodes[bettingNodeCount];
    bettingNodeCount++;

    std::stringstream sequence;
    if( parentNode != nullptr ) {
        sequence << parentNode->getSequence();
    }
    sequence << action;

    auto potSize = static_cast<float>(fmin(state.spent[0], state.spent[1]));
    if( state.finished ) {
        // The other player than the one which did the last action
        uint8_t player = (uint8_t)1 - parentNode->getPlayer();
        if (numFolded(game, &state) >= game->numPlayers - 1) {
            result->initialize(BettingNode::TERMINAL_FOLD_NODE, player, parentNode->getRound(), potSize, state.spent[0], state.spent[1], sequence.str(), bettingNodeCount-1);

        } else {
            result->initialize(BettingNode::TERMINAL_SHOWDOWN_NODE, player, parentNode->getRound(), potSize, state.spent[0], state.spent[1], sequence.str(), bettingNodeCount-1);
        }
    }
    else if (parentNode != nullptr && state.round != parentNode->getRound() && BettingNode::CHANCE_NODE != parentNode->getType()) {
        result->initialize(BettingNode::CHANCE_NODE, parentNode->getPlayer(), parentNode->getRound(), potSize, state.spent[0], state.spent[1], sequence.str(), bettingNodeCount-1);
        result->children[0] = createBettingNode(state, result, 'D');
        result->childCount = 1;
    }
    else {
        uint8_t  player = currentPlayer(game, &state);
        result->initialize(BettingNode::CHOICE_NODE, player, state.round, potSize, state.spent[0], state.spent[1], sequence.str(), bettingNodeCount-1 );
        setChoiceNodeChildren(result, state);
    }

    //Add to index
    bettingNodeIndex[result->getSequence()] = result;

    assert( result->getType() != BettingNode::EMPTY);
    return result;
}

void BettingTree::setChoiceNodeChildren(BettingNode *node, State state) {
    assert(node->getType() == BettingNode::CHOICE_NODE);

    Action actions[NUM_ACTIONS_FCPAD]{{a_fold, 0},{a_call, 0},{a_raise, 0},{a_invalid, 0},{a_invalid, 0}} ;
    char action_names[NUM_ACTIONS_FCPAD]{'f', 'c', 'p', 'a', 'd'};

    if( game->bettingType == noLimitBetting ){
        int32_t opponentBet = state.spent[1-node->getPlayer()];
        int32_t minRaise = 0, maxRaise = 0;
        if( raiseIsValid(game, &state, &minRaise, &maxRaise) ) {
            int32_t potBet = opponentBet + opponentBet * 2;
            if( potBet < minRaise || potBet >= maxRaise ){
                potBet = maxRaise;
                actions[ACTION_ALL_IN] = {a_raise, potBet};
            }
            else {
                actions[ACTION_BET_POT] = {a_raise, potBet};
            }

            if( maxRaise > potBet ) {
                actions[ACTION_ALL_IN] = {a_raise, maxRaise};
            }
        }
    }

    for( int a = 0; a < NUM_ACTIONS_FCPAD; a++ ) {
        char name = action_names[a];
        State deepCopyOfState = state;
        // Fold is always possible even when not logical
        if( actions[a].type == a_fold || isValidAction(game, &deepCopyOfState, 0, &actions[a]) ){
            doAction(game, &actions[a], &deepCopyOfState );
            BettingNode* child = this->createBettingNode(deepCopyOfState, node, name);
            node->children[node->childCount] = child;
            node->childCount++;
        }
    }



}

uint32_t BettingTree::getBettingNodeCount() const {
    return bettingNodeCount;
}

void BettingTree::printTree() {
    printNodeRecursive(rootNode, 0);
}

void BettingTree::printNodeRecursive(BettingNode *node, int depth) {
    for( int i = 0; i < depth; i++ ) {
        std::cout << "\t";
    }

    std::cout << "r" << (int)node->getRound() << " p" << (int)node->getPlayer() << " : ";
    if( node->getType() == BettingNode::CHANCE_NODE ) std::cout << "CHANCE_NODE" << " ";
    if( node->getType() == BettingNode::CHOICE_NODE ) std::cout << "CHOICE_NODE" << " ";
    if( node->getType() == BettingNode::TERMINAL_FOLD_NODE ) std::cout << "TERMINAL_FOLD_NODE" << " ";
    if( node->getType() == BettingNode::TERMINAL_SHOWDOWN_NODE ) std::cout << "TERMINAL_SHOWDOWN_NODE" << " ";

    std::cout << node->getBet0() << " / " << node->getBet1();
    std::cout << "\n";

    for( int i = 0; i < node->childCount; i++ ){
        printNodeRecursive(node->children[i], depth+1);
    }


}

BettingNode* BettingTree::getBettingNode(std::string bettingSequence) {
    assert(bettingNodeIndex.find(bettingSequence) != bettingNodeIndex.end());

    return bettingNodeIndex[bettingSequence];
}

BettingNode* BettingTree::getBettingNodes() {
    return bettingNodes;
}

