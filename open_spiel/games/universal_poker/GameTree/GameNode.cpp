//
// Created by Dennis Jöst on 03.05.18.
//

#include "GameNode.h"


GameNode::GameNode() {

}


GameNode::GameNode(std::string name, BettingNode *bettingNode, CardNode *cardNode, Game* game)
:bettingNode(bettingNode), cardNode(cardNode), name(name), game(game), depth(0)
{

}

uint8_t GameNode::getCurrentPlayer() const {
    return bettingNode->getPlayer();
}

BettingNode::BettingNodeType GameNode::getType() const {
    return bettingNode->getType();
}

uint8_t GameNode::getRound() const {
    return bettingNode->getRound();
}

float GameNode::getPotSize() const {
    return bettingNode->getPotSize();
}

size_t GameNode::getHandCount() {
    return cardNode->getHandCount();
}

Eigen::MatrixXf GameNode::getFoldMatrix() {
    return cardNode->viewFoldMatrix();
}

Eigen::MatrixXf GameNode::getCallMatrix() {
    return cardNode->viewCallMatrix();
}

Eigen::VectorXf GameNode::getHandMask(){
    return cardNode->viewHandMask();
}


CardSet GameNode::getBoardCards() {
    return cardNode->getBoardCards();
}

float GameNode::getBet0() {
    return bettingNode->getBet0();
}

float GameNode::getBet1() {
    return bettingNode->getBet1();
}

Eigen::MatrixXf GameNode::getConstantStrategy() {
    Eigen::MatrixXf result(children.size(), getHandCount());
    if( this->getType() == BettingNode::CHANCE_NODE) {
        float proba = 1.0f / (float) (children.size() - game->numHoleCards * game->numPlayers);
        result.fill(proba);

        int childIdx = 0;
        for (auto child : children) {
            int hand = 0;
            for (auto combo : cardNode->getHandCards()->combinations) {
                if (child->getBoardCards().isBlocking(combo)) {
                    result(childIdx, hand) = 0.0f;
                }
                hand++;
            }
            childIdx++;
        }
    }
    else {
        result.fill(1.0f / (float)children.size());
    }


    return result;
}

int GameNode::getChildCount() {
    return static_cast<int>(this->children.size());
}

GameNode *GameNode::getChild(int n) {
    assert(n < children.size());
    return children[n];
}

float GameNode::getBet0Normalized() {
    return getBet0() / (float)game->stack[0];
}

float GameNode::getBet1Normalized() {
    return getBet1() / (float)game->stack[1];
}

uint32_t GameNode::getDepth() const {
    return depth;
}
