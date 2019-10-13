//
// Created by Dennis Jöst on 03.05.18.
//

#include <cassert>
#include <string>
#include "BettingNode.h"

BettingNode::BettingNode()
        :type(EMPTY), player(0), round(0), potSize(0.0f), childCount(0), sequence()
{

}


void BettingNode::initialize(BettingNode::BettingNodeType type1, uint8_t player1, uint8_t round1, float potSize1, float pBet0, float pBet1, std::string pSequence, uint32_t index1) {
    type = type1;
    player = player1;
    round = round1;
    potSize = potSize1;
    bet0 = pBet0;
    bet1 = pBet1;
    sequence = pSequence;
    index = index1;

}

BettingNode::BettingNodeType BettingNode::getType() const {
    return type;
}

uint8_t BettingNode::getPlayer() const {
    return player;
}

uint8_t BettingNode::getRound() const {
    return round;
}

float BettingNode::getPotSize() const {
    return potSize;
}

float BettingNode::getBet0() const {
    return bet0;
}

float BettingNode::getBet1() const {
    return bet1;
}

std::string BettingNode::getSequence() const {
    return sequence;
}

uint32_t BettingNode::getIndex() const {
    return index;
}

