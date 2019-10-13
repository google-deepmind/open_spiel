//
// Created by Dennis Jöst on 01.05.18.
//

#include "CardNode.h"
#include "CardTree.h"
using namespace Eigen;

CardNode::CardNode(CardSet boardCards, CardSetIndex* handCards, CardTree* cardTree, uint8_t round, uint32_t index)
:boardCards(boardCards), handCards(handCards), handCount(handCards->combinations.size()), callMatrixCalculated(false), foldMatrixCalculated(false),
 cardTree(cardTree), round(round), index(index), handMaskCalculated(false)
{
    //calculateCallMatrix();
    //calculateFoldMatrix();
}


CardNode::CardNode() {

}


void CardNode::calculateCallMatrixLastRound() {
    callMatrix.resize(handCount, handCount);

    for( uint16_t hole_p1 = 0; hole_p1 < handCount; hole_p1++ ) {
        for( uint16_t hole_p2 = 0; hole_p2 < handCount; hole_p2++ ) {
            if( handsBlocking(hole_p1, hole_p2) ) {
                callMatrix(hole_p1, hole_p2) = 0.0f ;
            }
            else {
                CardSet cardsetP1 = handCards->combinations[hole_p1];
                CardSet cardsetP2 = handCards->combinations[hole_p2];

                cardsetP1.cs.cards = cardsetP1.cs.cards | boardCards.cs.cards;
                cardsetP2.cs.cards = cardsetP2.cs.cards | boardCards.cs.cards;

                int strengthP1 = cardsetP1.rank();
                int strengthP2 = cardsetP2.rank();

                if( strengthP1 > strengthP2 ) {
                    callMatrix(hole_p1, hole_p2) = -1.0f ;
                }
                else if ( strengthP1 < strengthP2 ) {
                    callMatrix(hole_p1, hole_p2) = 1.0f ;
                }
                else {
                    callMatrix(hole_p1, hole_p2) = 0.0f ;
                }
            }
        }
    }

    callMatrixCalculated = true;
}

void CardNode::calculateCallMatrix()
{
    if( cardTree->getGame()->numRounds-1 == round ) {
        this->calculateCallMatrixLastRound();
    }
    else {
        callMatrix.resize(handCount, handCount);
        callMatrix.fill(0.0f);

        auto childStates = cardTree->getChildStates(boardCards, round+1);
        for( CardNode* child: childStates ){
            callMatrix += child->viewCallMatrix();
        }

        callMatrix /= (float)(childStates.size() - cardTree->getGame()->numHoleCards * cardTree->getGame()->numPlayers) ;
    }


}


void CardNode::calculateFoldMatrix() {
    assert( handCount == handCards->combinations.size() );
    foldMatrix.resize(handCount, handCount);

    for( uint16_t hole_p1 = 0; hole_p1 < handCount; hole_p1++ ) {
        for( uint16_t hole_p2 = 0; hole_p2 < handCount; hole_p2++ ) {
            bool blocking = handsBlocking(hole_p1, hole_p2);
            if( blocking ){
                foldMatrix(hole_p1, hole_p2) = 0.0f ;
            }
            else {
                foldMatrix(hole_p1, hole_p2) = 1.0f ;
            }
        }
    }

    foldMatrixCalculated = true;
}


void CardNode::calculateHandMask() {
    handMask.resize(handCount);

    for( int h = 0; h < handCount; h++){
        handMask(h, 0) = handCards->combinations[h].isBlocking(boardCards) ? 0.0f : 1.0f;
    }

    handMaskCalculated = true;
}


bool CardNode::handsBlocking(int p1, int p2) {
    return handCards->combinations[p1].isBlocking(boardCards) ||
           handCards->combinations[p2].isBlocking(boardCards) ||
           handCards->combinations[p1].isBlocking(handCards->combinations[p2]);
}

MatrixXf CardNode::viewCallMatrix() {
    if( !callMatrixCalculated ){
        calculateCallMatrix();
    }

    return callMatrix;
}

MatrixXf CardNode::viewFoldMatrix() {
    if(!foldMatrixCalculated){
        calculateFoldMatrix();
    }

   return foldMatrix;

}

VectorXf CardNode::viewHandMask() {
    if( !handMaskCalculated ){
        calculateHandMask();
    }

    return handMask;
}



size_t CardNode::getHandCount() {
    return handCount;
}

CardSet CardNode::getBoardCards() {
    return boardCards;
}

CardSetIndex *CardNode::getHandCards() const {
    return handCards;
}

uint8_t CardNode::getRound() {
    return round;
}

uint32_t CardNode::getIndex() const {
    return index;
}

