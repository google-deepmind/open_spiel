//
// Created by Dennis Jöst on 03.05.18.
//

#include "open_spiel/games/universal_poker/acpc/game.h"
#include <iostream>
#include "CardTree.h"
#define DEBUG 0

CardTree::CardTree(Game *game)
:handCardIndex(game->numSuits, game->numRanks, game->numHoleCards), cardNodeCount(0), game(game)
{
    for(int r = 0; r < game->numRounds; r++){
        auto idx = CardSetIndex(game->numSuits, game->numRanks, game->numBoardCards[r]);
        indexPerRound.push_back(idx);
    }


    uint32_t totalBC = 0;
    for(int r = 0; r < game->numRounds; r++){
        totalBC += game->numBoardCards[r];
        auto idx = CardSetIndex(game->numSuits, game->numRanks, totalBC);
        assert(cardNodeCount + idx.combinations.size() <= MAX_CARD_NODES);
        for( auto cs: idx.combinations ){
            assert(cardNodeCount < MAX_CARD_NODES);
            if(DEBUG && cardNodeCount%100000 == 0) std::cout << cardNodeCount << "/" << MAX_CARD_NODES << std::endl;
            cardNodes[cardNodeCount] = CardNode(cs, &handCardIndex, this, r, cardNodeCount);
            CardNode* ptr = &cardNodes[cardNodeCount];
            cardNodeIndex[cs.cs.cards] = ptr;
            cardNodeCount++;
        }
    }
}

std::vector<CardNode*> CardTree::getChildStates(CardSet cs, int nextRound) {
    std::vector<CardNode*> result;
    for( CardSet idx : indexPerRound[nextRound].combinations )
    {
        if( !idx.isBlocking(cs) ) {
            idx.cs.cards |=  cs.cs.cards;
            result.push_back(getCardNode(idx));
        }
    }
    return result;
}

const CardSetIndex &CardTree::getHandCardIndex() const {
    return handCardIndex;
}

CardNode *CardTree::getCardNode(CardSet cs) {
    assert(cardNodeIndex.find(cs.cs.cards) != cardNodeIndex.end());
    return cardNodeIndex[cs.cs.cards];
}

int CardTree::getCardNodeCount() const {
    return cardNodeCount;
}

Game *CardTree::getGame() {
    return game;
}

CardNode *CardTree::getCardNodes() {
    return cardNodes;
}
