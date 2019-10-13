//
// Created by Dennis Jöst on 03.05.18.
//

#ifndef DEEPSTACK_CPP_CARDTREE_H
#define DEEPSTACK_CPP_CARDTREE_H

//Max combinations 1 + (52 über 3) + (52 über 4) + (52 über 5)
// 1+22100+270725+2598960 = 2891786
#define MAX_CARD_NODES (2891786)

#include <unordered_map>
#include "CardSetIndex.h"
#include "CardNode.h"

extern "C" {
    #include "open_spiel/games/universal_poker/ACPC/game.h"
};

class CardTree {
private:
    Game* game;
    std::vector<CardSetIndex> indexPerRound;
    CardSetIndex handCardIndex;
    CardNode cardNodes[MAX_CARD_NODES];
    std::unordered_map<int64_t, CardNode*> cardNodeIndex;
    int cardNodeCount;

public:
    CardTree(Game* game);

    CardNode *getCardNode(CardSet cs);
    std::vector<CardNode*> getChildStates(CardSet cs, int nextRound);
    const CardSetIndex &getHandCardIndex() const;

    int getCardNodeCount() const;
    CardNode* getCardNodes();
    Game* getGame();

};


#endif //DEEPSTACK_CPP_CARDTREE_H
