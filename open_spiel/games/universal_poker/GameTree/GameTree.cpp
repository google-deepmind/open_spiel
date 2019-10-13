//
// Created by Dennis Jöst on 03.05.18.
//

extern "C" {
    #include <ACPC/game.h>
};

#include <iostream>
#include <fstream>
#include <ACPC/game.h>
#include <random>
#include "GameTree.h"
#include <regex>


uint32_t choose(uint32_t n, uint32_t k) {
    if(k == 0) {
        return 1;
    }
    return (n * choose(n - 1, k - 1)) / k;
}


GameTree::GameTree(Game* g, std::string gameName)
:
game(*g),
bettingTree(std::make_unique<BettingTree>(&game)),
cardTree(std::make_unique<CardTree>(&game)),
generator(time(0)),
gameName(std::move(gameName))
{

}




void GameTree::buildGameTree(CardSet initialBoard, int bettingNodeIdx, int maxDepth) {
    gameNodeCount = 0;
    assert(bettingNodeIdx < bettingTree->getBettingNodeCount());

    BettingNode* bettingNode = &bettingTree->getBettingNodes()[bettingNodeIdx];
    int round = bettingNode->getRound();

    uint32_t nbBoardCards = 0;
    for( int r = 0; r <= round; r++){
        nbBoardCards += game.numBoardCards[r];
    }
    assert(nbBoardCards == initialBoard.countCards() );
    CardNode* cardNode = cardTree->getCardNode(initialBoard);

    root = createGameNode(bettingNode, cardNode, maxDepth);
}

GameNode *GameTree::createGameNode(BettingNode *bettingNode, CardNode *cardNode, int maxDepth) {
    assert(gameNodeCount < MAX_GAME_NODES);
    GameNode *result = &gameNodes[gameNodeCount];
    gameNodeCount++;

    std::stringstream n;
    n << "node_" << gameNodeCount-1;
    result->name = n.str();
    result->bettingNode = bettingNode;
    result->cardNode = cardNode;
    result->game = &game;

    if (maxDepth > 0) {
        std::vector<GameNode *> children;
        if (bettingNode->getType() == BettingNode::CHOICE_NODE) {
            for (int i = 0; i < bettingNode->childCount; i++) {
                GameNode *child = createGameNode(bettingNode->children[i], cardNode, maxDepth - 1);
                children.push_back(child);
                result->depth = std::max<uint32_t>(result->depth, child->depth+1);
            }
        }

        if (bettingNode->getType() == BettingNode::CHANCE_NODE) {
            assert(bettingNode->childCount == 1);

            std::vector<CardNode *> cardNodes = cardTree->getChildStates(cardNode->getBoardCards(), bettingNode->getRound()+1);
            for (auto cardNodeChild : cardNodes) {
                assert( bettingNode->childCount == 1);
                GameNode *child = createGameNode(bettingNode->children[0], cardNodeChild, maxDepth - 1);
                children.push_back(child);
                result->depth = std::max<uint32_t>(result->depth, child->depth+1);
            }
        }
        result->children = children;
    }

    return result;
}

size_t GameTree::getGameNodeCount() const {
    return gameNodeCount;
}

GameNode* GameTree::getRoot() {
    return root;
}

BettingTree *GameTree::getBettingTree() const {
    return bettingTree.get();
}

CardTree *GameTree::getCardTree() const {
    return cardTree.get();
}

std::string GameTree::getTreeAsString() {
    std::stringstream out;

    for(int i = 0; i < gameNodeCount; i++)
    {
        GameNode* n = &gameNodes[i];
        out << std::endl;
        out << "getType: " << n->getType() << std::endl;
        out << "getRound: " << (int)n->getRound() << std::endl;
        out << "getPotSize: " << n->getPotSize() << std::endl;
        out << "getHandCount: " << n->getHandCount() << std::endl;
        out << "getCurrentPlayer: " << (int)n->getCurrentPlayer() << std::endl;
        out << "boardCards: " << n->cardNode->getBoardCards().toString() << std::endl;

        if(n->getType() == BettingNode::TERMINAL_SHOWDOWN_NODE ){
            out << std::endl << getMatrixHead() << std::endl;
            out << n->getCallMatrix() << std::endl;
        }
        if(n->getType() == BettingNode::TERMINAL_FOLD_NODE ){
            out << std::endl << getMatrixHead() << std::endl;
            out << n->getFoldMatrix() << std::endl;
        }
        if(n->getType() == BettingNode::CHANCE_NODE ){
            out << std::endl << "Strategy" << std::endl;
            out << n->getConstantStrategy() << std::endl;
        }
        out << std::endl;
    }

    return out.str();
}

std::string GameTree::getMatrixHead() {
    std::stringstream out;
    for( auto combo : cardTree->getHandCardIndex().combinations )
    {
        out << combo.toString() << " ";
    }
    return out.str();


}

void GameTree::writeToDot(const char *filename) {

    std::ofstream dotfile;
    dotfile.open (filename);
    dotfile << "digraph g {  graph [ rankdir = \"LR\"];node [fontsize = \"16\" shape = \"ellipse\"]; edge [];" << std::endl;

    for( int i=0; i< gameNodeCount; i ++)
    {
        dotfile << gameNodes[i].name << "[ label=" << nodeToGraphviz(&gameNodes[i]) << " shape=record ]" << std::endl;
    }

    for( int i=0; i< gameNodeCount; i ++)
    {
        for( GameNode* child: gameNodes[i].children )
        {
            dotfile << gameNodes[i].name << "->" << child->name << std::endl;
        }

    }

    dotfile << "}" << std::endl;

    dotfile.close();
}

std::string GameTree::nodeToGraphviz(GameNode *node) {
    std::stringstream out;

    out << "\"<f0> Player " << (int)node->getCurrentPlayer();

    switch(node->getType())
    {
        case BettingNode::CHANCE_NODE:
            out << "| CHANCE_NODE";
            break;
        case BettingNode::CHOICE_NODE:
            out << "| CHOICE_NODE";
            break;
        case BettingNode::TERMINAL_FOLD_NODE:
            out << "| TERMINAL_FOLD_NODE";
            break;
        case BettingNode::TERMINAL_SHOWDOWN_NODE:
            out << "| TERMINAL_SHOWDOWN_NODE";
            break;
        case BettingNode::EMPTY:
            assert(false);
            break;
    }

    out << "| bet0: " << node->getBet0();
    out << "| bet1: " << node->getBet1();
    out << "| potsize: " << node->getPotSize();
    out << "| street: " << (int)node->getRound();
    out << "| board: " << node->getBoardCards().toString();
    out << "\"";

    return out.str();
}

void GameTree::buildRandomTree(int round, int maxDepth) {
    assert(round < game.numRounds);

    uint32_t nbBoardCards = 0;
    for( int r = 0; r <= round; r++){
        nbBoardCards += game.numBoardCards[r];
    }
    CardSetIndex cardSetIndex(game.numSuits, game.numRanks, nbBoardCards);

    std::uniform_int_distribution<int> distribution(0,static_cast<int>(cardSetIndex.combinations.size() - 1));
    int idx = distribution(generator);
    CardSet cardSet = cardSetIndex.combinations[idx];

    std::vector<int> bettingStatesIndexes;
    for( int i = 0; i < bettingTree->getBettingNodeCount(); i++){
        BettingNode* node = &bettingTree->getBettingNodes()[i];
        if(node->getRound() == round && node->getType() == BettingNode::CHOICE_NODE){
            bettingStatesIndexes.push_back(i);
        }
    }

    std::uniform_int_distribution<int> distribution2(0,static_cast<int>(bettingStatesIndexes.size() - 1));
    int idx2 = distribution2(generator);
    int bettingStateIndex = bettingStatesIndexes[idx2];

    buildGameTree(cardSet, bettingStateIndex, maxDepth);

}

std::unique_ptr<GameTree> GameTree::createFromGamedef(const char* fileName) {
    FILE* f = fopen(fileName, "r");
    assert(f != NULL);

    std::string s (fileName);
    std::stringstream out;
    std::smatch m;
    std::regex e ("[0-9a-zA-Z_\\.-]+.game$");

    if(std::regex_search (s,m,e)) {
        out << m[0] ;
    }

    Game* game = readGame( f );
    std::unique_ptr<GameTree> result = std::make_unique<GameTree>(game, out.str());
    free(game);
    fclose(f);
    return result;

}

void GameTree::buildGameTreeWithSeq(CardSet initialBoard, std::string bettingSequence, int maxDepth) {

    buildGameTree(initialBoard, bettingTree->getBettingNode(bettingSequence)->getIndex(), maxDepth);
}

std::vector<std::string> GameTree::getBettingSequencesForRound(uint8_t round) {
    std::vector<std::string> result;
    for( size_t i = 0; i < bettingTree->getBettingNodeCount(); i++)
    {
        BettingNode* node = &bettingTree->getBettingNodes()[i];
        if( node->getRound() == round ){
            result.push_back(node->getSequence());
        }
    }


    return result;
}

std::vector<CardSet> GameTree::getCardSetsForRound(uint8_t round) {
    std::vector<CardSet> result;
    for( int i = 0; i < cardTree->getCardNodeCount(); i++) {
        CardNode *cn = &cardTree->getCardNodes()[i];
        if (cn->getRound() == round) {
            result.push_back(cn->getBoardCards());
        }
    }

    return result;

}

uint32_t GameTree::getBettingNodeCount() {
    return bettingTree->getBettingNodeCount();
}

uint32_t GameTree::getCardNodeCount() {
    return cardTree->getCardNodeCount();
}


BettingNode *GameTree::getBettingNode(uint32_t index){
    assert(index < bettingTree->getBettingNodeCount());
    return &bettingTree->getBettingNodes()[index];
}


CardNode *GameTree::getCardNode(uint32_t index) {
    assert(index < cardTree->getCardNodeCount());
    return &cardTree->getCardNodes()[index];
}

std::string GameTree::getGameName() {
    return this->gameName;
}

uint8_t GameTree::getPlayerCount() {
    return game.numPlayers;
}

float GameTree::getBlind(uint8_t player) {
    assert( player < getPlayerCount() );

    return (float)game.blind[player];
}

uint32_t GameTree::getHandCount() {
    uint32_t deckSize = game.numSuits * game.numRanks;
    return choose( deckSize, game.numHoleCards );
}

uint8_t GameTree::getRounds() {
    return game.numRounds;
}

