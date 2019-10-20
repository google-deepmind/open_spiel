//
// Created by dj on 10/19/19.
//

#include "game_tree.h"
#include <sstream>
#include <assert.h>

namespace open_spiel::universal_poker::logic {
    GameTree::GameNode::GameNode( logic::GameTree* gameTree)
            : BettingNode(gameTree), deck_(gameTree->NumSuitsDeck(), gameTree->NumRanksDeck()), gameTree_(gameTree),
              actionCount_(GetPossibleActions().size())
    {
        for(uint8_t p=0; p < gameTree_->GetNbPlayers(); p++ ){
            holeCards_.emplace_back();
        }

        if(GetNodeType() == NODE_TYPE_CHANCE){
            actionCount_ = deck_.ToCardArray().size();
        }
    }

    void GameTree::GameNode::ApplyAction(uint32_t actionIdx) {
        if( GetNodeType() == NODE_TYPE_CHANCE ){
            BettingNode::ApplyDealCards();
            uint8_t card = deck_.ToCardArray()[actionIdx];
            deck_.RemoveCard(card);

            //Check where to add this card
            for(uint8_t p=0; p < gameTree_->GetNbPlayers(); p++ ){
                if( holeCards_[p].CountCards() < gameTree_->GetNbHoleCardsRequired() ){
                    holeCards_[p].AddCard(card);
                    break;
                }
            }

            if( boardCards_.CountCards() < gameTree_->GetNbBoardCardsRequired(GetRound())) {
                boardCards_.AddCard(card);
            }
        }
        else {
            BettingNode::ApplyChoiceAction(actionIdx);
        }

        if(GetNodeType() == NODE_TYPE_CHANCE ) {
            actionCount_ = deck_.ToCardArray().size();
        }
        else {
            actionCount_ = GetPossibleActions().size();
        }
    }

    uint32_t GameTree::GameNode::GetActionCount() const {
        return actionCount_;
    }

    std::string GameTree::GameNode::ToString() const {
            std::ostringstream buf;

            for(uint8_t p=0; p < gameTree_->GetNbPlayers(); p++ ){
                buf << "P" << (int)p << " Cards: " << holeCards_[p].ToString() << std::endl;
            }
            buf << "BoardCards " << boardCards_.ToString() << std::endl;


            if( GetNodeType() == NODE_TYPE_CHANCE ) {
                buf << "PossibleCardsToDeal " << deck_.ToString();
            }
            if( GetNodeType() == NODE_TYPE_TERMINAL_FOLD || GetNodeType() == NODE_TYPE_TERMINAL_SHOWDOWN){
                for(uint8_t p=0; p < gameTree_->GetNbPlayers(); p++ ){
                    buf << "P" << (int)p << " Reward: " << GetTotalReward(p) << std::endl;
                }

            }
            buf << BettingNode::ToString();


            return buf.str();
    }

    const CardSet& GameTree::GameNode::GetBoardCards() const {
        return boardCards_;
    }

    const CardSet& GameTree::GameNode::GetHoleCardsOfPlayer(uint8_t player) const {
        assert( player < holeCards_.size());
        return holeCards_[player];
    }

    double GameTree::GameNode::GetTotalReward(uint8_t player) const {
            assert(player < gameTree_->GetNbPlayers());
            // Copy Board Cards and Hole Cards
            uint8_t holeCards[10][3], boardCards[7], nbHoleCards[10], nbBoardCards;

            for( size_t p = 0; p < holeCards_.size(); p++){
                auto cards = holeCards_[p].ToCardArray();
                for( size_t c = 0; c < cards.size(); c++){
                    holeCards[p][c] = cards[c];
                }
                nbHoleCards[p] = cards.size();
            }

            auto bc = boardCards_.ToCardArray();
            for( size_t c = 0; c < bc.size(); c++){
                boardCards[c] = bc[c];
            }
            nbBoardCards = bc.size();

            SetHoleAndBoardCards(holeCards, boardCards, nbHoleCards, nbBoardCards);

            return ValueOfState(player);
    }

    GameTree::GameTree(const std::string &gameDef) : BettingTree(gameDef) {

        gameDepth_ = GameNode(this).GetDepth();

    }

    int GameTree::GetGameDepth() const {
        return gameDepth_;
    }
}
