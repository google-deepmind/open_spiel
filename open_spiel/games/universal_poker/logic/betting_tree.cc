//
// Created by dj on 10/19/19.
//

#include <assert.h>
#include "betting_tree.h"
#include <sstream>

namespace open_spiel::universal_poker::logic {

    BettingTree::BettingTree(const std::string& gameDef)
            : ACPCGame(gameDef) {

    }

    uint32_t BettingTree::GetMaxBettingActions() const {
        return IsLimitGame() ? 3 : 4;
    }

    BettingTree::BettingNode::BettingNode(BettingTree* bettingTree)
    : ACPCState(bettingTree), bettingTree_(bettingTree), nodeType_(NODE_TYPE_CHANCE), possibleActions_({ACTION_DEAL}),
      nbBoardCardsDealt_(0), nbHoleCardsDealtPerPlayer_{0,0,0,0,0,0,0,0,0,0}, potSize_(0), allInSize_(0)
    {
    }

    BettingTree::BettingNode::NodeType BettingTree::BettingNode::GetNodeType() const {
        return nodeType_;
    }

    void BettingTree::BettingNode::ApplyChoiceAction(uint32_t actionIdx) {
        assert(nodeType_ == NODE_TYPE_CHOICE);
        assert(actionIdx >= 0 && actionIdx < possibleActions_.size());

        ActionType actionType = possibleActions_[actionIdx];
        actionSequence_ += (char)actionType;
        switch(actionType)
        {
            case ACTION_FOLD:
                DoAction(ACPC_FOLD, 0);
                break;
            case ACTION_CHECK_CALL:
                DoAction(ACPC_CALL, 0);
                break;
            case ACTION_BET_POT:
                DoAction(ACPC_RAISE, potSize_);
                break;
            case ACTION_ALL_IN:
                DoAction(ACPC_RAISE, allInSize_);
                break;
            case ACTION_DEAL:
            default:
                assert(false);
                break;
        }

        _calculateActionsAndNodeType();
    }

    void BettingTree::BettingNode::ApplyDealCards() {
        assert(nodeType_ == NODE_TYPE_CHANCE);
        actionSequence_ += 'd';

        for(uint8_t p=0; p < bettingTree_->GetNbPlayers(); p++ ){
            if(nbHoleCardsDealtPerPlayer_[p] < bettingTree_->GetNbHoleCardsRequired() ){
                nbHoleCardsDealtPerPlayer_[p]++;
                _calculateActionsAndNodeType();
                return;
            }
        }

        if( nbBoardCardsDealt_ < bettingTree_->GetNbBoardCardsRequired(GetRound())) {
            nbBoardCardsDealt_++;
            _calculateActionsAndNodeType();
            return;
        }

        assert(false);
    }

    void BettingTree::BettingNode::_calculateActionsAndNodeType() {
        possibleActions_.clear();

        if(ACPCState::IsFinished()) {
            if( NumFolded() >= bettingTree_->GetNbPlayers() - 1){
                nodeType_ = NODE_TYPE_TERMINAL_FOLD;
            }
            else {
                if( nbBoardCardsDealt_ < bettingTree_->GetNbBoardCardsRequired(GetRound())) {
                    nodeType_ = NODE_TYPE_CHANCE;
                    possibleActions_.push_back(ACTION_DEAL);
                    return;
                }
                nodeType_= NODE_TYPE_TERMINAL_SHOWDOWN;
            }

        }
        else {
            // Check for sth to deal
            for(uint8_t p=0; p < bettingTree_->GetNbPlayers(); p++ ){
                if(nbHoleCardsDealtPerPlayer_[p] < bettingTree_->GetNbHoleCardsRequired() ){
                    nodeType_ = NODE_TYPE_CHANCE;
                    possibleActions_.push_back(ACTION_DEAL);
                    return;
                }
            }
            if( nbBoardCardsDealt_ < bettingTree_->GetNbBoardCardsRequired(GetRound())) {
                nodeType_ = NODE_TYPE_CHANCE;
                possibleActions_.push_back(ACTION_DEAL);
                return;
            }

            //Check for CHOICE Actions
            nodeType_ = NODE_TYPE_CHOICE;
            if( IsValidAction(ACPC_FOLD, 0) ){
                possibleActions_.push_back(ACTION_FOLD);
            }
            if( IsValidAction(ACPC_CALL, 0) ){
                possibleActions_.push_back(ACTION_CHECK_CALL);
            }

            potSize_ = 0;
            allInSize_ = 0;

            if( RaiseIsValid(&potSize_, &allInSize_) ){
                if(bettingTree_->IsLimitGame()){
                    potSize_ = 0;
                    possibleActions_.push_back(ACTION_BET_POT);
                }
                else {
                    int32_t currentPot = MaxSpend() * (bettingTree_->GetNbPlayers() - NumFolded());
                    potSize_ = currentPot > potSize_ ? currentPot : potSize_;
                    potSize_ = allInSize_ < potSize_ ? allInSize_ : potSize_;

                    possibleActions_.push_back(ACTION_BET_POT);
                    if( allInSize_ > potSize_ ) {
                        possibleActions_.push_back(ACTION_ALL_IN);
                    }
                }
            }
        }
    }

    const std::vector<BettingTree::ActionType> &BettingTree::BettingNode::GetPossibleActions() const {
        return possibleActions_;
    }

    std::string BettingTree::BettingNode::ToString() const {
        std::ostringstream buf;
        buf << "NodeType: ";
        buf << (nodeType_ == NODE_TYPE_CHANCE ? "NODE_TYPE_CHANCE" : "");
        buf << (nodeType_ == NODE_TYPE_CHOICE ? "NODE_TYPE_CHOICE" : "");
        buf << (nodeType_ == NODE_TYPE_TERMINAL_SHOWDOWN ? "NODE_TYPE_TERMINAL_SHOWDOWN" : "");
        buf << (nodeType_ == NODE_TYPE_TERMINAL_FOLD ? "NODE_TYPE_TERMINAL_FOLD" : "");
        buf << std::endl;

        buf << "PossibleActions (" << possibleActions_.size() << "): [";
        for( auto action : possibleActions_ ){
            buf << (action == ACTION_ALL_IN ? " ACTION_ALL_IN " : "");
            buf << (action == ACTION_BET_POT ? " ACTION_BET_POT " : "");
            buf << (action == ACTION_CHECK_CALL ? " ACTION_CHECK_CALL " : "");
            buf << (action == ACTION_FOLD ? " ACTION_FOLD " : "");
            buf << (action == ACTION_DEAL ? " ACTION_DEAL " : "");
        }
        buf << "]" << std::endl;
        buf << "Round: " << (int)GetRound() << std::endl;
        buf << "ACPC State: " << ACPCState::ToString() << std::endl;
        buf << "Action Sequence: " << actionSequence_ << std::endl;
        return buf.str();
    }

    int BettingTree::BettingNode::GetDepth() {
        int maxDepth = 0;
        for(size_t action = 0; action < possibleActions_.size(); action++){
            BettingNode child(*this);
            if( child.GetNodeType() == NODE_TYPE_CHANCE ){
                child.ApplyDealCards();
            }
            else if ( child.GetNodeType() == NODE_TYPE_CHOICE ){
                child.ApplyChoiceAction(action);
            }
            int depth = child.GetDepth();
            maxDepth = depth > maxDepth ? depth : maxDepth;
        }

        return 1+maxDepth;
    }

    std::string BettingTree::BettingNode::GetActionSequence() const {
        return actionSequence_;
    }

    bool BettingTree::BettingNode::IsFinished() const {
        bool finished = nodeType_== NODE_TYPE_TERMINAL_SHOWDOWN || nodeType_ == NODE_TYPE_TERMINAL_FOLD;
        assert( ACPCState::IsFinished() || !finished );

        return finished;
    }


}


