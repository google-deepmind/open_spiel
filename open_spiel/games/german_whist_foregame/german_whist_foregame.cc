
#include <filesystem>
//to do
//InfostateTensor implementation
// PR!!!!!


#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

namespace open_spiel {
namespace german_whist_foregame {


std::string kTTablePath="";
bool Triple::operator<(const Triple& triple)const{
    return (length < triple.length)|| (length == triple.length && sig < triple.sig);
}

inline int CardRank(int card, int suit) {
    uint64_t card_mask = ((uint64_t)1 << card);
    card_mask = (card_mask >> (suit * kNumRanks));
    return _tzcnt_u64(card_mask);
}
inline int CardSuit(int card) {
    uint64_t card_mask = ((uint64_t)1 << card);
    for (int i = 0; i < kNumSuits; ++i) {
        if (_mm_popcnt_u64(card_mask & kSuitMasks[i]) == 1) {
            return i;
        }
    }
    return kNumSuits;
}
std::string CardString(int card) {
    int suit = CardSuit(card);
    return { kSuitChar[suit],kRankChar[CardRank(card,suit)] };
}

std::vector<uint32_t> GenQuads(int size_endgames) {
    //Generates Suit splittings for endgames of a certain size//
    std::vector<uint32_t> v;
    for (char i = 0; i <= std::min(size_endgames * 2, kNumRanks); ++i) {
        int sum = size_endgames * 2 - i;
        for (char j = 0; j <= std::min(sum, kNumRanks); ++j) {
            for (char k = std::max((int)j, sum - j - kNumRanks); k <= std::min(sum - j, kNumRanks); ++k) {
                char l = sum - j - k;
                if (l < k) {
                    break;
                }
                else {
                    uint32_t num = 0;
                    num = num | (i);
                    num = num | (j << 4);
                    num = num | (k << 8);
                    num = num | (l << 12);
                    v.push_back(num);
                }
            }
        }
    }
    return v;
}
std::vector<std::vector<uint32_t>> BinCoeffs(uint32_t max_n) {
    //tabulates binomial coefficients//
    std::vector<std::vector<uint32_t>> C(max_n+1,std::vector<uint32_t>(max_n+1));
    for (uint32_t i = 1; i <= max_n; ++i) {
        C[0][i] = 0;
    }
    for (uint32_t i = 0; i <= max_n; ++i) {
        C[i][0] = 1;
    }
    for (uint32_t i = 1; i <= max_n; ++i) {
        for (uint32_t j = 1; j <= max_n; ++j) {
            C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
        }
    }
    return C;
}
uint32_t HalfColexer(uint32_t cards,const std::vector<std::vector<uint32_t>>* bin_coeffs) {
    //returns the colexicographical ranking of a combination of indices where the the size of the combination is half that of the set of indices//
    uint32_t out = 0;
    uint32_t count = 0;
    while (cards != 0) {
        uint32_t ind = _tzcnt_u32(cards);
        uint32_t val = bin_coeffs->at(ind)[count+1];
        out += val;
        cards = _blsr_u32(cards);
        count++;
    }
    return out;
}
void GenSuitRankingsRel(uint32_t size, std::unordered_map<uint32_t, uint32_t>* Ranks) {
    //Generates ranking Table for suit splittings for endgames of a certain size//
    std::vector<uint32_t> v=GenQuads(size);
    for (uint32_t i = 0; i < v.size(); ++i) {
        Ranks->insert({ v[i],i });
    }
}

vectorNa::vectorNa(size_t card_combs,size_t suit_splits,char val){
    data=std::vector<char>(card_combs*((suit_splits>>1)+1),val);
    inner_size =(suit_splits>>1)+1;
    outer_size = card_combs;
}
vectorNa::vectorNa(){
    data={};
    inner_size=0;
    outer_size=0;
}
size_t vectorNa::size() const{
    return data.size();
}
size_t vectorNa::GetInnerSize()const{
    return inner_size;
}
size_t vectorNa::GetOuterSize()const{
    return outer_size;
}
char const& vectorNa::operator[](size_t index) const{
    return data[index];
}
char vectorNa::GetChar(size_t i,size_t j)const{
    return data[i*inner_size+j];
}
void vectorNa::SetChar(size_t i,size_t j,char value){
    data[i*inner_size+j]=value;
}
char vectorNa::Get(size_t i,size_t j) const{
    int remainder = j&0b1;
    if(remainder==0){
        return 0b1111&data[i*inner_size+(j>>1)];
    }
    else{
        return ((0b11110000&data[i*inner_size+(j>>1)])>>4);
    }
}
void vectorNa::Set(size_t i,size_t j,char value){
    int remainder = j & 0b1;
    if (remainder == 0) {
        char datastore = 0b11110000 & data[i*inner_size+(j>>1)];
        data[i*inner_size+(j>>1)] = datastore|value;
    }
    else {
        char datastore = (0b1111 & data[i*inner_size+(j>>1)]);
        data[i*inner_size+(j>>1)] = datastore|(value << 4);
    }
}
vectorNa InitialiseTTable(int size,std::vector<std::vector<uint32_t>>& bin_coeffs) {
    //initialises TTable for a certain depth//
    size_t suit_size = GenQuads(size).size();
    return vectorNa(bin_coeffs[2 * size][size],suit_size, 0);
}
vectorNa LoadTTable(const std::string filename, int depth,std::vector<std::vector<uint32_t>>& bin_coeffs){
    //loads solution from a text file into a vector for use//
    std::cout<<"Loading Tablebase"<<std::endl;
    vectorNa v = InitialiseTTable(depth,bin_coeffs);
    std::ifstream file(filename,std::ios::binary);
    //std::cout<<file.is_open()<<std::endl;
    //std::cout<<"Current working directory "<<std::filesystem::current_path()<<std::endl;
    char c;
    for(int i =0;i<v.GetOuterSize();++i){
        for(int j =0;j<v.GetInnerSize();++j){
            file.get(c);
            v.SetChar(i,j,c);
        }
    }
    file.close();
    std::cout<<"Tablebase Loaded"<<std::endl;
    return v;
}

// Default parameters.

namespace {//namespace
// Facts about the game
const GameType kGameType{/*short_name=*/"german_whist_foregame",
    /*long_name=*/"german_whist_foregame",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
    return std::shared_ptr<const Game>(new GWhistFGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}//namespace

GWhistFGame::GWhistFGame(const GameParameters& params):Game(kGameType, params){
    bin_coeffs_=BinCoeffs(2*kNumRanks);
    std::unordered_map<uint32_t,uint32_t> temp;
    GenSuitRankingsRel(13,&temp);
    suit_ranks_=temp;
    ttable_ = LoadTTable(kTTablePath,13,bin_coeffs_);
};
std::unique_ptr<State> GWhistFGame::NewInitialState() const {
    const auto ptr=std::dynamic_pointer_cast<const GWhistFGame>(shared_from_this());
    return std::make_unique<GWhistFState>(ptr);
}


GWhistFState::GWhistFState(std::shared_ptr<const GWhistFGame> game):State(game) {
    player_ = kChancePlayerId;
    move_number_ = 0;
    trump_ = -1;
    deck_ = _bzhi_u64(~0,kNumRanks*kNumSuits);
    discard_ = 0;
    hands_ = { 0,0 };
    history_.reserve(78);
    ttable_ = &(game->ttable_);
    suit_ranks_ =&(game->suit_ranks_);
    bin_coeffs_=&(game->bin_coeffs_);
}
bool GWhistFState::Trick(int lead, int follow) const {
    int lead_suit = CardSuit(lead);
    int follow_suit = CardSuit(follow);
    int lead_rank = CardRank(lead,lead_suit);
    int follow_rank = CardRank(follow,follow_suit);
    return (lead_suit == follow_suit && lead_rank < follow_rank) || (lead_suit != follow_suit && follow_suit != trump_);
}
bool GWhistFState::IsTerminal() const {
    return(_mm_popcnt_u64(deck_) == 0);
}
uint64_t GWhistFState::EndgameKey(int player_to_move) const{
    //generates a 64 bit unsigned int where the first 32 are the suit ownerships from the perspective of the opponent using canonical rankings//
    //example: if Spade suit is to_move = A3, opp =2, suit = 0b100
    //least significant part of first 32 bits is the trump suit, then the remaining suits ascending length order.
    uint64_t cards_in_play = hands_[0]|hands_[1];
    std::vector<Triple> suit_lengths = {};
    int opp = (player_to_move==0)?1:0;
    //sort trump suits by length,then sig//
    for(int i =0;i<kNumSuits;++i){
        if(i!=trump_){
            uint64_t sig = _pext_u64(hands_[opp]&kSuitMasks[i],cards_in_play&kSuitMasks[i]);
            suit_lengths.push_back(Triple{i,_mm_popcnt_u64(kSuitMasks[i]&cards_in_play),sig});
        }
    }
    std::sort(suit_lengths.begin(),suit_lengths.end());
    std::array<uint64_t,kNumSuits> hand0;
    std::array<uint64_t,kNumSuits> hand1;
    hand0[0]=_pext_u64(hands_[0],kSuitMasks[trump_]);
    hand1[0]=_pext_u64(hands_[1],kSuitMasks[trump_]);
    for(int i =0;i<kNumSuits-1;++i){
        hand0[i+1]=_pext_u64(hands_[0],kSuitMasks[suit_lengths[i].index]);
        hand1[i+1]=_pext_u64(hands_[1],kSuitMasks[suit_lengths[i].index]);
    }
    std::array<uint64_t,2>hands_shuffled = {0,0};
    for(int i =0;i<kNumSuits;++i){
        hands_shuffled[0]=hands_shuffled[0]|(hand0[i]<<(kNumRanks*i));
        hands_shuffled[1]=hands_shuffled[1]|(hand1[i]<<(kNumRanks*i));
    }
    uint64_t suit_sig =0;
    suit_sig = _mm_popcnt_u64(kSuitMasks[trump_]&cards_in_play);
    for(int i =0;i<kNumSuits-1;++i){
        suit_sig = suit_sig|((uint64_t)suit_lengths[i].length << (4*(i+1)));
    }
    suit_sig = (suit_sig<<32);
    cards_in_play = hands_shuffled[0]|hands_shuffled[1];
    uint64_t cards = _pext_u64(hands_shuffled[opp],cards_in_play);
    uint64_t key = cards|suit_sig;
    return key;
}
std::vector<double> GWhistFState::Returns() const{
    if(IsTerminal()){
        std::vector<double> out = {0,0};
        int lead_win = Trick(history_[move_number_ - 3].action, history_[move_number_ - 2].action);
        int player_to_move=(lead_win)?history_[move_number_-3].player:history_[move_number_-2].player;
        int opp = (player_to_move==0)?1:0;
        uint64_t key = EndgameKey(player_to_move);
        uint32_t cards = (key&_bzhi_u64(~0,32));
        uint32_t colex = HalfColexer(cards,bin_coeffs_);
        uint32_t suits = (key&(~0^_bzhi_u64(~0,32)))>>32;
        uint32_t suit_rank = suit_ranks_->at(suits);
        char value =ttable_->Get(colex,suit_rank);
        out[player_to_move] = 2*value-kNumRanks;
        out[opp]=-out[player_to_move];
        return out;
    }
    else{
        std::vector<double> out = {0,0};
        return out;
    }
}


int GWhistFState::CurrentPlayer() const { return player_; }

std::vector<std::pair<Action, double>> GWhistFState::ChanceOutcomes() const {
    std::vector<std::pair<Action, double>> outcomes;
    std::vector<Action> legal_actions = LegalActions();
    for(int i =0;i<legal_actions.size();++i){
        std::pair<Action,double> pair;
        pair.first =legal_actions[i];
        pair.second = 1/double(legal_actions.size());
        outcomes.push_back(pair);
    }
    return outcomes;
}
std::string GWhistFState::ActionToString(Player player,Action move) const {
    return CardString(move);
}
std::string GWhistFState::ToString() const{
    std::string out;
    for (int i = 0; i < history_.size(); ++i) {
        out += ActionToString(history_[i].player, history_[i].action);
        out += "\n";
    }
    return out;
}
std::unique_ptr<State> GWhistFState::Clone() const{
    return std::unique_ptr<State>(new GWhistFState(*this));
}

std::string GWhistFState::StateToString() const {
    //doesnt use history in case of a resampled state with unreconciled history//
    std::string out;
    uint64_t copy_deck = deck_;
    uint64_t copy_discard = discard_;
    std::array<uint64_t,2> copy_hands = hands_;
    std::vector<int> deck_cards;
    std::vector<int> player0_cards;
    std::vector<int> player1_cards;
    std::vector<int> discard;
    while (copy_deck != 0) {
        deck_cards.push_back(_tzcnt_u64(copy_deck));
        copy_deck = _blsr_u64(copy_deck);
    }
    while (copy_discard != 0) {
        discard.push_back(_tzcnt_u64(copy_discard));
        copy_discard = _blsr_u64(copy_discard);
    }

    while (copy_hands[0] != 0) {
        player0_cards.push_back(_tzcnt_u64(copy_hands[0]));
        copy_hands[0] = _blsr_u64(copy_hands[0]);
    }
    while (copy_hands[1] != 0) {
        player1_cards.push_back(_tzcnt_u64(copy_hands[1]));
        copy_hands[1] = _blsr_u64(copy_hands[1]);
    }
    out += "Deck \n";
    for (int i = 0; i < deck_cards.size(); ++i) {
        out += CardString(deck_cards[i]) + "\n";
    }
    out += "Discard \n";
    for (int i = 0; i < discard.size(); ++i) {
        out += CardString(discard[i]) + "\n";
    }

    for (int i = 0; i < 2; ++i) {
        out += "Player " + std::to_string(i) + "\n";
        std::vector<int> var;
        if (i == 0) {
            var = player0_cards;
        }
        else {
            var = player1_cards;
        }
        for (int j = 0; j < var.size(); ++j) {
            out += CardString(var[j]) + "\n";
        }
    }
    return out;
}
std::string GWhistFState::InformationStateString(Player player) const{
    //THIS IS WHAT A PLAYER IS SHOWN WHEN PLAYING//
    std::string p = std::to_string(player)+",";
    std::string cur_hand = "";
    std::string observations="";
    std::vector<int> v_hand = {};
    uint64_t p_hand = hands_[player];
    while(p_hand!=0){
        v_hand.push_back(_tzcnt_u64(p_hand));
        p_hand = _blsr_u64(p_hand);
    }
    std::sort(v_hand.begin(),v_hand.end());
    for(int i =0;i<v_hand.size();++i){
        cur_hand=cur_hand+CardString(v_hand[i]);
        cur_hand=cur_hand+",";
    }
    cur_hand+="\n";
    for(int i =2*kNumRanks;i<history_.size();++i){
        int index =(i-2*kNumRanks)%4;
        switch(index){
            case 0:
                observations=observations + "c_public:"+CardString(history_[i].action)+",";
                break;
            case 1:
                observations=observations+"p"+std::to_string(history_[i].player)+":"+CardString(history_[i].action)+",";
                break;
            case 2:
                observations=observations+"p"+std::to_string(history_[i].player)+":"+CardString(history_[i].action)+",";
                break;
            case 3:
                int lead_win = Trick(history_[i - 2].action, history_[i - 1].action);
                int loser = ((lead_win) ^ (history_[i - 2].player == 0)) ? 0 : 1;
                if(loser==player){
                    observations=observations+"c_observed:"+CardString(history_[i].action)+"\n";
                }
                else{
                    observations=observations+"c_unobserved:"+"\n";
                }
                break;
        }
    }
    return p+cur_hand+observations;
}
std::unique_ptr<State> GWhistFState::ResampleFromInfostate(int player_id,std::function<double()> rng) const{
        //only valid when called from a position where a player can act//
        auto resampled_state = std::unique_ptr<GWhistFState>(new GWhistFState(*this));
        //seeding mt19937//
        std::random_device rd;
        std::mt19937 gen(rd());
        uint64_t necessary_cards = 0;
        for (int i = 2 * kNumRanks; i < history_.size(); i+=4) {
            //face up cards from deck//
            necessary_cards = (necessary_cards | (uint64_t(1) << history_[i].action));
        }
        int move_index = move_number_ - ((kNumRanks * kNumSuits) / 2);
        int move_remainder = move_index % 4;
        int opp = (player_id == 0) ? 1 : 0;
        int recent_faceup = move_number_ - move_remainder;
        uint64_t recent_faceup_card = (uint64_t(1) << history_[recent_faceup].action);
        // if a face up card from the deck is not in players hand or discard it must be in opps unless it is the most recent face up//
        necessary_cards = (necessary_cards & (~(hands_[player_id] | discard_|recent_faceup_card)));
        //sufficient cards are all cards not in players hand,the discard, or the recent face up//
        uint64_t sufficient_cards = (_bzhi_u64(~0, kNumRanks * kNumSuits) ^(hands_[player_id] | discard_|recent_faceup_card));
        //sufficient_cards are not necessary //
        sufficient_cards = (sufficient_cards & (~(necessary_cards)));
        //we must now take into account the observation of voids//
        std::array<int, kNumSuits> when_voided = {0,0,0,0};
        std::array<int, kNumSuits> voids = {-1,-1,-1,-1};
        std::vector<int> opp_dealt_hidden;
        for (int i = 2 * kNumRanks; i < history_.size(); ++i) {
            if (history_[i - 1].player == player_id && history_[i].player == (opp) && CardSuit(history_[i-1].action)!=CardSuit(history_[i].action)) {
                when_voided[CardSuit(history_[i - 1].action)] = i - 1;
            }
            if (history_[i - 1].player == player_id && history_[i].player == (opp) && Trick(history_[i - 1].action, history_[i].action)) {
                opp_dealt_hidden.push_back(i - 1);
            }
            if (history_[i - 1].player == (opp) && history_[i].player == (player_id) && !Trick(history_[i - 1].action, history_[i].action)) {
                opp_dealt_hidden.push_back(i - 1);
            }
        }
        //now voids contains the number of hidden cards dealt to opp since it showed a void in that suit, i.e the maximum number of cards held in that suit//
        //if the suit is unvoided, then this number is -1//
        for (int i = 0; i < kNumSuits; ++i) {
            if (when_voided[i] != 0) {
                voids[i] = 0;
                for (int j = 0; j < opp_dealt_hidden.size(); ++j) {
                    if (opp_dealt_hidden[j] >= when_voided[i]) {
                        voids[i] += 1;
                    }
                }
            }
        }
        //we now perform a sequence of shuffles to generate a possible opponent hand, and make no attempt to reconcile the history with this new deal//
        int nec = _mm_popcnt_u64(necessary_cards);
        for (int i = 0; i < kNumSuits; ++i) {
            if (voids[i] != -1&&_mm_popcnt_u64(sufficient_cards&kSuitMasks[i])>voids[i]) {
                uint64_t suit_subset = (sufficient_cards & kSuitMasks[i]);
                std::vector<int> temp;
                while (suit_subset != 0) {
                    temp.push_back(_tzcnt_u64(suit_subset));
                    suit_subset = _blsr_u64(suit_subset);
                }
                std::shuffle(temp.begin(), temp.end(), gen);
                sufficient_cards = (sufficient_cards &~(kSuitMasks[i]));
                for (int j = 0; j < voids[i]; ++j) {
                    sufficient_cards = (sufficient_cards | (uint64_t(1) << temp[j]));
                }
            }
        }
        //finally generating a possible hand for opponent//
        std::vector<int> hand_vec;
        while (sufficient_cards != 0) {
            hand_vec.push_back(_tzcnt_u64(sufficient_cards));
            sufficient_cards = _blsr_u64(sufficient_cards);
        }
        std::shuffle(hand_vec.begin(), hand_vec.end(), gen);
        uint64_t suff_hand = 0;
        uint64_t opp_hand=0;
        for (int i = 0; i < _mm_popcnt_u64(hands_[opp])-nec; ++i) {
            suff_hand = suff_hand | (uint64_t(1) << hand_vec[i]);
        }
        opp_hand = suff_hand | necessary_cards;
        resampled_state->hands_[opp] = opp_hand;
        resampled_state->deck_ = _bzhi_u64(~0, kNumRanks * kNumSuits) ^ (discard_ | opp_hand | hands_[player_id]|recent_faceup_card);
        return resampled_state;
    }
std::string GWhistFState::ObservationString(Player player) const {
    //note this is a lie, this is not the observation state string but it is used for ISMCTS to label nodes//
    std::string p = "p"+std::to_string(player)+",";
    std::string cur_hand="";
    std::string public_info = "";
    uint64_t p_hand = hands_[player];
    std::vector<int> v_hand = {};
    while(p_hand!=0){
        v_hand.push_back(_tzcnt_u64(p_hand));
        p_hand = _blsr_u64(p_hand);
    }
    std::sort(v_hand.begin(),v_hand.end());
    for(int i =0;i<v_hand.size();++i){
        cur_hand=cur_hand+CardString(v_hand[i])+",";
    }
    for(int i =2*kNumRanks;i<history_.size();++i){
        int index =(i-2*kNumRanks)%4;
        if(index!=3){
            public_info=public_info + std::to_string(history_[i].player)+":"+CardString(history_[i].action)+",";
        }
    }
    return p+cur_hand+public_info;
}

std::vector<Action> GWhistFState::LegalActions() const{
    std::vector<Action> actions;
    if (IsTerminal()) return {};
    if (IsChanceNode()) {
        actions.reserve(_mm_popcnt_u64(deck_));
        uint64_t copy_deck = deck_;
        while (copy_deck != 0) {
            actions.push_back(_tzcnt_u64(copy_deck));
            copy_deck = _blsr_u64(copy_deck);
        }
    }
    else {
        //lead//
        actions.reserve(kNumRanks);
        if (history_.back().player == kChancePlayerId) {
            uint64_t copy_hand = hands_[player_];
            while (copy_hand != 0) {
                actions.push_back(_tzcnt_u64(copy_hand));
                copy_hand = _blsr_u64(copy_hand);
            }
        }

        //follow//
        else {
            uint64_t copy_hand = hands_[player_] & kSuitMasks[CardSuit(history_.back().action)];
            if (copy_hand == 0) {
                copy_hand = hands_[player_];
            }
            while (copy_hand != 0) {
                actions.push_back(_tzcnt_u64(copy_hand));
                copy_hand = _blsr_u64(copy_hand);
            }
        }
    }
    return actions;
}

void GWhistFState::DoApplyAction(Action move) {
    //initial deal//
    int player_start = player_;
    if (move_number_ < (kNumSuits * kNumRanks) / 2) {
        hands_[move_number_ % 2] = (hands_[move_number_ % 2] |((uint64_t)1 << move));
        deck_ = (deck_ ^ ((uint64_t)1 << move));
    }
    else if (move_number_ == (kNumSuits * kNumRanks / 2)) {
        trump_ = CardSuit(move);
        deck_ = (deck_ ^ ((uint64_t)1 << move));
        player_ = 0;
    }
    //cardplay//
    else if (move_number_ > (kNumSuits * kNumRanks) / 2) {
        int move_index = (move_number_ - ((kNumSuits * kNumRanks) / 2)) % 4;
        switch (move_index) {
            bool lead_win;
            int winner;
            int loser;
        case 0:
            //revealing face up card//
            deck_ = (deck_ ^ ((uint64_t)1 << move));
            lead_win = Trick(history_[move_number_ - 3].action, history_[move_number_ - 2].action);
            winner = ((lead_win) ^ (history_[move_number_ - 3].player == 0)) ? 1 : 0;
            player_ = winner;
            break;
        case 1:
            //establishing lead//
            discard_ = (discard_|((uint64_t)1<<move));
            hands_[player_] = (hands_[player_] ^ ((uint64_t)1 << move));
            (player_ == 0) ? player_ = 1 : player_ = 0;
            break;
        case 2:
            //following and awarding face up//
            discard_ = (discard_ | ((uint64_t)1 << move));
            hands_[player_] = (hands_[player_] ^ ((uint64_t)1 << move));
            lead_win = Trick(history_[move_number_ - 1].action, move);
            winner = ((lead_win) ^ (history_[move_number_ - 1].player == 0)) ? 1 : 0;
            hands_[winner] = (hands_[winner] | ((uint64_t)1 << history_[move_number_ - 2].action));
            player_ = kChancePlayerId;
            break;
        case 3:
            //awarding face down//
            deck_ = (deck_ ^ ((uint64_t)1 << move));
            lead_win = Trick(history_[move_number_ - 2].action, history_[move_number_ - 1].action);
            loser = ((lead_win) ^ (history_[move_number_ - 2].player == 0)) ? 0 : 1;
            hands_[loser] = (hands_[loser] | ((uint64_t)1 << move));
            if(IsTerminal()){
                player_=kTerminalPlayerId;
            }
            break;
        }
    }
#ifdef DEBUG
    std::cout << ActionToString(player_start, move) << std::endl;
    std::cout << move << std::endl;
#endif
}

}  // namespace german_whist_foregame
}  // namespace open_spiel
