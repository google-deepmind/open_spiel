#ifndef OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
#define OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <x86intrin.h>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

//The imperfect information part of 2 player whist variant
//https://en.wikipedia.org/wiki/German_Whist
//
//

//

namespace open_spiel {
namespace german_whist_foregame {



class GWhistFGame;
class GWhistFObserver;

inline constexpr int kNumRanks = 13;
inline constexpr int kNumSuits = 4;
inline constexpr char kRankChar[] = "AKQJT98765432";
inline constexpr char kSuitChar[] = "CDHS";
inline const std::array<uint64_t, 4> kSuitMasks = { _bzhi_u64(~0,kNumRanks),_bzhi_u64(~0,2 * kNumRanks) ^ _bzhi_u64(~0,kNumRanks),_bzhi_u64(~0,3 * kNumRanks) ^ _bzhi_u64(~0,2 * kNumRanks),_bzhi_u64(~0,4 * kNumRanks) ^ _bzhi_u64(~0,3 * kNumRanks) };
extern std::string kTTablePath ;
struct Triple{
    char index;
    char length;
    uint32_t sig;
    bool operator<(const Triple& triple) const;
};
std::vector<uint32_t> GenQuads(int size_endgames);
std::vector<std::vector<uint32_t>> BinCoeffs(uint32_t max_n);
uint32_t HalfColexer(uint32_t cards,const std::vector<std::vector<uint32_t>>* bin_coeffs);
void GenSuitRankingsRel(uint32_t size, std::unordered_map<uint32_t,uint32_t>* Ranks);
class vectorNa{
private:
    std::vector<char> data;
public:
    vectorNa(size_t num,char val);
    size_t size()const;
    char const& operator[](size_t index)const;
    void SetChar(size_t index,char value);
    char Get(size_t index) const;
    void Set(size_t index,char value);
};
std::vector<vectorNa> InitialiseTTable(int size,std::vector<std::vector<uint32_t>>& bin_coeffs);
std::vector<vectorNa> LoadTTable(const std::string filename,int depth,std::vector<std::vector<uint32_t>>& bin_coeffs);
class GWhistFGame : public Game {
public:
    explicit GWhistFGame(const GameParameters& params);
    int NumDistinctActions() const override { return kNumRanks*kNumSuits; }
    std::unique_ptr<State> NewInitialState() const override;
    int MaxChanceOutcomes() const override { return kNumRanks*kNumSuits ; }
    int NumPlayers() const override { return num_players_; }
    double MinUtility() const override {return -kNumRanks;};
    double MaxUtility() const override {return kNumRanks;};
    absl::optional<double> UtilitySum() const override { return 0; };
    int MaxGameLength() const override{kNumRanks*(kNumSuits+2);};
    int MaxChanceNodesInHistory() const override{return kNumRanks*kNumSuits;};
    std::vector<vectorNa> ttable_;
    std::unordered_map<uint32_t,uint32_t> suit_ranks_;
    std::vector<std::vector<uint32_t>>bin_coeffs_;
private:
    // Number of players.
    int num_players_=2;
};
class GWhistFState : public State {
public:
    explicit GWhistFState(std::shared_ptr<const GWhistFGame> game);
    GWhistFState(const GWhistFState&) = default;
    Player CurrentPlayer() const override;
    std::string ActionToString(Player player, Action move) const override;
    std::string ToString() const override;
    bool IsTerminal() const override;
    std::vector<double> Returns() const override;
    std::unique_ptr<State> Clone() const override;
    ActionsAndProbs ChanceOutcomes() const override;
    std::vector<Action> LegalActions() const override;
    std::string InformationStateString(Player player) const override;
    std::string ObservationString(Player player) const override;
    std::unique_ptr<State> ResampleFromInfostate(int player_id,std::function<double()> rng) const override;
    std::string StateToString() const ;
    uint64_t EndgameKey(int player_to_move) const;
protected:
    void DoApplyAction(Action move) override;
private:
    uint64_t deck_;
    uint64_t discard_;
    const std::vector<vectorNa>* ttable_;
    const std::unordered_map<uint32_t,uint32_t>* suit_ranks_;
    const std::vector<std::vector<uint32_t>>* bin_coeffs_;
    std::array<uint64_t,2> hands_;
    int player_;
    int trump_;
    bool Trick(int lead,int follow) const;
    
    
    // The move history and number of players are sufficient information to
    // specify the state of the game. We keep track of more information to make
    // extracting legal actions and utilities easier.
    // The cost of the additional book-keeping is more complex ApplyAction() and
    // UndoAction() functions
    
};
}//g_whist_foregame
}//open_spiel


#endif OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
