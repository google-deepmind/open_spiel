#ifndef OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
#define OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// The imperfect information part of 2 player whist variant
// https://en.wikipedia.org/wiki/German_Whist

namespace open_spiel {
namespace german_whist_foregame {



class GWhistFGame;
class GWhistFObserver;

inline constexpr int kNumRanks = 13;
inline constexpr int kNumSuits = 4;
inline constexpr char kRankChar[] = "AKQJT98765432";
inline constexpr char kSuitChar[] = "CDHS";

extern std::string kTTablePath;

// Reimplementing bmi2 intrinsics with bit operations that will work on all platforms//
uint32_t tzcnt_u32(uint32_t a);
uint64_t tzcnt_u64(uint64_t a);
uint32_t bzhi_u32(uint32_t a,uint32_t b);
uint64_t bzhi_u64(uint64_t a,uint64_t b);
uint32_t blsr_u32(uint32_t a);
uint64_t blsr_u64(uint64_t a);
uint32_t popcnt_u32(uint32_t a);
uint64_t popcnt_u64(uint64_t a);
uint64_t pext_u64(uint64_t a,uint64_t b);

// containers of cards are 64 bits,with the least significant 52bits being the suits CDHS,with the least sig bit of each suit being the highest rank card//
// this container of masks is used to extract only the cards from a suit//
inline const std::array<uint64_t, 4> kSuitMasks = { bzhi_u64(~0,kNumRanks),bzhi_u64(~0,2 * kNumRanks) ^ bzhi_u64(~0,kNumRanks),bzhi_u64(~0,3 * kNumRanks) ^ bzhi_u64(~0,2 * kNumRanks),bzhi_u64(~0,4 * kNumRanks) ^ bzhi_u64(~0,3 * kNumRanks) };


struct Triple{
    char index;
    char length;
    uint32_t sig;
    bool operator<(const Triple& triple) const;
};
std::vector<uint32_t> GenQuads(int size_endgames);
std::vector<std::vector<uint32_t>> BinCoeffs(uint32_t max_n);
uint32_t HalfColexer(uint32_t cards,const std::vector<std::vector<uint32_t>>* bin_coeffs);
void GenSuitRankingsRel(uint32_t size,std::unordered_map<uint32_t,uint32_t>* Ranks);
class vectorNa{
private:
    std::vector<char> data;
    size_t inner_size;
    size_t outer_size;
public:
    vectorNa(size_t card_combs,size_t suit_splits,char val);
    vectorNa();
    size_t size()const;
    size_t GetInnerSize()const;
    size_t GetOuterSize()const;
    char const& operator[](size_t index)const;
    char GetChar(size_t i,size_t j)const;
    void SetChar(size_t i,size_t j,char value);
    char Get(size_t i,size_t j) const;
    void Set(size_t i,size_t j,char value);
};
vectorNa InitialiseTTable(int size,const std::vector<std::vector<uint32_t>>& bin_coeffs);
vectorNa LoadTTable(const std::string filename,int depth,const std::vector<std::vector<uint32_t>>& bin_coeffs);
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
    int MaxGameLength() const override{return kNumRanks*(kNumSuits+2);};
    int MaxChanceNodesInHistory() const override{return kNumRanks*kNumSuits;};
    vectorNa ttable_;
    std::unordered_map<uint32_t,uint32_t> suit_ranks_;
    std::vector<std::vector<uint32_t>>bin_coeffs_;
private:
    // Number of players.
    int num_players_ = 2;
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
    std::string StateToString() const;
    uint64_t EndgameKey(int player_to_move) const;
protected:
    void DoApplyAction(Action move) override;
private:
    uint64_t deck_;
    uint64_t discard_;
    const vectorNa* ttable_;
    const std::unordered_map<uint32_t,uint32_t>* suit_ranks_;
    const std::vector<std::vector<uint32_t>>* bin_coeffs_;
    std::array<uint64_t,2> hands_;
    int player_;
    int trump_;
    bool Trick(int lead,int follow) const;
};
}// namespace german_whist_foregame
}// namespace open_spiel


#endif OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
