#ifndef OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
#define OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

//The imperfect information part of 2 player whist variant
//https://en.wikipedia.org/wiki/German_Whist
//
//

//
// Parameters:
//     kNumSuits, kNumRanks

namespace open_spiel {
namespace german_whist_foregame {


enum ActionType { kPass = 0, kBet = 1 };

class GermanWhistForegameGame;
class GermanWhistForegameƒObserver;

class GermanWhistForegameState : public State {
public:
    explicit GermanWhistForegameState(std::shared_ptr<const Game> game);
    GermanWhistForegameState(const GermanWhistForegameState&) = default;
    
    Player CurrentPlayer() const override;
    
    std::string ActionToString(Player player, Action move) const override;
    std::string ToString() const override;
    bool IsTerminal() const override;
    std::vector<double> Returns() const override;
    std::string InformationStateString(Player player) const override;
    std::string ObservationString(Player player) const override;
    void InformationStateTensor(Player player,
                                absl::Span<float> values) const override;
    void ObservationTensor(Player player,
                           absl::Span<float> values) const override;
    std::unique_ptr<State> Clone() const override;
    void UndoAction(Player player, Action move) override;
    std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
    std::vector<Action> LegalActions() const override;
    std::vector<int> hand() const { return {card_dealt_[CurrentPlayer()]}; }
    std::unique_ptr<State> ResampleFromInfostate(
                                                 int player_id, std::function<double()> rng) const override;
    
    const std::vector<int>& CardDealt() const { return card_dealt_; }
    
protected:
    void DoApplyAction(Action move) override;
    
private:
    friend class GermanWhistForegameObserver;
    
    // Whether the specified player made a bet
    bool DidBet(Player player) const;
    
    // The move history and number of players are sufficient information to
    // specify the state of the game. We keep track of more information to make
    // extracting legal actions and utilities easier.
    // The cost of the additional book-keeping is more complex ApplyAction() and
    // UndoAction() functions.
    int first_bettor_;             // the player (if any) who was first to bet
    std::vector<int> card_dealt_;  // the player (if any) who has each card
    int winner_;                   // winning player, or kInvalidPlayer if the
    // game isn't over yet.
    int pot_;                      // the size of the pot
    // How much each player has contributed to the pot, indexed by pid.
    std::vector<int> ante_;
};

class GermanWhistForegameGame : public Game {
public:
    explicit GermanWhistForegameGame(const GameParameters& params);
    int NumDistinctActions() const override { return 2; }
    std::unique_ptr<State> NewInitialState() const override;
    int MaxChanceOutcomes() const override { return num_players_ + 1; }
    int NumPlayers() const override { return num_players_; }
    double MinUtility() const override;
    double MaxUtility() const override;
    absl::optional<double> UtilitySum() const override { return 0; }
    std::vector<int> InformationStateTensorShape() const override;
    std::vector<int> ObservationTensorShape() const override;
    int MaxGameLength() const override { return num_players_ * 2 - 1; }
    int MaxChanceNodesInHistory() const override { return num_players_; }
    std::shared_ptr<Observer> MakeObserver(
                                           absl::optional<IIGObservationType> iig_obs_type,
                                           const GameParameters& params) const override;
    
    // Used to implement the old observation API.
    std::shared_ptr<GermanWhistForegameObserver> default_observer_;
    std::shared_ptr<GermanWhistForegameObserver> info_state_observer_;
    std::shared_ptr<GermanWhistForegameObserver> public_observer_;
    std::shared_ptr<GermanWhistForegameObserver> private_observer_;
    
private:
    // Number of players.
    int num_players_;
};

#endif OPEN_SPIEL_GAMES_GERMAN_WHIST_FOREGAME_H
