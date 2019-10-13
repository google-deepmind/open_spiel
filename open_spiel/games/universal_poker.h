#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace universal_poker {

// Default parameters.

// TODO(b/127425075): Use std::optional instead of sentinel values once absl is
// added as a dependency.
inline constexpr int kInvalidCard = -10000;
inline constexpr int kDefaultPlayers = 2;
inline constexpr int kNumSuits = 2;
inline constexpr int kFirstRaiseAmount = 2;
inline constexpr int kSecondRaiseAmount = 4;
inline constexpr int kTotalRaisesPerRound = 2;
inline constexpr int kMaxRaises = 2;
inline constexpr int kStartingMoney = 100;
inline constexpr int kNumInfoStates =
    936;  // Number of info state in the 2P game.

class UniversalPokerGame;

enum ActionType { kFold = 0, kCall = 1, kRaise = 2 };

class UniversalPokerState : public State {
 public:
  explicit UniversalPokerState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationState(Player player) const override;
  std::string Observation(Player player) const override;
  void InformationStateAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
};

class UniversalPokerGame : public Game {
 public:
  explicit UniversalPokerGame(const GameParameters& params);

  int NumDistinctActions() const override { return 3; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
  }
  std::vector<int> InformationStateNormalizedVectorShape() const override;
  std::vector<int> ObservationNormalizedVectorShape() const override;
  int MaxGameLength() const override;

};

}  // namespace universal_poker
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
