#ifndef OPEN_SPIEL_GAMES_SCHNAPSEN_H_
#define OPEN_SPIEL_GAMES_SCHNAPSEN_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "spiel_globals.h"

namespace open_spiel {
namespace schnapsen {

inline constexpr int kNumPlayers = 2;
inline constexpr int kSuits = 4;
inline constexpr int kValues = 5;
inline constexpr int kCards = kSuits * kValues;
inline constexpr int kHandSize = 5;
inline constexpr int kWinningScore = 66;
inline constexpr int kNoCard = -1;

// TODO: Add history, etc.
inline constexpr int kInformationStateTensorSize = kCards  // Current hand
                                                   + 1;    // Who goes first

class SchnapsenState : public State {
 public:
  SchnapsenState(std::shared_ptr<const Game> game);

  SchnapsenState(const SchnapsenState&) = default;
  SchnapsenState& operator=(const SchnapsenState&) = default;

  bool IsTerminal() const override;
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<double> Returns() const override;

  std::string ToString() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;

  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

  // Required to be overridden.
  std::string ActionToString(Player player, Action action_id) const override;
  std::unique_ptr<State> Clone() const override;

  // Required in tests.
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override {
    return;
  }

 protected:
  void DoApplyAction(Action action_id) override;
  void ApplyChanceAction(Action action);
  void ApplyPlayerAction(Action action);
  Player GetWinner() const;
  std::array<std::array<bool, kCards>, kNumPlayers> hands_;
  std::array<bool, kCards> played_cards_;
  // For convenience, so it does not need to be recomputed every time.
  std::array<bool, kCards> stack_cards_;
  int open_card_;
  int attout_suit_;
  int attout_open_card_ = kNoCard;
  std::array<int, kNumPlayers> scores_;
  Player active_player_;
  Player last_trick_winner_ = kInvalidPlayer;
};

class SchnapsenGame : public Game {
 public:
  explicit SchnapsenGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new SchnapsenState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  // TODO: Update for Zwanziger and Vierziger
  int NumDistinctActions() const override { return kCards; }
  double MinUtility() const override { return -3; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 3; }
  int MaxGameLength() const override { return kCards; }
  int MaxChanceOutcomes() const override { return kCards; }
  int MaxChanceNodesInHistory() const override { return kCards; }

  std::vector<int> InformationStateTensorShape() const override {
    return {kInformationStateTensorSize};
  };

  // Required in tests.
  std::vector<int> ObservationTensorShape() const override { return {0}; }
};

}  // namespace schnapsen
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SCHNAPSEN_H_
