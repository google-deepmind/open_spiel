#ifndef OPEN_SPIEL_GAMES_SOLITAIRE_H
#define OPEN_SPIEL_GAMES_SOLITAIRE_H

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <map>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/solitaire/solitaire_common.h"
#include "open_spiel/spiel.h"
#include <open_spiel/spiel_utils.h>

// An implementation of klondike solitaire:
// https://en.wikipedia.org/wiki/Klondike_(solitaire) More specifically, it is
// K+ solitaire, which allows the player to play any card from the deck/waste
// that would normally become playable after some number of draws in standard
// klondike solitaire. For a more in-depth description of K+ solitaire, see
// http://web.engr.oregonstate.edu/~afern/papers/solitaire.pdf. This
// implementation also gives rewards at intermediate states like most electronic
// versions of solitaire do, rather than only at terminal states.

namespace open_spiel::solitaire {

class SolitaireGame : public Game {
 public:
  // Constructor
  explicit SolitaireGame(const GameParameters &params);

  // Overridden Methods
  int NumDistinctActions() const override;
  int MaxGameLength() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;

  std::vector<int> ObservationTensorShape() const override;
  std::unique_ptr<State> NewInitialState() const override;
  std::shared_ptr<const Game> Clone() const override;

 private:
  int num_players_;
  int depth_limit_;
  bool is_colored_;
};

class SolitaireState : public State {
 public:
  // Constructors
  explicit SolitaireState(std::shared_ptr<const Game> game);

  // Overridden Methods
  Player CurrentPlayer() const override;
  std::unique_ptr<State> Clone() const override;
  bool IsTerminal() const override;
  bool IsChanceNode() const override;
  std::string ToString() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         std::vector<double> *values) const override;
  void DoApplyAction(Action move) override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  // Other Methods
  std::vector<Card> Targets(
      const absl::optional<LocationType> &location = LocationType::kMissing) const;
  std::vector<Card> Sources(
      const absl::optional<LocationType> &location = LocationType::kMissing) const;
  std::vector<Move> CandidateMoves() const;
  Pile *GetPile(const Card &card) const;
  void MoveCards(const Move &move);
  bool IsReversible(const Card &source, Pile *source_pile) const;

 private:
  Waste waste;
  std::vector<Foundation> foundations;
  std::vector<Tableau> tableaus;
  std::vector<Action> revealed_cards;

  bool is_finished = false;
  bool is_reversible = false;
  int current_depth = 0;

  std::set<std::size_t> previous_states = {};
  std::map<Card, PileID> card_map;

  double current_returns = 0.0;
  double current_rewards = 0.0;

  // Parameters
  int depth_limit = kDefaultDepthLimit;
  bool is_colored = kDefaultIsColored;
};

}  // namespace open_spiel::solitaire

#endif  // OPEN_SPIEL_GAMES_SOLITAIRE_H