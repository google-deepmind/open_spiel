// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This game is a simplified version of the Sheriff of Nottingham board
// game, as introduced in [1].
//
// Game dynamics
// =============
//
// Player 1 (the "smuggler") selects the number of `num_items` (0 or more)
// illegal items to be placed in the cargo. The selected number is unknown to
// Player 2 (the "sheriff").
//
// Then, the game proceeds for `num_rounds` bargaining rounds. In each round,
// the following happens:
//
// - The smuggler selects an integer bribe amount, in the range 0 to `max_bribe`
//   (inclusive). The selected amount is public information. However, the
//   smuggler does *not* actually give money to the sheriff, unless this is the
//   final round.
// - Then, the sheriff tells the smuggler whether he is planning to inspect the
//   cargo. However, no cargo is actually inspected other than in the final
//   round. The sheriff can change his mind in later rounds, except for the
//   final round.
//
// Payoff computation
// ------------------
//
// At the end of the game, the payoffs of the players are computed as follows:
//
// - If the sheriff did *not* inspect the cargo, the smuggler gains a payoff
//   equal to `num_items * item_value - bribe_amount`, and the sheriff gets a
//   payoff equal to `bribe_amount`, where `bribe_amount` is the *last* bribe
//   amount.
// - If the sheriff inspects the cargo, and no illegal items were present, the
//   smuggler gains a payoff equal to `sheriff_penalty`, while the sheriff loses
//   `sheriff_penalty` value.
// - Finally, if the sheriff inspects the cargo and finds `num_items` (1 or
//   more) illegal items, the smuggler loses a total value computed as
//   `-num_item * item_penalty`, while the sheriff gains value `num_items *
//   item_penalty`.
//
//
// Game size
// ---------
//
// +-------+-------+--------+-----------------+----------------+----------+
// |  Max  |  Max  |   Num  |  Num sequences  |  Num infosets  | Terminal |
// | bribe | items | rounds |   pl 0 |   pl 1 |  pl 0 |   pl 1 |  states  |
// +-------+-------+--------+--------+--------+-------+--------+----------+
// |     3 |     3 |      1 |     21 |      0 |     5 |      4 |       32 |
// |     3 |     5 |      2 |    223 |     73 |    55 |     36 |      384 |
// |     3 |     3 |      3 |   1173 |    585 |   293 |    292 |     2048 |
// |     3 |     5 |      4 |  14047 |   4681 |  3511 |   2340 |    24576 |
// +-------+-------+--------+--------+--------+-------+--------+----------+
// |     5 |     3 |      1 |     29 |     13 |     5 |      6 |       48 |
// |     5 |     3 |      2 |    317 |    157 |    53 |     78 |      576 |
// |     5 |     5 |      3 |   5659 |   1885 |   943 |    942 |    10368 |
// +-------+-------+--------+--------+--------+-------+--------+----------+
//
//
//
// Game parameters
// ===============
//
//     "item_penalty"    double     Penalty (per item) incurred by the smuggler
//                                  for carrying illegal goods  (default = 2.0)
//     "item_value"      double     Value of each successfully smuggled item
//                                                              (default = 1.0)
//     "sheriff_penalty" double     Sheriff's penalty for inspecting a cargo
//                                  that does not contain illegal items
//                                                              (default = 3.0)
//     "max_bribe"          int     Maximum bribe amount, per round
//                                                              (default = 3)
//     "max_items"          int     Maximum numbers of items that fit the cargo
//                                                              (default = 3)
//     "num_rounds"         int     Number of bargaining rounds (default = 4)
//
// References
// ==========
//
// If you want to reference the paper that introduced the benchmark game, here
// is a Bibtex citation:
//
// ```
// @inproceedings{Farina19:Correlation,
//    title=    {Correlation in Extensive-Form Games: Saddle-Point Formulation
//               and Benchmarks},
//    author=   {Farina, Gabriele and Ling, Chun Kai and Fang, Fei and
//               Sandholm, Tuomas},
//    booktitle={Conference on Neural Information Processing Systems
//               (NeurIPS)},
//    year={2019}
// }
// ```
//
// [1]:
// https://papers.nips.cc/paper/9122-correlation-in-extensive-form-games-saddle-point-formulation-and-benchmarks.pdf

#ifndef OPEN_SPIEL_GAMES_SHERIFF_H_
#define OPEN_SPIEL_GAMES_SHERIFF_H_

#include <memory>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace sheriff {

inline constexpr double kDefaultItemPenalty = 2.0;
inline constexpr double kDefaultItemValue = 1.0;
inline constexpr double kDefaultSheriffPenalty = 3.0;
inline constexpr int kDefaultMaxBribe = 3;
inline constexpr int kDefaultMaxItems = 3;
inline constexpr int kDefaultNumRounds = 4;

class SheriffGame final : public Game {
 public:
  explicit SheriffGame(const GameParameters& params);

  // Virtual functions inherited by OpenSpiel's `Game` interface
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return 2; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override;
  int MaxGameLength() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::vector<int> InformationStateTensorShape() const override;

  // Information about the specific variant being played.
  uint32_t num_rounds() const { return conf.num_rounds; }
  uint32_t max_items() const { return conf.max_items; }
  uint32_t max_bribe() const { return conf.max_bribe; }

  // Action (de)serialization routines
  // =================================
  //
  // The inspection feedback for the sheriff player is serialized to action ids
  // 0 (= will not inspect) and 1 (= will inspect). All other actions belong to
  // the smuggler player. Actions [2, 2 + num_items] correspond to placements
  // of illegal items in the cargo (action id 2 means "0 illegal items placed
  // in the cargo").Actions [3 + num_items, 4 + num_items + num_bribes]
  // correspond to bribing actions (action 3 + num_items means that a bribe of
  // 0 is selected.

  Action SerializeItemPlacementAction(const uint32_t num_illegal_items) const;
  Action SerializeBribe(const uint32_t bribe) const;
  Action SerializeInspectionFeedback(const bool feedback) const;

  uint32_t DeserializeItemPlacementAction(const Action action_id) const;
  uint32_t DeserializeBribe(const Action action_id) const;
  bool DeserializeInspectionFeedback(const Action action_id) const;

  // Members
  // =======

  struct SheriffGameConfiguration {
    double item_penalty;
    double item_value;
    double sheriff_penalty;

    uint32_t max_items;
    uint32_t max_bribe;
    uint32_t num_rounds;
  } conf;

 private:
  std::shared_ptr<const SheriffGame> sheriff_game_;
};

class SheriffState final : public State {
 public:
  explicit SheriffState(const std::shared_ptr<const SheriffGame> sheriff_game);
  ~SheriffState() = default;

  // Virtual functions inherited by OpenSpiel's `State` interface
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  std::string InformationStateString(Player player) const override;
  void UndoAction(Player player, Action action_id) override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  absl::optional<uint32_t> num_illegal_items_;
  std::vector<uint32_t> bribes_;
  std::vector<bool> inspection_feedback_;

  std::shared_ptr<const SheriffGame> sheriff_game_;
};

}  // namespace sheriff
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_SHERIFF_H_
