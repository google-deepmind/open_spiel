#include "open_spiel/games/german_whist_foregame/german_whist_foregame.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace german_whist_foregame {
namespace {

// Default parameters.


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
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kDefaultPlayers)}},
                         /*default_loadable=*/true,
                         /*provides_factored_observation_string=*/true,
                        };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new GermanWhistForegameGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);
}  // namespace

class GermanWhistForegameObserver : public Observer {
 public:
  GermanWhistForegameObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
  }

  std::string StringFrom(const State& observed_state,
                         int player) const override {
  }

 private:
  IIGObservationType iig_obs_type_;
};

GermanWhistForegameState::GermanWhistForegameState(std::shared_ptr<const Game> game)
    : State(game),
      first_bettor_(kInvalidPlayer),
      card_dealt_(game->NumPlayers() + 1, kInvalidPlayer),
      winner_(kInvalidPlayer),
      pot_(kAnte * game->NumPlayers()),
      // How much each player has contributed to the pot, indexed by pid.
      ante_(game->NumPlayers(), kAnte) {}

int GermanWhistForegameState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return (history_.size() < num_players_) ? kChancePlayerId
                                            : history_.size() % num_players_;
  }
}

void GermanWhistForegameState::DoApplyAction(Action move) {
  // Additional book-keeping
  if (history_.size() < num_players_) {
    // Give card `move` to player `history_.size()` (CurrentPlayer will return
    // kChancePlayerId, so we use that instead).
    card_dealt_[move] = history_.size();
  } else if (move == ActionType::kBet) {
    if (first_bettor_ == kInvalidPlayer) first_bettor_ = CurrentPlayer();
    pot_ += 1;
    ante_[CurrentPlayer()] += kAnte;
  }

  // We undo that before exiting the method.
  // This is used in `DidBet`.
  history_.push_back({CurrentPlayer(), move});

  // Check for the game being over.
  const int num_actions = history_.size() - num_players_;
  if (first_bettor_ == kInvalidPlayer && num_actions == num_players_) {
    // Nobody bet; the winner is the person with the highest card dealt,
    // which is either the highest or the next-highest card.
    // Losers lose 1, winner wins 1 * (num_players - 1)
    winner_ = card_dealt_[num_players_];
    if (winner_ == kInvalidPlayer) winner_ = card_dealt_[num_players_ - 1];
  } else if (first_bettor_ != kInvalidPlayer &&
             num_actions == num_players_ + first_bettor_) {
    // There was betting; so the winner is the person with the highest card
    // who stayed in the hand.
    // Check players in turn starting with the highest card.
    for (int card = num_players_; card >= 0; --card) {
      const Player player = card_dealt_[card];
      if (player != kInvalidPlayer && DidBet(player)) {
        winner_ = player;
        break;
      }
    }
    SPIEL_CHECK_NE(winner_, kInvalidPlayer);
  }
  history_.pop_back();
}

std::vector<Action> GermanWhistForegameState::LegalActions() const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) {
    std::vector<Action> actions;
    for (int card = 0; card < card_dealt_.size(); ++card) {
      if (card_dealt_[card] == kInvalidPlayer) actions.push_back(card);
    }
    return actions;
  } else {
    return {ActionType::kPass, ActionType::kBet};
  }
}

std::string GermanWhistForegameState::ActionToString(Player player, Action move) const {
  if (player == kChancePlayerId)
    return absl::StrCat("Deal:", move);
  else if (move == ActionType::kPass)
    return "Pass";
  else
    return "Bet";
}

std::string GermanWhistForegameState::ToString() const {
  // The deal: space separated card per player
  std::string str;
  for (int i = 0; i < history_.size() && i < num_players_; ++i) {
    if (!str.empty()) str.push_back(' ');
    absl::StrAppend(&str, history_[i].action);
  }

  // The betting history: p for Pass, b for Bet
  if (history_.size() > num_players_) str.push_back(' ');
  for (int i = num_players_; i < history_.size(); ++i) {
    str.push_back(history_[i].action ? 'b' : 'p');
  }

  return str;
}

bool GermanWhistForegameState::IsTerminal() const { return winner_ != kInvalidPlayer; }

std::vector<double> GermanWhistForegameState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (auto player = Player{0}; player < num_players_; ++player) {
    const int bet = DidBet(player) ? 2 : 1;
    returns[player] = (player == winner_) ? (pot_ - bet) : -bet;
  }
  return returns;
}

std::string GermanWhistForegameState::InformationStateString(Player player) const {
  const GermanWhistForegameGame& game = open_spiel::down_cast<const GermanWhistForegameGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

std::string GermanWhistForegameState::ObservationString(Player player) const {
  const GermanWhistForegameGame& game = open_spiel::down_cast<const GermanWhistForegameGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void GermanWhistForegameState::InformationStateTensor(Player player,
                                       absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const GermanWhistForegameGame& game = open_spiel::down_cast<const GermanWhistForegameGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void GermanWhistForegameState::ObservationTensor(Player player,
                                  absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const GermanWhistForegameGame& game = open_spiel::down_cast<const GermanWhistForegameGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> GermanWhistForegameState::Clone() const {
  return std::unique_ptr<State>(new GermanWhistForegameState(*this));
}

void GermanWhistForegameState::UndoAction(Player player, Action move) {
  if (history_.size() <= num_players_) {
    // Undoing a deal move.
    card_dealt_[move] = kInvalidPlayer;
  } else {
    // Undoing a bet / pass.
    if (move == ActionType::kBet) {
      pot_ -= 1;
      if (player == first_bettor_) first_bettor_ = kInvalidPlayer;
    }
    winner_ = kInvalidPlayer;
  }
  history_.pop_back();
  --move_number_;
}

std::vector<std::pair<Action, double>> GermanWhistForegameState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;
  const double p = 1.0 / (num_players_ + 1 - history_.size());
  for (int card = 0; card < card_dealt_.size(); ++card) {
    if (card_dealt_[card] == kInvalidPlayer) outcomes.push_back({card, p});
  }
  return outcomes;
}

bool GermanWhistForegameState::DidBet(Player player) const {
  if (first_bettor_ == kInvalidPlayer) {
    return false;
  } else if (player == first_bettor_) {
    return true;
  } else if (player > first_bettor_) {
    return history_[num_players_ + player].action == ActionType::kBet;
  } else {
    return history_[num_players_ * 2 + player].action == ActionType::kBet;
  }
}

std::unique_ptr<State> GermanWhistForegameState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> state = game_->NewInitialState();
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < game_->NumPlayers(); ++p) {
    if (p == history_.size()) return state;
    if (p == player_id) {
      state->ApplyAction(player_chance);
    } else {
      Action other_chance = player_chance;
      while (other_chance == player_chance) {
        other_chance = SampleAction(state->ChanceOutcomes(), rng()).first;
      }
      state->ApplyAction(other_chance);
    }
  }
  SPIEL_CHECK_GE(state->CurrentPlayer(), 0);
  if (game_->NumPlayers() == history_.size()) return state;
  for (int i = game_->NumPlayers(); i < history_.size(); ++i) {
    state->ApplyAction(history_.at(i).action);
  }
  return state;
}

GermanWhistForegameGame::GermanWhistForegameGame(const GameParameters& params)
    : Game(kGameType, params), num_players_(ParameterValue<int>("players")) {
  SPIEL_CHECK_GE(num_players_, kGameType.min_num_players);
  SPIEL_CHECK_LE(num_players_, kGameType.max_num_players);
  default_observer_ = std::make_shared<GermanWhistForegameObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<GermanWhistForegameObserver>(kInfoStateObsType);
  private_observer_ = std::make_shared<GermanWhistForegameObserver>(
      IIGObservationType{/*public_info*/false,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kSinglePlayer});
  public_observer_ = std::make_shared<GermanWhistForegameObserver>(
      IIGObservationType{/*public_info*/true,
                         /*perfect_recall*/false,
                         /*private_info*/PrivateInfoType::kNone});
}

std::unique_ptr<State> GermanWhistForegameGame::NewInitialState() const {
  return std::unique_ptr<State>(new GermanWhistForegameState(shared_from_this()));
}

std::vector<int> GermanWhistForegameGame::InformationStateTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by 2 (n - 1 + n) bits for betting sequence (longest sequence:
  // everyone except one player can pass and then everyone can bet/pass).
  // n + n + 1 + 2 (n-1 + n) = 6n - 1.
  return {6 * num_players_ - 1};
}

std::vector<int> GermanWhistForegameGame::ObservationTensorShape() const {
  // One-hot for whose turn it is.
  // One-hot encoding for the single private card. (n+1 cards = n+1 bits)
  // Followed by the contribution of each player to the pot (n).
  // n + n + 1 + n = 3n + 1.
  return {3 * num_players_ + 1};
}

double GermanWhistForegameGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // Everyone puts a chip in at the start, and then they each have one more
  // chip. Most that a player can gain is (#opponents)*2.
  return (num_players_ - 1) * 2;
}

double GermanWhistForegameGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end
  // of the game minus then money the player had before starting the game.
  // In GermanWhistForegame, the most any one player can lose is the single chip they paid
  // to play and the single chip they paid to raise/call.
  return -2;
}

std::shared_ptr<Observer> GermanWhistForegameGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<GermanWhistForegameObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

TabularPolicy GetAlwaysPassPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<GermanWhistForegameGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kPass});
}

TabularPolicy GetAlwaysBetPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<GermanWhistForegameGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kBet});
}

TabularPolicy GetOptimalPolicy(double alpha) {
  SPIEL_CHECK_GE(alpha, 0.);
  SPIEL_CHECK_LE(alpha, 1. / 3);
  const double three_alpha = 3 * alpha;
  std::unordered_map<std::string, ActionsAndProbs> policy;

  // All infostates have two actions: Pass (0) and Bet (1).
  // Player 0
  policy["0"] = {{0, 1 - alpha}, {1, alpha}};
  policy["0pb"] = {{0, 1}, {1, 0}};
  policy["1"] = {{0, 1}, {1, 0}};
  policy["1pb"] = {{0, 2. / 3. - alpha}, {1, 1. / 3. + alpha}};
  policy["2"] = {{0, 1 - three_alpha}, {1, three_alpha}};
  policy["2pb"] = {{0, 0}, {1, 1}};

  // Player 1
  policy["0p"] = {{0, 2. / 3.}, {1, 1. / 3.}};
  policy["0b"] = {{0, 1}, {1, 0}};
  policy["1p"] = {{0, 1}, {1, 0}};
  policy["1b"] = {{0, 2. / 3.}, {1, 1. / 3.}};
  policy["2p"] = {{0, 0}, {1, 1}};
  policy["2b"] = {{0, 0}, {1, 1}};
  return TabularPolicy(policy);
}

}  // namespace GermanWhistForegame_poker
}  // namespace open_spiel
