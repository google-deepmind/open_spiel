#include "open_spiel/games/schnapsen.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <string>

#include "absl/strings/str_join.h"
#include "leduc_poker.h"
#include "spiel_globals.h"
#include "spiel_utils.h"

namespace open_spiel {
namespace schnapsen {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"schnapsen",
    /*long_name=*/"Schnapsen",
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
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new SchnapsenGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
}  // namespace

Player GetOtherPlayer(Player player) { return Player{(player + 1) % 2}; }

bool All(const std::array<bool, kCards> &cards) {
  return std::all_of(cards.cbegin(), cards.cend(), [](bool v) { return v; });
}
bool Any(const std::array<bool, kCards> &cards) {
  return std::any_of(cards.cbegin(), cards.cend(), [](bool v) { return v; });
}
int Count(const std::array<bool, kCards> &cards) {
  return std::count(cards.cbegin(), cards.cend(), 1);
}

std::vector<Action> Actions(const std::array<bool, kCards> &cards) {
  std::vector<Action> numbers;
  for (int card = 0; card < kCards; card++) {
    if (cards[card]) {
      numbers.push_back(card);
    }
  }

  return numbers;
}

// Cards are ordered by suit, i.e. J,Q,K,T,A of the first suit, etc.
int GetSuit(int action) {
  return action == kNoCard ? kNoCard : action / kValues;
}
int GetValue(int action) {
  return action == kNoCard ? kNoCard : action % kValues;
}

Action ToAction(int value, int suit) { return suit * kValues + value; }

int GetCardScore(int action) {
  switch (GetValue(action)) {
    case 0:
      return 2;
    case 1:
      return 3;
    case 2:
      return 4;
    case 3:
      return 10;
    case 4:
      return 11;
    default:
      SpielFatalError("Cannot compute score for card value: " +
                      std::to_string(GetValue(action)));
  }
}

std::string ValueString(int value) {
  switch (value) {
    case kNoCard:
      return "X";
    case 0:
      return "J";
    case 1:
      return "Q";
    case 2:
      return "K";
    case 3:
      return "T";
    case 4:
      return "A";
    default:
      SpielFatalError("Unknown card value: " + std::to_string(value));
  }
}

std::string SuitString(int suit) {
  // Arbitrary order, only used for printing.
  switch (suit) {
    case kNoCard:
      return "X";
    case 0:
      return "♠";
    case 1:
      return "♥";
    case 2:
      return "♣";
    case 3:
      return "♦";
    default:
      SpielFatalError("Unknown card suit: " + std::to_string(suit));
  }
}

std::string CardString(int action) {
  return ValueString(GetValue(action)) + SuitString(GetSuit(action));
}

std::string CardsString(std::array<bool, kCards> cards) {
  std::vector<Action> values = Actions(cards);
  std::vector<std::string> stringValues;
  std::transform(values.begin(), values.end(), std::back_inserter(stringValues),
                 CardString);

  return absl::StrJoin(stringValues, "");
}

std::string SchnapsenState::ActionToString(Player player,
                                           Action action_id) const {
  return CardString(action_id);
}

std::string SchnapsenState::ToString() const {
  return ("Open card: " + CardString(open_card_) + "\n") +
         ("Hand player 0: " + CardsString(hands_[0]) + "\n") +
         ("Hand player 1: " + CardsString(hands_[1]) + "\n") +
         ("Attout: " + ValueString(attout_open_card_) +
          SuitString(attout_suit_) + "\n") +
         ("Points player 0: " + std::to_string(scores_[0]) + "\n") +
         ("Points player 1: " + std::to_string(scores_[1]) + "\n") +
         ("Stack size: " + std::to_string(Count(stack_cards_)) + "\n") +
         ("Stack: " + CardsString(stack_cards_) + "\n") +
         ("Played cards " + CardsString(played_cards_) + "\n");
}

std::string SchnapsenState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  return (CardString(open_card_) + "\n") +
         (CardsString(hands_[player]) + "\n") +
         (ValueString(attout_open_card_) + SuitString(attout_suit_) + "\n") +
         (std::to_string(scores_[player]) + "\n") +
         (std::to_string(scores_[GetOtherPlayer(player)]) + "\n") +
         (CardsString(played_cards_) + "\n");
}

// TODO: There is probably a better way to assign vector values to a span.
int Assign(std::array<bool, kCards> to_insert, absl::Span<float> values,
           int i) {
  for (int v : to_insert) {
    values[i] = v;
    i++;
  }
  return i;
}

void SchnapsenState::InformationStateTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  int i = 0;

  // Open card value, if any
  if (open_card_ != kNoCard) {
    values[i + GetValue(open_card_)] = 1;
  }
  i += kValues;

  // Open card suit, if any
  if (open_card_ != kNoCard) {
    values[i + GetSuit(open_card_)] = 1;
  }
  i += kSuits;

  // Current hand
  i = Assign(hands_[player], values, i);

  // Open card attout, if any
  if (attout_open_card_ != kNoCard) {
    values[i + attout_open_card_] = 1;
  }
  i += kValues;

  // Attout suit
  values[i + attout_suit_] = 1;
  i += kSuits;

  // Scores (normalized)
  values[i] = double(scores_[player]) / kWinningScore;
  i++;
  values[i] = double(scores_[GetOtherPlayer(player)]) / kWinningScore;
  i++;

  // Played cards
  i = Assign(played_cards_, values, i);

  SPIEL_CHECK_EQ(i, kInformationStateTensorSize);
}

std::string SchnapsenState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  return "";
}

bool GetWinnerReturn(int loser_score) {
  if (loser_score == 0) {
    return 3;
  } else if (loser_score < kCutOffScore) {
    return 2;
  } else {
    return 1;
  }
}

Player SchnapsenState::GetWinner() const {
  for (int i = 0; i < kNumPlayers; i++) {
    if (scores_[i] > kWinningScore) {
      return Player{i};
    }
  }
  if (All(played_cards_)) {
    return last_trick_winner_;
  }

  return kInvalidPlayer;
}

bool SchnapsenState::IsTerminal() const {
  return GetWinner() != kInvalidPlayer;
}

std::vector<double> SchnapsenState::Returns() const {
  int winner = GetWinner();
  if (winner == kInvalidPlayer) {
    return {0, 0};
  }

  int winner_return = GetWinnerReturn(scores_[GetOtherPlayer(winner)]);
  std::vector<double> returns(kNumPlayers);
  returns[winner] = winner_return;
  returns[GetOtherPlayer(winner)] = -winner_return;
  return returns;
}

Player GetTrickWinner(int first_action, int second_action, Player first_player,
                      int attout_suit) {
  if (GetSuit(first_action) != GetSuit(second_action)) {
    return GetSuit(second_action) != attout_suit ? first_player
                                                 : GetOtherPlayer(first_player);
  }

  return GetValue(first_action) > GetValue(second_action)
             ? first_player
             : GetOtherPlayer(first_player);
}

// TODO: Add "zudrehen", "zwanziger", "vierziger"
bool SchnapsenState::CanDrawCard() const {
  return attout_open_card_ != kNoCard;
}

void SchnapsenState::DealAttout(Action action) {
  attout_open_card_ = GetValue(action);
  attout_suit_ = GetSuit(action);
  stack_cards_[action] = 0;
}

void SchnapsenState::DrawCard(Action action, Player player) {
  SPIEL_CHECK_FALSE(hands_[player][action]);
  hands_[player][action] = 1;

  if (action == ToAction(attout_open_card_, attout_suit_)) {
    attout_open_card_ = kNoCard;
  } else {
    stack_cards_[action] = 0;
  }
}

void SchnapsenState::PlayCard(Action action, Player player) {
  SPIEL_CHECK_TRUE(hands_[player][action]);
  SPIEL_CHECK_FALSE(played_cards_[action]);

  hands_[player][action] = false;
  played_cards_[action] = true;

  if (open_card_ == kNoCard) {
    open_card_ = action;
    next_actions_.emplace(
        PlayerActionType{GetOtherPlayer(player), ActionType::kPlay});

  } else {
    Player winner = GetTrickWinner(open_card_, action, GetOtherPlayer(player),
                                   attout_suit_);
    scores_[winner] += GetCardScore(open_card_) + GetCardScore(action);
    open_card_ = kNoCard;
    last_trick_winner_ = winner;

    if (CanDrawCard()) {
      next_actions_.push(PlayerActionType{winner, ActionType::kDraw});
      next_actions_.push(
          PlayerActionType{GetOtherPlayer(winner), ActionType::kDraw});
    }
    next_actions_.push(PlayerActionType{winner, ActionType::kPlay});
  }
}

void SchnapsenState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(next_actions_.empty());
  PlayerActionType player_action_type = next_actions_.front();
  next_actions_.pop();

  switch (player_action_type.action_type) {
    case ActionType::kDealAttout:
      DealAttout(action);
      break;
    case ActionType::kDraw:
      DrawCard(action, player_action_type.player);
      break;
    case ActionType::kPlay:
      PlayCard(action, player_action_type.player);
      break;
    default:
      SpielFatalError("Unknown action type");
  }
}

std::vector<Action> SchnapsenState::LegalActions() const {
  if (IsTerminal()) return {};

  SPIEL_CHECK_FALSE(next_actions_.empty());
  PlayerActionType player_action_type = next_actions_.front();

  if (player_action_type.action_type == ActionType::kDealAttout ||
      player_action_type.action_type == ActionType::kDraw) {
    std::vector<Action> actions = Actions(stack_cards_);

    if (actions.size() == 0) {
      SPIEL_CHECK_NE(attout_open_card_, kNoCard);
      actions.emplace_back(ToAction(attout_open_card_, attout_suit_));
    }
    return actions;
  } else if (attout_open_card_ != kNoCard || open_card_ == kNoCard) {
    return Actions(hands_[player_action_type.player]);
  }
  // Farbzwang, Stichzwang
  else {
    int open_suit = GetSuit(open_card_);
    int open_value = GetValue(open_value);
    std::vector<Action> cards = Actions(hands_[player_action_type.player]);
    std::vector<Action> actions;

    // Larger card in the same suit
    std::copy_if(cards.cbegin(), cards.cend(), std::back_inserter(actions),
                 [open_suit, open_value](int card) {
                   return GetSuit(card) == open_suit &&
                          GetValue(card) > open_value;
                 });
    if (actions.size() > 0) {
      return actions;
    }

    // Smaller card in the same suit
    std::copy_if(cards.cbegin(), cards.cend(), std::back_inserter(actions),
                 [open_suit](int card) { return GetSuit(card) == open_suit; });
    if (actions.size() > 0) {
      return actions;
    }

    // Attout
    std::copy_if(cards.cbegin(), cards.cend(), std::back_inserter(actions),
                 [this](int card) { return GetSuit(card) == attout_suit_; });
    if (actions.size() > 0) {
      return actions;
    }

    // Other cards
    return cards;
  }
}

Player SchnapsenState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }

  PlayerActionType player_action_type = next_actions_.front();

  if (player_action_type.action_type == ActionType::kDealAttout ||
      player_action_type.action_type == ActionType::kDraw) {
    return kChancePlayerId;
  } else {
    return player_action_type.player;
  }
}

std::vector<std::pair<Action, double>> SchnapsenState::ChanceOutcomes() const {
  std::vector<Action> legal_actions = LegalActions();
  std::vector<std::pair<Action, double>> chance_outcomes;

  for (Action action : legal_actions) {
    chance_outcomes.push_back({action, 1.0 / double(legal_actions.size())});
  }

  return chance_outcomes;
};

std::unique_ptr<State> SchnapsenState::Clone() const {
  return std::unique_ptr<State>(new SchnapsenState(*this));
}

SchnapsenState::SchnapsenState(std::shared_ptr<const Game> game) : State(game) {
  for (auto &hand : hands_) {
    std::fill(hand.begin(), hand.end(), 0);
  }
  std::fill(played_cards_.begin(), played_cards_.end(), 0);
  std::fill(stack_cards_.begin(), stack_cards_.end(), 1);

  next_actions_.emplace(
      PlayerActionType{kInvalidPlayer, ActionType::kDealAttout});
  for (int i = 0; i < kHandSize; i++) {
    next_actions_.emplace(
        PlayerActionType{kDefaultPlayerId, ActionType::kDraw});
    next_actions_.emplace(
        PlayerActionType{GetOtherPlayer(kDefaultPlayerId), ActionType::kDraw});
  }
  next_actions_.emplace(PlayerActionType{kDefaultPlayerId, ActionType::kPlay});
}

SchnapsenGame::SchnapsenGame(const GameParameters &params)
    : Game(kGameType, params) {}

}  // namespace schnapsen
}  // namespace open_spiel
