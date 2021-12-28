// Copyright 2021 DeepMind Technologies Limited
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

#include "open_spiel/python/pybind11/python_games.h"

#include <memory>

// Interface code for using Python Games and States from C++.

#include "open_spiel/abseil-cpp/absl/strings/escaping.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/string_view.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

namespace py = ::pybind11;

PyGame::PyGame(GameType game_type, GameInfo game_info,
               GameParameters game_parameters)
    : Game(game_type, game_parameters), info_(game_info) {}

std::unique_ptr<State> PyGame::NewInitialState() const {
  PYBIND11_OVERLOAD_PURE_NAME(std::unique_ptr<State>, Game, "new_initial_state",
                              NewInitialState);
}

std::unique_ptr<State> PyGame::NewInitialStateForPopulation(
    int population) const {
  PYBIND11_OVERLOAD_PURE_NAME(std::unique_ptr<State>, Game,
                              "new_initial_state_for_population",
                              NewInitialStateForPopulation, population);
}

int PyGame::MaxChanceNodesInHistory() const {
  PYBIND11_OVERLOAD_PURE_NAME(int, Game,
                              "max_chance_nodes_in_history",
                              MaxChanceNodesInHistory);
}

const Observer& PyGame::default_observer() const {
  if (!default_observer_) default_observer_ = MakeObserver(kDefaultObsType, {});
  return *default_observer_;
}

const Observer& PyGame::info_state_observer() const {
  if (!info_state_observer_)
    info_state_observer_ = MakeObserver(kInfoStateObsType, {});
  return *info_state_observer_;
}

PyState::PyState(std::shared_ptr<const Game> game) : State(game) {}

Player PyState::CurrentPlayer() const {
  PYBIND11_OVERLOAD_PURE_NAME(Player, State, "current_player", CurrentPlayer);
}

std::vector<Action> PyState::LegalActions() const {
  return LegalActions(CurrentPlayer());
}

std::vector<Action> PyState::LegalActions(Player player) const {
  if (IsTerminal()) return {};
  if (IsChanceNode()) return LegalChanceOutcomes();
  if ((player == CurrentPlayer()) || (player >= 0 && IsSimultaneousNode())) {
    PYBIND11_OVERLOAD_PURE_NAME(std::vector<Action>, State, "_legal_actions",
                                LegalActions, player);
  } else if (player < 0) {
    SpielFatalError(
        absl::StrCat("Called LegalActions for psuedo-player ", player));
  } else {
    return {};
  }
}

std::string PyState::ActionToString(Player player, Action action_id) const {
  PYBIND11_OVERLOAD_PURE_NAME(std::string, State, "_action_to_string",
                              ActionToString, player, action_id);
}

std::string PyState::ToString() const {
  PYBIND11_OVERLOAD_PURE_NAME(std::string, State, "__str__", ToString);
}

bool PyState::IsTerminal() const {
  PYBIND11_OVERLOAD_PURE_NAME(bool, State, "is_terminal", IsTerminal);
}

std::vector<double> PyState::Returns() const {
  PYBIND11_OVERRIDE_PURE_NAME(std::vector<double>, State, "returns", Returns);
}

std::vector<double> PyState::Rewards() const {
  PYBIND11_OVERRIDE_NAME(std::vector<double>, State, "rewards", Rewards);
}

void PyState::DoApplyAction(Action action_id) {
  PYBIND11_OVERLOAD_PURE_NAME(void, State, "_apply_action", DoApplyAction,
                              action_id);
}

void PyState::DoApplyActions(const std::vector<Action>& actions) {
  PYBIND11_OVERLOAD_PURE_NAME(void, State, "_apply_actions", DoApplyActions,
                              actions);
}

ActionsAndProbs PyState::ChanceOutcomes() const {
  PYBIND11_OVERLOAD_PURE_NAME(ActionsAndProbs, State, "chance_outcomes",
                              ChanceOutcomes);
}

std::unique_ptr<State> PyState::Clone() const {
  // Create a new State of the right type.
  auto rv = game_->NewInitialState();

  // Copy the Python-side properties of the state.
  py::function deepcopy = py::module::import("copy").attr("deepcopy");
  py::object py_state = py::cast(*rv);
  for (auto [k, v] : PyDict(*this)) {
    py_state.attr(k) = deepcopy(v);
  }

  // Copy the C++-side properties of the state (all on the parent class).
  // Since we started with a valid initial state, we only need to copy
  // properties that change during the life of the state - hence num_players,
  // num_distinct_actions are omitted.
  PyState* state = open_spiel::down_cast<PyState*>(rv.get());
  state->history_ = history_;
  state->move_number_ = move_number_;

  return rv;
}

std::vector<std::string> PyState::DistributionSupport() {
  PYBIND11_OVERLOAD_PURE_NAME(std::vector<std::string>, State,
                              "distribution_support", DistributionSupport);
}
void PyState::UpdateDistribution(const std::vector<double>& distribution) {
  PYBIND11_OVERLOAD_PURE_NAME(void, State, "update_distribution",
                              UpdateDistribution, distribution);
}

// Register a Python game.
void RegisterPyGame(const GameType& game_type, py::function creator) {
  GameRegisterer::RegisterGame(
      game_type, [game_type, creator](const GameParameters& game_parameters) {
        py::dict params = py::cast(game_parameters);
        for (const auto& [k, v] : game_type.parameter_specification) {
          if (game_parameters.count(k) == 0) {
            params[pybind11::str(k)] = v;
          }
        }
        auto py_game = creator(params);
        return py::cast<std::shared_ptr<Game>>(py_game);
      });
}

// Observers and observations. We implement the C++ Observer in terms of the
// Python one.

// Wrapper for using a Python observer from C++.
// This is not a 'trampoline' class, just a wrapper.
class PyObserver : public Observer {
 public:
  PyObserver(py::object py_observer);
  void WriteTensor(const State& state, int player,
                   Allocator* allocator) const override;
  std::string StringFrom(const State& state, int player) const override;

 private:
  py::object py_observer_;
  py::function set_from_;
  py::function string_from_;
};

PyObserver::PyObserver(py::object py_observer)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      py_observer_(py_observer),
      set_from_(py_observer_.attr("set_from")),
      string_from_(py_observer_.attr("string_from")) {
  has_tensor_ = !py_observer_.attr("tensor").is_none();
}

void PyObserver::WriteTensor(const State& state, int player,
                             Allocator* allocator) const {
  using Array = py::array_t<float, py::array::c_style | py::array::forcecast>;
  const PyState& py_state = open_spiel::down_cast<const PyState&>(state);
  set_from_(py_state, player);
  py::dict dict = py_observer_.attr("dict");
  for (auto [k, v] : dict) {
    auto a = py::cast<Array>(v);
    const int dims = a.ndim();
    absl::InlinedVector<int, 4> shape(dims);
    for (int i = 0; i < dims; ++i) shape[i] = a.shape(i);
    SpanTensor out = allocator->Get(k.cast<std::string>(), shape);
    std::copy(a.data(), a.data() + a.size(), out.data().begin());
  }
}

std::string PyObserver::StringFrom(const State& state, int player) const {
  const PyState& py_state = open_spiel::down_cast<const PyState&>(state);
  return py::cast<std::string>(string_from_(py_state, player));
}

std::shared_ptr<Observer> PyGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  py::object h = py::cast(this);
  py::function f = h.attr("make_py_observer");
  if (!f) SpielFatalError("make_py_observer not implemented");
  py::object observer = (iig_obs_type.has_value() ?
      f(iig_obs_type.value(), params) : f(params));
  return std::make_shared<PyObserver>(observer);
}

std::string PyState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  return game.info_state_observer().StringFrom(*this, player);
}

std::string PyState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  return game.default_observer().StringFrom(*this, player);
}

void PyState::InformationStateTensor(Player player,
                                     absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  ContiguousAllocator allocator(values);
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  game.info_state_observer().WriteTensor(*this, player, &allocator);
}

namespace {
std::vector<int> TensorShape(const TrackingVectorAllocator& allocator) {
  switch (allocator.tensors_info().size()) {
    case 0:
      return {};
    case 1:
      return allocator.tensors_info().front().vector_shape();
    default: {
      int size = 0;
      for (const auto& info : allocator.tensors_info()) {
        size += info.size();
      }
      return {size};
    }
  }
}
}  // namespace

std::vector<int> PyGame::InformationStateTensorShape() const {
  TrackingVectorAllocator allocator;
  auto state = NewInitialState();
  info_state_observer().WriteTensor(*state, kDefaultPlayerId, &allocator);
  return TensorShape(allocator);
}

void PyState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  ContiguousAllocator allocator(values);
  const PyGame& game = open_spiel::down_cast<const PyGame&>(*game_);
  game.default_observer().WriteTensor(*this, player, &allocator);
}

std::vector<int> PyGame::ObservationTensorShape() const {
  TrackingVectorAllocator allocator;
  auto state = NewInitialState();
  default_observer().WriteTensor(*state, kDefaultPlayerId, &allocator);
  return TensorShape(allocator);
}

py::dict PyDict(const State& state) {
  py::object obj = py::cast(&state);
  if (py::hasattr(obj, "__dict__")) {
    return obj.attr("__dict__");
  } else {
    return py::dict();
  }
}

std::unique_ptr<State> PyGame::DeserializeState(const std::string& str) const {
  std::unique_ptr<State> state = NewInitialState();
  open_spiel::down_cast<PyState*>(state.get())->Deserialize(str);
  return state;
}

// Serialization form for the Python-side attributes is a b64-encoded pickled
// Python dict (the __dict__ member of the Python object).

py::dict decode_dict(const absl::string_view str) {
  std::string bytes;
  SPIEL_CHECK_TRUE(absl::Base64Unescape(str, &bytes));
  py::function pickle_loads = py::module::import("pickle").attr("loads");
  return pickle_loads(py::bytes(bytes));
}

std::string encode_dict(py::dict dict) {
  py::function pickle_dumps = py::module::import("pickle").attr("dumps");
  py::bytes bytes = pickle_dumps(dict);
  return absl::Base64Escape(std::string(bytes));
}

inline constexpr const absl::string_view kTagHistory = "history=";
inline constexpr const absl::string_view kTagMoveNumber = "move_number=";
inline constexpr const absl::string_view kTagDict = "__dict__=";

void PyState::Deserialize(const std::string& str) {
  std::vector<absl::string_view> pieces =
      absl::StrSplit(str, absl::MaxSplits('\n', 2));
  SPIEL_CHECK_EQ(pieces.size(), 3);

  SPIEL_CHECK_EQ(pieces[0].substr(0, kTagHistory.size()), kTagHistory);
  auto history_str = pieces[0].substr(kTagHistory.size());
  if (!history_str.empty()) {
    for (auto& h : absl::StrSplit(history_str, ',')) {
      std::vector<absl::string_view> p = absl::StrSplit(h, ':');
      SPIEL_CHECK_EQ(p.size(), 2);
      int player, action;
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(p[0], &player));
      SPIEL_CHECK_TRUE(absl::SimpleAtoi(p[1], &action));
      history_.push_back({player, action});
    }
  }

  SPIEL_CHECK_EQ(pieces[1].substr(0, kTagMoveNumber.size()), kTagMoveNumber);
  SPIEL_CHECK_TRUE(
      absl::SimpleAtoi(pieces[1].substr(kTagMoveNumber.size()), &move_number_));

  SPIEL_CHECK_EQ(pieces[2].substr(0, kTagDict.size()), kTagDict);
  py::object py_state = py::cast(*this);
  for (const auto& [k, v] : decode_dict(pieces[2].substr(kTagDict.size()))) {
    py_state.attr(k) = v;
  }
}

std::string PyState::Serialize() const {
  return absl::StrCat(
      // C++ Attributes
      kTagHistory,
      absl::StrJoin(history_, ",",
                    [](std::string* out, const PlayerAction& pa) {
                      absl::StrAppend(out, pa.player, ":", pa.action);
                    }),
      "\n", kTagMoveNumber, move_number_, "\n",
      // Python attributes
      kTagDict, encode_dict(PyDict(*this)));
}

int PyState::MeanFieldPopulation() const {
  // Use a python population() implementation if available.
  PYBIND11_OVERRIDE_IMPL(int, State, "mean_field_population");

  // Otherwise, default to behavior from the base class.
  return State::MeanFieldPopulation();
}

}  // namespace open_spiel
