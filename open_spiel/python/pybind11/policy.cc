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

#include "open_spiel/python/pybind11/policy.h"

// Python bindings for policies and algorithms handling them.

#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/cfr_br.h"
#include "open_spiel/algorithms/deterministic_policy.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/external_sampling_mccfr.h"
#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/algorithms/outcome_sampling_mccfr.h"
#include "open_spiel/algorithms/tabular_best_response_mdp.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/policy.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/python/pybind11/python_policy.h"
#include "open_spiel/spiel.h"
#include "pybind11/include/pybind11/detail/common.h"

namespace open_spiel {
namespace {

using ::open_spiel::ActionsAndProbs;
using ::open_spiel::Policy;
using ::open_spiel::TabularPolicy;
using ::open_spiel::algorithms::Exploitability;
using ::open_spiel::algorithms::NashConv;
using ::open_spiel::algorithms::TabularBestResponse;
using ::open_spiel::algorithms::TabularBestResponseMDP;
using ::open_spiel::algorithms::TabularBestResponseMDPInfo;

namespace py = ::pybind11;
}  // namespace

void init_pyspiel_policy(py::module& m) {
  py::classh<Policy, PyPolicy> policy(m, "Policy");
  policy.def(py::init<>())
      .def("action_probabilities",
           py::overload_cast<const State&>(&Policy::GetStatePolicyAsMap,
                                           py::const_),
           py::arg("state"),
           "Returns a dictionary mapping actions to probabilities for the "
           "policy at the given "
           "state.")
      .def("action_probabilities",
           py::overload_cast<const std::string&>(&Policy::GetStatePolicyAsMap,
                                                 py::const_),
           py::arg("info_state"),
           "Returns a dictionary mapping actions to probabilities for the "
           "policy at the given "
           "information state.")
      .def("get_state_policy",
           py::overload_cast<const State&>(&Policy::GetStatePolicy, py::const_),
           py::arg("state"),
           "Returns a list of (action, prob) pairs for the policy at the given "
           "state.")
      .def("get_state_policy",
           py::overload_cast<const State&, Player>(&Policy::GetStatePolicy,
                                                   py::const_),
           py::arg("state"), py::arg("player"),
           "Returns a list of (action, prob) pairs for the policy for the "
           "specified player at the "
           "given state.")
      .def("get_state_policy",
           py::overload_cast<const std::string&>(&Policy::GetStatePolicy,
                                                 py::const_),
           py::arg("info_state"),
           "Returns a list of (action, prob) pairs for the policy at the given "
           "info state.")
      .def("get_state_policy_as_parallel_vectors",
           py::overload_cast<const State&>(
               &Policy::GetStatePolicyAsParallelVectors, py::const_),
           py::arg("state"),
           "Returns a pair of parallel vectors (actions, probs) for the policy "
           "at the given state.")
      .def("get_state_policy_as_parallel_vectors",
           py::overload_cast<const std::string&>(
               &Policy::GetStatePolicyAsParallelVectors, py::const_),
           py::arg("info_state"),
           "Returns a pair of parallel vectors (actions, probs) for the policy "
           "at the given "
           "information state.")
      .def("serialize", &Policy::Serialize, py::arg("double_precision") = -1,
           py::arg("delimiter") = "<~>", "Serializes the policy to a string.");

  auto ptt = m.def_submodule(
      "_policy_trampoline_testing",
      "Internal test functions for calling policy member functions.");
  ptt.def("call_action_probabilities",
          [](const Policy& policy, const State& state) {
            return policy.GetStatePolicyAsMap(state);
          });
  ptt.def("call_action_probabilities",
          [](const Policy& policy, const std::string& info_state) {
            return policy.GetStatePolicyAsMap(info_state);
          });
  ptt.def("call_get_state_policy",
          [](const Policy& policy, const State& state) {
            return policy.GetStatePolicy(state);
          });
  ptt.def("call_get_state_policy",
          [](const Policy& policy, const State& state, Player player) {
            return policy.GetStatePolicy(state, player);
          });
  ptt.def("call_get_state_policy",
          [](const Policy& policy, const std::string& info_state) {
            return policy.GetStatePolicy(info_state);
          });
  ptt.def("call_get_state_policy_as_parallel_vectors",
          [](const Policy& policy, const State& state) {
            return policy.GetStatePolicyAsParallelVectors(state);
          });
  ptt.def("call_get_state_policy_as_parallel_vectors",
          [](const Policy& policy, const std::string& info_state) {
            return policy.GetStatePolicyAsParallelVectors(info_state);
          });
  ptt.def("call_serialize", [](const Policy& policy, int precision,
                               const std::string& delimiter = "<~>") {
    return policy.Serialize(precision, delimiter);
  });

  py::class_<TabularBestResponse>(m, "TabularBestResponse")
      .def(py::init<const open_spiel::Game&, int,
                    const std::unordered_map<std::string,
                                             open_spiel::ActionsAndProbs>&>())
      .def(py::init<const open_spiel::Game&, int, const open_spiel::Policy*>())
      .def(py::init<
           const open_spiel::Game&, int,
           const std::unordered_map<std::string, open_spiel::ActionsAndProbs>&,
           const float, const float>())
      .def(py::init<const open_spiel::Game&, int, const open_spiel::Policy*,
                    const float, const float>())
      .def("value",
           py::overload_cast<const std::string&>(&TabularBestResponse::Value))
      .def("value_from_state", py::overload_cast<const open_spiel::State&>(
                                   &TabularBestResponse::Value))
      .def("get_best_response_policy",
           &TabularBestResponse::GetBestResponsePolicy)
      .def("get_best_response_actions",
           &TabularBestResponse::GetBestResponseActions)
      .def("set_policy", py::overload_cast<const std::unordered_map<
                             std::string, open_spiel::ActionsAndProbs>&>(
                             &TabularBestResponse::SetPolicy))
      .def("set_policy",
           py::overload_cast<const Policy*>(&TabularBestResponse::SetPolicy));

  // A tabular policy represented internally as a map. Note that this
  // implementation is not directly compatible with the Python TabularPolicy
  // implementation; the latter is implemented as a table of size
  // [num_states, num_actions], while this is implemented as a map. It is
  // non-trivial to convert between the two, but we have a function that does so
  // in the open_spiel/python/policy.py file.
  py::classh<TabularPolicy, Policy>(m, "TabularPolicy")
      .def(py::init<const std::unordered_map<std::string, ActionsAndProbs>&>())
      .def("__str__", &TabularPolicy::ToString)
      .def("__repr__", &TabularPolicy::ToString)
      .def("__len__", &TabularPolicy::size)
      .def("get_state_policy", &TabularPolicy::GetStatePolicy)
      .def("policy_table",
           py::overload_cast<>(&TabularPolicy::PolicyTable))
      .def("size", &TabularPolicy::size)
      .def("to_string", &TabularPolicy::ToString);

  py::classh<open_spiel::PartialTabularPolicy, open_spiel::Policy>(
      m, "PartialTabularPolicy")
      .def(py::init<>())
      .def(py::init<const std::unordered_map<std::string, ActionsAndProbs>&>())
      .def(py::init<const std::unordered_map<std::string, ActionsAndProbs>&,
                    std::shared_ptr<Policy>>())
      .def("get_state_policy",
           (ActionsAndProbs(open_spiel::Policy::*)(const State&) const) &
               open_spiel::PartialTabularPolicy::GetStatePolicy)
      .def(
          "get_state_policy",
          (ActionsAndProbs(open_spiel::Policy::*)(const State&, Player) const) &
              open_spiel::PartialTabularPolicy::GetStatePolicy)
      .def("get_state_policy",
           (ActionsAndProbs(open_spiel::Policy::*)(const std::string&) const) &
               open_spiel::PartialTabularPolicy::GetStatePolicy)
      .def("set_prob", &open_spiel::PartialTabularPolicy::SetProb)
      .def("set_state_policy",
           &open_spiel::PartialTabularPolicy::SetStatePolicy)
      .def("policy_table",
           py::overload_cast<>(&open_spiel::PartialTabularPolicy::PolicyTable));

  m.def("GetRandomPolicy", &open_spiel::GetRandomPolicy,
      py::arg("game"), py::arg("seed"), py::arg("player") = -1);
  m.def("GetFlatDirichletPolicy", &open_spiel::GetFlatDirichletPolicy,
      py::arg("game"), py::arg("seed"), py::arg("player") = -1);
  m.def("GetRandomDeterministicPolicy",
      &open_spiel::GetRandomDeterministicPolicy,
      py::arg("game"), py::arg("seed"), py::arg("player") = -1);
  m.def("GetRandomDeterministicVisitPolicy",
      &open_spiel::GetRandomDeterministicVisitPolicy,
      py::arg("game"), py::arg("seed"), py::arg("player") = -1);
  m.def("UniformRandomPolicy", &open_spiel::GetUniformPolicy);

  py::classh<open_spiel::UniformPolicy, open_spiel::Policy>(m, "UniformPolicy")
      .def(py::init<>())
      .def("get_state_policy", &open_spiel::UniformPolicy::GetStatePolicy);

  py::classh<open_spiel::PreferredActionPolicy, open_spiel::Policy>(
      m, "PreferredActionPolicy")
      .def(py::init<const std::vector<Action>&>())
      .def("get_state_policy",
           &open_spiel::PreferredActionPolicy::GetStatePolicy);

  py::class_<open_spiel::algorithms::CFRSolver>(m, "CFRSolver")
      .def(py::init([](std::shared_ptr<const Game> game) {
        return new algorithms::CFRSolver(*game);
      }))
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy", &open_spiel::algorithms::CFRSolver::AveragePolicy)
      .def("tabular_average_policy",
           &open_spiel::algorithms::CFRSolver::TabularAveragePolicy)
      .def("tabular_current_policy",
           &open_spiel::algorithms::CFRSolver::TabularCurrentPolicy)
      .def(py::pickle(
          [](const open_spiel::algorithms::CFRSolver& solver) {  // __getstate__
            return solver.Serialize();
          },
          [](const std::string& serialized) {  // __setstate__
            return open_spiel::algorithms::DeserializeCFRSolver(serialized);
          }));

  py::class_<open_spiel::algorithms::CFRPlusSolver>(m, "CFRPlusSolver")
      .def(py::init([](std::shared_ptr<const Game> game) {
        return new algorithms::CFRPlusSolver(*game);
      }))
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy)
      .def("tabular_average_policy",
           &open_spiel::algorithms::CFRPlusSolver::TabularAveragePolicy)
      .def(py::pickle(
          [](const open_spiel::algorithms::CFRPlusSolver&
                 solver) {  // __getstate__
            return solver.Serialize();
          },
          [](const std::string& serialized) {  // __setstate__
            return open_spiel::algorithms::DeserializeCFRPlusSolver(serialized);
          }));

  py::class_<open_spiel::algorithms::CFRBRSolver>(m, "CFRBRSolver")
      .def(py::init([](std::shared_ptr<const Game> game) {
        return new algorithms::CFRBRSolver(*game);
      }))
      .def("evaluate_and_update_policy",
           &open_spiel::algorithms::CFRPlusSolver::EvaluateAndUpdatePolicy)
      .def("current_policy", &open_spiel::algorithms::CFRSolver::CurrentPolicy)
      .def("average_policy",
           &open_spiel::algorithms::CFRPlusSolver::AveragePolicy)
      .def(py::pickle(
          [](const open_spiel::algorithms::CFRBRSolver&
                 solver) {  // __getstate__
            return solver.Serialize();
          },
          [](const std::string& serialized) {  // __setstate__
            return open_spiel::algorithms::DeserializeCFRBRSolver(serialized);
          }));

  py::enum_<open_spiel::algorithms::AverageType>(m, "MCCFRAverageType")
      .value("SIMPLE", open_spiel::algorithms::AverageType::kSimple)
      .value("FULL", open_spiel::algorithms::AverageType::kFull);

  py::class_<open_spiel::algorithms::ExternalSamplingMCCFRSolver>(
      m, "ExternalSamplingMCCFRSolver")
      .def(py::init([](std::shared_ptr<const Game> game, int seed,
                       algorithms::AverageType average_type) {
             return new algorithms::ExternalSamplingMCCFRSolver(*game, seed,
                                                                average_type);
           }),
           py::arg("game"), py::arg("seed") = 0,
           py::arg("avg_type") = open_spiel::algorithms::AverageType::kSimple)
      .def("run_iteration",
           py::overload_cast<>(&open_spiel::algorithms::
                                   ExternalSamplingMCCFRSolver::RunIteration))
      .def("average_policy",
           &open_spiel::algorithms::ExternalSamplingMCCFRSolver::AveragePolicy)
      .def(py::pickle(
          [](const open_spiel::algorithms::ExternalSamplingMCCFRSolver&
                 solver) {  // __getstate__
            return solver.Serialize();
          },
          [](const std::string& serialized) {  // __setstate__
            return open_spiel::algorithms::
                DeserializeExternalSamplingMCCFRSolver(serialized);
          }));

  py::class_<open_spiel::algorithms::OutcomeSamplingMCCFRSolver>(
      m, "OutcomeSamplingMCCFRSolver")
      .def(py::init(
               [](std::shared_ptr<const Game> game, double epsilon, int seed) {
                 return new algorithms::OutcomeSamplingMCCFRSolver(
                     *game, epsilon, seed);
               }),
           py::arg("game"),
           py::arg("epsilon") = open_spiel::algorithms::
               OutcomeSamplingMCCFRSolver::kDefaultEpsilon,
           py::arg("seed") = -1)
      .def("run_iteration",
           py::overload_cast<>(&open_spiel::algorithms::
                                   OutcomeSamplingMCCFRSolver::RunIteration))
      .def("average_policy",
           &open_spiel::algorithms::OutcomeSamplingMCCFRSolver::AveragePolicy)
      .def(py::pickle(
          [](const open_spiel::algorithms::OutcomeSamplingMCCFRSolver&
                 solver) {  // __getstate__
            return solver.Serialize();
          },
          [](const std::string& serialized) {  // __setstate__
            return open_spiel::algorithms::
                DeserializeOutcomeSamplingMCCFRSolver(serialized);
          }));

  py::class_<TabularBestResponseMDPInfo>(m, "TabularBestResponseMDPInfo")
      .def_readonly("br_values", &TabularBestResponseMDPInfo::br_values)
      .def_readonly("br_policies", &TabularBestResponseMDPInfo::br_policies)
      .def_readonly("on_policy_values",
                    &TabularBestResponseMDPInfo::on_policy_values)
      .def_readonly("deviation_incentives",
                    &TabularBestResponseMDPInfo::deviation_incentives)
      .def_readonly("nash_conv", &TabularBestResponseMDPInfo::nash_conv)
      .def_readonly("exploitability",
                    &TabularBestResponseMDPInfo::exploitability);

  py::class_<TabularBestResponseMDP>(m, "TabularBestResponseMDP")
      .def(py::init<const open_spiel::Game&, const open_spiel::Policy&>())
      .def("compute_best_responses",  // Takes no arguments.
           &TabularBestResponseMDP::ComputeBestResponses)
      .def("compute_best_response",  // Takes one argument: Player max_player.
           &TabularBestResponseMDP::ComputeBestResponse, py::arg("max_player"))
      .def("nash_conv", &TabularBestResponseMDP::NashConv)
      .def("exploitability", &TabularBestResponseMDP::Exploitability);

  m.def(
      "expected_returns",
      py::overload_cast<const State&, const std::vector<const Policy*>&, int,
                        bool, float>(&open_spiel::algorithms::ExpectedReturns),
      "Computes the undiscounted expected returns from a depth-limited "
      "search.",
      py::arg("state"), py::arg("policies"), py::arg("depth_limit"),
      py::arg("use_infostate_get_policy"), py::arg("prob_cut_threshold") = 0.0);

  m.def("expected_returns",
        py::overload_cast<const State&, const Policy&, int, bool, float>(
            &open_spiel::algorithms::ExpectedReturns),
        "Computes the undiscounted expected returns from a depth-limited "
        "search.",
        py::arg("state"), py::arg("joint_policy"), py::arg("depth_limit"),
        py::arg("use_infostate_get_policy"),
        py::arg("prob_cut_threshold") = 0.0);

  m.def("expected_returns_of_deterministic_policies_from_seeds",
        py::overload_cast<const State&, const std::vector<int>&>(
            &open_spiel::algorithms::
                ExpectedReturnsOfDeterministicPoliciesFromSeeds),
        py::call_guard<py::gil_scoped_release>(),
        "Computes the undiscounted expected returns from seeds.",
        py::arg("state"), py::arg("policy_seeds"));

  m.def("expected_returns_of_deterministic_policies_from_seeds",
        py::overload_cast<const State&, const std::vector<int>&,
                          const std::vector<const Policy*>&>(
            &open_spiel::algorithms::
                ExpectedReturnsOfDeterministicPoliciesFromSeeds),
        py::call_guard<py::gil_scoped_release>(),
        "Computes the expected returns from seeds and policies.",
        py::arg("state"), py::arg("policy_seeds"), py::arg("policies"));

  m.def(
      "exploitability",
      [](std::shared_ptr<const Game> game, const Policy& policy) {
        return Exploitability(*game, policy);
      },
      "Returns the sum of the utility that a best responder wins when when "
      "playing against 1) the player 0 policy contained in `policy` and 2) "
      "the player 1 policy contained in `policy`."
      "This only works for two player, zero- or constant-sum sequential "
      "games, and raises a SpielFatalError if an incompatible game is passed "
      "to it.");

  m.def(
      "exploitability",
      [](std::shared_ptr<const Game> game,
         const std::unordered_map<std::string, ActionsAndProbs>& policy) {
        return Exploitability(*game, policy);
      },
      "Returns the sum of the utility that a best responder wins when when "
      "playing against 1) the player 0 policy contained in `policy` and 2) "
      "the player 1 policy contained in `policy`."
      "This only works for two player, zero- or constant-sum sequential "
      "games, and raises a SpielFatalError if an incompatible game is passed "
      "to it.");

  m.def(
      "nash_conv",
      [](std::shared_ptr<const Game> game, const Policy& policy,
         bool use_state_get_policy) {
        return NashConv(*game, policy, use_state_get_policy);
      },
      "Calculates a measure of how far the given policy is from a Nash "
      "equilibrium by returning the sum of the improvements in the value "
      "that each player could obtain by unilaterally changing their strategy "
      "while the opposing player maintains their current strategy (which "
      "for a Nash equilibrium, this value is 0). The third parameter is to "
      "indicate whether to use the Policy::GetStatePolicy(const State&) "
      "instead of Policy::GetStatePolicy(const std::string& info_state) for "
      "computation of the on-policy expected values.",
      py::arg("game"), py::arg("policy"),
      py::arg("use_state_get_policy") = false);

  m.def(
      "nash_conv",
      [](std::shared_ptr<const Game> game,
         const std::unordered_map<std::string, ActionsAndProbs>& policy) {
        return NashConv(*game, policy);
      },
      "Calculates a measure of how far the given policy is from a Nash "
      "equilibrium by returning the sum of the improvements in the value "
      "that each player could obtain by unilaterally changing their strategy "
      "while the opposing player maintains their current strategy (which "
      "for a Nash equilibrium, this value is 0).");

  m.def("num_deterministic_policies",
        &open_spiel::algorithms::NumDeterministicPolicies,
        "Returns number of determinstic policies in this game for a player, "
        "or -1 if there are more than 2^64 - 1 policies.");

  m.def("to_joint_tabular_policy", &open_spiel::ToJointTabularPolicy,
        "Returns a merged tabular policy from a list of TabularPolicy. The "
        "second argument is a bool which, if true, checks that there is no "
        "overlap among all the policies.");
}
}  // namespace open_spiel
