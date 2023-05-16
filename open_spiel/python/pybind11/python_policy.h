
#ifndef OPEN_SPIEL_PYTHON_POLICY_H
#define OPEN_SPIEL_PYTHON_POLICY_H


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <tuple>
#include <unordered_map>

#include "open_spiel/policy.h"
#include "pybind11/trampoline_self_life_support.h"

namespace open_spiel {
namespace py = pybind11;

class PyPolicy: public Policy, public py::trampoline_self_life_support {
  public:
   ~PyPolicy() override = default;
   PyPolicy() = default;

   std::pair< std::vector< Action >, std::vector< double > > GetStatePolicyAsParallelVectors(
      const State& state
   ) const override;

   std::pair< std::vector< Action >, std::vector< double > > GetStatePolicyAsParallelVectors(
      const std::string& info_state
   ) const override;

   std::unordered_map< Action, double > GetStatePolicyAsMap(const State& state) const override;

   std::unordered_map< Action, double > GetStatePolicyAsMap(const std::string& info_state
   ) const override;

   ActionsAndProbs GetStatePolicy(const State& state) const override;

   ActionsAndProbs GetStatePolicy(const State& state, Player player) const override;

   ActionsAndProbs GetStatePolicy(const std::string& info_state) const override;

   std::string Serialize(int double_precision, std::string delimiter) const override;
};
}  // namespace open_spiel
#endif  // OPEN_SPIEL_PYTHON_POLICY_H
