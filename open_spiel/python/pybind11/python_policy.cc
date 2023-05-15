

#include "open_spiel/python/pybind11/python_policy.h"

#include "open_spiel/spiel_utils.h"


#ifndef SINGLE_ARG
   #define SINGLE_ARG(...) __VA_ARGS__
#endif

namespace open_spiel {

std::pair< std::vector< Action >, std::vector< double > >
PyPolicy::GetStatePolicyAsParallelVectors(const State& state) const
{
   PYBIND11_OVERRIDE(
      SINGLE_ARG(std::pair< std::vector< Action >, std::vector< double > >),
      Policy,
      GetStatePolicyAsParallelVectors,
      state
   );
}
std::pair< std::vector< Action >, std::vector< double > >
PyPolicy::GetStatePolicyAsParallelVectors(const std::string info_state) const
{
   PYBIND11_OVERRIDE(
      SINGLE_ARG(std::pair< std::vector< Action >, std::vector< double > >),
      Policy,
      GetStatePolicyAsParallelVectors,
      info_state
   );
}
std::unordered_map< Action, double > PyPolicy::GetStatePolicyAsMap(
   const State& state
) const
{
   PYBIND11_OVERRIDE(
      SINGLE_ARG(std::unordered_map< Action, double >), Policy, GetStatePolicyAsMap, state
   );
}
std::unordered_map< Action, double > PyPolicy::GetStatePolicyAsMap(
   const std::string& info_state
) const
{
   PYBIND11_OVERRIDE(
      SINGLE_ARG(std::unordered_map< Action, double >), Policy, GetStatePolicyAsMap, info_state
   );
}
ActionsAndProbs PyPolicy::GetStatePolicy(const State& state
) const
{
   PYBIND11_OVERRIDE(ActionsAndProbs, Policy, GetStatePolicy, state);
}
ActionsAndProbs PyPolicy::GetStatePolicy(
   const State& state,
   Player player
) const
{
   PYBIND11_OVERRIDE(ActionsAndProbs, Policy, GetStatePolicy, state, player);
}
ActionsAndProbs PyPolicy::GetStatePolicy(const std::string& info_state
) const
{
   PYBIND11_OVERRIDE(ActionsAndProbs, Policy, GetStatePolicy, info_state);
}
std::string PyPolicy::Serialize(int double_precision, std::string delimiter) const
{
   PYBIND11_OVERRIDE(std::string, Policy, Serialize, double_precision, delimiter);
}

}  // namespace open_spiel