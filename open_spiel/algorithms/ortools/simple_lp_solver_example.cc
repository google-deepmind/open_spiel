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

#include "open_spiel/spiel.h"
#include "ortools/linear_solver/linear_solver.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

namespace opres = operations_research;

// Example use of OR-Tools adapted from here:
// https://developers.google.com/optimization/introduction/cpp
void TestSimpleLpProgram() {
  // Create the linear solver with the GLOP backend.
  opres::MPSolver solver("simple_lp_program",
                         opres::MPSolver::GLOP_LINEAR_PROGRAMMING);

  // Create the variables x and y.
  opres::MPVariable* const x = solver.MakeNumVar(0.0, 1, "x");
  opres::MPVariable* const y = solver.MakeNumVar(0.0, 2, "y");

  std::cout << "Number of variables = " << solver.NumVariables() << std::endl;

  // Create a linear constraint, 0 <= x + y <= 2.
  opres::MPConstraint* const ct = solver.MakeRowConstraint(0.0, 2.0, "ct");
  ct->SetCoefficient(x, 1);
  ct->SetCoefficient(y, 1);

  std::cout << "Number of constraints = " << solver.NumConstraints()
            << std::endl;

  // Create the objective function, 3 * x + y.
  opres::MPObjective* const objective = solver.MutableObjective();
  objective->SetCoefficient(x, 3);
  objective->SetCoefficient(y, 1);
  objective->SetMaximization();

  solver.Solve();

  std::cout << "Solution:" << std::endl;
  std::cout << "Objective value = " << objective->Value() << std::endl;
  std::cout << "x = " << x->solution_value() << std::endl;
  std::cout << "y = " << y->solution_value() << std::endl;
}
}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) { algorithms::ortools::TestSimpleLpProgram(); }
