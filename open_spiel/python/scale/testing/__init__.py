# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract checkers and fixtures for authors of components.

check_env                 the Env contract over random playouts
check_policy              the Policy contract (carry, outputs, resets, seqs)
check_decide              known answers for a decide implementation
check_forward_dtypes      a network forward keeps its declared dtypes under x64
check_f64                 every float64 value in a traced program
check_operator            the conformance test of a search operator
infoset_invariance        a policy's output depends only on its seat's info
host_operator_fixtures    a host-driven test operator and its slot factory
fixtures                  small policies with known outputs, every capability
populations               kuhn tabular populations with known equilibria
multidevice               probes under N forced host devices, tree comparisons
optional_pyspiel_games    skips for pyspiel games a build may leave out
steps                     SeatStep builders for contract tests
errors                    ContractError, raised by every checker
"""
