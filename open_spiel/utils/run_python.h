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

#ifndef OPEN_SPIEL_UTILS_RUN_PYTHON_H_
#define OPEN_SPIEL_UTILS_RUN_PYTHON_H_

#include <string>
#include <vector>

namespace open_spiel {

inline constexpr const char* kDefaultPythonCommand = "python3";

// Spawn a python interpreter and run the python script referenced by a module
// path, with args. This is equivalent to `<python command> -m module <args>`.
// Returns true on success, false on failure.
bool RunPython(const std::string& python_command,
               const std::string& module,
               const std::vector<std::string>& args);

// Same as above using the default python command,.
bool RunPython(const std::string& module,
               const std::vector<std::string>& args);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_UTILS_RUN_PYTHON_H_
