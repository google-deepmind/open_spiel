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

#include "open_spiel/games/ludii/state.h"

namespace open_spiel {
namespace ludii {

State::State(JNIEnv *env, jobject state) : env(env), state(state) {}

std::vector<ContainerState> State::ContainerStates() const {
  std::vector<ContainerState> containerStateVector;

  jclass stateClass = env->FindClass("util/state/State");
  jmethodID containerStates_id =
      env->GetMethodID(stateClass, "containerStates",
                       "()[Lutil/state/containerState/ContainerState;");
  jobjectArray containerStateArray =
      (jobjectArray)env->CallObjectMethod(state, containerStates_id);
  int containerStateCount = env->GetArrayLength(containerStateArray);

  for (int i = 0; i < containerStateCount; i++) {
    jobject containerStateObj =
        env->GetObjectArrayElement(containerStateArray, i);
    containerStateVector.push_back(ContainerState(env, containerStateObj));
  }

  return containerStateVector;
}

int State::Mover() const {
  jclass stateClass = env->FindClass("util/state/State");
  jmethodID mover_id = env->GetMethodID(stateClass, "mover", "()I");

  return (int)env->CallIntMethod(state, mover_id);
}

}  // namespace ludii
}  // namespace open_spiel
