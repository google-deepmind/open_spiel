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

#include "open_spiel/games/ludii/container_state.h"

namespace open_spiel {
namespace ludii {

ContainerState::ContainerState(JNIEnv *env, jobject container_state)
    : env(env), container_state(container_state) {}

Region ContainerState::Empty() const {
  jclass ContainerStateClass =
      env->FindClass("util/state/containerState/ContainerState");
  jmethodID empty_id =
      env->GetMethodID(ContainerStateClass, "empty", "()Lutil/Region;");
  jobject region_obj = env->CallObjectMethod(container_state, empty_id);

  return Region(env, region_obj);
}

ChunkSet ContainerState::CloneWho() const {
  jclass ContainerStateClass =
      env->FindClass("util/state/containerState/ContainerState");
  jmethodID cloneWho_id =
      env->GetMethodID(ContainerStateClass, "cloneWho", "()Lutil/ChunkSet;");
  jobject chunkset_obj = env->CallObjectMethod(container_state, cloneWho_id);

  return ChunkSet(env, chunkset_obj);
}

ChunkSet ContainerState::CloneWhat() const {
  jclass ContainerStateClass =
      env->FindClass("util/state/containerState/ContainerState");
  jmethodID cloneWhat_id =
      env->GetMethodID(ContainerStateClass, "cloneWhat", "()Lutil/ChunkSet;");
  jobject chunkset_obj = env->CallObjectMethod(container_state, cloneWhat_id);

  return ChunkSet(env, chunkset_obj);
}

}  // namespace ludii
}  // namespace open_spiel
