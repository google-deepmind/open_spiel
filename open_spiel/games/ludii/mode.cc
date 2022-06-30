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

#include "open_spiel/games/ludii/mode.h"

namespace open_spiel {
namespace ludii {

Mode::Mode(JNIEnv *env, jobject mode) : env(env), mode(mode) {}

int Mode::NumPlayers() const {
  jclass gameClass = env->FindClass("game/mode/Mode");
  jmethodID stateFlags_id = env->GetMethodID(gameClass, "numPlayers", "()I");
  return (int)env->CallIntMethod(mode, stateFlags_id);
}

}  // namespace ludii
}  // namespace open_spiel
