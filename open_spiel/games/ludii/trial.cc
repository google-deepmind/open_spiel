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

#include "open_spiel/games/ludii/trial.h"

namespace open_spiel {
namespace ludii {

Trial::Trial(JNIEnv *env, Game game) : env(env) {
  jclass trial_class = env->FindClass("util/Trial");
  jmethodID trial_const_id =
      env->GetMethodID(trial_class, "<init>", "(Lgame/Game;)V");
  jobject trial_obj =
      env->NewObject(trial_class, trial_const_id, game.GetObj());

  trial = trial_obj;
}

jobject Trial::GetObj() const { return trial; }

State Trial::GetState() const {
  jclass trial_class = env->FindClass("util/Trial");
  jmethodID state_id =
      env->GetMethodID(trial_class, "state", "()Lutil/state/State;");
  jobject state_obj = env->CallObjectMethod(trial, state_id);

  return State(env, state_obj);
}

bool Trial::Over() const {
  jclass trial_class = env->FindClass("util/Trial");
  jmethodID over_id = env->GetMethodID(trial_class, "over", "()Z");

  return (bool)env->CallObjectMethod(trial, over_id);
}

}  // namespace ludii
}  // namespace open_spiel
