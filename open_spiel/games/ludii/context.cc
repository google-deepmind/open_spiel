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

#include "open_spiel/games/ludii/context.h"

#include "open_spiel/games/ludii/game.h"

namespace open_spiel {
namespace ludii {

Context::Context(JNIEnv *env, Game game, Trial trial) : env(env) {
  jclass context_class = env->FindClass("util/Context");
  jmethodID context_const_id =
      env->GetMethodID(context_class, "<init>", "(Lgame/Game;Lutil/Trial;)V");
  jobject context_obj = env->NewObject(context_class, context_const_id,
                                       game.GetObj(), trial.GetObj());

  context = context_obj;
}

jobject Context::GetObj() const { return context; }

}  // namespace ludii
}  // namespace open_spiel
