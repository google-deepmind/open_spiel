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

#include "open_spiel/games/ludii/moves.h"

namespace open_spiel {
namespace ludii {

Moves::Moves(JNIEnv *env, jobject moves) : env(env), moves(moves) {}

std::vector<Move> Moves::GetMoves() const {
  std::vector<Move> moveVector;

  jclass moves_class = env->FindClass("game/rules/play/moves/Moves");
  jmethodID moves_id =
      env->GetMethodID(moves_class, "moves", "()Lmain/FastArrayList;");
  jobject moveFastArray_obj = env->CallObjectMethod(moves, moves_id);

  jclass fastArray_class = env->FindClass("main/FastArrayList");
  jmethodID fastArraySize_id = env->GetMethodID(fastArray_class, "size", "()I");
  jmethodID fastArrayGet_id =
      env->GetMethodID(fastArray_class, "get", "(I)Ljava/lang/Object;");

  jint fastArraySize = env->CallIntMethod(moveFastArray_obj, fastArraySize_id);

  for (int i = 0; i < fastArraySize; i++) {
    jobject move_obj =
        env->CallObjectMethod(moveFastArray_obj, fastArrayGet_id, i);
    moveVector.push_back(Move(env, move_obj));
  }

  return moveVector;
}

}  // namespace ludii
}  // namespace open_spiel
