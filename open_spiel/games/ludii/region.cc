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

#include "open_spiel/games/ludii/region.h"

namespace open_spiel {
namespace ludii {

Region::Region(JNIEnv *env, jobject region) : env(env), region(region) {}

ChunkSet Region::BitSet() const {
  jclass regionClass = env->FindClass("util/Region");
  jmethodID bitSet_id =
      env->GetMethodID(regionClass, "bitSet", "()Lutil/ChunkSet;");
  jobject chunkset_obj = env->CallObjectMethod(region, bitSet_id);

  return ChunkSet(env, chunkset_obj);
}

}  // namespace ludii
}  // namespace open_spiel
