// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LUDII_H_
#define LUDII_H_

#include "jni.h"
#include <string>
#include <vector>
#include "game.h"

class GameLoader
{

public:

    GameLoader(JNIEnv *env_const);
    std::vector<std::string> ListGames() const;
    Game LoadGame(std::string game_name) const;

private:

    JNIEnv *env;

};

#endif