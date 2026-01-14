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

#ifndef __GO_OPEN_SPIEL_H__
#define __GO_OPEN_SPIEL_H__

/* A pure C API that wraps the C++ OpenSpiel core. */

#ifdef __cplusplus
extern "C" {
#endif

/* Test */
void Test();

/* Game functions. */
void* LoadGame(const char* name);
void DeleteGame(void* game_ptr);
char* GameShortName(const void* game_ptr);
char* GameLongName(const void* game_ptr);
void* GameNewInitialState(const void* game_ptr);
int GameNumPlayers(const void* game_ptr);
int GameMaxGameLength(const void* game_ptr);
int GameNumDistinctActions(const void* game_ptr);

/* State functions. */
void DeleteState(void* state_ptr);
void* StateClone(const void* state_ptr);
char* StateToString(const void* state_ptr);
int StateNumLegalActions(const void* state_ptr);
int StateNumDistinctActions(const void* state_ptr);
void StateFillLegalActions(const void* state_ptr, void* array_ptr);
void StateFillLegalActionsMask(const void* state_ptr, void* array_ptr);
int StateSizeObservation(const void* state_ptr);
void StateFillObservation(const void* state_ptr, void* array_ptr);
void StateFillObservationPlayer(const void* state_ptr, void* array_ptr,
                                int player);
double StateObservation(const void* observation_ptr, int idx);
int StateSizeInformationState(const void* state_ptr);
void StateFillInformationState(const void* state_ptr, void* array_ptr);
void StateFillInformationStatePlayer(const void* state_ptr, void* array_ptr,
                                     int player);
double StateInformationState(const void* information_state_ptr, int idx);

int StateSizeChanceOutcomes(const void* state_ptr);
void StateFillChanceOutcomes(const void* state_ptr, void* action_ptr,
                             void* proba_ptr);

int StateIsTerminal(const void* state_ptr);
int StateIsChanceNode(const void* state_ptr);
int StateCurrentPlayer(const void* state_ptr);
char* StateActionToString(const void* state_ptr, int player, int action);
void StateApplyAction(void* state_ptr, int action);
double StatePlayerReturn(const void* state_ptr, int player);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
