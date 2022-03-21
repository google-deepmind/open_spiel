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

#ifndef __RUST_OPEN_SPIEL_H__
#define __RUST_OPEN_SPIEL_H__

/* A pure C API that wraps the C++ OpenSpiel core. */

#ifdef __cplusplus
extern "C" {
#endif

/* GameParameters functions */
void* NewGameParameters();
void DeleteGameParameters(void* params_ptr);
void GameParametersSetInt(void* params_ptr, const char* key, int value);
void GameParametersSetDouble(void* params_ptr, const char* key, double value);
void GameParametersSetString(void* params_ptr, const char* key,
                             const char* value);

/* Game functions. */
void* LoadGame(const char* name);
void* LoadGameFromParameters(const void* params_ptr);
void DeleteGame(void* game_ptr);
char* GameShortName(const void* game_ptr, unsigned long* length);  /* NOLINT */
char* GameLongName(const void* game_ptr, unsigned long* length);  /* NOLINT */
void* GameNewInitialState(const void* game_ptr);
int GameNumPlayers(const void* game_ptr);
int GameMaxGameLength(const void* game_ptr);
int GameNumDistinctActions(const void* game_ptr);
int* GameObservationTensorShape(const void* game_ptr, int* size);
int* GameInformationStateTensorShape(const void* game_ptri, int* size);

/* State functions. */
void DeleteState(void* state_ptr);
void* StateClone(const void* state_ptr);
char* StateToString(const void* state_ptr, unsigned long* length);  /* NOLINT */
long* StateLegalActions(const void* state_ptr, int* num_legal_actions);  /* NOLINT */
int StateCurrentPlayer(const void* state_ptr);
char* StateActionToString(const void* state_ptr, int player, long action,  /* NOLINT */
                          unsigned long* length);  /* NOLINT */
int StateIsTerminal(const void* state_ptr);
int StateIsChanceNode(const void* state_ptr);
int StateNumPlayers(const void* state_ptr);
void StateApplyAction(void* state_ptr, long action);  /* NOLINT */
void StateReturns(const void* state_ptr, double* returns_buf);
double StatePlayerReturn(const void* state_ptr, int player);
double* StateChanceOutcomeProbs(const void* state_ptr, int* size);
char* StateObservationString(const void* state_ptr,
                             unsigned long* length);  /* NOLINT */
char* StateInformationStateString(const void* state_ptr,
                                  unsigned long* length);  /* NOLINT */
int StateInformationStateTensorSize(const void* state_ptr);
int StateObservationTensorSize(const void* state_ptr);
void StateObservationTensor(const void* state_ptr, float* obs_buf, int length);
void StateInformationStateTensor(const void* state_ptr,
                                 float* infostate_buf, int length);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
