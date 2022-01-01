#ifndef __RUST_OPEN_SPIEL_H__
#define __RUST_OPEN_SPIEL_H__

/* A pure C API that wraps the C++ OpenSpiel core. */

#ifdef __cplusplus
extern "C" {
#endif

void test();

/* Game functions. */
void* LoadGame(const char* name);
void DeleteGame(void* game_ptr);
char* GameShortName(const void* game_ptr);
char* GameLongName(const void* game_ptr);
void* GameNewInitialState(const void* game_ptr);
int GameNumPlayers(const void* game_ptr);
int GameMaxGameLength(const void* game_ptr);
int GameNumDistinctActions(const void* game_ptr);
int* GameObservationTensorShape(const void* game_ptr, int* size);
int* GameInformationStateTensorShape(const void* game_ptri, int* size);

/* State functions. */
void DeleteState(void* state_ptr);
void* StateClone(const void* state_ptr);
char* StateToString(const void* state_ptr);
long* StateLegalActions(const void* state_ptr, int* num_legal_actions);
int StateCurrentPlayer(const void* state_ptr);
char* StateActionToString(const void* state_ptr, int player, long action);
int StateIsTerminal(const void* state_ptr);
int StateIsChanceNode(const void* state_ptr);
int StateNumPlayers(const void* state_ptr);
void StateApplyAction(void* state_ptr, long action);
double* StateReturns(const void* state_ptr);
double StatePlayerReturn(const void* state_ptr, int player);
double* StateChanceOutcomeProbs(const void* state_ptr, int* size);
char* StateObservationString(const void* state_ptr);
char* StateInformationStateString(const void* state_ptr);
float* StateObservationTensor(const void* state_ptr, int *size);
float* StateInformationStateTensor(const void* state_ptr, int *size);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif
