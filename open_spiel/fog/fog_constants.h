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

#ifndef OPEN_SPIEL_FOG_FOG_CONSTANTS_H_
#define OPEN_SPIEL_FOG_FOG_CONSTANTS_H_

// This file describes constants that are used for games that implement
// factored observations. See README.md for details.
namespace open_spiel {

// In the initial state of the game (root node of the world tree)
// all players receive a "dummy" public observation that the game just started.
// This corresponds to the initial observations $O^0$ from the FOG paper.
//
// This is automatically tested for the State::PublicObservationString() method.
inline const char* kStartOfGamePublicObservation = "start game";

// Imagine the following:
//
// All players sit in a room and play a game. There is a clock on the wall.
// Everyone can see this clock, and therefore perceive the flow of time.
// The clock ticking is a public observation: all players see that time goes on,
// and all players know that everyone knows this.
//
// Therefore, if there is no specific public observation other than clock
// ticking, the state should return this constant. However, if there is another
// observation, you do not need to encode the time. This is because it is
// then done implicitly through that specific observation: the length of
// the list of observations changes, indicating the implicit passing of time.
inline const char* kClockTickPublicObservation = "clock tick";

// Constant representing an invalid public observation.
//
// Empty string is an invalid public observation on purpose: players always
// receive some public observation. If nothing else, players perceive the flow
// of time: in this case, you should return kClockTickPublicObservation.
inline const char* kInvalidPublicObservation = "";

// Constant representing that player received no private observation.
// Perfect information games should return only this value in
// State::PrivateObservationString()
inline const char* kNothingPrivateObservation = "";

}  // namespace open_spiel

#endif  // OPEN_SPIEL_FOG_FOG_CONSTANTS_H_
