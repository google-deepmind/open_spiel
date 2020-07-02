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

#ifndef OPEN_SPIEL_FOG_CONSTANTS_H_
#define OPEN_SPIEL_FOG_CONSTANTS_H_

// This file describes constants that are used for games that implement
// factored observations. We use them for Factored-Observation Games
// (FOGs) [1], which are games where players can distinguish between so-called
// public and private observations. The implementation tries to follow the
// formalism in [1] closely, but there are some important differences, which
// will be noted below.
//
// The public observations correspond to the information that all the players
// know that all the players know, like upward-facing cards on a table or
// passing of time (clock ticking).
// The private observation is then the remaining information, distinct from
// the public observation, with the exception of the (shared) observations
// of start of the game and passing of time. Thus private and public
// observations are almost non-overlaping (except for the mentioned start of the
// game or time) and their union corresponds to the (undistinguished)
// player observation available as State::ObservationString(). While there is
// no equality operator defined between the union and ObservationString, they
// should partition the game tree in the same way.
//
// The public / private observations can be accessed by
// State::PublicObservationString() and by
// State::PrivateObservationString(Player).
// Currently Tensor versions are not provided.
//
// There are some subtle, but important differences from FOGs:
//
// - In FOGs, there is a concept of a "world state", which represents the true
//   current state of the world from a perspective of an omniscient being
//   (that knows everything). The players may not be certain which world state
//   they are currently in, as they only receive observations in the game, and
//   they have to deduce what world states are compatible with these
//   observations.
//   In OpenSpiel, the abstract class State tracks the (omniscient) history
//   of actions of all players, but this history is not what defines the state:
//   it's the variables inside the derived State object (for each game).
//   There may be two distinct action histories that represent the same world
//   state.
// - In FOGs players take joint actions, and each player has at least one
//   action. In OpenSpiel, there is a version of simultaneous-move games, which
//   does this. The sequential game can be interpreted as simultaneous-move game
//   where all the other players just played a "no-op" action.
// - However, after the joint actions are taken, in FOGs there is always a
//   transition function to new world states, which encodes stochasticity, and
//   is deterministic if there is only one outcome. In OpenSpiel, this is done
//   via explicit chance `State`.
// - In FOGs the players receive observations only after the stochastic
//   transition. In OpenSpiel we can ask for the newest observations at any
//   `State`: a player or terminal `State`, and importantly also at chance
//   `State`.
// - In FOGs the "world history" is encoded as a sequence of tuples
//   (world, joint actions), because the stochastic transition can put the
//   players into different world states. In OpenSpiel, the equivalent (action)
//   history is sufficient, because chance player also contributes its action to
//   the action history.
// - In FOGs the imperfect information is encoded by using Action-Observation
//   histories (AOH), where Observation here represents the joint private and
//   public observations. In OpenSpiel, besides AOH there is also an
//   "information state", which is a non-factored variant of AOH (it is not
//   factored into the respective actions and observations). Both of these ways
//   of representation have a string and tensor representation, and they
//   partition the game tree in the same way.
//
// The factored observations can be used to deduce observations (as
// concatenation of private and public observations). Observations can be used
// to deduce information states (as a concatenation of AOH). Therefore factored
// observations are the most general formalism to describe imperfect information
// games supported by OpenSpiel. They are also used for automatic generation
// of public state API via game transformations, see public_states/README.md
// for more details.
//
// [1] https://arxiv.org/abs/1906.11110
namespace open_spiel {

// In the initial state of the game (root node of the world tree)
// all players receive a "dummy" observation that the game just started.
// This corresponds to the initial observations $O^0$ from the FOG paper [1].
//
// This is tested for both private / public observations methods.
inline const char* kStartOfGameObservation = "start game";

// Imagine the following:
//
// All players sit in a room and play a game. There is a clock on the wall.
// Everyone can see this clock, and therefore perceive the flow of time.
// The clock ticking is a public observation: all players see that time goes on,
// and all players are aware of this fact.
//
// Therefore, if there is no other public observation than clock ticking,
// the state should return this constant. However, if there is another
// observation, you do not need to encode the time. This is because it is
// then done implicitly through that observation: the length of the list of
// observations changes, indicating the implicit passing of time.
//
// This "implicit-coding-of-time-through-observations" also happens for private
// observations. For technical reasons, the state should return this value
// for private observations as well, if there is no other private observation
// available. We do this to make sure that there is no discrepancy between
// lengths of the lists of private observations between different histories.
inline const char* kClockTickObservation = "clock tick";

}  // namespace open_spiel

#endif  // OPEN_SPIEL_FOG_CONSTANTS_H_
