// Copyright 2026 DeepMind Technologies Limited
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

// Controllers, seats and frames for the Players tests. The controllers are
// the ones the server registers, with the schemas it gives them, plus one
// whose settings no form can show.

import type {Controller, Frame, SeatView} from '../../contract/contract.ts';

/** The person at the board, a controller with no settings. */
export const HUMAN_CONTROLLER: Controller = {
  name: 'human',
  label: 'Human',
  config_schema: {},
  unavailable_reason: null,
};
/** The uniformly random bot, a controller with no settings. */
export const RANDOM: Controller = {
  name: 'random',
  label: 'Random',
  config_schema: {},
  unavailable_reason: null,
};
/** The MCTS bot, whose settings are a whole number, a number and a box. */
export const MCTS: Controller = {
  name: 'mcts',
  label: 'MCTS',
  config_schema: {
    type: 'object',
    properties: {
      max_simulations: {
        type: 'integer',
        default: 1000,
        description: 'Simulations per move, each ending in a random rollout.',
        minimum: 1,
        maximum: 10000,
      },
      uct_c: {
        type: 'number',
        default: 2,
        description: 'The exploration constant of the UCT selection rule.',
        minimum: 0,
        maximum: 100,
      },
      solve: {
        type: 'boolean',
        default: true,
        description:
          'Whether proven wins and losses are backed up the search tree.',
      },
    },
    additionalProperties: false,
  },
  unavailable_reason: null,
};
/**
 * A controller whose one setting, a list of moves, no form can show: its
 * `items` keyword is a constraint a flat form cannot enforce.
 */
export const SCRIPTED: Controller = {
  name: 'scripted',
  label: 'Scripted',
  config_schema: {
    type: 'object',
    properties: {
      moves: {type: 'array', items: {type: 'integer'}, default: []},
    },
  },
  unavailable_reason: null,
};
/**
 * A bot that plays a trained checkpoint, as a plugin declares it: a
 * required text setting with no default, and two settings with defaults.
 */
export const CHECKPOINT: Controller = {
  name: 'checkpoint',
  label: 'Checkpoint',
  config_schema: {
    type: 'object',
    properties: {
      checkpoint: {
        type: 'string',
        description: 'The checkpoint to play.',
        minLength: 1,
      },
      deterministic: {
        type: 'boolean',
        default: true,
        description: 'Whether it plays its most likely move.',
      },
      temperature: {
        type: 'number',
        default: 1,
        description: 'The temperature it samples its moves at.',
        minimum: 0,
      },
    },
    additionalProperties: false,
    required: ['checkpoint'],
  },
  unavailable_reason: null,
};
/** Every controller above, in the order a frame offers them. */
export const CONTROLLERS = [
  HUMAN_CONTROLLER,
  RANDOM,
  MCTS,
  SCRIPTED,
  CHECKPOINT,
];

/**
 * A seat played by `controller`. Like the server, it holds the settings it
 * was given, none unless `overrides` gives some, and not their defaults.
 */
export function seatView(
  seat: number,
  controller: Controller = HUMAN_CONTROLLER,
  overrides: Partial<SeatView> = {},
): SeatView {
  return {
    seat,
    controller: controller.name,
    config: {},
    autoplay: false,
    failure: null,
    status: 'ready',
    message: null,
    ...overrides,
  };
}

/** A live frame of a two-seat game, as `overrides` change it. */
export function frameWith(
  seats: SeatView[],
  overrides: Partial<Frame> = {},
): Frame {
  return {
    contract_version: 7,
    session: {
      id: 'session-1',
      game: 'kuhn_poker',
      parameters: {},
      num_seats: seats.length,
      chance_policy: 'auto',
      turn_based: false,
      hides_information: true,
      seed: 7,
      controllers: CONTROLLERS,
    },
    viewer: {kind: 'oracle', seat: null},
    omniscient: true,
    status: 'playing',
    seats,
    ply: 0,
    num_plies: 0,
    to_move: {kind: 'seat', seat: 0},
    legal_actions: [],
    chance_outcomes: [],
    history: [],
    rewards: null,
    returns: null,
    state_text: '',
    observation_text: null,
    information_state_text: null,
    struct: null,
    observation_struct: null,
    presentation: null,
    ...overrides,
  };
}
