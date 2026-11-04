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

// A catalogue entry and frames of "board", a made-up game whose parameters
// cover every parameter type and kind.

import type {
  CatalogueEntry,
  Controller,
  Frame,
  ParameterSpec,
  Viewer,
} from '../../contract/contract.ts';

const FREE = {
  kind: 'free',
  minimum: null,
  maximum: null,
  choices: null,
} as const;

/** A whole-number parameter within a range, with a description. */
export const BOARD_SIZE: ParameterSpec = {
  name: 'board_size',
  type: 'int',
  kind: 'range',
  default: 9,
  minimum: 5,
  maximum: 19,
  choices: null,
  description: 'The width of the board.',
};
/** A number parameter with no bounds. */
export const KOMI: ParameterSpec = {
  ...FREE,
  name: 'komi',
  type: 'float',
  default: 7.5,
  description: null,
};
/** A whole-number parameter with no bounds. */
export const PLAYERS: ParameterSpec = {
  ...FREE,
  name: 'players',
  type: 'int',
  default: 2,
  description: null,
};
/** A text parameter that takes one of a list of values. */
export const RULES: ParameterSpec = {
  name: 'rules',
  type: 'string',
  kind: 'choice',
  default: 'area',
  minimum: null,
  maximum: null,
  choices: ['area', 'territory'],
  description: null,
};
/** A true-or-false parameter, with a description. */
export const SUICIDE: ParameterSpec = {
  ...FREE,
  name: 'suicide',
  type: 'bool',
  default: false,
  description: 'Whether a move may capture its own stones.',
};
/** A text parameter that takes any text. */
export const TITLE: ParameterSpec = {
  ...FREE,
  name: 'title',
  type: 'string',
  default: 'Board',
  description: null,
};

/** The catalogue entry of "board", listing every parameter above. */
export const ENTRY: CatalogueEntry = {
  name: 'board',
  long_name: 'Board',
  dynamics: 'sequential',
  chance_mode: 'deterministic',
  information: 'perfect_information',
  min_seats: 2,
  max_seats: 4,
  num_players: 2,
  parameters: [BOARD_SIZE, KOMI, PLAYERS, RULES, SUICIDE, TITLE],
  loadable: true,
  unavailable_reason: null,
  controllers: ['human', 'random'],
  renderer: null,
};

/** The values ENTRY's parameters take by default. */
export const DEFAULTS = {
  board_size: 9,
  komi: 7.5,
  players: 2,
  rules: 'area',
  suicide: false,
  title: 'Board',
};

/** What a frame of "board" may differ in. */
export interface FrameOptions {
  readonly viewer?: Viewer;
  readonly hidesInformation?: boolean;
  readonly parameters?: Frame['session']['parameters'];
  readonly history?: readonly number[];
  readonly id?: string;
}

/** A controller with no settings that can play "board". */
function controller(name: string, label: string): Controller {
  return {name, label, config_schema: {}, unavailable_reason: null};
}

/** A live frame of "board" after `history`, with seat 1 played at random. */
export function boardFrame({
  viewer = {kind: 'oracle', seat: null},
  hidesInformation = false,
  parameters = DEFAULTS,
  history = [],
  id = 'session-1',
}: FrameOptions = {}): Frame {
  const omniscient = viewer.kind === 'oracle' || !hidesInformation;
  return {
    contract_version: 8,
    session: {
      id,
      game: 'board',
      parameters: {...parameters},
      num_seats: 2,
      chance_policy: 'expose',
      turn_based: false,
      hides_information: hidesInformation,
      seed: viewer.kind === 'oracle' ? 3 : null,
      controllers: [
        controller('human', 'Human'),
        controller('random', 'Random'),
      ],
    },
    viewer,
    omniscient,
    status: 'playing',
    seats: [
      {
        seat: 0,
        controller: 'human',
        config: {},
        autoplay: false,
        failure: null,
        status: 'ready',
        message: null,
        notice: null,
      },
      {
        seat: 1,
        controller: 'random',
        config: {depth: 2},
        autoplay: true,
        failure: null,
        status: 'ready',
        message: null,
        notice: null,
      },
    ],
    ply: history.length,
    num_plies: history.length,
    to_move: {kind: 'seat', seat: history.length % 2},
    legal_actions: omniscient ? [{id: 0, label: 'pass'}] : null,
    chance_outcomes: omniscient ? [] : null,
    history: history.map((action, ply) => ({
      seat: ply % 2,
      public_label: null,
      ...(omniscient
        ? {action, label: String(action), rewards: [0, 0]}
        : {action: null, label: null, rewards: null}),
    })),
    rewards: history.length === 0 ? null : omniscient ? [0, 0] : null,
    returns: null,
    state_text: omniscient ? 'board' : null,
    observation_text: viewer.kind === 'seat' ? 'board' : null,
    information_state_text: null,
    struct: null,
    observation_struct: null,
    presentation: null,
  };
}
