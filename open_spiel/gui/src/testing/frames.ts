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

// Frames and sessions built directly, for components that take a session as
// a property.

import {vi} from 'vitest';

import type {Action, Frame, HistoryEntry} from '../contract/contract.ts';
import type {SessionCommands} from '../session/session_store.ts';
import type {Session} from '../session/use_session.ts';

/** `count` actions with ids 0 up, labelled by `label`. */
export function actions(
  count: number,
  label: (id: number) => string = id => `action ${String(id)}`,
): Action[] {
  return Array.from({length: count}, (_, id) => ({id, label: label(id)}));
}

/** A visible history entry for each label, seats alternating from 0. */
export function entries(labels: readonly string[]): HistoryEntry[] {
  return labels.map((label, index) => ({
    seat: index % 2,
    action: index,
    label,
    rewards: [0, 0],
  }));
}

/**
 * The live frame of a two-seat game of human seats, shown to the oracle, with
 * seat 0 to move; `fields` replace any of its fields.
 */
export function frame(fields: Partial<Frame> = {}): Frame {
  const history = fields.history ?? [];
  return {
    contract_version: 2,
    session: {
      id: 'session-1',
      game: 'game',
      parameters: {},
      num_seats: 2,
      chance_policy: 'auto',
      turn_based: false,
      hides_information: false,
      seed: null,
      controllers: [
        {
          name: 'human',
          label: 'Human',
          config_schema: {},
          unavailable_reason: null,
        },
        {
          name: 'random',
          label: 'Uniform random',
          config_schema: {},
          unavailable_reason: null,
        },
      ],
    },
    viewer: {kind: 'oracle', seat: null},
    omniscient: true,
    status: 'playing',
    seats: [0, 1].map(seat => ({
      seat,
      controller: 'human',
      config: {},
      autoplay: false,
      failure: null,
    })),
    ply: history.length,
    num_plies: history.length,
    to_move: {kind: 'seat', seat: 0},
    legal_actions: actions(2),
    chance_outcomes: [],
    history,
    rewards: null,
    returns: null,
    state_text: '',
    observation_text: null,
    information_state_text: null,
    struct: null,
    ...fields,
  };
}

/** A session showing `shown`, whose commands are mocks. */
export function session(shown: Frame | null, busy = false): Session {
  const commands: SessionCommands = {
    create: vi.fn(),
    open: vi.fn(),
    refresh: vi.fn(),
    goTo: vi.fn(),
    step: vi.fn(),
    act: vi.fn(),
    chooseChance: vi.fn(),
    advance: vi.fn(),
    undo: vi.fn(),
    reset: vi.fn(),
    setSeat: vi.fn(),
  };
  return {
    frame: shown,
    busy,
    error: null,
    live: shown !== null && shown.ply === shown.num_plies,
    commands,
  };
}
