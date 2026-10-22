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

import {describe, expect, it} from 'vitest';

import type {Frame} from '../contract/contract.ts';
import {entries, frame} from '../testing/frames.ts';
import {
  controllerToMove,
  seatToHandOver,
  statusLine,
} from './gameplay_state.ts';

// A frame whose seat 1 is played by `controller`, with seat 1 to move.
function seatOneToMove(controller: string, fields: Partial<Frame> = {}): Frame {
  const shown = frame({to_move: {kind: 'seat', seat: 1}, ...fields});
  return {
    ...shown,
    seats: shown.seats.map(seat =>
      seat.seat === 1 ? {...seat, controller} : seat,
    ),
  };
}

function hidden(shown: Frame, seat: number): Frame {
  return {
    ...shown,
    session: {...shown.session, hides_information: true},
    viewer: {kind: 'seat', seat},
    omniscient: false,
  };
}

describe('controllerToMove', () => {
  it('names the seat a controller plays when it is to move', () => {
    expect(controllerToMove(seatOneToMove('random'))).toBe(1);
  });

  it('is null when a person is to move', () => {
    expect(controllerToMove(seatOneToMove('human'))).toBeNull();
  });

  it.each<[string, Partial<Frame>]>([
    ['at an earlier position', {ply: 0, num_plies: 1}],
    ['while paused', {status: 'paused'}],
    ['once the game is over', {status: 'terminal'}],
  ])('is null %s', (_, fields) => {
    expect(controllerToMove(seatOneToMove('random', fields))).toBeNull();
  });

  it('refuses a mover that is not seated', () => {
    const shown = frame({to_move: {kind: 'seat', seat: 5}});
    expect(() => controllerToMove(shown)).toThrow('not seated');
  });
});

describe('seatToHandOver', () => {
  it('names another person’s seat to move in a game that hides information', () => {
    expect(seatToHandOver(hidden(seatOneToMove('human'), 0))).toBe(1);
  });

  it('is null when the viewer’s own seat is to move', () => {
    expect(seatToHandOver(hidden(seatOneToMove('human'), 1))).toBeNull();
  });

  it('is null when a controller is to move', () => {
    expect(seatToHandOver(hidden(seatOneToMove('random'), 0))).toBeNull();
  });

  it('is null when the game hides nothing', () => {
    const shown: Frame = {
      ...seatOneToMove('human'),
      viewer: {kind: 'seat', seat: 0},
    };
    expect(seatToHandOver(shown)).toBeNull();
  });

  it('is null for the oracle', () => {
    const shown = hidden(seatOneToMove('human'), 0);
    expect(
      seatToHandOver({...shown, viewer: {kind: 'oracle', seat: null}}),
    ).toBeNull();
  });
});

describe('statusLine', () => {
  it('gives the turn and who is to move', () => {
    expect(statusLine(frame())).toBe('Turn 0 · Player 0');
    expect(statusLine(frame({to_move: {kind: 'chance', seat: null}}))).toBe(
      'Turn 0 · Chance',
    );
  });

  it('says a controller is to play, autoplaying or not', () => {
    const waiting = seatOneToMove('random');
    expect(statusLine(waiting)).toBe(
      'Turn 0 · Player 1 · Waiting for bot step…',
    );
    const autoplaying = {
      ...waiting,
      seats: waiting.seats.map(seat => ({...seat, autoplay: seat.seat === 1})),
    };
    // Autoplay that left its seat to move has stopped, so it waits too.
    expect(statusLine(autoplaying)).toBe(
      'Turn 0 · Player 1 · Waiting for bot step…',
    );
  });

  it('names the viewer when it is a seat or a spectator', () => {
    expect(statusLine(frame({viewer: {kind: 'seat', seat: 0}}))).toBe(
      'Turn 0 · Player 0 · Seat: Player 0',
    );
    expect(statusLine(frame({viewer: {kind: 'spectator', seat: null}}))).toBe(
      'Turn 0 · Player 0 · Spectating',
    );
  });

  it('gives the returns at the end, and the turn of the position shown', () => {
    const over = frame({
      history: entries(['a', 'b']),
      status: 'terminal',
      to_move: {kind: 'terminal', seat: null},
      returns: [1, -1],
    });
    expect(statusLine(over)).toBe(
      'Turn 2 · Terminal · Returns: seat 0: 1, seat 1: -1.',
    );
    expect(statusLine(frame({history: entries(['a', 'b']), ply: 1}))).toBe(
      'Turn 1 · Player 0',
    );
  });

  it('says a game is simultaneous, and that a session is paused', () => {
    const shown = frame({status: 'paused'});
    expect(
      statusLine({...shown, session: {...shown.session, turn_based: true}}),
    ).toBe('Turn 0 · Player 0 · Paused · Simultaneous, one seat at a time');
  });
});
