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

import type {Frame, SeatView, Viewer} from '../contract/contract.ts';
import {entries, frame} from '../testing/frames.ts';
import {playBlocker} from './play_blocker.ts';

const SEAT_0: Viewer = {kind: 'seat', seat: 0};
const SEAT_1: Viewer = {kind: 'seat', seat: 1};
const SPECTATOR: Viewer = {kind: 'spectator', seat: null};

// The live frame with seat 1 played by the random controller.
function withRandomSeat1(fields: Partial<Frame>): Frame {
  const base = frame(fields);
  const seats: SeatView[] = base.seats.map(seat =>
    seat.seat === 1 ? {...seat, controller: 'random'} : seat,
  );
  return {...base, seats};
}

describe('playBlocker', () => {
  it('lets the oracle and the seat to move play a human seat', () => {
    expect(playBlocker(frame())).toBeNull();
    expect(playBlocker(frame({viewer: SEAT_0}))).toBeNull();
  });

  it('keeps a seat that is not to move waiting', () => {
    expect(playBlocker(frame({viewer: SEAT_1}))).toBe('Waiting for your turn…');
  });

  it('never lets a spectator play', () => {
    expect(playBlocker(frame({viewer: SPECTATOR}))).toBe(
      'Spectating. Claim a seat to act.',
    );
  });

  it('names the controller of a seat a person does not play', () => {
    const mover = {kind: 'seat', seat: 1} as const;
    expect(playBlocker(withRandomSeat1({to_move: mover, viewer: SEAT_1}))).toBe(
      'Seat controlled by Uniform random.',
    );
    expect(playBlocker(withRandomSeat1({to_move: mover, viewer: SEAT_0}))).toBe(
      'Waiting for opponent…',
    );
    expect(playBlocker(withRandomSeat1({to_move: mover}))).toBe(
      'Waiting for opponent…',
    );
  });

  it('falls back to the name of a controller the session does not list', () => {
    const base = frame({to_move: {kind: 'seat', seat: 1}, viewer: SEAT_1});
    const seats = base.seats.map(seat => ({...seat, controller: 'mystery'}));
    expect(playBlocker({...base, seats})).toBe('Seat controlled by mystery.');
  });
});

describe('playBlocker at an earlier position', () => {
  it('lets only an omniscient viewer play from an earlier position', () => {
    const history = entries(['a', 'b']);
    expect(playBlocker(frame({history, ply: 1}))).toBeNull();
    expect(
      playBlocker(frame({history, ply: 1, omniscient: false, viewer: SEAT_0})),
    ).toBe('Viewing an earlier position. Go to the live position to play.');
  });

  it('plays from an earlier position of a paused session', () => {
    const history = entries(['a', 'b']);
    const paused = frame({history, ply: 0, status: 'paused'});
    const seats = paused.seats.map(seat =>
      seat.seat === 1
        ? {...seat, failure: {error: 'ValueError', message: null}}
        : seat,
    );
    expect(playBlocker({...paused, seats})).toBeNull();
  });
});

describe('playBlocker where nobody plays', () => {
  it('says why nobody plays at an ended, paused or chance position', () => {
    expect(playBlocker(frame({to_move: {kind: 'terminal', seat: null}}))).toBe(
      'Game finished. No legal actions.',
    );
    expect(playBlocker(frame({to_move: {kind: 'chance', seat: null}}))).toBe(
      'Chance to move.',
    );
    const paused = withRandomSeat1({status: 'paused'});
    const seats = paused.seats.map(seat =>
      seat.seat === 1
        ? {...seat, failure: {error: 'ValueError', message: null}}
        : seat,
    );
    expect(playBlocker({...paused, seats})).toBe(
      'Paused: the controller of seat 1 failed.',
    );
  });

  it('refuses a frame whose mover has no seat', () => {
    expect(() =>
      playBlocker(frame({to_move: {kind: 'seat', seat: 5}})),
    ).toThrow('Seat 5 is to move but not seated.');
    const unnamed = {kind: 'seat', seat: null} as const;
    expect(() => playBlocker(frame({to_move: unnamed}))).toThrow(
      'A seat is to move but unnamed.',
    );
    const replay = frame({history: entries(['a']), ply: 0, to_move: unnamed});
    expect(() => playBlocker(replay)).toThrow('A seat is to move but unnamed.');
  });

  it('takes the human controller to be the first the session lists', () => {
    const base = frame({viewer: SEAT_0});
    const controllers = [...base.session.controllers].reverse();
    const reordered = {...base, session: {...base.session, controllers}};
    expect(playBlocker(reordered)).toBe('Seat controlled by Human.');
  });
});
