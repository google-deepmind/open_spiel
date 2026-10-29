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

import {frame} from '../testing/frames.ts';
import {boardText, missingTextReason, returnsSentence} from './board_text.ts';

const SEAT_1 = {kind: 'seat', seat: 1} as const;
const SPECTATOR = {kind: 'spectator', seat: null} as const;

// What a viewer that is not omniscient receives: no full state.
const hidden = {omniscient: false, state_text: null, legal_actions: null};

describe('boardText', () => {
  it('shows an omniscient viewer the full state, verbatim', () => {
    const text = ' x | o \n---+---\n   | x \n';
    expect(boardText(frame({state_text: text}))).toEqual({
      title: 'State',
      text,
    });
  });

  it('shows a seat of a game that hides nothing the full state', () => {
    const shown = frame({
      viewer: SEAT_1,
      state_text: 'x.o',
      observation_text: 'x.o',
    });
    expect(boardText(shown)).toEqual({title: 'State', text: 'x.o'});
  });

  it('shows an empty state as an empty text, not as no text', () => {
    expect(boardText(frame({state_text: ''}))).toEqual({
      title: 'State',
      text: '',
    });
  });

  it('shows a seat its information state in preference to its observation', () => {
    const shown = frame({
      ...hidden,
      viewer: SEAT_1,
      observation_text: '211',
      information_state_text: '2pb',
    });
    expect(boardText(shown)).toEqual({
      title: "Seat 1's information state",
      text: '2pb',
    });
  });

  it('shows a seat its observation when the game gives no information state', () => {
    const shown = frame({...hidden, viewer: SEAT_1, observation_text: '211'});
    expect(boardText(shown)).toEqual({
      title: "Seat 1's observation",
      text: '211',
    });
  });

  it('shows the full state only to a viewer the frame calls omniscient', () => {
    const shown = frame({
      viewer: SEAT_1,
      omniscient: false,
      state_text: 'x.o',
      observation_text: '..o',
    });
    expect(boardText(shown)).toEqual({
      title: "Seat 1's observation",
      text: '..o',
    });
  });

  it('shows nothing when the frame carries no text', () => {
    expect(boardText(frame({...hidden, viewer: SPECTATOR}))).toBeNull();
  });
});

describe('missingTextReason', () => {
  it('says that the game gives the seat no text', () => {
    expect(missingTextReason(frame({...hidden, viewer: SEAT_1}))).toBe(
      'The game gives seat 1 no text.',
    );
  });

  it('says why a spectator is shown no text', () => {
    expect(missingTextReason(frame({...hidden, viewer: SPECTATOR}))).toBe(
      'The game hides information, so a spectator is shown no text.',
    );
  });
});

describe('returnsSentence', () => {
  it('names each seat with its return', () => {
    expect(returnsSentence([1, -1, 0.5])).toBe(
      'Returns: seat 0: 1, seat 1: -1, seat 2: 0.5.',
    );
  });

  it('writes an integer return exactly', () => {
    expect(returnsSentence([12345678, -987654321012345])).toBe(
      'Returns: seat 0: 12345678, seat 1: -987654321012345.',
    );
  });

  it('writes a fractional return without floating-point noise', () => {
    expect(returnsSentence([0.1 + 0.2, 1e-7, -2 / 3])).toBe(
      'Returns: seat 0: 0.3, seat 1: 0.0000001, seat 2: -0.666666666666667.',
    );
  });
});
