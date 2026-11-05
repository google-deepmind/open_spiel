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

import {act} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import type {BotOutput, Frame, HistoryEntry} from '../contract/contract.ts';
import {frame} from '../testing/frames.ts';
import {
  BotOutputView,
  percent,
  shownOutput,
  signedValue,
} from './bot_output.tsx';

const OUTPUT: BotOutput = {
  policy: [
    {action: 7, label: 'Discard 7h', prob: 0.25},
    {action: 3, label: 'Discard Kd', prob: 0.7},
    {action: 9, label: 'Knock', prob: 0.05},
  ],
  value: 0.18,
  decided_by: 'network',
};

function entry(
  seat: number | null,
  action: number,
  output: BotOutput | null = null,
): HistoryEntry {
  return {
    seat,
    action,
    label: `move ${String(action)}`,
    rewards: [0, 0],
    public_label: null,
    bot_output: output,
  };
}

// A human move, the bot's move 3 with OUTPUT, then a human move.
const HISTORY: HistoryEntry[] = [entry(0, 1), entry(1, 3, OUTPUT), entry(0, 5)];

let container: HTMLElement;
let root: Root;

function show(shown: Frame | null): void {
  act(() => {
    root.render(<BotOutputView frame={shown} />);
  });
}

// Each policy row's label and probability, and whether it is the move played.
function rows(): Array<[string, string, boolean]> {
  return Array.from(container.querySelectorAll('li'), row => [
    row.querySelector('.bot-output__label')?.textContent ?? '',
    row.querySelector('.bot-output__prob')?.textContent ?? '',
    row.getAttribute('aria-current') === 'true',
  ]);
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
});

describe('BotOutputView', () => {
  it('lists the policy most probable first, the move played marked', () => {
    show(frame({history: HISTORY, ply: 2}));
    expect(rows()).toEqual([
      ['Discard Kd', '70.0%', true],
      ['Discard 7h', '25.0%', false],
      ['Knock', '5.0%', false],
    ]);
    expect(container.textContent).toContain('#2 · Seat 1 played move 3');
    expect(container.textContent).toContain('Decided bynetwork');
    expect(container.textContent).toContain('Value+0.180');
  });

  it('steps with the History: each position shows the latest bot move', () => {
    show(frame({history: HISTORY, ply: 1, num_plies: 3}));
    expect(rows()).toEqual([]);
    expect(container.textContent).toBe('No bot output up to this position.');
    show(frame({history: HISTORY, ply: 3, num_plies: 3}));
    expect(rows()).toHaveLength(3);
    expect(container.textContent).toContain('#2 · Seat 1');
  });

  it('shows the later of two bot moves once it is reached', () => {
    const later: BotOutput = {
      policy: [{action: 9, label: 'Knock', prob: 1}],
      value: null,
      decided_by: null,
    };
    const history = [...HISTORY, entry(1, 9, later)];
    show(frame({history, ply: 3}));
    expect(rows()[0]?.[0]).toBe('Discard Kd');
    show(frame({history, ply: 4}));
    expect(rows()).toEqual([['Knock', '100.0%', true]]);
    expect(container.textContent).toContain('Decided byNot given');
    expect(container.textContent).toContain('ValueNot given');
  });

  it('says when the bot gave no policy', () => {
    const output: BotOutput = {policy: [], value: -0.5, decided_by: 'forced'};
    show(frame({history: [entry(1, 2, output)], ply: 1}));
    expect(container.textContent).toContain(
      'The bot gave no policy for this move.',
    );
    expect(container.textContent).toContain('Value-0.500');
  });
});

describe('BotOutputView of a move withheld from the viewer', () => {
  it('tells a viewer who is not omniscient whom a bot output goes to', () => {
    // In a game that hides information, the server sends seat 0 no output
    // for the bot's move: it would tell of the bot's hand.
    const withheld: HistoryEntry = {
      seat: 1,
      action: null,
      label: null,
      rewards: null,
      public_label: '7h',
      bot_output: null,
    };
    const humans = frame({
      history: [entry(0, 1), withheld],
      ply: 2,
      viewer: {kind: 'seat', seat: 0},
      omniscient: false,
    });
    const withBot: Frame = {
      ...humans,
      seats: humans.seats.map(seat =>
        seat.seat === 1 ? {...seat, controller: 'remote'} : seat,
      ),
    };
    show(withBot);
    expect(rows()).toEqual([]);
    expect(container.textContent).toBe(
      "No bot output up to this position. In a game that hides information, a bot's output goes only to its own seat.",
    );
    // With no bot at the table, nothing is withheld, and the panel says
    // only that there is no output.
    show(humans);
    expect(container.textContent).toBe('No bot output up to this position.');
  });

  it('shows no position before a frame arrives', () => {
    show(null);
    expect(container.textContent).toBe('No position to show.');
  });
});

describe('helpers', () => {
  it('finds no output at the initial position', () => {
    expect(shownOutput(frame({history: HISTORY, ply: 0}))).toBeNull();
  });

  it('formats probabilities and values', () => {
    expect(percent(0.123)).toBe('12.3%');
    expect(signedValue(0.5)).toBe('+0.500');
    expect(signedValue(0)).toBe('0.000');
  });
});
