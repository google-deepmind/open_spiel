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

import type {HistoryEntry} from '../contract/contract.ts';
import {historyRows, rowText} from './history_rows.ts';

function hidden(seat: number | null): HistoryEntry {
  return {seat, action: null, label: null, rewards: null};
}

function shown(seat: number | null, label: string): HistoryEntry {
  return {seat, action: 0, label, rewards: null};
}

// Each row's two columns, joined by a space.
function texts(history: readonly HistoryEntry[]): string[] {
  return historyRows(history).map(row => {
    const {turn, label} = rowText(row);
    return `${turn} ${label}`;
  });
}

describe('historyRows', () => {
  it('starts at the initial position', () => {
    expect(texts([])).toEqual(['#0 Initial']);
  });

  it('gives each visible action its own row', () => {
    expect(texts([shown(0, 'e4'), shown(1, 'e5'), shown(0, 'e4')])).toEqual([
      '#0 Initial',
      '#1 e4',
      '#2 e5',
      '#3 e4',
    ]);
  });

  it('collapses a run of withheld actions by the same mover', () => {
    const deal = [hidden(null), hidden(null), hidden(null), hidden(null)];
    expect(texts([...deal, shown(0, 'Bid 1'), hidden(1), hidden(1)])).toEqual([
      '#0 Initial',
      '#1–4 Hidden chance outcome ×4',
      '#5 Bid 1',
      '#6–7 Hidden action of seat 1 ×2',
    ]);
  });

  it('keeps withheld actions by different movers apart', () => {
    expect(texts([hidden(0), hidden(1), hidden(null)])).toEqual([
      '#0 Initial',
      '#1 Hidden action of seat 0',
      '#2 Hidden action of seat 1',
      '#3 Hidden chance outcome',
    ]);
  });

  it('never folds a withheld action into a visible one worded alike', () => {
    const lookalike = shown(null, 'Hidden chance outcome');
    expect(texts([lookalike, hidden(null)])).toEqual([
      '#0 Initial',
      '#1 Hidden chance outcome',
      '#2 Hidden chance outcome',
    ]);
  });

  it('puts the span of a collapsed row in its turn column', () => {
    expect(
      rowText({first: 1, last: 3, label: 'Hidden chance outcome'}),
    ).toEqual({turn: '#1–3', label: 'Hidden chance outcome ×3'});
  });

  it('spans the plies of a collapsed row', () => {
    expect(historyRows([hidden(null), hidden(null)])).toEqual([
      {first: 0, last: 0, label: 'Initial'},
      {first: 1, last: 2, label: 'Hidden chance outcome'},
    ]);
  });
});
