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

import type {Action} from '../contract/contract.ts';
import {
  OVERSCAN_ROWS,
  ROW_STRIDE_PX,
  clampScrollTop,
  duplicateLabelIds,
  scrollTopShowing,
  sortById,
  windowRange,
} from './action_list.ts';

function labelled(...labels: string[]): Action[] {
  return labels.map((label, id) => ({id, label}));
}

describe('sortById', () => {
  it('orders by id and leaves its input alone', () => {
    const given = [
      {id: 3, label: 'c'},
      {id: 1, label: 'a'},
      {id: 2, label: 'b'},
    ];
    expect(sortById(given).map(action => action.id)).toEqual([1, 2, 3]);
    expect(given.map(action => action.id)).toEqual([3, 1, 2]);
  });
});

describe('duplicateLabelIds', () => {
  it('names every action whose label is shared', () => {
    const all = labelled('Draw', 'Play', 'Draw', 'Pass', 'Draw', 'Play2');
    expect([...duplicateLabelIds(all)].sort()).toEqual([0, 2, 4]);
  });

  it('is empty when every label differs', () => {
    expect(duplicateLabelIds(labelled('a', 'b')).size).toBe(0);
  });
});

describe('windowRange', () => {
  const view = 10 * ROW_STRIDE_PX;

  it('renders the rows in view and the overscan below at the top', () => {
    expect(windowRange(400, 0, view)).toEqual({
      start: 0,
      end: 10 + OVERSCAN_ROWS,
    });
  });

  it('renders the overscan on both sides in the middle', () => {
    expect(windowRange(400, 100 * ROW_STRIDE_PX, view)).toEqual({
      start: 100 - OVERSCAN_ROWS,
      end: 110 + OVERSCAN_ROWS,
    });
  });

  it('counts a partly visible row as in view', () => {
    const {start, end} = windowRange(400, 100 * ROW_STRIDE_PX + 1, view);
    expect([start, end]).toEqual([100 - OVERSCAN_ROWS, 111 + OVERSCAN_ROWS]);
  });

  it('treats a scrollTop past the end as the last full view', () => {
    expect(windowRange(50, 400 * ROW_STRIDE_PX, view)).toEqual({
      start: 40 - OVERSCAN_ROWS,
      end: 50,
    });
  });

  it('renders the whole of a list shorter than its view', () => {
    expect(windowRange(5, 0, view)).toEqual({start: 0, end: 5});
    expect(windowRange(0, 0, view)).toEqual({start: 0, end: 0});
  });

  it('renders only the overscan in a view not yet measured', () => {
    expect(windowRange(400, 0, 0)).toEqual({start: 0, end: OVERSCAN_ROWS});
  });

  it('refuses a count, scrollTop or view it cannot place', () => {
    expect(() => windowRange(-1, 0, view)).toThrow('whole number');
    expect(() => windowRange(2.5, 0, view)).toThrow('whole number');
    expect(() => windowRange(10, Number.NaN, view)).toThrow('No window');
    expect(() => windowRange(10, 0, -1)).toThrow('No window');
    expect(() => windowRange(10, 0, Number.NaN)).toThrow('No window');
  });
});

describe('clampScrollTop', () => {
  it('keeps a scrollTop inside the list', () => {
    const view = 10 * ROW_STRIDE_PX;
    expect(clampScrollTop(100, -5, view)).toBe(0);
    expect(clampScrollTop(100, 3 * ROW_STRIDE_PX, view)).toBe(
      3 * ROW_STRIDE_PX,
    );
    expect(clampScrollTop(100, 1e9, view)).toBe(90 * ROW_STRIDE_PX);
    expect(clampScrollTop(5, 1e9, view)).toBe(0);
  });
});

describe('scrollTopShowing', () => {
  const view = 10 * ROW_STRIDE_PX;
  const at = 50 * ROW_STRIDE_PX;

  it('leaves a row already in view where it is', () => {
    expect(scrollTopShowing(50, at, view)).toBe(at);
    expect(scrollTopShowing(59, at, view)).toBe(at);
  });

  it('scrolls a row above the view to its top', () => {
    expect(scrollTopShowing(49, at, view)).toBe(49 * ROW_STRIDE_PX);
  });

  it('scrolls a row below the view to its bottom', () => {
    expect(scrollTopShowing(60, at, view)).toBe(51 * ROW_STRIDE_PX);
  });
});
