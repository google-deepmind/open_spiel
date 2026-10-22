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

import {formatHistory, historyActions, parseHistory} from './position_text.ts';
import {boardFrame} from './testing/fixtures.ts';

describe('parseHistory', () => {
  it.each([
    ['0 4 8', [0, 4, 8]],
    ['0,4,8', [0, 4, 8]],
    [' [0, 4,\n8] ', [0, 4, 8]],
    ['[ 12 ]', [12]],
    ['[]', []],
    ['{"history": [1, 4, 9]}', [1, 4, 9]],
    ['{"actions": [2]}', [2]],
    ['{"sequence": []}', []],
  ])('reads %j', (text, history) => {
    expect(parseHistory(text)).toEqual({ok: true, history});
  });

  it.each([
    ['', 'Enter the action ids to replay.'],
    ['[ , ]', 'Enter the action ids to replay.'],
    ['[0, 4', 'Close the brackets around the actions.'],
    ['0 4]', 'Close the brackets around the actions.'],
    ['0 -4', '-4 is not an action id.'],
    ['0 4.5', '4.5 is not an action id.'],
    ['0 e4', 'e4 is not an action id.'],
    ['99999999999999999999', '99999999999999999999 is not an action id.'],
    ['{"history": [1, "x"]}', '"x" is not an action id.'],
    [
      '{"moves": [1]}',
      'The object holds no history, actions or sequence list.',
    ],
    ['{history: [1]}', 'The object is not valid JSON.'],
  ])('refuses %j', (text, problem) => {
    expect(parseHistory(text)).toEqual({ok: false, problem});
  });

  it('reads back the histories it writes', () => {
    expect(formatHistory([0, 4, 8])).toBe('[0, 4, 8]');
    expect(parseHistory(formatHistory([3, 1]))).toEqual({
      ok: true,
      history: [3, 1],
    });
    expect(parseHistory(formatHistory([]))).toEqual({ok: true, history: []});
  });
});

describe('historyActions', () => {
  it('reads the actions of a history shown in full', () => {
    expect(historyActions(boardFrame({history: [3, 1, 4]}))).toEqual([3, 1, 4]);
  });

  it('refuses a history that withholds an action', () => {
    const frame = boardFrame({
      viewer: {kind: 'seat', seat: 0},
      hidesInformation: true,
      history: [3],
    });
    expect(() => historyActions(frame)).toThrow('The history withholds ply 0.');
  });
});
