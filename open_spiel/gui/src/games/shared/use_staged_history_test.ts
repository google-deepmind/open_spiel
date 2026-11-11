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

// The seat's move and the bot's reply, arriving in one frame, are drawn one
// after the other; a finished two-seat board says who won.
import {describe, expect, it} from 'vitest';

import type {OpenSpielState} from './board_session.ts';
import {alternatingStagedState, boardResultText} from './use_staged_history.ts';

const LIVE: OpenSpielState = {
  history: [4, 0, 8],
  historyStr: 'x(1,1) o(0,0) x(2,2)',
  currentPlayer: 1,
  currentPlayerLabel: 'player 1',
  isTerminal: false,
  legalActions: [{action: 1, label: 'o(0,1)'}],
  returns: null,
  turnNumber: 3,
};

describe('alternatingStagedState', () => {
  it('is the live state once every action is drawn', () => {
    expect(alternatingStagedState(LIVE, 3)).toBe(LIVE);
  });

  it('is the position after the seat move, the bot to move, nothing legal', () => {
    const staged = alternatingStagedState(LIVE, 2);
    expect(staged.history).toEqual([4, 0]);
    expect(staged.historyStr).toBe('x(1,1) o(0,0)');
    expect(staged.currentPlayer).toBe(0);
    expect(staged.legalActions).toEqual([]);
    expect(staged.turnNumber).toBe(2);
  });

  it('holds back the result of a finished game until its last move', () => {
    const over = {
      ...LIVE,
      isTerminal: true,
      returns: [1, -1],
      currentPlayer: -4,
    };
    const staged = alternatingStagedState(over, 2);
    expect(staged.isTerminal).toBe(false);
    expect(staged.returns).toBeNull();
  });
});

describe('boardResultText', () => {
  it('names the winning side, and the viewer win or loss', () => {
    expect(boardResultText([1, -1], ['X', 'O'], null)).toBe('X wins');
    expect(boardResultText([-1, 1], ['Red', 'Yellow'], 0)).toBe(
      'Yellow wins · You lose',
    );
    expect(boardResultText([1, -1], ['X', 'O'], 0)).toBe('X wins · You win');
  });

  it('says Draw, never the raw returns', () => {
    expect(boardResultText([0, 0], ['X', 'O'], 1)).toBe('Draw');
  });

  it('fails loud on returns for another seat count', () => {
    expect(() => boardResultText([1], ['X', 'O'], null)).toThrow(/1 returns/);
  });
});
