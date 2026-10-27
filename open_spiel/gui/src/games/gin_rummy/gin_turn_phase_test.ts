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

// The turn line's phase: what the player can do.

import {describe, expect, it} from 'vitest';

import {turnPhaseLabel} from './gin_derived_parse.ts';
import type {GinState, LegalAction, SessionPayload} from './gin_session.ts';
import {deriveGinBase} from './use_gin_rummy_derived_state.ts';

const DRAW_STOCK: LegalAction = {action: 53, label: 'Draw stock'};
const DRAW_UPCARD: LegalAction = {action: 52, label: 'Draw upcard'};
const PASS: LegalAction = {action: 54, label: 'Pass'};

function session(
  phase: string,
  legalActions: readonly LegalAction[],
): SessionPayload {
  const state: GinState = {
    struct: {phase, current_player: 'Player_0'},
    currentPlayerLabel: null,
    text: '',
    lastAction: null,
    turnNumber: 3,
    isTerminal: false,
    legalActions,
    returns: null,
    historyPlayers: [],
    historyActionStrings: [],
  };
  return {sessionId: 's', mode: 'game', state, players: []};
}

describe('turnPhaseLabel', () => {
  it('says to draw from the stock when Draw stock is the only legal move in FirstUpcard', () => {
    expect(turnPhaseLabel('FirstUpcard', [DRAW_STOCK])).toBe(
      'Draw from the stock',
    );
  });

  it('says First upcard while the upcard can still be taken or passed', () => {
    expect(turnPhaseLabel('FirstUpcard', [DRAW_UPCARD, PASS])).toBe(
      'First upcard',
    );
  });

  it("names the engine's phase otherwise", () => {
    expect(turnPhaseLabel('Draw', [DRAW_UPCARD, DRAW_STOCK])).toBe('Draw');
    expect(turnPhaseLabel('Discard', [])).toBe('Discard');
    expect(turnPhaseLabel(undefined, [])).toBeNull();
  });
});

describe('the derived turn line', () => {
  it('after both pass the upcard reads "Your turn • Draw from the stock"', () => {
    const base = deriveGinBase(session('FirstUpcard', [DRAW_STOCK]), {
      viewerSeat: 0,
    });
    expect(base.statusText).toBe('Your turn • Draw from the stock');
  });

  it('reads "Your turn • First upcard" at the upcard choice', () => {
    const base = deriveGinBase(session('FirstUpcard', [DRAW_UPCARD, PASS]), {
      viewerSeat: 0,
    });
    expect(base.statusText).toBe('Your turn • First upcard');
  });
});
