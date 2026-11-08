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

// The board's Pass button: the Pass it plays (the first upcard's and the
// wall's), and whether it is enabled at the wall for the seat to move, for
// a seat not to move and in analysis.

import {act, createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {describe, expect, it, vi} from 'vitest';

import '../../testing/render.tsx';
import {useGinMoves, type GinMoves} from './gin_gameplay_moves.ts';
import type {GinTurn} from './gin_gameplay_turn.ts';
import type {GinView} from './gin_gameplay_view.ts';
import {passKnock} from './gin_rummy_gameplay_panel.tsx';
import type {LegalAction, SessionContextValue} from './gin_session.ts';
import {passActionOf} from './hooks/use_gin_actions.ts';
import type {GinRummyDerivedState} from './use_gin_rummy_derived_state.ts';

const DRAW_UPCARD: LegalAction = {
  action: 52,
  label: 'Player: 0 Action: Draw upcard',
};
const DRAW_STOCK: LegalAction = {
  action: 53,
  label: 'Player: 0 Action: Draw stock',
};
const PASS: LegalAction = {action: 54, label: 'Player: 0 Action: Pass'};
const KNOCK: LegalAction = {action: 55, label: 'Player: 0 Action: Knock'};

describe('passActionOf', () => {
  it('plays the first-upcard pass', () => {
    expect(passActionOf([DRAW_UPCARD, PASS], 'firstupcard')).toBe(54);
  });

  it('plays the pass at the wall, alone or beside a knock', () => {
    expect(passActionOf([PASS], 'wall')).toBe(54);
    expect(passActionOf([PASS, KNOCK], 'wall')).toBe(54);
  });

  it('plays no pass in other phases: a laydown pass goes through Done', () => {
    expect(passActionOf([PASS], 'knock')).toBeNull();
    expect(passActionOf([PASS], 'layoff')).toBeNull();
    expect(passActionOf([DRAW_UPCARD, DRAW_STOCK], 'draw')).toBeNull();
  });

  it('plays nothing when Pass is not legal', () => {
    expect(passActionOf([DRAW_STOCK], 'firstupcard')).toBeNull();
  });
});

interface Seat {
  readonly active: boolean;
  readonly analysis: boolean;
  readonly busy: boolean;
}

/** The wall with Pass and Knock legal, as `seat` sees it. */
function atWall(seat: Seat): {
  buttons: ReturnType<typeof passKnock>;
  submitted: number[];
} {
  const submitted: number[] = [];
  const s = {
    isBusy: seat.busy,
    submitAction: vi.fn((action: number) => {
      submitted.push(action);
      return Promise.resolve();
    }),
  } as unknown as SessionContextValue;
  const d = {
    normalizedPhase: 'wall',
    isTerminal: false,
    knockAction: 55,
    passAction: 54,
    drawUpcardAction: null,
    drawStockAction: null,
  } as unknown as GinRummyDerivedState;
  const view = {isAnalysisMode: seat.analysis} as unknown as GinView;
  const turn = {
    viewerIsActiveSeat: seat.active,
    canInteract: seat.active && !seat.busy,
  } as unknown as GinTurn;
  const out: {moves: GinMoves | null} = {moves: null};
  function Probe(): null {
    out.moves = useGinMoves(s, d, view, turn);
    return null;
  }
  const root = createRoot(document.createElement('div'));
  act(() => {
    root.render(createElement(Probe));
  });
  if (out.moves === null) throw new Error('useGinMoves did not render.');
  const buttons = passKnock(d, out.moves, turn, seat.busy, seat.analysis);
  act(() => {
    root.unmount();
  });
  return {buttons, submitted};
}

describe('Pass at the wall', () => {
  it('is enabled for the seat to move, and plays the pass', () => {
    const {buttons, submitted} = atWall({
      active: true,
      analysis: false,
      busy: false,
    });
    expect(buttons.passDisabled).toBe(false);
    expect(buttons.knockDisabled).toBe(false);
    buttons.onPass?.();
    expect(submitted).toEqual([54]);
  });

  it('is disabled for a seat not to move', () => {
    const {buttons} = atWall({active: false, analysis: false, busy: false});
    expect(buttons.passDisabled).toBe(true);
    expect(buttons.onPass).toBeUndefined();
  });

  it('is enabled in analysis, for either seat', () => {
    const {buttons} = atWall({active: false, analysis: true, busy: false});
    expect(buttons.passDisabled).toBe(false);
  });

  it('is disabled while a move is in flight', () => {
    const {buttons} = atWall({active: true, analysis: false, busy: true});
    expect(buttons.passDisabled).toBe(true);
  });
});
