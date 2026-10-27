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

// The checkers board's tests' session: a session as the renderer gives it,
// and the board mounted on it, re-rendered when the test changes it.

import {act} from 'react';
import {type Root, createRoot} from 'react-dom/client';
import {expect, vi} from 'vitest';

import '../../testing/render.tsx';
import {CheckersGameplay} from './checkers_gameplay_panel.tsx';
import {
  CheckersSessionContext,
  type CheckersState,
  type SessionContextValue,
} from './checkers_session.ts';

/** The fields of the session's state a test sets; the rest default. */
export type StateOverrides = Partial<CheckersState>;

/** A session at the opening of a two-seat game, viewer seat 0, to move. */
export function sessionContext(
  state: StateOverrides,
  mode: 'analysis' | 'game' = 'analysis',
): SessionContextValue {
  return {
    session: {
      sessionId: 's',
      mode,
      state: {
        movableSources: [],
        currentPlayer: 0,
        isTerminal: false,
        legalActions: [],
        returns: null,
        history: [],
        historyPlayers: [],
        historyActionStrings: [],
        ...state,
      },
      players: [
        {slot: 0, controller: 'human', autoPlay: false},
        {slot: 1, controller: 'human', autoPlay: false},
      ],
    },
    error: null,
    viewer: {seat: 0, isParticipant: true, isActiveSeat: true},
    isBusy: false,
    submitAction: vi.fn(() => Promise.resolve()),
  };
}

/** The board mounted on a session, and a re-render on the holder's next. */
export interface Mounted {
  readonly container: HTMLElement;
  /** The one element whose data-testid is `id`. */
  getByTestId: (id: string) => HTMLElement;
  /** Renders the board again with the holder's current session. */
  again: () => void;
  unmount: () => void;
}

let mounted: Array<{root: Root; container: HTMLElement}> = [];

/** Mounts the board on `holder.ctx`. */
export function mount(holder: {ctx: SessionContextValue}): Mounted {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({root, container});
  const draw = (): void => {
    act(() => {
      root.render(
        <CheckersSessionContext.Provider value={holder.ctx}>
          <CheckersGameplay />
        </CheckersSessionContext.Provider>,
      );
    });
  };
  draw();
  return {
    container,
    getByTestId: id => {
      const found = container.querySelector(`[data-testid="${id}"]`);
      if (!(found instanceof HTMLElement)) {
        throw new Error(`No element has data-testid ${id}.`);
      }
      return found;
    },
    again: draw,
    unmount: () => {
      act(() => {
        root.unmount();
      });
      container.remove();
    },
  };
}

/** Unmounts every board mounted since the last cleanup. */
export function cleanup(): void {
  for (const {root, container} of mounted) {
    act(() => {
      root.unmount();
    });
    container.remove();
  }
  mounted = [];
}

/** The square at `row`, `col` (row 0 is rank 1). */
export function square(root: HTMLElement, row: number, col: number): Element {
  const el = root.querySelector(
    `[data-row="${String(row)}"][data-col="${String(col)}"]`,
  );
  expect(el, `square ${String(row)},${String(col)}`).not.toBeNull();
  if (el === null) throw new Error('unreachable');
  return el;
}

/** Whether `element` carries `name` in its classes. */
export function hasClass(element: Element | null, name: string): boolean {
  return element?.classList.contains(name) ?? false;
}
