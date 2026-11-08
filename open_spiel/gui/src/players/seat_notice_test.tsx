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
import {type Root, createRoot} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {Frame} from '../contract/contract.ts';
import {stubResizeObserver} from '../testing/dom.ts';
import {only} from '../testing/render.tsx';
import {PlayersPanel} from './players_panel.tsx';
import {RANDOM, frameWith, seatView} from './testing/seats.ts';

const NOTE = 'Recovered from a legal-set mismatch at ply 12.';

let container: HTMLElement;
let root: Root;

beforeEach(() => {
  stubResizeObserver();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  vi.unstubAllGlobals();
});

function show(frame: Frame): void {
  act(() => {
    root.render(
      <PlayersPanel
        frame={frame}
        busy={false}
        commands={{setSeat: vi.fn(), advance: vi.fn()}}
      />,
    );
  });
}

function seat(index: number): HTMLElement {
  const row = container.querySelectorAll<HTMLElement>('li.seat')[index];
  if (row === undefined) throw new Error(`No seat ${String(index)}.`);
  return row;
}

describe('a seat’s notice', () => {
  it('follows the controller’s name, on the same line', () => {
    show(frameWith([seatView(0), seatView(1, RANDOM, {notice: NOTE})]));
    const notice = only(seat(1), '.seat__controller-line .seat__notice');
    expect(notice.textContent).toBe(NOTE);
    expect(notice.getAttribute('title')).toBe(NOTE);
    expect(only(seat(1), '.seat__controller').textContent).toBe('Random');
    expect(seat(0).querySelector('.seat__notice')).toBeNull();
  });

  it('is information only: no alert, and play is not paused', () => {
    show(frameWith([seatView(0), seatView(1, RANDOM, {notice: NOTE})]));
    expect(seat(1).querySelector('[role="alert"]')).toBeNull();
    expect(seat(1).textContent).not.toContain('paused');
  });

  it('is gone when the frame carries none', () => {
    show(frameWith([seatView(0), seatView(1, RANDOM, {notice: NOTE})]));
    show(frameWith([seatView(0), seatView(1, RANDOM)]));
    expect(seat(1).querySelector('.seat__notice')).toBeNull();
  });
});
