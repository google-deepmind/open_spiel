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

import type {Frame, HistoryEntry} from '../contract/contract.ts';
import type {Session} from '../session/use_session.ts';
import {layOut, restoreLayout, scrollTo} from '../testing/dom.ts';
import {entries, frame, session} from '../testing/frames.ts';
import {only} from '../testing/render.tsx';
import {History} from './history.tsx';

const WITHHELD: HistoryEntry = {
  seat: null,
  action: null,
  label: null,
  rewards: null,
};

let container: HTMLElement;
let root: Root;

function show(shown: Session): void {
  act(() => {
    root.render(<History session={shown} />);
  });
}

// Each row's turn and label, joined by a space, its standing and whether it
// is current.
function rows(): Array<Array<string | null>> {
  return Array.from(container.querySelectorAll('li'), row => [
    Array.from(row.querySelectorAll('span'), span => span.textContent).join(
      ' ',
    ),
    row.getAttribute('data-standing'),
    row.getAttribute('aria-current'),
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
  restoreLayout();
});

describe('History', () => {
  it('lets an omniscient viewer jump to any position', () => {
    const shown = session(
      frame({history: entries(['e4', 'e5', 'Nf3']), ply: 1}),
    );
    show(shown);
    expect(only(container, 'ol').getAttribute('aria-label')).toBe('History');
    expect(rows()).toEqual([
      ['#0 Initial', 'played', null],
      ['#1 e4', 'current', 'step'],
      ['#2 e5', 'ahead', null],
      ['#3 Nf3', 'ahead', null],
    ]);
    const jumps = container.querySelectorAll('button');
    expect(jumps).toHaveLength(4);
    act(() => {
      jumps[3]?.click();
    });
    expect(shown.commands.goTo).toHaveBeenCalledExactlyOnceWith(3);
  });

  it('shows a seat its history read-only, with withheld runs collapsed', () => {
    const seen: Frame = frame({
      viewer: {kind: 'seat', seat: 0},
      omniscient: false,
      history: [WITHHELD, WITHHELD, WITHHELD, ...entries(['Bid'])],
    });
    show(session(seen));
    expect(container.querySelector('button')).toBeNull();
    expect(rows()).toEqual([
      ['#0 Initial', 'played', null],
      ['#1–3 Hidden chance outcome ×3', 'played', null],
      ['#4 Bid', 'current', 'step'],
    ]);
  });

  it('marks a collapsed row current at any ply inside it', () => {
    const history = [WITHHELD, WITHHELD, WITHHELD];
    show(session(frame({omniscient: false, history, ply: 2})));
    expect(rows()[1]).toEqual([
      '#1–3 Hidden chance outcome ×3',
      'current',
      'step',
    ]);
  });

  it('refuses a jump while a request is in flight', () => {
    const shown = session(frame({history: entries(['e4', 'e5'])}), true);
    show(shown);
    const jumps = container.querySelectorAll('button');
    expect(jumps[1]?.getAttribute('aria-disabled')).toBe('true');
    act(() => {
      jumps[1]?.click();
    });
    expect(shown.commands.goTo).not.toHaveBeenCalled();
  });

  it('tells a seat that nothing has been played', () => {
    show(
      session(
        frame({
          viewer: {kind: 'seat', seat: 0},
          omniscient: false,
          history: [],
        }),
      ),
    );
    expect(container.querySelector('ol')).toBeNull();
    expect(only(container, '[role="status"]').textContent).toBe(
      'No actions yet.',
    );
  });

  it('says so when it has no frame', () => {
    show(session(null));
    expect(only(container, '[role="status"]').textContent).toBe(
      'No position to show.',
    );
  });
});

describe('History scrolling', () => {
  // 600 px of rows in a 200 px view: the bottom is at scrollTop 400.
  beforeEach(() => {
    layOut(HTMLOListElement.prototype, {clientHeight: 200, scrollHeight: 600});
  });

  function played(count: number, ply = count): Session {
    const labels = Array.from({length: count}, (_, index) => String(index));
    return session(frame({history: entries(labels), ply}));
  }

  it('follows appended actions while the reader is at the bottom', () => {
    show(played(3));
    const list = only(container, 'ol');
    expect(list.scrollTop).toBe(600);
    scrollTo(list, 400);
    show(played(4));
    expect(list.scrollTop).toBe(600);
  });

  it('does not scroll when stepping through a replay', () => {
    show(played(5));
    const list = only(container, 'ol');
    list.scrollTop = 250;
    show(played(5, 1));
    expect(list.scrollTop).toBe(250);
  });
});
