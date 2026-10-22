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
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {HistoryEntry} from '../contract/contract.ts';
import {entries, frame, session} from '../testing/frames.ts';
import {type Rendered, only, render} from '../testing/render.tsx';
import {restoreProperties, stubProperty} from '../testing/stub_property.ts';
import {HistoryText, historyJson} from './history_text.tsx';

const PLAYED = entries(['x(0,0)', 'o(1,1)']);
const WITHHELD: HistoryEntry = {
  seat: 1,
  action: null,
  label: null,
  rewards: null,
};

let rendered: Rendered | undefined;

afterEach(() => {
  rendered?.unmount();
  rendered = undefined;
  restoreProperties();
});

function button(container: HTMLElement, name: string): HTMLButtonElement {
  const found = Array.from(container.querySelectorAll('button')).find(
    candidate => candidate.textContent === name,
  );
  if (found === undefined) throw new Error(`No button ${name}.`);
  return found;
}

describe('historyJson', () => {
  it('writes labels or ids, on one line or one per line', () => {
    expect(historyJson(PLAYED, 'strings', 'multiline')).toBe(
      '[\n  "x(0,0)",\n  "o(1,1)"\n]',
    );
    expect(historyJson(PLAYED, 'strings', 'oneline')).toBe(
      '["x(0,0)","o(1,1)"]',
    );
    expect(historyJson(PLAYED, 'ints', 'oneline')).toBe('[0,1]');
    expect(historyJson(PLAYED, 'ints', 'multiline')).toBe('[\n  0,\n  1\n]');
    expect(historyJson([], 'ints', 'multiline')).toBe('[]');
  });

  it('writes nothing when an entry is withheld', () => {
    expect(historyJson([...PLAYED, WITHHELD], 'ints', 'oneline')).toBeNull();
    expect(historyJson([WITHHELD], 'strings', 'oneline')).toBeNull();
  });
});

describe('HistoryText', () => {
  it('switches between labels and ids and between layouts', () => {
    rendered = render(
      <HistoryText session={session(frame({history: PLAYED}))} />,
    );
    const {container} = rendered;
    const body = only(container, 'pre');
    expect(body.textContent).toBe('[\n  "x(0,0)",\n  "o(1,1)"\n]');
    act(() => {
      button(container, 'Ints').click();
    });
    act(() => {
      button(container, 'One line').click();
    });
    expect(body.textContent).toBe('[0,1]');
    expect(button(container, 'Ints').getAttribute('aria-pressed')).toBe('true');
  });

  it('copies the text it shows', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    stubProperty(navigator, 'clipboard', {writeText});
    rendered = render(
      <HistoryText session={session(frame({history: PLAYED}))} />,
    );
    const copy = only(rendered.container, '.copyable .copy-button');
    expect(copy.getAttribute('aria-label')).toBe('Copy to clipboard');
    await act(async () => {
      copy.click();
      await Promise.resolve();
    });
    expect(writeText).toHaveBeenCalledExactlyOnceWith(
      '[\n  "x(0,0)",\n  "o(1,1)"\n]',
    );
  });

  it('offers no copy of a history with withheld actions', () => {
    const history = [...PLAYED, WITHHELD];
    rendered = render(
      <HistoryText session={session(frame({omniscient: false, history}))} />,
    );
    const {container} = rendered;
    expect(container.querySelector('pre')).toBeNull();
    expect(container.querySelector('.copy-button')).toBeNull();
    expect(only(container, '.history-text > [role="status"]').textContent).toBe(
      'Some actions are withheld from this viewer, so the history cannot be ' +
        'written out.',
    );
  });

  it('says that nothing has been played, with no text to copy', () => {
    rendered = render(<HistoryText session={session(frame())} />);
    const {container} = rendered;
    expect(container.querySelector('pre')).toBeNull();
    expect(only(container, '.history-text > [role="status"]').textContent).toBe(
      'No actions yet.',
    );
  });

  it('names its pills for screen readers only', () => {
    rendered = render(
      <HistoryText session={session(frame({history: PLAYED}))} />,
    );
    const groups = rendered.container.querySelectorAll(
      '[role="group"]:not(pre)',
    );
    expect(Array.from(groups, group => group.textContent)).toEqual([
      'ValuesStringsInts',
      'LayoutMulti-lineOne line',
    ]);
  });

  it('says so when it has no frame', () => {
    rendered = render(<HistoryText session={session(null)} />);
    expect(rendered.container.textContent).toBe('No position to show.');
  });
});
