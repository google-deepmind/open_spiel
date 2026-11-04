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

import {act, type JSX, useRef} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {only} from '../testing/render.tsx';
import {useKeepFocus} from './use_keep_focus.ts';

let container: HTMLElement;
let root: Root;

// Rows of buttons, or of text when `text` is set, under a focusable root.
function Rows({
  labels,
  text = false,
}: {
  labels: readonly string[];
  text?: boolean;
}): JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  const keepFocus = useKeepFocus(ref);
  return (
    <div ref={ref} className="root" tabIndex={-1} onFocus={keepFocus}>
      <input aria-label="Outside the rows" />
      {labels.map((label, index) => (
        <div key={label} data-row-index={index}>
          {text ? <span>{label}</span> : <button type="button">{label}</button>}
        </div>
      ))}
    </div>
  );
}

function show(labels: readonly string[], text = false): void {
  act(() => {
    root.render(<Rows labels={labels} text={text} />);
  });
}

function focus(selector: string): void {
  act(() => {
    only(container, selector).focus();
  });
}

function focusButton(label: string): void {
  const button = Array.from(container.querySelectorAll('button')).find(
    candidate => candidate.textContent === label,
  );
  if (button === undefined) throw new Error(`No button ${label}.`);
  act(() => {
    button.focus();
  });
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
});

describe('useKeepFocus', () => {
  it('moves the focus to the row now at the index of the one removed', () => {
    show(['a', 'b', 'c']);
    focusButton('b');
    show(['a', 'c']);
    expect(document.activeElement?.textContent).toBe('c');
  });

  it('moves the focus to the nearest row when no row has that index', () => {
    show(['a', 'b', 'c']);
    focusButton('c');
    show(['a', 'b']);
    expect(document.activeElement?.textContent).toBe('b');
  });

  it('moves the focus to the root when no row is focusable', () => {
    show(['a', 'b']);
    focusButton('a');
    show(['a', 'b'], true);
    expect(document.activeElement).toBe(only(container, '.root'));
    show([]);
    expect(document.activeElement).toBe(only(container, '.root'));
  });

  it('keeps a focus that is not on a removed element', () => {
    show(['a', 'b']);
    focus('input');
    show(['b']);
    expect(document.activeElement).toBe(only(container, 'input'));
  });

  it('leaves a focus that moved out of the root before the removal', () => {
    const outside = document.createElement('button');
    document.body.append(outside);
    show(['a', 'b']);
    focusButton('a');
    act(() => {
      outside.focus();
    });
    show(['b']);
    expect(document.activeElement).toBe(outside);
    outside.remove();
  });
});
