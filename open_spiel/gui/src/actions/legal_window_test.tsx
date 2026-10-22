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

import {act, type JSX} from 'react';
import {createRoot, type Root} from 'react-dom/client';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import type {Action} from '../contract/contract.ts';
import {
  layOut,
  press,
  restoreLayout,
  scrollTo,
  stubResizeObserver,
} from '../testing/dom.ts';
import {actions} from '../testing/frames.ts';
import {only} from '../testing/render.tsx';
import {OVERSCAN_ROWS, ROW_STRIDE_PX} from './action_list.ts';
import {LegalWindow} from './legal_window.tsx';

const VIEW_ROWS = 16;
const COUNT = 400;
const LIST = actions(COUNT);

let container: HTMLElement;
let root: Root;

function button(action: Action, tabIndex: 0 | -1): JSX.Element {
  return (
    <button type="button" className="row" tabIndex={tabIndex}>
      {action.label}
    </button>
  );
}

function text(action: Action): JSX.Element {
  return <span className="row">{action.label}</span>;
}

function show(focusableRows: boolean, list: readonly Action[] = LIST): void {
  act(() => {
    root.render(
      <LegalWindow
        actions={list}
        label="Legal"
        focusableRows={focusableRows}
        renderRow={focusableRows ? button : text}
      />,
    );
  });
}

function scroller(): HTMLElement {
  return only(container, '[role="list"]');
}

function rows(): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>('.row'));
}

function labels(): Array<string | null> {
  return rows().map(row => row.textContent);
}

// Every element of the list in the tab order.
function tabStops(): HTMLElement[] {
  const all = [scroller(), ...scroller().querySelectorAll<HTMLElement>('*')];
  return all.filter(element => element.tabIndex === 0);
}

// Presses `key` on the element with the focus; returns the one it has after.
function pressFocused(key: string): string | null {
  const focused = document.activeElement;
  if (!(focused instanceof HTMLElement)) throw new Error('Nothing focused.');
  press(focused, key);
  return document.activeElement?.textContent ?? null;
}

beforeEach(() => {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  stubResizeObserver();
  layOut(HTMLDivElement.prototype, {
    clientHeight: VIEW_ROWS * ROW_STRIDE_PX,
    scrollHeight: COUNT * ROW_STRIDE_PX,
  });
});

afterEach(() => {
  act(() => {
    root.unmount();
  });
  container.remove();
  restoreLayout();
  vi.unstubAllGlobals();
});

describe('LegalWindow', () => {
  it('renders the rows in view and the overscan, over the full height', () => {
    show(false);
    expect(labels()).toHaveLength(VIEW_ROWS + OVERSCAN_ROWS);
    expect(only(container, '.legal-window__track').style.height).toBe(
      `${String(COUNT * ROW_STRIDE_PX)}px`,
    );
    scrollTo(scroller(), 200 * ROW_STRIDE_PX);
    expect(labels()).toHaveLength(VIEW_ROWS + 2 * OVERSCAN_ROWS);
    expect(labels()[0]).toBe(`action ${String(200 - OVERSCAN_ROWS)}`);
    const slot = only(container, '[data-action-id="200"]');
    expect(slot.getAttribute('role')).toBe('listitem');
    expect(slot.getAttribute('aria-posinset')).toBe('201');
    expect(slot.getAttribute('aria-setsize')).toBe(String(COUNT));
    expect(slot.getAttribute('data-row-index')).toBe('200');
  });

  it('shows the last rows when the list shrinks under its scroll position', () => {
    show(false);
    scrollTo(scroller(), 390 * ROW_STRIDE_PX);
    show(false, LIST.slice(0, 250));
    expect(labels()).toHaveLength(VIEW_ROWS + OVERSCAN_ROWS);
    expect(labels().at(-1)).toBe('action 249');
  });
});

describe('LegalWindow tab order', () => {
  it('is the scroller alone when the rows are text', () => {
    show(false);
    expect(tabStops()).toEqual([scroller()]);
  });

  it('is one row when the rows are focusable: the first in view', () => {
    show(true);
    scrollTo(scroller(), 200 * ROW_STRIDE_PX);
    expect(tabStops().map(stop => stop.textContent)).toEqual(['action 200']);
    expect(rows().filter(row => row.tabIndex === -1)).toHaveLength(
      VIEW_ROWS + 2 * OVERSCAN_ROWS - 1,
    );
  });

  it('is the row last focused, wherever the list scrolls', () => {
    show(true);
    act(() => {
      rows()[3]?.focus();
    });
    scrollTo(scroller(), 200 * ROW_STRIDE_PX);
    expect(tabStops().map(stop => stop.textContent)).toEqual(['action 3']);
  });
});

describe('LegalWindow by keyboard', () => {
  it('moves the focus with the arrow keys, Home and End', () => {
    show(true);
    act(() => {
      rows()[0]?.focus();
    });
    expect(pressFocused('ArrowUp')).toBe('action 0');
    expect(pressFocused('ArrowDown')).toBe('action 1');
    expect(pressFocused('ArrowUp')).toBe('action 0');
    expect(pressFocused('End')).toBe('action 399');
    expect(scroller().scrollTop).toBe((COUNT - VIEW_ROWS) * ROW_STRIDE_PX);
    expect(pressFocused('ArrowDown')).toBe('action 399');
    expect(pressFocused('a')).toBe('action 399');
    expect(pressFocused('Home')).toBe('action 0');
    expect(scroller().scrollTop).toBe(0);
  });

  it('leaves the keys to the scroller when the rows are text', () => {
    show(false);
    const key = new KeyboardEvent('keydown', {key: 'End', cancelable: true});
    act(() => {
      scroller().dispatchEvent(key);
    });
    expect(key.defaultPrevented).toBe(false);
    expect(scroller().scrollTop).toBe(0);
  });
});

describe('LegalWindow focused row', () => {
  it('stays rendered, first, when the list scrolls down past it', () => {
    show(true);
    const first = rows()[0];
    act(() => {
      first?.focus();
    });
    scrollTo(scroller(), 300 * ROW_STRIDE_PX);
    expect(rows()[0]).toBe(first);
    expect(document.activeElement).toBe(first);
    expect(labels()).toHaveLength(VIEW_ROWS + 2 * OVERSCAN_ROWS + 1);
  });

  it('stays rendered, last, when the list scrolls up past it', () => {
    show(true);
    scrollTo(scroller(), 300 * ROW_STRIDE_PX);
    const below = rows().at(-1);
    act(() => {
      below?.focus();
    });
    scrollTo(scroller(), 0);
    expect(rows().at(-1)).toBe(below);
    expect(below?.textContent).toBe(
      `action ${String(300 + VIEW_ROWS + OVERSCAN_ROWS - 1)}`,
    );
    expect(document.activeElement).toBe(below);
  });

  it('is rendered no longer once the focus leaves it', () => {
    show(true);
    const first = rows()[0];
    act(() => {
      first?.focus();
    });
    scrollTo(scroller(), 300 * ROW_STRIDE_PX);
    act(() => {
      first?.blur();
    });
    expect(rows()).not.toContain(first);
    expect(labels()).toHaveLength(VIEW_ROWS + 2 * OVERSCAN_ROWS);
  });
});
