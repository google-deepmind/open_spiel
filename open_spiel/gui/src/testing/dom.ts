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

// Input as a user gives it, and the layout jsdom does not do.

import {act} from 'react';
import {vi} from 'vitest';

/** Types `text` into `input`, replacing its value, as React observes it. */
export function typeInto(input: HTMLInputElement, text: string): void {
  act(() => {
    // React tracks the value through the prototype's setter, not the field.
    Reflect.set(HTMLInputElement.prototype, 'value', text, input);
    input.dispatchEvent(new Event('input', {bubbles: true}));
  });
}

/** Presses `key` on `element`. */
export function press(element: HTMLElement, key: string): void {
  act(() => {
    element.dispatchEvent(new KeyboardEvent('keydown', {key, bubbles: true}));
  });
}

/** Scrolls `element` to `scrollTop` and sends the scroll event. */
export function scrollTo(element: HTMLElement, scrollTop: number): void {
  act(() => {
    element.scrollTop = scrollTop;
    element.dispatchEvent(new Event('scroll'));
  });
}

/** The sizes layOut() gives an element, in pixels. */
export interface Layout {
  readonly clientHeight: number;
  readonly scrollHeight: number;
}

const laidOut: object[] = [];

/**
 * Gives every element made from `prototype` the sizes in `layout` and a
 * scrollTop that keeps what is set: jsdom lays nothing out. Undo it with
 * restoreLayout().
 */
export function layOut(prototype: HTMLElement, layout: Layout): void {
  if (laidOut.includes(prototype)) {
    throw new Error('The prototype is already laid out.');
  }
  Object.defineProperties(prototype, {
    clientHeight: {value: layout.clientHeight, configurable: true},
    scrollHeight: {value: layout.scrollHeight, configurable: true},
    scrollTop: {value: 0, writable: true, configurable: true},
  });
  laidOut.push(prototype);
}

/** Removes every layout layOut() gave. */
export function restoreLayout(): void {
  for (const prototype of laidOut.splice(0)) {
    for (const name of ['clientHeight', 'scrollHeight', 'scrollTop']) {
      Reflect.deleteProperty(prototype, name);
    }
  }
}

/**
 * Replaces ResizeObserver with one that notifies once, as observation
 * starts, as browsers do for a box already laid out. Undo it with
 * vi.unstubAllGlobals().
 */
export function stubResizeObserver(): void {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      private readonly notify: () => void;
      constructor(notify: () => void) {
        this.notify = notify;
      }
      observe(): void {
        this.notify();
      }
      disconnect(): void {
        // Nothing to stop: it never notifies again.
      }
    },
  );
}
