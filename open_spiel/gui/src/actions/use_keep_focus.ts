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

import {
  type FocusEvent,
  type RefObject,
  useCallback,
  useLayoutEffect,
  useRef,
} from 'react';

const FOCUSABLE = 'button, [tabindex]';

/** The element that held the focus, and the index of its row, if any. */
interface Holder {
  readonly element: Element;
  readonly index: number | null;
}

function rowIndex(element: Element): number | null {
  const row = element.closest('[data-row-index]');
  return row === null ? null : Number(row.getAttribute('data-row-index'));
}

// The focusable element of the row under `root` whose index is nearest
// `index`, or null when no row has one.
function nearestRow(root: HTMLElement, index: number): HTMLElement | null {
  let nearest: HTMLElement | null = null;
  let distance = Infinity;
  for (const row of root.querySelectorAll<HTMLElement>('[data-row-index]')) {
    const target = row.matches(FOCUSABLE)
      ? row
      : row.querySelector<HTMLElement>(FOCUSABLE);
    const away = Math.abs(Number(row.getAttribute('data-row-index')) - index);
    if (target !== null && away < distance) {
      nearest = target;
      distance = away;
    }
  }
  return nearest;
}

/**
 * Keeps the focus inside `root` when the element holding it leaves the
 * document, as a played action leaves the legal list. The focus moves to
 * the focusable row, marked `data-row-index`, nearest the row that held it,
 * or to `root` itself, which must be focusable, when there is none. A focus
 * that has moved elsewhere is left alone.
 *
 * @returns The focus handler `root` must be given.
 */
export function useKeepFocus(
  root: RefObject<HTMLElement | null>,
): (event: FocusEvent) => void {
  const holder = useRef<Holder | null>(null);
  useLayoutEffect(() => {
    const held = holder.current;
    const element = root.current;
    if (held === null || element === null || held.element.isConnected) return;
    holder.current = null;
    const active = document.activeElement;
    if (active !== null && active !== document.body) return;
    const row = held.index === null ? null : nearestRow(element, held.index);
    (row ?? element).focus();
  });
  return useCallback((event: FocusEvent) => {
    holder.current = {element: event.target, index: rowIndex(event.target)};
  }, []);
}
