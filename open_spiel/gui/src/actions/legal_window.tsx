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
  type JSX,
  type KeyboardEvent,
  type RefObject,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import type {Action} from '../contract/contract.ts';
import {
  ROW_GAP_PX,
  ROW_STRIDE_PX,
  clampScrollTop,
  scrollTopShowing,
  windowRange,
} from './action_list.ts';

import './actions.css';

/** The properties of a LegalWindow. */
export interface LegalWindowProps<Row extends Action = Action> {
  readonly actions: readonly Row[];
  /** The accessible name of the list. */
  readonly label: string;
  /** Whether the rows are focusable, as buttons are, rather than text. */
  readonly focusableRows: boolean;
  /**
   * Renders a row. A focusable row takes `tabIndex`: 0 for the one row in
   * the tab order, -1 for every other.
   */
  readonly renderRow: (action: Row, tabIndex: 0 | -1) => JSX.Element;
}

// The row a key moves the focus to from row `from` of `count`, or null for a
// key that does not move it.
function rowAfterKey(key: string, from: number, count: number): number | null {
  switch (key) {
    case 'ArrowDown':
      return Math.min(count - 1, from + 1);
    case 'ArrowUp':
      return Math.max(0, from - 1);
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    default:
      return null;
  }
}

// Focuses the rendered row of action `id`; false when it is not rendered.
function focusRow(list: HTMLElement, id: number): boolean {
  const row = list.querySelector(`[data-action-id="${String(id)}"] > *`);
  if (!(row instanceof HTMLElement)) return false;
  row.focus();
  return true;
}

// The client height of the element `ref` holds, kept current as it resizes.
function useClientHeight(ref: RefObject<HTMLElement | null>): number {
  const [height, setHeight] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (element === null) return;
    // Read now, so the first paint has the rows in view.
    setHeight(element.clientHeight);
    const observer = new ResizeObserver(() => {
      setHeight(element.clientHeight);
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, [ref]);
  return height;
}

// The rows to render, with their indices, in list order: `start` up to
// `end`, and the row at `focused`, if any, wherever it is.
function renderedRows<Row extends Action>(
  actions: readonly Row[],
  start: number,
  end: number,
  focused: number,
): Array<readonly [number, Row]> {
  const rows = actions
    .slice(start, end)
    .map((action, offset) => [start + offset, action] as const);
  const action = actions[focused];
  if (action !== undefined && (focused < start || focused >= end)) {
    rows.splice(focused < start ? 0 : rows.length, 0, [focused, action]);
  }
  return rows;
}

/**
 * A long list that renders only the rows in view, and is scrollable as a
 * whole: its full height is the whole list's. The list is one stop in the
 * tab order. For text rows it is the scroller, so the keyboard scrolls it.
 * For focusable rows it is the first row in view or the row last focused,
 * and the arrow keys, Home and End move the focus between rows, scrolling it
 * into view. The focused row stays rendered when it scrolls out of view.
 * Each row is a list item that gives its place in the whole list, and its
 * index in `data-row-index`.
 */
export function LegalWindow<Row extends Action>({
  actions,
  label,
  focusableRows,
  renderRow,
}: LegalWindowProps<Row>): JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  // A row to focus once it is rendered.
  const pendingFocus = useRef<number | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const viewportPx = useClientHeight(ref);
  const [focusedId, setFocusedId] = useState<number | null>(null);

  useLayoutEffect(() => {
    const id = pendingFocus.current;
    if (id !== null && ref.current !== null) {
      pendingFocus.current = null;
      focusRow(ref.current, id);
    }
  });

  const top = clampScrollTop(actions.length, scrollTop, viewportPx);
  const {start, end} = windowRange(actions.length, top, viewportPx);
  const focused = actions.findIndex(action => action.id === focusedId);
  const tabStop = focused === -1 ? Math.floor(top / ROW_STRIDE_PX) : focused;
  const rows = renderedRows(actions, start, end, focused);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const element = ref.current;
    // With no row focused, the focus is on the scroller, which the keys
    // scroll.
    if (focused === -1 || element === null) return;
    const next = rowAfterKey(event.key, focused, actions.length);
    const action = next === null ? undefined : actions[next];
    if (next === null || action === undefined) return;
    event.preventDefault();
    const shown = scrollTopShowing(next, top, viewportPx);
    element.scrollTop = shown;
    setScrollTop(shown);
    if (!focusRow(element, action.id)) pendingFocus.current = action.id;
  };
  return (
    <div
      ref={ref}
      className="legal-window"
      role="list"
      aria-label={label}
      tabIndex={focusableRows ? undefined : 0}
      onScroll={event => {
        setScrollTop(event.currentTarget.scrollTop);
      }}
      onKeyDown={onKeyDown}
    >
      <div
        className="legal-window__track"
        style={{height: actions.length * ROW_STRIDE_PX}}
      >
        {rows.map(([index, action]) => (
          <div
            key={action.id}
            className="legal-window__slot"
            role="listitem"
            aria-setsize={actions.length}
            aria-posinset={index + 1}
            data-action-id={action.id}
            data-row-index={index}
            style={{
              top: index * ROW_STRIDE_PX,
              height: ROW_STRIDE_PX - ROW_GAP_PX,
            }}
            onFocus={() => {
              setFocusedId(action.id);
            }}
            onBlur={() => {
              setFocusedId(current => (current === action.id ? null : current));
            }}
          >
            {renderRow(action, index === tabStop ? 0 : -1)}
          </div>
        ))}
      </div>
    </div>
  );
}
