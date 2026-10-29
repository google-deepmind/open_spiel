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
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type JSX,
  type ReactNode,
  type RefObject,
} from 'react';

import {CloseIcon, KebabIcon} from '../controls/icons.tsx';

import '../controls/controls.css';

/** A panel the drawer can reopen. */
export interface ClosedPanel {
  readonly id: string;
  readonly title: string;
}

interface PanelDrawerProps {
  /** In the order they are listed. */
  readonly closed: readonly ClosedPanel[];
  readonly onOpen: (id: string) => void;
  /** Further options, below the closed panels. */
  readonly children?: ReactNode;
}

// The buttons that reopen a panel.
const REOPEN_BUTTONS = '.panel-drawer-list button';

/**
 * The "Panel options" button, three dots, and the drawer it opens: a sheet
 * that slides in at the right of the window, titled "Panels", which lists
 * the closed panels, reopening one when it is chosen, above any further
 * options. Focus moves into the drawer when it opens, and after each reopen
 * to the panel listed next; it rests on the close button when no panel is
 * listed. The drawer stays open while panels are reopened, and closes on its
 * close button, on Escape inside it, which returns focus to the button, and
 * on a press anywhere outside it.
 */
export function PanelDrawer({
  closed,
  onOpen,
  children,
}: PanelDrawerProps): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const drawer = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const sheet = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const recordReopen = useFocusAfterReopen(closed, sheet, close);
  const sheetId = useId();
  const headingId = useId();

  useFocusInside(expanded, setExpanded, drawer, sheet, close);

  return (
    <div
      className="panel-drawer"
      ref={drawer}
      onKeyDown={({key}) => {
        if (expanded && key === 'Escape') {
          setExpanded(false);
          trigger.current?.focus();
        }
      }}
    >
      <button
        type="button"
        className="button button--icon panel-drawer-trigger"
        ref={trigger}
        aria-label="Panel options"
        title="Panel options"
        aria-haspopup="dialog"
        aria-expanded={expanded}
        aria-controls={sheetId}
        onClick={() => {
          setExpanded(value => !value);
        }}
      >
        <KebabIcon />
      </button>
      <section
        id={sheetId}
        ref={sheet}
        className="panel-drawer-sheet"
        role="dialog"
        aria-labelledby={headingId}
        hidden={!expanded}
      >
        <div className="panel-drawer-header">
          <h2 id={headingId}>Panels</h2>
          <button
            type="button"
            className="button button--icon"
            ref={close}
            aria-label="Close panel options"
            title="Close"
            onClick={() => {
              setExpanded(false);
            }}
          >
            <CloseIcon />
          </button>
        </div>
        <div className="panel-drawer-content">
          <ClosedPanelList
            closed={closed}
            onOpen={(id, index) => {
              recordReopen(id, index);
              onOpen(id);
            }}
          />
          {children}
        </div>
      </section>
    </div>
  );
}

/**
 * While the drawer is `expanded`, moves focus into it, to the first panel
 * listed or else to `close`, and collapses it on a press outside `drawer`.
 */
function useFocusInside(
  expanded: boolean,
  setExpanded: (expanded: boolean) => void,
  drawer: RefObject<HTMLElement | null>,
  sheet: RefObject<HTMLElement | null>,
  close: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    if (!expanded) {
      return undefined;
    }
    const first = sheet.current?.querySelector<HTMLElement>(REOPEN_BUTTONS);
    (first ?? close.current)?.focus();
    const pressed = ({target}: PointerEvent): void => {
      if (!(target instanceof Node && drawer.current?.contains(target))) {
        setExpanded(false);
      }
    };
    document.addEventListener('pointerdown', pressed);
    return () => {
      document.removeEventListener('pointerdown', pressed);
    };
  }, [expanded, setExpanded, drawer, sheet, close]);
}

/**
 * Once a panel reopened from position `index` of the list has left it, moves
 * focus to the panel listed there now, or the last one, or to the close button
 * when none is left. Returns the function that records the reopen.
 */
function useFocusAfterReopen(
  closed: readonly ClosedPanel[],
  sheet: RefObject<HTMLElement | null>,
  close: RefObject<HTMLElement | null>,
): (id: string, index: number) => void {
  const reopened = useRef<{id: string; index: number} | null>(null);
  useEffect(() => {
    const pending = reopened.current;
    if (pending === null || closed.some(({id}) => id === pending.id)) {
      return;
    }
    reopened.current = null;
    const items = [
      ...(sheet.current?.querySelectorAll<HTMLElement>(REOPEN_BUTTONS) ?? []),
    ];
    (
      items[Math.min(pending.index, items.length - 1)] ?? close.current
    )?.focus();
  }, [closed, sheet, close]);

  return useCallback((id: string, index: number) => {
    reopened.current = {id, index};
  }, []);
}

function ClosedPanelList({
  closed,
  onOpen,
}: {
  readonly closed: readonly ClosedPanel[];
  readonly onOpen: (id: string, index: number) => void;
}): JSX.Element {
  if (closed.length === 0) {
    return <p className="panel-drawer-empty">All panels are open</p>;
  }
  return (
    <>
      <h3 className="panel-drawer-section" aria-hidden>
        Closed panels
      </h3>
      <ul className="panel-drawer-list" aria-label="Closed panels">
        {closed.map(({id, title}, index) => (
          <li key={id}>
            <button
              type="button"
              className="button panel-drawer-item"
              onClick={() => {
                onOpen(id, index);
              }}
            >
              {title}
            </button>
          </li>
        ))}
      </ul>
    </>
  );
}
