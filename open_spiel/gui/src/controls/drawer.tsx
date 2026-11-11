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

/**
 * A drawer: a modal side panel on a wide screen, a full-screen sheet on a
 * narrow one.
 *
 * - `role="dialog"` with `aria-modal` and a labelled heading.
 * - Focus moves into the drawer when it opens and is trapped there (Tab and
 *   Shift+Tab cycle inside); Escape closes it; on close focus returns to the
 *   element that had it before (or to `returnFocus()` when given).
 * - It is `position: fixed` over the page, so opening and closing never move
 *   the page under it: no scroll lock, no width change. There is no
 *   dimming backdrop, so the page stays readable; a transparent layer over the
 *   page takes the click that closes the drawer, so nothing behind it is
 *   clicked by accident.
 * - At 640 px or less it is a full-screen sheet whose close button reads
 *   "Back", as the spec's wireframe has it.
 */
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  type JSX,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from 'react';

import './drawer.css';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

function focusables(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    el =>
      !el.hasAttribute('hidden') && el.getAttribute('aria-hidden') !== 'true',
  );
}

/** A drawer: its title, content and how it closes. */
export interface DrawerProps {
  /** The heading; also the dialog's accessible name. */
  title: ReactNode;
  onClose: () => void;
  children: ReactNode;
  /** Where focus goes when the drawer closes; default: where it was. */
  returnFocus?: () => HTMLElement | null | undefined;
  testId?: string | undefined;
  /** Extra class on the panel. */
  className?: string | undefined;
  /** What the drawer shows; a new key scrolls its body back to the top. */
  contentKey?: string | undefined;
}

// Focus moves into the drawer on open and back out on close, and new
// content starts at its top with focus on the drawer (not on a control of
// the content it replaced).
function useDrawerFocus(
  panelRef: RefObject<HTMLDivElement | null>,
  bodyRef: RefObject<HTMLDivElement | null>,
  returnRef: RefObject<(() => HTMLElement | null | undefined) | undefined>,
  contentKey: string | undefined,
): void {
  useEffect(() => {
    const before =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const panel = panelRef.current;
    panel?.focus({preventScroll: true});
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps -- the focus goes back to what returnFocus names at close.
      const target = returnRef.current?.() ?? before;
      if (target && document.contains(target)) {
        target.focus({preventScroll: true});
      }
    };
  }, [panelRef, returnRef]);
  const firstKey = useRef(contentKey);
  useEffect(() => {
    if (contentKey === firstKey.current) return;
    firstKey.current = contentKey;
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
    panelRef.current?.focus({preventScroll: true});
  }, [contentKey, panelRef, bodyRef]);
}

// Escape closes; Tab and Shift+Tab cycle inside the panel.
function drawerKeys(
  e: KeyboardEvent<HTMLDivElement>,
  panel: HTMLDivElement | null,
  close: () => void,
): void {
  if (e.key === 'Escape') {
    e.stopPropagation();
    e.preventDefault();
    close();
    return;
  }
  if (e.key !== 'Tab' || !panel) return;
  const items = focusables(panel);
  const first = items[0];
  const last = items.at(-1);
  if (first === undefined || last === undefined) {
    e.preventDefault();
    panel.focus();
    return;
  }
  const active = document.activeElement;
  if (e.shiftKey && (active === first || active === panel)) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  }
}

/** A side drawer over the page, closed by Escape or its button. */
export function Drawer({
  title,
  onClose,
  children,
  returnFocus,
  testId,
  className,
  contentKey,
}: DrawerProps): JSX.Element {
  const panelRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const headingId = useId();
  const onCloseRef = useRef(onClose);
  const returnRef = useRef(returnFocus);
  useLayoutEffect(() => {
    onCloseRef.current = onClose;
    returnRef.current = returnFocus;
  });
  useDrawerFocus(panelRef, bodyRef, returnRef, contentKey);
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>): void => {
    drawerKeys(e, panelRef.current, () => {
      onCloseRef.current();
    });
  };

  return (
    <div
      className="drawer-root"
      data-testid={testId ? `${testId}-root` : undefined}
    >
      <div
        className="drawer-scrim"
        onClick={() => {
          onCloseRef.current();
        }}
        aria-hidden="true"
      />
      <div
        ref={panelRef}
        className={`drawer-panel${className ? ` ${className}` : ''}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        data-testid={testId}
      >
        <div className="drawer-head">
          <button
            type="button"
            className="drawer-close"
            onClick={() => {
              onCloseRef.current();
            }}
            data-testid={testId ? `${testId}-close` : undefined}
          >
            <span className="drawer-close__back">‹ Back</span>
            <span className="drawer-close__x" aria-hidden="true">
              ✕
            </span>
            <span className="drawer-close__label">Close</span>
          </button>
          <h2 className="drawer-title" id={headingId}>
            {title}
          </h2>
        </div>
        <div className="drawer-body" ref={bodyRef}>
          {children}
        </div>
      </div>
    </div>
  );
}
