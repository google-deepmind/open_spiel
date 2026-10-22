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
  type CSSProperties,
  type JSX,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import {createPortal} from 'react-dom';

import type {SavedLayout} from './layout_store.ts';
import {LayoutsMenu} from './layouts_menu.tsx';
import type {SavedLayouts} from './use_saved_layouts.ts';

/**
 * The layouts bar in the page's header: every saved layout a one-click
 * button, in order, the active one pressed; this game's after a "This game"
 * label, then those for all games after an "All games" label. When the
 * buttons outgrow the bar it scrolls sideways, keeping the active one in
 * view. With no layout active the bar says, in words, what is on screen.
 * "Layouts ▾" opens the menu of commands, under a line that says which
 * layout is in use and why. A failure is said in the menu, and on the menu
 * button as a word until it is dismissed.
 */
export function SavedLayoutsBar({
  layouts,
}: {
  readonly layouts: SavedLayouts;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const strip = useRef<HTMLDivElement>(null);
  const more = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const position = useMenuPosition(open, setOpen, root, more, menu);
  useKeepActiveInView(strip, layouts.active?.id ?? null);
  const {word, gameLayouts, allLayouts, error} = layouts;
  return (
    <div ref={root} className="layouts" role="group" aria-label="Saved layouts">
      <div ref={strip} className="layouts__strip">
        {word !== null && (
          <span className="layouts__state" title={layouts.using}>
            {word}
          </span>
        )}
        <LayoutButtons
          label="This game"
          list={gameLayouts}
          level="game"
          layouts={layouts}
        />
        <LayoutButtons
          label="All games"
          list={allLayouts}
          level="all"
          layouts={layouts}
        />
      </div>
      <button
        ref={more}
        type="button"
        className="button layouts__more"
        aria-haspopup="menu"
        aria-expanded={open}
        title={
          error === null
            ? 'Save, switch and manage layouts and defaults'
            : `Save, switch and manage layouts and defaults. The last action failed: ${error}`
        }
        onClick={() => {
          setOpen(!open);
        }}
      >
        {error === null ? (
          'Layouts ▾'
        ) : (
          <>
            Layouts <span className="layouts__failed">(failed)</span> ▾
          </>
        )}
      </button>
      <span className="visually-hidden" role="status" aria-live="polite">
        {layouts.status}
      </span>
      {open &&
        createPortal(
          <LayoutsMenu
            ref={menu}
            style={position}
            layouts={layouts}
            onDone={() => {
              setOpen(false);
              more.current?.focus();
            }}
          />,
          document.body,
        )}
    </div>
  );
}

function LayoutButtons({
  label,
  list,
  level,
  layouts,
}: {
  readonly label: string;
  readonly list: readonly SavedLayout[];
  readonly level: 'game' | 'all';
  readonly layouts: SavedLayouts;
}): JSX.Element | null {
  if (list.length === 0) return null;
  const {active, modified, gameDefault, allDefault} = layouts;
  // The word on a default's button; its group's label names the level.
  const tag = (layout: SavedLayout): string | null => {
    const forGame = gameDefault?.id === layout.id;
    if (level === 'game') return forGame ? '(default)' : null;
    if (allDefault?.id === layout.id) return '(default)';
    return forGame ? '(default here)' : null;
  };
  const note = (layout: SavedLayout): string =>
    gameDefault?.id === layout.id
      ? ' (default for this game)'
      : allDefault?.id === layout.id
        ? ' (default for all games)'
        : '';
  return (
    <>
      <span className="layouts__scope" role="separator" aria-label={label}>
        {label}
      </span>
      {list.map(layout => {
        const isActive = layout.id === active?.id;
        const shown = tag(layout);
        return (
          <button
            key={layout.id}
            type="button"
            className="button layouts__button"
            aria-pressed={isActive}
            title={`${layout.name}${note(layout)}${
              isActive && modified
                ? ' (modified: the dock differs from the saved layout)'
                : ''
            }`}
            onClick={() => {
              layouts.apply(layout);
            }}
          >
            <span className="layouts__name">{layout.name}</span>
            {shown !== null && <span className="layouts__tag">{shown}</span>}
          </button>
        );
      })}
    </>
  );
}

/**
 * Where the open menu goes: under its button, right-aligned with it, kept
 * there as the window resizes. Closes it on a press outside both and on
 * Escape.
 */
function useMenuPosition(
  open: boolean,
  setOpen: (open: boolean) => void,
  root: RefObject<HTMLElement | null>,
  more: RefObject<HTMLElement | null>,
  menu: RefObject<HTMLElement | null>,
): CSSProperties {
  const [position, setPosition] = useState<CSSProperties>({});
  useEffect(() => {
    if (!open) return undefined;
    const place = (): void => {
      const rect = more.current?.getBoundingClientRect();
      if (rect !== undefined) {
        setPosition({
          top: rect.bottom + 4,
          right: Math.max(4, window.innerWidth - rect.right),
        });
      }
    };
    place();
    const pressed = ({target}: PointerEvent): void => {
      if (!(target instanceof Node)) return;
      if (root.current?.contains(target) === true) return;
      if (menu.current?.contains(target) === true) return;
      setOpen(false);
    };
    const key = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        setOpen(false);
        more.current?.focus();
      }
    };
    window.addEventListener('resize', place);
    document.addEventListener('pointerdown', pressed);
    document.addEventListener('keydown', key);
    return () => {
      window.removeEventListener('resize', place);
      document.removeEventListener('pointerdown', pressed);
      document.removeEventListener('keydown', key);
    };
  }, [open, setOpen, root, more, menu]);
  return position;
}

/** The strip's scroll that shows [left, left + width) with least movement. */
export function scrollToShow(
  scrollLeft: number,
  viewport: number,
  left: number,
  width: number,
): number {
  if (left < scrollLeft) return left;
  if (left + width > scrollLeft + viewport) {
    return Math.max(0, left + width - viewport);
  }
  return scrollLeft;
}

// Scrolls `strip` sideways, as little as it can, to show `button`.
function scrollInto(strip: HTMLElement, button: HTMLElement): void {
  strip.scrollLeft = scrollToShow(
    strip.scrollLeft,
    strip.clientWidth,
    button.offsetLeft,
    button.offsetWidth,
  );
}

function useKeepActiveInView(
  strip: RefObject<HTMLElement | null>,
  activeId: string | null,
): void {
  useLayoutEffect(() => {
    const element = strip.current;
    if (element === null) return undefined;
    const show = (): void => {
      const pressed = element.querySelector<HTMLElement>(
        '[aria-pressed="true"]',
      );
      if (pressed !== null) scrollInto(element, pressed);
    };
    show();
    const observer = new ResizeObserver(show);
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, [strip, activeId]);
}
