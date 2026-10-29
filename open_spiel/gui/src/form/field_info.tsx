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
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {createPortal} from 'react-dom';

import './form.css';

/** Pixels a popover keeps from the window's edges. */
const VIEWPORT_MARGIN = 8;

/** A rectangle's position and size, in pixels. */
interface Box {
  readonly left: number;
  readonly top: number;
  readonly bottom: number;
}

/**
 * Where a popover of `size` goes under `anchor`: its left edge on the
 * anchor's, moved left to stay inside `viewport`, and above the anchor when
 * there is no room below. Pixels.
 */
export function placePopover(
  anchor: Box,
  size: {readonly width: number; readonly height: number},
  viewport: {readonly width: number; readonly height: number},
): {left: number; top: number} {
  const maxLeft = viewport.width - size.width - VIEWPORT_MARGIN;
  const left = Math.max(VIEWPORT_MARGIN, Math.min(anchor.left, maxLeft));
  const below = anchor.bottom + 4;
  const fitsBelow = below + size.height <= viewport.height - VIEWPORT_MARGIN;
  const top = fitsBelow
    ? below
    : Math.max(VIEWPORT_MARGIN, anchor.top - 4 - size.height);
  return {left, top};
}

/**
 * Calls `close` while `open` on Escape, on a press outside every element of
 * `inside`, and when the window is resized.
 */
export function useDismiss(
  open: boolean,
  inside: ReadonlyArray<RefObject<HTMLElement | null>>,
  close: () => void,
): void {
  useEffect(() => {
    if (!open) return undefined;
    const dismiss = (): void => {
      close();
    };
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') dismiss();
    };
    const onPointerDown = (event: MouseEvent): void => {
      const {target} = event;
      const within =
        target instanceof Node &&
        inside.some(element => element.current?.contains(target) === true);
      if (!within) dismiss();
    };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('mousedown', onPointerDown);
    window.addEventListener('resize', dismiss);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('mousedown', onPointerDown);
      window.removeEventListener('resize', dismiss);
    };
  }, [open, inside, close]);
}

/**
 * A small "i" toggle that shows a setting's description in a popover, so the
 * label and the control stay on one compact row. It sits beside the field's
 * label, never inside it, so the label alone names the control. The popover
 * is drawn over the page, kept inside the window, and closes on Escape, on a
 * press outside it and when the window is resized.
 */
export function FieldInfo({
  description,
}: {
  readonly description: string;
}): JSX.Element {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState<{
    left: number;
    top: number;
  } | null>(null);
  const popoverId = useId();
  const button = useRef<HTMLButtonElement>(null);
  const popover = useRef<HTMLSpanElement>(null);

  useLayoutEffect(() => {
    const anchor = button.current?.getBoundingClientRect();
    const box = popover.current?.getBoundingClientRect();
    if (!open || anchor === undefined || box === undefined) {
      setPosition(null);
      return;
    }
    setPosition(
      placePopover(
        anchor,
        {width: box.width, height: box.height},
        {width: window.innerWidth, height: window.innerHeight},
      ),
    );
  }, [open, description]);

  const inside = useMemo(() => [button, popover], []);
  const close = useCallback(() => {
    setOpen(false);
  }, []);
  useDismiss(open, inside, close);

  return (
    <span className="field-info">
      <button
        ref={button}
        type="button"
        className="field-info__button"
        aria-label="More info"
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={() => {
          setOpen(!open);
        }}
      >
        {'\u2139'}
      </button>
      {open &&
        createPortal(
          <span
            ref={popover}
            id={popoverId}
            role="tooltip"
            className="field-info__popover"
            data-placed={position !== null}
            style={position ?? {left: 0, top: 0}}
          >
            {description}
          </span>,
          document.body,
        )}
    </span>
  );
}
