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

import type {JSX} from 'react';
import {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';

import './setup.css';

// The confirmation's width and the height it is allowed for, and the least
// gap it keeps from the window's edges. Pixels.
const CONFIRM_WIDTH = 280;
const CONFIRM_HEIGHT = 140;
const MARGIN = 8;

/** Where the confirmation goes, in pixels from the window's edges. */
export interface ConfirmPlace {
  /** From the top edge, when it opens below its button. */
  readonly top?: number;
  /** From the bottom edge, when there is no room below and it opens above. */
  readonly bottom?: number;
  readonly left: number;
}

/**
 * Where the confirmation goes for a Reset button at `rect` in a window
 * `height` pixels tall: under the button, its right edge on the button's,
 * kept inside the window, and over the button when there is no room under
 * it.
 */
export function placeConfirm(
  rect: {readonly top: number; readonly bottom: number; readonly right: number},
  height: number,
): ConfirmPlace {
  const left = Math.max(MARGIN, rect.right - CONFIRM_WIDTH);
  return rect.bottom + 6 + CONFIRM_HEIGHT <= height - MARGIN
    ? {top: rect.bottom + 6, left}
    : {bottom: height - rect.top + 6, left};
}

/** The properties of ResetControl. */
export interface ResetControlProps {
  readonly busy: boolean;
  /** Whether the game has history that a reset clears. */
  readonly hasHistory: boolean;
  readonly onReset: () => void;
}

/**
 * A Reset button that asks first: "Reset Game?", with a Reset that starts
 * the game again and a Cancel. Escape cancels too.
 */
export function ResetControl({
  busy,
  hasHistory,
  onReset,
}: ResetControlProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<ConfirmPlace>({top: 0, left: 0});
  const button = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      button.current?.focus();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);
  return (
    <>
      <button
        ref={button}
        type="button"
        className="button button--ghost"
        aria-expanded={open}
        aria-haspopup="dialog"
        disabled={busy && !open}
        onClick={() => {
          const rect = button.current?.getBoundingClientRect();
          if (rect !== undefined) {
            setPlace(placeConfirm(rect, window.innerHeight));
          }
          setOpen(!open);
        }}
      >
        Reset
      </button>
      {open && (
        <ResetConfirm
          place={place}
          busy={busy}
          hasHistory={hasHistory}
          onReset={() => {
            onReset();
            setOpen(false);
          }}
          onCancel={() => {
            setOpen(false);
          }}
        />
      )}
    </>
  );
}

interface ResetConfirmProps {
  /** Where the confirmation goes, in pixels from the window's corner. */
  readonly place: ConfirmPlace;
  readonly busy: boolean;
  readonly hasHistory: boolean;
  readonly onReset: () => void;
  readonly onCancel: () => void;
}

// The confirmation, drawn over the page so that no panel clips it.
function ResetConfirm({
  place,
  busy,
  hasHistory,
  onReset,
  onCancel,
}: ResetConfirmProps): JSX.Element {
  const titleId = useId();
  return createPortal(
    <div
      className="setup-confirm"
      role="dialog"
      aria-labelledby={titleId}
      style={place}
    >
      <div className="setup-confirm__warning">
        <strong id={titleId}>Reset Game?</strong>
        <p>
          {hasHistory
            ? 'This will clear all history and start a new game.'
            : 'This will start a new game.'}
        </p>
      </div>
      <div className="setup-set-state__actions">
        <button
          type="button"
          className="button button--danger"
          disabled={busy}
          onClick={onReset}
        >
          Reset
        </button>
        <button
          type="button"
          className="button button--ghost"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>
    </div>,
    document.body,
  );
}
