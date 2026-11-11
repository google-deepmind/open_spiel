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
  createContext,
  isValidElement,
  useContext,
  useEffect,
  useState,
  type JSX,
  type ReactNode,
} from 'react';
import {createPortal} from 'react-dom';
import './gameplay_notice.css';

/**
 * Status and error notices drawn OVER the gameplay table.
 *
 * A notice that appears inside the layout flow pushes the board down when it
 * shows and pulls it back when it goes: 52-64 px for every rejected move, and
 * once per bot turn for "Bot is thinking…". A status must not displace the
 * table.
 *
 * The board's panel owns one notice layer, an absolutely positioned stack
 * over the panel. Anything rendered through `<GameplayNotice>` is portalled
 * into that layer, wherever in the panel it is written, so a game can keep
 * its notice next to the code that decides it without the notice taking
 * layout space. Several notices at once stack in the layer, in render order.
 *
 * A game that would rather say it in the table itself ("Waiting for West…" on
 * the felt, a result plaque) renders that instead and does not use this.
 */

/** What a notice says: news, a turn, a wait, a warning or an error. */
export type GameplayNoticeTone =
  'info' | 'turn' | 'waiting' | 'warning' | 'error';

/** Which edge of the table the notice stack hugs. */
export type GameplayNoticePlacement = 'top' | 'bottom';

/**
 * `null` outside any shell; `{ layer: null }` inside a shell for the one
 * commit before the layer element is attached.
 */
type NoticeLayerValue = {layer: HTMLElement | null} | null;

const NoticeLayerContext = createContext<NoticeLayerValue>(null);

/** Provides the panel's notice layer to the notices in it. */
export const GameplayNoticeLayerProvider = NoticeLayerContext.Provider;

/** The notice layer element: absolutely positioned, no layout footprint. */
export function GameplayNoticeLayer({
  placement,
  layerRef,
}: {
  readonly placement: GameplayNoticePlacement;
  readonly layerRef: (element: HTMLDivElement | null) => void;
}): JSX.Element {
  return (
    <div
      ref={layerRef}
      className={`gameplay-notice-layer gameplay-notice-layer--${placement}`}
      data-testid="gameplay-notice-layer"
    />
  );
}

/**
 * A caller-chosen fade for transient errors. Not the default: an error often
 * asks the player to do something ("press Done to resume"), and a polling
 * failure repeats the same text without re-rendering, so a default fade would
 * hide it for good. Errors stay until dismissed or cleared by their owner.
 */
export const ERROR_AUTO_HIDE_MS = 10_000;

interface GameplayNoticeProps {
  tone?: GameplayNoticeTone;
  /** Show a close button. Defaults to true for errors and warnings. */
  dismissible?: boolean;
  /**
   * Hide after this many ms. Default: never. A notice lasts exactly as long
   * as the condition that renders it (or until dismissed).
   */
  autoHideMs?: number | null;
  /** Stable hook for tests and the displacement guard. */
  testId?: string;
  children?: ReactNode;
}

function isEmpty(node: ReactNode): boolean {
  return node === null || node === undefined || node === false || node === '';
}

/**
 * Text identity of a message, so a NEW message re-shows after a dismissal.
 * Walks JSX too (`Error: {x}`, the disconnect countdown), so a non-string
 * message that changes is a new message, not a stuck dismissal.
 */
function messageText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') {
    return '';
  }
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(messageText).join('');
  if (isValidElement<{children?: ReactNode}>(node)) {
    return messageText(node.props.children);
  }
  return '';
}

/** A notice over the board (see the file comment). */
export function GameplayNotice({
  tone = 'info',
  dismissible,
  autoHideMs,
  testId,
  children,
}: GameplayNoticeProps): JSX.Element | null {
  const shell = useContext(NoticeLayerContext);
  const key = messageText(children);
  const [hiddenKey, setHiddenKey] = useState<string | undefined>(undefined);
  const hidden = hiddenKey !== undefined && hiddenKey === key;

  const canDismiss = dismissible ?? (tone === 'error' || tone === 'warning');
  const hideAfter = autoHideMs ?? null;

  const empty = isEmpty(children);
  // A message that goes away and comes back is a new message: forget the
  // dismissal, so the same "Illegal move" twice shows twice.
  useEffect(() => {
    if (empty) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- a message that went away forgets its dismissal.
      setHiddenKey(undefined);
    }
  }, [empty]);
  useEffect(() => {
    if (hideAfter === null || hidden || empty) {
      return;
    }
    const timer = setTimeout(() => {
      setHiddenKey(key);
    }, hideAfter);
    return () => {
      clearTimeout(timer);
    };
  }, [hideAfter, hidden, key, empty]);

  if (empty || hidden) {
    return null;
  }

  const alerting = tone === 'error' || tone === 'warning';
  const notice = (
    <div
      className={`gameplay-notice gameplay-notice--${tone}`}
      role={alerting ? 'alert' : 'status'}
      aria-live={alerting ? 'assertive' : 'polite'}
      data-notice-tone={tone}
      data-testid={testId}
      title={key || undefined}
    >
      <span className="gameplay-notice__text">{children}</span>
      {canDismiss && (
        <button
          type="button"
          className="gameplay-notice__dismiss"
          aria-label="Dismiss"
          onClick={() => {
            setHiddenKey(key);
          }}
        >
          ×
        </button>
      )}
    </div>
  );

  // Outside a board's panel (a unit test rendering a board on its own) there
  // is no layer to portal into. The notice still renders, in place, so the
  // message is never silently dropped; inside the app every gameplay panel
  // renders in the shell. Inside a shell whose layer is not attached yet
  // (the first commit) it waits a frame rather than flash into the flow.
  if (shell === null) {
    return notice;
  }
  return shell.layer ? createPortal(notice, shell.layer) : null;
}
