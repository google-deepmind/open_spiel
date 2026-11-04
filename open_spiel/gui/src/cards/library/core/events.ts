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

// The event stream: the library decides what an event IS, the game what
// it MEANS. Pure types plus the `handle` dispatch helper.
import type {CardId, Point} from './identity.ts';
import type {ItemRef, Target} from './targets.ts';

/** The modifier keys held during an event. */
export interface Modifiers {
  shift: boolean;
  ctrl: boolean;
  alt: boolean;
  meta: boolean;
}
/** No modifier keys held. */
export const NO_MODIFIERS: Modifiers = Object.freeze({
  shift: false,
  ctrl: false,
  alt: false,
  meta: false,
});
/** The cards a drag carries, and the slot they come from. */
export interface DragSource {
  ids: readonly CardId[];
  from: string;
  refs: readonly ItemRef[];
}
/** What every table event carries. */
export interface EventBase {
  target: Target;
  via: 'mouse' | 'touch' | 'pen' | 'keyboard' | 'synthetic';
  modifiers: Modifiers;
  /** Table px (null for keyboard). */
  point: Point | null;
  /** Viewport px (null for keyboard and synthetic). */
  client: Point | null;
  /** The raw DOM event; null only when via === 'synthetic'. */
  native: Event | null;
  /** Which mounted Table (see CardScene). */
  tableId: string;
}
/** An event on a table, as the library names it. */
export type TableEvent =
  | (EventBase & {type: 'activate'}) // click (detail 1), tap, Enter
  | (EventBase & {type: 'doubleActivate'}) // click detail 2 (instead of a 2nd activate), Shift+Enter
  | (EventBase & {type: 'contextMenu'}) // right-click, Shift+F10, ContextMenu key
  | (EventBase & {type: 'longPress'}) // held >= longPressMs without moving past threshold
  // One member per type, so Extract<TableEvent, { type: K }> (handle's
  // parameter types) finds each: a shared 'hoverStart' | 'hoverEnd' | …
  // member extracted to never, and handle({ hoverStart }) did not compile.
  | (EventBase & {type: 'hoverStart'})
  | (EventBase & {type: 'hoverEnd'})
  | (EventBase & {type: 'focus'})
  | (EventBase & {type: 'blur'})
  | (EventBase & {
      type: 'dragStart';
      source: DragSource;
      legal: readonly Target[];
    })
  | (EventBase & {
      type: 'dragOver';
      source: DragSource;
      over: Target | null;
      legal: boolean;
    })
  | (EventBase & {type: 'drop'; source: DragSource}) // target = resolved LEGAL target
  | (EventBase & {type: 'dragCancel'; source: DragSource})
  | (EventBase & {type: 'key'; key: string}); // keys the library does not bind
/** The kinds of table event. */
export type TableEventType = TableEvent['type'];
/** hold: keep the dropped cards at their drop pose until the next `items`
 *  change (server round-trip). After holdTimeoutMs (default 3000) they animate
 *  home and a console.error names the ids. */
// eslint-disable-next-line @typescript-eslint/no-invalid-void-type -- a handler that returns nothing is the common case; only a drop returns a hold.
export type EventResult = void | {hold: true};

/** A handler per event type, for `handle`. */
export type EventHandlers = {
  [K in TableEventType]?: (e: Extract<TableEvent, {type: K}>) => EventResult;
};

/** Typed dispatch table: a pure helper, not a prop. Unlisted types are ignored.
 *  handle({ activate: e => ..., drop: e => ({ hold: true }) }) -> onEvent */
export function handle(map: EventHandlers): (e: TableEvent) => EventResult {
  return e => {
    const f = map[e.type] as ((ev: TableEvent) => EventResult) | undefined;
    return f ? f(e) : undefined;
  };
}
