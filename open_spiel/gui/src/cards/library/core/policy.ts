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

// InteractionPolicy: everything a game says about interaction. Pure types.
import type {CardId, TableItem} from './identity.ts';
import type {EventResult, DragSource, TableEvent} from './events.ts';
import type {ItemRef, Target, TargetKind} from './targets.ts';

/** Everything a game says about interaction on a table. */
export interface InteractionPolicy {
  /** Every gesture, drag and key. The library decides what an event IS;
   *  the game decides what it MEANS. */
  onEvent(e: TableEvent): EventResult;
  /** true: the table is inert for now (the bot's turn, a submit in flight,
   *  the game over) but keeps every WORD: label, tooltip, describe,
   *  slotTooltip, slotLabel, slotSummary and role still apply. Nothing is a
   *  button or a tab stop, no event is delivered and nothing drags. Prefer
   *  this to dropping the policy, which also drops the words. Default false. */
  inert?: boolean;
  /** Passive events for a card that is NOT interactive (a face-down card, a
   *  covered card): true delivers hoverStart, hoverEnd, contextMenu and
   *  longPress on it (a hover peek, a right-click menu) WITHOUT making it
   *  activatable, draggable, focusable or a button; it keeps its role.
   *  Interactive cards get these events anyway. Default: false. */
  observe?(ref: ItemRef): boolean;
  /** Does this interactive card draw the hover look (the library 'hover'
   *  tag, CardStyle.tags.hover) under the pointer? false keeps the card
   *  interactive (pointer cursor, events, drag) but unlit: a pile a click
   *  draws from, shown as a clickable stock is, by the pointer alone.
   *  Default: true. */
  hoverLook?(ref: ItemRef): boolean;
  /** Can this slot be activated and focused as a slot (e.g. a face-down stock
   *  whose cards are inert)? Default: false. Pointer input on a non-interactive
   *  card resolves to the slot target when this returns true, and to nothing
   *  otherwise. */
  slotInteractive?(slot: string): boolean;
  /** Pressing (pointer, or Space on a focused card) picks up these ids, or false.
   *  Default: nothing drags. The ids must be in one slot and include ref.id. */
  canDrag?(ref: ItemRef, e: TableEvent): readonly CardId[] | false;
  /** Evaluated once per candidate target at dragStart, then cached. Must be pure.
   *  Default: false. */
  canDrop?(source: DragSource, target: Target): boolean;
  /** Candidate target kinds enumerated at dragStart. Default ['slot'].
   *  'insert' enumerates index 0..len for every slot in a hand mode or a fan. */
  dropKinds?: readonly TargetKind[];
  /** 'overlap': largest overlap of the dragged lead card's box with a legal
   *  target. 'pointer': the target under the pointer. Default 'overlap'. */
  dropResolve?: 'overlap' | 'pointer';
  // ---- a11y text hooks (the game owns words; the library ships neutral defaults) ----
  /** Accessible name. Default: cardLabel(card), or 'Face-down card'; the token's label. */
  label?(ref: ItemRef): string | undefined;
  /** Sets title=. Default: none. */
  tooltip?(ref: ItemRef): string | undefined;
  /** The card's role. An interactive card is always a 'button' (anything else
   *  throws); a non-interactive one may be 'img' (named) or 'none'
   *  (decorative: aria-hidden, no name). Default: 'button' when interactive;
   *  otherwise 'img' when the face is drawn or the game labels it (label()),
   *  and 'none' for an unlabelled back. */
  role?(ref: ItemRef): 'button' | 'img' | 'none' | undefined;
  /** aria-pressed on an interactive card, for cards whose activation TOGGLES
   *  something (a pass selection). undefined: no aria-pressed, which is the
   *  default: a card that plays or moves when activated is a plain button,
   *  not a toggle, whatever its tags. */
  pressed?(ref: ItemRef): boolean | undefined;
  /** aria-disabled on an interactive card the game will not play now (an
   *  unplayable card on your turn). The card stays interactive, so a tap
   *  still answers with why not, but the accessibility tree no longer
   *  offers it as a move.
   *  undefined or false: no aria-disabled. */
  disabled?(ref: ItemRef): boolean | undefined;
  /** aria-describedby text. Default: none. */
  describe?(ref: ItemRef): string | undefined;
  /** Sets title= on the slot, empty or not (e.g. 'Reset stock'). Default: none. */
  slotTooltip?(slot: string): string | undefined;
  /** Accessible name of a slot: used when it is interactive or summarised,
   *  and for an EMPTY slot's placeholder (without one the placeholder is
   *  decorative: aria-hidden, never announced). Slot ids are never names. */
  slotLabel?(slot: string): string | undefined;
  /** The group a slot belongs to in the accessibility tree: consecutive
   *  slots (spec order) with the same id are drawn inside one element, a
   *  role="group" named by groupLabel (a player's hand laid out over four
   *  suit slots, the four cards of a trick). The element has no box, so
   *  nothing moves; a group id should not come and go between renders (its
   *  slots would remount). Default: no groups. */
  slotGroup?(slot: string): string | undefined;
  /** Accessible name of a slot group (slotGroup). undefined: the group
   *  element has no role and no name. */
  groupLabel?(group: string): string | undefined;
  /** Returning a string collapses the slot to ONE a11y node with that label, and
   *  its cards become aria-hidden. Used for hidden hands: 'West: 13 cards'. */
  slotSummary?(slot: string, items: readonly TableItem[]): string | undefined;
  /** aria-live text for an event. undefined: the neutral default for drag
   *  events ('Picked up 3 cards', 'Dropped', 'Cancelled'), nothing for others. */
  announce?(e: TableEvent): string | undefined;
}

/** The neutral announcement for an event when the policy gives none. */
export function defaultAnnouncement(e: TableEvent): string | undefined {
  switch (e.type) {
    case 'dragStart': {
      const n = e.source.ids.length;
      return n === 1 ? 'Picked up 1 card' : `Picked up ${String(n)} cards`;
    }
    case 'drop':
      return 'Dropped';
    case 'dragCancel':
      return 'Cancelled';
    default:
      return undefined;
  }
}

/** The policy that answers events, drags and focus: none while inert. */
export function activePolicy(
  policy: InteractionPolicy | undefined,
): InteractionPolicy | undefined {
  return policy && !policy.inert ? policy : undefined;
}

/** policy.announce, falling back to the neutral default. */
export function announcementFor(
  policy: InteractionPolicy | undefined,
  e: TableEvent,
): string | undefined {
  const own = policy?.announce?.(e);
  return own ?? defaultAnnouncement(e);
}
