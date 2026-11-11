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

// The hearts table's InteractionPolicy: every card gesture (click,
// double-click, drag to the trick, keyboard) arrives through it as a card
// code, with the tooltips, pressed states, summaries and announcements.
import {
  cardLabel,
  handle,
  trickSlotOf,
  type InteractionPolicy,
  type ItemRef,
  type TableCard,
  type Target,
  type TrickLayout,
} from '../../../cards/library/index.ts';
import {SEAT_SLOT} from '../hearts_layout.ts';
import {toSharedCard, type CardCode, type PlayerPosition} from '../types.ts';
import {COMPASS, SEAT_OF_SLOT, points, type TableRules} from './table_items.ts';
import {PASS_COUNT, type TableLayoutProps} from './table_types.ts';

/** What the policy reads beside the rules. */
export interface PolicyInputs {
  readonly props: TableLayoutProps;
  readonly rules: TableRules;
  readonly trickLayout: TrickLayout;
  readonly passPending: boolean;
}

/** The seat a hand or 'toward' trick slot belongs to; throws for another. */
function seatOfSlot(slot: string): PlayerPosition {
  const seat = SEAT_OF_SLOT.get(slot);
  if (seat === undefined) {
    throw new Error(`hearts: slot '${slot}' belongs to no seat`);
  }
  return seat;
}

/** Whether `slot` is a seat's hand. */
function handSlot(slot: string): boolean {
  const seat = SEAT_OF_SLOT.get(slot);
  return seat !== undefined && SEAT_SLOT[seat].hand === slot;
}

function cardOf(t: Target): CardCode | null {
  return t.kind === 'card' ? (t.ref.id as CardCode) : null;
}

/** The tooltip of a card: what it is, and what a gesture on it does. */
function tooltipOf(
  i: PolicyInputs,
  seatName: (seat: PlayerPosition) => string,
  ref: ItemRef,
): string | undefined {
  const {isPassPhase, isChosen, isSending} = i.rules;
  const it = ref.item as TableCard;
  if (!it.card) return undefined;
  const code = ref.id as CardCode;
  const name = `${cardLabel(it.card)}${points(code)}`;
  if (!handSlot(ref.slot)) {
    // Who played each card of the trick (an overlapping trick's slot is no
    // seat's).
    const played = i.props.trickPile.find(t => t.card === code)?.position;
    const seat = played ?? SEAT_OF_SLOT.get(ref.slot);
    if (seat === undefined) {
      throw new Error(
        `hearts: card '${ref.id}' in slot '${ref.slot}' belongs to no seat`,
      );
    }
    return `${seatName(seat)} played the ${name}`;
  }
  if (ref.slot === SEAT_SLOT.bottom.hand && isSending(code)) {
    return `${name}: playing it now`;
  }
  if (isPassPhase && i.passPending && isChosen(code)) {
    return `${name}: passing it now`;
  }
  if (it.interactive === false) {
    return isPassPhase ? name : `${name}: not playable now`;
  }
  if (isPassPhase) {
    return isChosen(code)
      ? `${name}: chosen to pass, click to keep`
      : `${name}: click to pass it`;
  }
  return `${name}: click, drag to the trick, or press Enter to play`;
}

/** What a screen reader hears of an activation or a drop. */
function announceOf(
  i: PolicyInputs,
  e: Parameters<NonNullable<InteractionPolicy['announce']>>[0],
): string | undefined {
  const {isPassPhase, isChosen} = i.rules;
  const code =
    e.type === 'drop'
      ? (e.source.ids[0] as CardCode)
      : e.type === 'activate'
        ? cardOf(e.target)
        : null;
  if (!code) return undefined;
  const name = cardLabel(toSharedCard(code));
  if (!isPassPhase) return `Played the ${name}`;
  if (isChosen(code)) return `Kept the ${name}`;
  const n = i.props.selectedCards?.size ?? 0;
  return n < PASS_COUNT
    ? `Chose the ${name} to pass, ${String(n + 1)} of ${String(PASS_COUNT)}`
    : `Already ${String(PASS_COUNT)} cards chosen`;
}

/** The table's policy (see the file comment). */
export function tablePolicy(
  i: PolicyInputs,
  seatName: (seat: PlayerPosition) => string,
): InteractionPolicy {
  const {isPassPhase, faceUp, isChosen} = i.rules;
  const {onCardClick} = i.props;
  const choose = (code: CardCode | null): void => {
    if (code && onCardClick) onCardClick(code);
  };
  return {
    onEvent: handle({
      activate: e => {
        choose(cardOf(e.target));
      },
      // The first click of a double-click already chose the card; the second
      // is not another choice (no double play, no un-select).
      drop: e => {
        choose(e.source.ids[0] as CardCode);
        return {hold: true};
      },
    }),
    // In play, a playable card can be dragged to its seat's trick position.
    // The hover peek marks what a click would act on; a card already chosen
    // to pass is lifted by its selection, so it takes no second lift.
    hoverLook: (ref: ItemRef) =>
      !(isPassPhase && handSlot(ref.slot) && isChosen(ref.id as CardCode)),
    canDrag: ref => (!isPassPhase && handSlot(ref.slot) ? [ref.id] : false),
    dropKinds: ['slot'],
    canDrop: (src, t) =>
      t.kind === 'slot' &&
      t.slot === trickSlotOf(i.trickLayout, COMPASS[seatOfSlot(src.from)]),
    tooltip: (ref: ItemRef) => tooltipOf(i, seatName, ref),
    // Choosing pass cards toggles (click again to keep the card): those cards
    // say so. Playing a card is not a toggle, so in play there is no
    // aria-pressed.
    pressed: (ref: ItemRef) =>
      isPassPhase &&
      handSlot(ref.slot) &&
      (ref.item as TableCard).interactive !== false
        ? isChosen(ref.id as CardCode)
        : undefined,
    slotSummary: (slot, its) =>
      handSlot(slot) && !faceUp(seatOfSlot(slot))
        ? `${seatName(seatOfSlot(slot))}: ${String(its.length)} cards`
        : undefined,
    announce: e => announceOf(i, e),
  };
}
