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

// The gin board's InteractionPolicy: what each gesture on the table means
// (gin_board_parts.ts lists them), and the words a screen reader and a
// tooltip give.

import {
  cardLabel,
  handle,
  type DragSource,
  type InteractionPolicy,
  type ItemRef,
  type Target,
  type TableEvent,
} from '../../cards/library/index.ts';
import {
  describeState,
  stateTooltip,
  type CardCellState,
} from './components/card_state.ts';
import {
  codeOf,
  isGhostRef,
  zoneOf,
  type GinRummyBoardSurfaceProps,
} from './gin_board_parts.ts';
import type {BoardFrame} from './gin_board_staging.ts';
import {GIN_SLOT, GRID_PREFIX, ginCard} from './gin_rummy_layout.ts';
import type {HandOrder} from './hooks/use_hand_order.ts';

type Ref = Pick<ItemRef, 'id' | 'slot'>;

/** What the policy reads: the props, the frame, the hand and its order. */
export interface PolicyContext {
  readonly p: GinRummyBoardSurfaceProps;
  readonly frame: BoardFrame;
  readonly handHidden: boolean;
  readonly fanned: boolean;
  readonly hand: HandOrder;
  /** Cards of the viewer's own fanned hand (no ghosts). */
  readonly ownCount: number;
}

function stateOf(c: PolicyContext, ref: Ref): CardCellState | undefined {
  const zone = zoneOf(ref.slot);
  const code = codeOf(ref.id);
  if (zone === null || code === null) return undefined;
  const states =
    zone === 'player' ? c.p.playerCardStates : c.frame.shownOpponentStates;
  return states.get(code);
}

function own(c: PolicyContext, ref: Ref): boolean {
  return !c.handHidden && ref.slot === GIN_SLOT.hand && !isGhostRef(ref);
}

/** A card of the actor's hand that can be laid off now. */
function layingOff(c: PolicyContext, ref: Ref): boolean {
  const drop = c.p.layoffDrop ?? null;
  const code = codeOf(ref.id);
  return (
    drop !== null &&
    code !== null &&
    !isGhostRef(ref) &&
    zoneOf(ref.slot) === drop.actorZone &&
    drop.canLayOff(code)
  );
}

/** A drop of `src` on a card of the knocker's meld it extends. */
function layoffFits(
  c: PolicyContext,
  src: Ref | undefined,
  onto: Ref,
): boolean {
  const drop = c.p.layoffDrop ?? null;
  if (drop === null || src === undefined || !layingOff(c, src)) return false;
  const zone = zoneOf(onto.slot);
  const srcCode = codeOf(src.id);
  const ontoCode = codeOf(onto.id);
  return (
    zone !== null &&
    zone !== drop.actorZone &&
    srcCode !== null &&
    ontoCode !== null &&
    drop.fits(srcCode, ontoCode)
  );
}

function act(c: PolicyContext, ref: ItemRef): void {
  if (c.frame.staging) {
    c.frame.settle();
    return;
  }
  if (ref.slot === GIN_SLOT.discard) c.frame.drawDiscard?.();
  else stateOf(c, ref)?.onDoubleClick?.();
}

function canDrag(c: PolicyContext, ref: ItemRef): readonly string[] | false {
  const {picked, order} = c.hand;
  if (own(c, ref)) {
    return picked.has(ref.id) ? order.filter(id => picked.has(id)) : [ref.id];
  }
  const staging = c.frame.staging;
  if (
    ref.slot === GIN_SLOT.discard &&
    c.fanned &&
    c.p.onDrawDiscard !== undefined &&
    !staging
  ) {
    return [ref.id];
  }
  if (layingOff(c, ref)) return [ref.id];
  if (
    ref.slot.startsWith(GRID_PREFIX.player) &&
    stateOf(c, ref)?.action === 'discard' &&
    !staging
  ) {
    return [ref.id];
  }
  return false;
}

function canDrop(c: PolicyContext, src: DragSource, t: Target): boolean {
  const staging = c.frame.staging;
  if (t.kind === 'card') {
    return src.ids.length === 1 && layoffFits(c, src.refs[0], t.ref);
  }
  if (t.kind === 'insert') {
    return (
      t.slot === GIN_SLOT.hand &&
      t.index <= c.ownCount &&
      (src.from === GIN_SLOT.hand ||
        (src.from === GIN_SLOT.discard && !staging))
    );
  }
  if (t.kind === 'slot' && t.slot === GIN_SLOT.discard) {
    const first = src.refs[0];
    return (
      !staging &&
      src.ids.length === 1 &&
      src.from !== GIN_SLOT.discard &&
      first !== undefined &&
      stateOf(c, first)?.action === 'discard'
    );
  }
  return false;
}

type DropEvent = Extract<TableEvent, {type: 'drop'}>;

function onDrop(c: PolicyContext, e: DropEvent): {hold: true} | undefined {
  const t = e.target;
  const first = e.source.ids[0];
  if (t.kind === 'insert') {
    if (e.source.from === GIN_SLOT.hand) {
      c.hand.move(e.source.ids, t.index);
      return undefined;
    }
    if (c.frame.staging || first === undefined) return undefined;
    c.hand.placeNext(first, t.index);
    c.p.onDrawDiscard?.();
    return {hold: true};
  }
  if (t.kind === 'slot' && t.slot === GIN_SLOT.discard) {
    const ref = e.source.refs[0];
    if (c.frame.staging || ref === undefined) return undefined;
    stateOf(c, ref)?.onDoubleClick?.();
    return {hold: true};
  }
  const code = first === undefined ? null : codeOf(first);
  if (t.kind === 'card' && code !== null) c.p.layoffDrop?.layOff(code);
  return undefined;
}

function onEvent(
  c: PolicyContext,
): (e: TableEvent) => ReturnType<InteractionPolicy['onEvent']> {
  return handle({
    activate: e => {
      const t = e.target;
      if (t.kind === 'slot') {
        if (t.slot === GIN_SLOT.stock && e.via === 'keyboard') {
          c.frame.drawStock?.();
        }
        return;
      }
      if (t.kind !== 'card') return;
      if (e.modifiers.shift && own(c, t.ref)) {
        c.hand.togglePick(t.ref.id);
        return;
      }
      // Enter acts on any card that does something; a click, on the fanned
      // hand's actionable cards.
      if (e.via === 'keyboard') act(c, t.ref);
      else if (own(c, t.ref) && stateOf(c, t.ref)?.isActionable === true) {
        act(c, t.ref);
      }
    },
    doubleActivate: e => {
      const t = e.target;
      if (t.kind === 'slot' && t.slot === GIN_SLOT.stock) c.frame.drawStock?.();
      else if (t.kind === 'card') act(c, t.ref);
    },
    key: e => {
      const t = e.target;
      if (
        (e.key === 'x' || e.key === 'X') &&
        t.kind === 'card' &&
        own(c, t.ref)
      ) {
        c.hand.togglePick(t.ref.id);
      }
    },
    drop: e => onDrop(c, e),
  });
}

function describe(c: PolicyContext, ref: ItemRef): string | undefined {
  const code = codeOf(ref.id);
  if (code === null) return undefined;
  const {cued} = c.frame;
  const last = ref.id === cued.discard ? ' Discarded last turn.' : '';
  if (ref.slot === GIN_SLOT.discard) {
    return c.p.onDrawDiscard === undefined
      ? `Upcard.${last}`
      : `Upcard.${last} Double-click to draw it.`;
  }
  const words = describeState(stateOf(c, ref), {
    ghost: isGhostRef(ref),
    picked: c.hand.picked.has(ref.id),
  });
  if (ref.id !== cued.taken) return words;
  return [words, 'Taken from the discard pile last turn.']
    .filter((w): w is string => w !== undefined && w !== '')
    .join(' ');
}

function announce(e: TableEvent): string | undefined {
  if (e.type !== 'drop') return undefined;
  const t = e.target;
  const names = e.source.ids
    .map(id => {
      const code = codeOf(id);
      return code === null ? id : cardLabel(ginCard(code));
    })
    .join(', ');
  if (t.kind === 'insert') return `${names} to position ${String(t.index + 1)}`;
  if (t.kind === 'slot' && t.slot === GIN_SLOT.discard) {
    return `Discarding ${names}`;
  }
  if (t.kind === 'card') return `Laying off ${names}`;
  return undefined;
}

/** The board's interaction policy for this render. */
export function ginBoardPolicy(c: PolicyContext): InteractionPolicy {
  const {p, frame, handHidden} = c;
  return {
    slotInteractive: s => s === GIN_SLOT.stock && p.onDrawStock !== undefined,
    slotLabel: s =>
      s === GIN_SLOT.stock
        ? `Stock, ${String(p.stockCount)} cards${frame.cued.stock === null ? '' : ', drawn from last turn'}: double-click to draw`
        : undefined,
    slotSummary: (s, its) =>
      s === GIN_SLOT.opponent
        ? `Opponent's hand, ${String(its.length)} cards`
        : s === GIN_SLOT.hand && handHidden
          ? `Hand, ${String(its.length)} cards face down`
          : undefined,
    // The upcard is never lit: it takes a draw as the stock does, shown by
    // the pointer and its tooltip alone.
    hoverLook: ref => ref.slot !== GIN_SLOT.discard,
    canDrag: ref => canDrag(c, ref),
    dropKinds:
      p.layoffDrop === null || p.layoffDrop === undefined
        ? ['insert', 'slot']
        : ['insert', 'slot', 'card'],
    canDrop: (src, t) => canDrop(c, src, t),
    onEvent: onEvent(c),
    describe: ref => describe(c, ref),
    tooltip: ref => {
      const code = codeOf(ref.id);
      if (code === null) return undefined;
      if (ref.slot === GIN_SLOT.discard) {
        return p.onDrawDiscard === undefined
          ? undefined
          : `${cardLabel(ginCard(code))}: double-click to draw`;
      }
      return stateTooltip(code, stateOf(c, ref));
    },
    announce,
  };
}
