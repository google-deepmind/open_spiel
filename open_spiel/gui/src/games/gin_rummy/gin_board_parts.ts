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

// The gin board's props and the pure helpers its parts share: the item id
// scheme, the face-down hands, and card identity across deals.

import {
  isToken,
  type ItemRef,
  type TableItem,
} from '../../cards/library/index.ts';
import {ginItem, type CardCellState} from './components/card_state.ts';
import type {GinHandResult, GinSeatNamer} from './gin_hand_result.ts';
import type {GinBoardCue} from './gin_move_cues.ts';
import {GIN_SLOT, GRID_PREFIX, type GinZone} from './gin_rummy_layout.ts';
import type {PlayerMeld} from './types.ts';

/** The finished hand, its few words for the stats and how seats are named. */
export interface BoardHandResult {
  result: GinHandResult;
  figure: {label: string; outcome: string};
  name: GinSeatNamer;
}

/** The defender's lay-off drop: a hand card onto a knocker's meld it extends. */
export interface LayoffDrop {
  actorZone: GinZone;
  canLayOff: (card: string) => boolean;
  fits: (card: string, onto: string) => boolean;
  layOff: (card: string) => void;
}

/**
 * The gin board: one shared <Table> over ginTable's spec, fitted to the
 * panel by the library. All data arrives as props, so it stays decoupled
 * from the session and could be driven from any source that produces the
 * same data.
 *
 * Interaction is one InteractionPolicy (gin_board_policy.ts):
 *  - double-click a card (or Shift+Enter) runs its state's action; a click
 *    (or Enter) on an actionable card of the fanned hand does too;
 *  - double-click the stock or the upcard (or Enter on it) draws; neither
 *    is lit on hover (policy.hoverLook): the pointer is the affordance;
 *  - drag a hand card (or the shift+clicked group) to a gap to reorder, a
 *    discardable card onto the discard pile to discard, the upcard into a
 *    gap of the hand to draw it there; Space starts the same drag from the
 *    keyboard, arrows pick the gap, Enter drops;
 *  - shift+click (or X) picks hand cards to move together;
 *  - in the defender's lay-off stage, drag a card that lays off onto the
 *    knocker's meld it extends to lay it off (layoffDrop).
 */
export interface GinRummyBoardSurfaceProps {
  showOpponentCards: boolean;
  opponentCardStates: Map<string, CardCellState>;
  opponentHandCount: number;
  stockCount: number;
  discardCard: string | null;
  knockCard: number | null;
  playerDeadwood: number | null;
  /**
   * The finished hand (ginHandResult), its few words for the stats
   * (ginResultFigure) and how seats are named; null while it is played. The
   * action bar's Result button opens its full accounting.
   */
  handResult?: BoardHandResult | null | undefined;
  /**
   * Each player's return, for a board with no hand result; ignored when
   * `handResult` is given.
   */
  returns?: readonly number[] | undefined;
  statusText: string;
  onDrawStock?: (() => void) | undefined;
  onDrawDiscard?: (() => void) | undefined;
  onPass?: (() => void) | undefined;
  onKnock?: (() => void) | undefined;
  passDisabled?: boolean | undefined;
  knockDisabled?: boolean | undefined;
  /** Laydown compose in progress: Done takes the action region. */
  laydown?:
    | {stage: string | null; onDone?: (() => void) | undefined}
    | null
    | undefined;
  waitingOnBot?: boolean | undefined;
  currentAutoPlay?: boolean | undefined;
  interactionBlockReason?: string | null | undefined;
  canInteract?: boolean | undefined;
  playerCardStates: Map<string, CardCellState>;
  playerHand: readonly string[];
  playerMelds: readonly PlayerMeld[];
  playerBestMeldIds: readonly string[];
  sortTrigger?: number | undefined;
  /**
   * Cards NOT in the hand that must still show in the fanned hand (as
   * ghosts, after it) so their action stays reachable: staged layoffs on
   * this player's spread, taken back by double-click.
   */
  playerGhostCards?: readonly string[] | undefined;
  onSort?: (() => void) | undefined;
  /**
   * The defender's lay-off stage: a card of the actor's hand (`actorZone`)
   * that can be laid off drags onto a card of the knocker's meld it extends
   * (`fits`), and the drop lays it off.
   */
  layoffDrop?: LayoffDrop | null | undefined;
  /**
   * The near hand is face down to this viewer (a spectator mid-hand:
   * `playerHand` is all backs). Drawn as a fanned row of backs, like the far
   * hand; nothing in it can be picked, dragged or reordered.
   */
  playerHandHidden?: boolean | undefined;
  /**
   * What the last turn did (lastTurnCue) and which zone its seat sits in.
   * The board marks the pile it drew from, the upcard it took and the card
   * it discarded until the next turn begins; a turn whose draw and discard
   * arrive together is drawn in two quick legs.
   */
  moveCue?: GinBoardCue | null | undefined;
  /** The discard pile, bottom first (the card under a staged discard). */
  discardPile?: readonly string[] | undefined;
}

/**
 * A face-down hand of `count` cards, ids `${prefix}-${i}`; the upcard it took
 * last turn (public), if any, face up at its end in place of a back.
 */
export function backs(
  prefix: string,
  count: number,
  taken: string | null,
): TableItem[] {
  return [
    ...Array.from(
      {length: Math.max(0, count - (taken === null ? 0 : 1))},
      (_, i): TableItem => ({id: `${prefix}-${String(i)}`, card: null}),
    ),
    ...(taken === null
      ? []
      : [ginItem(taken, taken, {type: 'card'}, {interactive: false})]),
  ];
}

/** A card's gin code from its table id: its code, or a ghost's `${zone}ghost-${code}`. */
export function codeOf(id: string): string | null {
  const m = /(?:^|ghost-)([2-9TJQKA][shdc])$/.exec(id);
  return m?.[1] ?? null;
}

/** Whether a table item is a ghost (a card drawn where it is not held). */
export function isGhostRef(ref: Pick<ItemRef, 'id'>): boolean {
  return ref.id.includes('ghost-');
}

/** Which zone a slot belongs to: the player's, the opponent's, or neither. */
export function zoneOf(slot: string): GinZone | null {
  if (slot === GIN_SLOT.hand || slot.startsWith(GRID_PREFIX.player)) {
    return 'player';
  }
  return slot.startsWith(GRID_PREFIX.opponent) ? 'opponent' : null;
}

function isKnown(i: TableItem): boolean {
  return !isToken(i) && i.card !== null;
}

/**
 * Identity continuity (TableCard.was) between two deals: a card that shows
 * up under a new id is the same physical card as one that just vanished:
 * the same face under another id (a staged layoff jumping to the other
 * grid), or a face-down card (the stock's top drawn into the hand; the
 * opponent's card discarded face up). So the library flies (and flips) it
 * instead of fading.
 */
export function withContinuity(
  prev: Readonly<Record<string, readonly TableItem[]>> | null,
  next: Record<string, TableItem[]>,
): Record<string, TableItem[]> {
  if (prev === null) return next;
  const prevItems = Object.values(prev).flat();
  const nextIds = new Set(
    Object.values(next)
      .flat()
      .map(i => i.id),
  );
  const prevIds = new Set(prevItems.map(i => i.id));
  const vanished = prevItems.filter(i => !nextIds.has(i.id));
  if (vanished.length === 0) return next;
  const used = new Set<string>();
  const take = (pred: (i: TableItem) => boolean): string | undefined => {
    // The last match is the top of its pile (the stock's top card).
    const hit = [...vanished].reverse().find(i => !used.has(i.id) && pred(i));
    if (hit !== undefined) used.add(hit.id);
    return hit?.id;
  };
  const out: Record<string, TableItem[]> = {};
  for (const [slot, list] of Object.entries(next)) {
    out[slot] = list.map(item => {
      if (prevIds.has(item.id) || isToken(item)) return item;
      if (item.was !== undefined && item.was !== '') return item;
      const code = codeOf(item.id);
      const was = isKnown(item)
        ? (take(i => codeOf(i.id) === code) ?? take(i => !isKnown(i)))
        : (take(isKnown) ?? take(i => !isKnown(i)));
      return was === undefined ? item : {...item, was};
    });
  }
  return out;
}
