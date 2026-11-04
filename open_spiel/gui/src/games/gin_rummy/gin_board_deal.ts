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

// What the board deals onto the table: the layout spec for the view, and
// the items in each slot (the grids or fanned hands, the stock, the upcard).

import {type RefObject, useLayoutEffect, useMemo} from 'react';

import {
  getDeck,
  measureHands,
  useCardStyle,
  type TableItem,
  type TableLayout,
} from '../../cards/library/index.ts';
import {
  ginItem,
  gridItems,
  type CardCellState,
} from './components/card_state.ts';
import type {DeckConfig} from './deck_context.tsx';
import {
  backs,
  withContinuity,
  type GinRummyBoardSurfaceProps,
} from './gin_board_parts.ts';
import type {BoardFrame} from './gin_board_staging.ts';
import {ginTable} from './gin_layout_table.ts';
import {oneCardOnePlace} from './gin_placement.ts';
import {
  GIN_HAND_WIDTHS,
  GIN_SLOT,
  GRID_PREFIX,
  MAX_HAND,
  gridCodes,
  type GinTableOptions,
} from './gin_rummy_layout.ts';
import type {GridStyle, HandView} from './gin_view_settings.ts';

const NO_GHOSTS: readonly string[] = [];

/** What grids, hands and piles the board shows, from the settings and props. */
export interface BoardView {
  readonly handView: HandView;
  readonly gridStyle: GridStyle;
  readonly deckConfig: DeckConfig;
  readonly handHidden: boolean;
  readonly fanned: boolean;
}

/** The codes a grid holds as cards, as one string. */
function heldKey(
  deck: DeckConfig,
  states: ReadonlyMap<string, CardCellState>,
): string {
  return gridCodes(deck)
    .filter(c => states.get(c)?.type === 'card')
    .join(' ');
}

/** The table's options and its layout. */
export function useBoardLayout(
  p: GinRummyBoardSurfaceProps,
  v: BoardView,
  frame: BoardFrame,
  ownCount: number,
): {opts: GinTableOptions; spec: TableLayout} {
  const style = useCardStyle();
  const deck = getDeck(style.deck);
  const measure = useMemo(
    () => measureHands(deck, style, GIN_HAND_WIDTHS),
    [deck, style],
  );
  // What each grid holds: the layout paints a grid's empty cells under its
  // cards (it never moves a cell).
  const playerHeld = heldKey(v.deckConfig, p.playerCardStates);
  const opponentHeld = heldKey(v.deckConfig, frame.shownOpponentStates);
  // The fanned hand keeps room for a full hand; staged layoffs (drawn as
  // ghosts after it) may ask for more while they are staged.
  const ghostCount = (p.playerGhostCards ?? NO_GHOSTS).filter(
    c => p.playerCardStates.get(c)?.type === 'card',
  ).length;
  const handCards = v.handHidden
    ? p.playerHand.length
    : v.fanned
      ? ownCount + ghostCount
      : undefined;
  const opponent = p.showOpponentCards ? 'grid' : 'hidden';
  const player =
    v.handView === 'arc' ? 'hand-arc' : v.fanned ? 'hand-overlap' : 'grid';
  // Only a hand past MAX_HAND changes the room the layout keeps.
  const handRoom = Math.max(MAX_HAND, handCards ?? 0);
  const {gridStyle, deckConfig} = v;
  const opts = useMemo(
    (): GinTableOptions => ({
      opponent,
      player,
      grid: gridStyle,
      deck: deckConfig,
      held: {
        player: playerHeld === '' ? [] : playerHeld.split(' '),
        opponent: opponentHeld === '' ? [] : opponentHeld.split(' '),
      },
      handCards: handRoom,
    }),
    [
      opponent,
      player,
      gridStyle,
      deckConfig,
      playerHeld,
      opponentHeld,
      handRoom,
    ],
  );
  const spec = useMemo(() => ginTable(opts, measure), [opts, measure]);
  return {opts, spec};
}

/** Everything a deal reads. */
export interface DealFields {
  readonly deckConfig: DeckConfig;
  readonly showOpponentCards: boolean;
  readonly shownOpponentStates: ReadonlyMap<string, CardCellState>;
  readonly shownOpponentCount: number;
  readonly shownHiddenHandCount: number;
  readonly takenFaceUp: string | null;
  readonly cueZone: 'player' | 'opponent' | null;
  readonly shownStockCount: number;
  readonly shownDiscard: string | null;
  readonly staging: boolean;
  readonly canDrawDiscard: boolean;
  readonly handHidden: boolean;
  readonly fanned: boolean;
  readonly order: readonly string[];
  readonly picked: ReadonlySet<string>;
  readonly playerGhostCards: readonly string[];
  readonly playerCardStates: ReadonlyMap<string, CardCellState>;
}

function fannedHand(d: DealFields): TableItem[] {
  return [
    ...d.order.map(c =>
      ginItem(c, c, d.playerCardStates.get(c), {
        picked: d.picked.has(c),
        interactive: true,
      }),
    ),
    ...d.playerGhostCards.flatMap(c => {
      const state = d.playerCardStates.get(c);
      return state?.type === 'card'
        ? [ginItem(`${GRID_PREFIX.player}ghost-${c}`, c, state, {ghost: true})]
        : [];
    }),
  ];
}

/** The items of every slot, one physical card in one place. */
export function dealItems(d: DealFields): {
  items: Record<string, TableItem[]>;
  faults: string[];
} {
  const out: Record<string, TableItem[]> = {};
  if (d.showOpponentCards) {
    Object.assign(
      out,
      gridItems('opponent', d.shownOpponentStates, d.deckConfig),
    );
  } else {
    out[GIN_SLOT.opponent] = backs(
      'opp',
      d.shownOpponentCount,
      d.cueZone === 'opponent' ? d.takenFaceUp : null,
    );
  }
  out[GIN_SLOT.stock] = Array.from(
    {length: Math.max(0, d.shownStockCount)},
    (_, i): TableItem => ({id: `stock-${String(i)}`, card: null}),
  );
  out[GIN_SLOT.discard] =
    d.shownDiscard === null
      ? []
      : [
          ginItem(
            d.shownDiscard,
            d.shownDiscard,
            {type: 'card'},
            {
              interactive: d.canDrawDiscard && !d.staging,
            },
          ),
        ];
  if (d.handHidden) {
    out[GIN_SLOT.hand] = backs(
      'own',
      d.shownHiddenHandCount,
      d.cueZone === 'player' ? d.takenFaceUp : null,
    );
  } else if (d.fanned) {
    out[GIN_SLOT.hand] = fannedHand(d);
  } else {
    Object.assign(out, gridItems('player', d.playerCardStates, d.deckConfig));
  }
  return oneCardOnePlace(out);
}

/**
 * The dealt items, with each card's identity carried over from the deal
 * last drawn (withContinuity), which is then this deal.
 */
export function useDeal(
  d: DealFields,
  lastDealRef: RefObject<Record<string, readonly TableItem[]> | null>,
): {items: Record<string, TableItem[]>; faults: string[]} {
  const {deckConfig, showOpponentCards, shownOpponentStates} = d;
  const {shownOpponentCount, shownHiddenHandCount, takenFaceUp, cueZone} = d;
  const {shownStockCount, shownDiscard, staging, canDrawDiscard} = d;
  const {
    handHidden,
    fanned,
    order,
    picked,
    playerGhostCards,
    playerCardStates,
  } = d;
  const dealt = useMemo(
    () =>
      dealItems({
        deckConfig,
        showOpponentCards,
        shownOpponentStates,
        shownOpponentCount,
        shownHiddenHandCount,
        takenFaceUp,
        cueZone,
        shownStockCount,
        shownDiscard,
        staging,
        canDrawDiscard,
        handHidden,
        fanned,
        order,
        picked,
        playerGhostCards,
        playerCardStates,
      }),
    [
      deckConfig,
      showOpponentCards,
      shownOpponentStates,
      shownOpponentCount,
      shownHiddenHandCount,
      takenFaceUp,
      cueZone,
      shownStockCount,
      shownDiscard,
      staging,
      canDrawDiscard,
      handHidden,
      fanned,
      order,
      picked,
      playerGhostCards,
      playerCardStates,
    ],
  );
  // `was` is computed against the deal last drawn.
  // Keyed on the deal alone: the deal last drawn is read when the deal
  // changes, not tracked (a re-render of the same deal keeps its `was`).
  const items = useMemo(
    // eslint-disable-next-line react-hooks/refs
    () => withContinuity(lastDealRef.current, dealt.items),
    [dealt, lastDealRef],
  );
  useLayoutEffect(() => {
    lastDealRef.current = items;
  }, [items, lastDealRef]);
  return {items, faults: dealt.faults};
}
