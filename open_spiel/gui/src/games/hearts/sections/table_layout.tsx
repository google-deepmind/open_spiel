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

// The Hearts table: four seats around the trick (backs turned to face their
// players, faces upright for you), drawn by the shared <HandTable> from
// hearts_layout.ts. Names, the pass button, game over and the status
// message are table regions (table_centre.tsx); every card gesture (click,
// double-click, drag to the trick, keyboard) arrives through one
// InteractionPolicy as a card code (table_policy.ts).
import type {JSX} from 'react';

import {
  CardStyleProvider,
  HandTable,
  trickLayoutRecipe,
} from '../../../cards/library/index.ts';
import {
  DEFAULT_HEARTS_HAND_SHAPE,
  HEARTS_MAX_CARD_WIDTH,
  HEARTS_MIN_CARD_WIDTH,
  HEARTS_TABLES,
  SEAT_SLOT,
  type HeartsHandShape,
} from '../hearts_layout.ts';
import type {PlayerPosition} from '../types.ts';
import {centreOf, regionsOf} from './table_centre.tsx';
import {heartsItems, rulesOf, trickLayoutOf} from './table_items.ts';
import {tablePolicy} from './table_policy.ts';
import type {TableLayoutProps} from './table_types.ts';

export {heartsItems} from './table_items.ts';
export {
  PASS_COUNT,
  passSeats,
  type CardSize,
  type PassFlight,
  type TableLayoutProps,
} from './table_types.ts';

/** The table's recipes for the hand shape `shape`; throws on another. */
function tablesOf(
  shape: HeartsHandShape,
): (typeof HEARTS_TABLES)[HeartsHandShape] {
  const tables = HEARTS_TABLES[shape] as
    (typeof HEARTS_TABLES)[HeartsHandShape] | undefined;
  if (tables === undefined) {
    throw new Error(`hearts: unknown hand shape ${JSON.stringify(shape)}`);
  }
  return tables;
}

/** The Hearts table (see the file comment). */
export function TableLayout(props: TableLayoutProps): JSX.Element {
  const {
    players,
    trickPile,
    showOpponentCards = false,
    canShowAll = false,
    passPending = false,
    collect = null,
    cardSize = 'md',
    handShape = DEFAULT_HEARTS_HAND_SHAPE,
    deck,
    className = '',
  } = props;
  const maxCardWidth = HEARTS_MAX_CARD_WIDTH[cardSize] as number | undefined;
  if (maxCardWidth === undefined) {
    throw new Error(`hearts: unknown card size ${JSON.stringify(cardSize)}`);
  }
  const tables = tablesOf(handShape);
  const trickLayout = trickLayoutOf(props);
  const rules = rulesOf(props);
  const bySeat = new Map(players.map(p => [p.position, p] as const));
  const centre = centreOf(props);
  const items = heartsItems(props, {trick: centre === null, trickLayout});
  const seatName = (seat: PlayerPosition): string =>
    bySeat.get(seat)?.name ??
    `${seat.charAt(0).toUpperCase()}${seat.slice(1)} player`;
  const policy = tablePolicy(
    {props, rules, trickLayout, passPending},
    seatName,
  );
  const table = (
    <HandTable
      recipe={trickLayoutRecipe(
        tables[showOpponentCards ? 'open' : canShowAll ? 'hidden' : 'closed'],
        trickLayout,
      )}
      items={items}
      policy={policy}
      regions={regionsOf(props, centre)}
      fit="contain"
      minCardWidth={HEARTS_MIN_CARD_WIDTH}
      maxCardWidth={maxCardWidth}
      testId="hearts-cards"
      collect={
        collect && centre === null
          ? {
              key: collect.key,
              ids: trickPile.map(t => t.card),
              toward: SEAT_SLOT[collect.winner].hand,
              durationMs: collect.durationMs,
            }
          : null
      }
    />
  );
  return (
    <div
      className={`hearts-table ${className}`}
      data-testid="hearts-table"
      data-layout="standard"
      data-trick-layout={trickLayout}
    >
      {deck === undefined ? (
        table
      ) : (
        <CardStyleProvider value={{deck}}>{table}</CardStyleProvider>
      )}
    </div>
  );
}
