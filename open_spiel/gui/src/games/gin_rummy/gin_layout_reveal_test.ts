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

// The hand's end reveals the opponent's hand in the room the table kept for
// it: the supply row and the player's own hand do not move.

import {describe, expect, it} from 'vitest';

import {
  DEFAULT_CARD_STYLE,
  getDeck,
  measureHands,
  tableClearance,
  type TableSpec,
} from '../../cards/library/index.ts';
import {FULL_DECK} from './deck_context.tsx';
import {ginTable} from './gin_layout_table.ts';
import {
  GIN_HAND_WIDTHS,
  GIN_REGION,
  GIN_SLOT,
  GRID_PREFIX,
  type GinTableOptions,
} from './gin_rummy_layout.ts';
import type {GridStyle} from './gin_view_settings.ts';

const STYLES: readonly GridStyle[] = ['normal', 'slots', 'no-bg', 'compact'];
const PLAYERS: ReadonlyArray<GinTableOptions['player']> = [
  'grid',
  'hand-overlap',
  'hand-arc',
];

/** A layout the library may draw: its spec, cost and card-width gate. */
interface Drawn {
  readonly spec: Omit<TableSpec, 'variants' | 'minWidth' | 'minDrawnCardWidth'>;
  readonly cost: number | undefined;
  readonly minDrawnCardWidth: number | undefined;
}

/** The base layout, then each variant. */
function layouts(o: GinTableOptions): Drawn[] {
  const deck = getDeck(DEFAULT_CARD_STYLE.deck);
  const m = measureHands(deck, DEFAULT_CARD_STYLE, GIN_HAND_WIDTHS);
  const clear = tableClearance(deck, DEFAULT_CARD_STYLE, GIN_HAND_WIDTHS);
  const base = ginTable(o, m)(clear);
  return [
    {spec: base, cost: 1, minDrawnCardWidth: base.minDrawnCardWidth},
    ...(base.variants ?? []).map(v => ({
      spec: v.spec,
      cost: v.cost ?? 1,
      minDrawnCardWidth: v.minDrawnCardWidth,
    })),
  ];
}

function isOpponent(id: string): boolean {
  return id === GIN_SLOT.opponent || id.startsWith(GRID_PREFIX.opponent);
}

/** Everything but the opponent's own cards and panel. */
function fixedPart(d: Drawn): unknown {
  return {
    width: d.spec.width,
    height: d.spec.height,
    cost: d.cost,
    minDrawnCardWidth: d.minDrawnCardWidth,
    slots: d.spec.slots.filter(s => !isOpponent(s.id)),
    regions: d.spec.regions?.filter(r => r.id !== GIN_REGION.opponentPanel),
  };
}

describe('the gin table at the hand end', () => {
  for (const grid of STYLES) {
    for (const player of PLAYERS) {
      it(`keeps the supply and the ${player} hand in place when the opponent is revealed (${grid})`, () => {
        const options = {grid, player, deck: FULL_DECK, handCards: 11};
        const hidden = layouts({...options, opponent: 'hidden'});
        const shown = layouts({...options, opponent: 'grid'});
        expect(hidden.length).toBeGreaterThan(1);
        expect(hidden.length).toBe(shown.length);
        hidden.forEach((drawn, i) => {
          const other = shown[i];
          if (other === undefined) throw new Error('no matching variant');
          expect(fixedPart(drawn)).toEqual(fixedPart(other));
        });
      });
    }
  }

  it('draws the face-down fan inside the room kept for the grid', () => {
    const options = {
      grid: 'normal' as const,
      player: 'grid' as const,
      deck: FULL_DECK,
    };
    const [hidden] = layouts({...options, opponent: 'hidden'});
    const [shown] = layouts({...options, opponent: 'grid'});
    if (hidden === undefined || shown === undefined) throw new Error('no spec');
    const fan = hidden.spec.slots.find(s => s.id === GIN_SLOT.opponent);
    const cells = shown.spec.slots.filter(s =>
      s.id.startsWith(GRID_PREFIX.opponent),
    );
    if (fan === undefined || cells.length === 0) throw new Error('no slots');
    const top = Math.min(...cells.map(s => s.y));
    const bottom = Math.max(...cells.map(s => s.y));
    expect(fan.y).toBeGreaterThanOrEqual(top);
    expect(fan.y).toBeLessThanOrEqual(bottom);
  });
});
