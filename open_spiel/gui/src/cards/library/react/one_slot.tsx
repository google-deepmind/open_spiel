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

// <Pile items> and <Hand items>: the presentational wrappers as a
// <Table> over a one-slot spec sized to what the slot draws, so a lone pile
// or hand gets the same events, policy, tags, a11y and escape hatches.
import {type JSX, useMemo} from 'react';
import {getDeck} from '../registry.ts';
import {useCardStyle} from '../style.ts';
import type {FreePlacement, SlotSpec, TableSpec} from '../table_spec.ts';
import type {TableItem} from '../core/identity.ts';
import {isFaceUp, isToken} from '../core/identity.ts';
import type {InteractionPolicy} from '../core/policy.ts';
import {slotBox} from '../core/scene.ts';
import {Table, type TableProps} from './table.tsx';

/** A one-slot table: its slot, cards and policy. */
export interface OneSlotProps {
  /** Items bottom first (TableCard; ids required). In a fan pile an entry
   *  may be null: a positional hole (see TableProps.items). */
  items: ReadonlyArray<(TableItem & Partial<FreePlacement>) | null>;
  /** Absent: an inert display (no buttons, no tab stops). */
  policy?: InteractionPolicy | undefined;
  renderOverlay?: TableProps['renderOverlay'] | undefined;
  renderCard?: TableProps['renderCard'] | undefined;
  /** The slot id reported in events (default 'pile' / 'hand'). */
  slotId?: string | undefined;
  tableId?: string | undefined;
}

/** A table of one slot: a hand or a pile on its own. */
export function OneSlotTable({
  slot,
  cardWidth,
  items,
  policy,
  renderOverlay,
  renderCard,
  tableId,
}: OneSlotProps & {
  slot: Omit<SlotSpec, 'x' | 'y'>;
  cardWidth: number;
}): JSX.Element {
  const style = useCardStyle();
  const deck = getDeck(style.deck);
  if (!(cardWidth > 0)) {
    throw new Error(
      `one-slot table card width must be > 0, got ${String(cardWidth)}`,
    );
  }
  const spec = useMemo<TableSpec>(() => {
    const s: SlotSpec = {...slot, x: 0, y: 0};
    const cards = items.map(i =>
      i === null
        ? null
        : {
            faceUp: !isToken(i) && isFaceUp(i),
            selected: (i.tags ?? []).includes('selected'),
          },
    );
    const b = slotBox(s, cards, deck, style, cardWidth);
    return {
      width: Math.max(1, b.width / cardWidth),
      height: Math.max(1 / deck.aspect, b.height / cardWidth),
      slots: [s],
    };
  }, [slot, items, deck, style, cardWidth]);
  return (
    <div
      data-one-slot={slot.id}
      style={{display: 'inline-block', flex: 'none'}}
    >
      <Table
        spec={spec}
        items={{[slot.id]: items}}
        policy={policy}
        cardWidth={cardWidth}
        renderOverlay={renderOverlay}
        renderCard={renderCard}
        tableId={tableId}
      />
    </div>
  );
}
