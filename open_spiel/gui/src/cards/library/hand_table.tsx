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

// <HandTable>: a hand-and-table recipe (a CardStyle override plus a layout
// from the deck's HandMeasure to a TableSpec) drawn with <Table>. The recipe's CardStyle override applies to everything inside, and
// its layout is resolved from the hand measures of the deck and style in
// effect over every card width the table can draw, so a deck whose index
// needs more room gets a bigger table instead of a covered index.
import {type JSX, useMemo} from 'react';
import {getDeck} from './registry.ts';
import type {Deck} from './types.ts';
import {
  CardStyleProvider,
  useCardStyle,
  type CardStyle,
  type CardStyleOverride,
} from './style.ts';
import {measureHands, type HandMeasure} from './hand_measure.ts';
import {MAX_CARD_WIDTH, MIN_CARD_WIDTH} from './layouts.tsx';
import {Table, type TableProps} from './react/table.tsx';
import type {TableSpec} from './table_spec.ts';

/** A table recipe: where it comes from, the style it plays in, its layout. */
export interface HandTableRecipe {
  id: string;
  label: string;
  /** Where the layout comes from (a source, rules, or a reference), for
   *  display. */
  source: string;
  style: CardStyleOverride;
  layout: (m: HandMeasure) => TableSpec;
}

/** The spec a recipe draws for one deck and style, for cards `widths` px wide. */
export function resolveHandTable(
  recipe: HandTableRecipe,
  deck: Deck,
  style: CardStyle,
  widths: {min: number; max: number},
): TableSpec {
  return recipe.layout(measureHands(deck, style, widths));
}

/** A recipe drawn by <Table>: every Table prop but spec. */
export type HandTableProps = Omit<TableProps, 'spec'> & {
  recipe: HandTableRecipe;
};

function Resolved({recipe, ...table}: HandTableProps) {
  const style = useCardStyle();
  const deck = getDeck(style.deck);
  // The card widths <Table> can draw: an explicit width, or the fitted range.
  const lo = table.cardWidth ?? table.minCardWidth ?? MIN_CARD_WIDTH;
  const hi =
    table.cardWidth ?? Math.max(lo, table.maxCardWidth ?? MAX_CARD_WIDTH);
  const spec = useMemo(
    () => resolveHandTable(recipe, deck, style, {min: lo, max: hi}),
    [recipe, deck, style, lo, hi],
  );
  return <Table {...table} spec={spec} />;
}

/** A <Table> laid out by a hand-table recipe, fitted to its deck and style. */
export function HandTable(props: HandTableProps): JSX.Element {
  return (
    <CardStyleProvider value={props.recipe.style}>
      <Resolved {...props} />
    </CardStyleProvider>
  );
}
