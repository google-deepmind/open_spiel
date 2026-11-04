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

// Fitting a table to its box: one responsive card width for the whole table.
import {
  minSlotScale,
  resolveTableSpec,
  tableClearance,
  validateTableSpec,
  type LayoutFit,
  type TableFit,
  type TableSource,
  type TableSpec,
} from '../table_spec.ts';
import {MAX_CARD_WIDTH, MIN_CARD_WIDTH} from '../card_widths.ts';
import type {SceneEnv, WidthBounds} from './scene_types.ts';

type Sized = Pick<TableSpec, 'width' | 'height'> &
  Partial<Pick<TableSpec, 'slots'>>;

// The largest card width at which the spec fits the box, unclamped.
function rawWidth(
  spec: Sized,
  box: {width: number; height: number},
  fit: TableFit,
): number {
  let raw = Math.floor(box.width / spec.width);
  if (fit === 'contain' && box.height > 0) {
    raw = Math.min(raw, Math.floor(box.height / spec.height));
  }
  return raw;
}

/** Card width for a spec in a box: the largest that fits, clamped to [min, max].
 *  Scaled slots raise the floor: the smallest slot's cards must meet `min`. */
export function fitTableCardWidth(
  spec: Sized,
  box: {width: number; height: number},
  fit: TableFit,
  bounds: WidthBounds = {},
): {cardWidth: number; overflow: boolean} {
  const min = Math.ceil(
    (bounds.min ?? MIN_CARD_WIDTH) /
      (spec.slots ? minSlotScale({slots: spec.slots}) : 1),
  );
  const max = bounds.max ?? MAX_CARD_WIDTH;
  const raw = rawWidth(spec, box, fit);
  return {cardWidth: Math.max(min, Math.min(max, raw)), overflow: raw < min};
}

/** How a layout fits a box, for selectTableVariant: its fitted card width,
 *  whether that is under the minimum, whether its width alone is (it would
 *  scroll sideways), and the unclamped width. */
export function fitLayout(
  spec: Sized,
  box: {width: number; height: number},
  fit: TableFit,
  bounds: WidthBounds = {},
): LayoutFit {
  const f = fitTableCardWidth(spec, box, fit, bounds);
  const across = fitTableCardWidth(spec, box, 'width', bounds);
  return {
    cardWidth: f.cardWidth,
    overflow: f.overflow,
    sideways: across.overflow,
    raw: rawWidth(spec, box, fit),
  };
}

/** The spec a table draws for a deck and style (a TableLayout resolved over
 *  every card width the table can draw), validated. */
export function resolveSceneSpec(
  source: TableSource,
  env: Pick<
    SceneEnv,
    'deck' | 'style' | 'cardWidth' | 'minCardWidth' | 'maxCardWidth'
  >,
): TableSpec {
  const lo = env.cardWidth ?? env.minCardWidth ?? MIN_CARD_WIDTH;
  const hi = env.cardWidth ?? Math.max(lo, env.maxCardWidth ?? MAX_CARD_WIDTH);
  const spec = resolveTableSpec(
    source,
    tableClearance(env.deck, env.style, {min: lo, max: hi}),
  );
  validateTableSpec(spec);
  return spec;
}
