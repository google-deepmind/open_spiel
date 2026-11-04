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

// The rules each slot of a TableSpec keeps (table_validate.ts runs them).
import {assertTagName, isLibraryTag} from './core/tags.ts';
import {HAND_WRAPS, isHandMode} from './hand_geometry.ts';
import {
  CAPTION_HEIGHT,
  CUE_WIDTH,
  TABLE_PINS,
  type SlotBlock,
  type SlotSpec,
  type TableSpec,
} from './table_spec.ts';
import {validateBlock} from './table_validate.ts';

const RIGHT_ANGLES = [0, 90, 180, 270];

/** Throws unless slot `s` (named `w`) keeps the id rules. */
export function checkSlotId(
  s: SlotSpec,
  w: string,
  seen: Set<string>,
  blocks: Map<string, {slot: string; block: SlotBlock}>,
): void {
  if (s.block !== undefined) validateBlock(s, s.block, w, blocks);
  if (s.pin !== undefined && !TABLE_PINS.includes(s.pin)) {
    throw new Error(
      `${w} pin must be one of ${TABLE_PINS.join('/')}, got ${JSON.stringify(s.pin)}`,
    );
  }
  if (!s.id) throw new Error('table slot without an id');
  if (s.tags !== undefined) {
    const tags: unknown = s.tags;
    if (!Array.isArray(tags)) {
      throw new Error(`${w} tags must be an array of tag names`);
    }
    for (const t of s.tags) {
      assertTagName(t, `${w} tags`);
      if (isLibraryTag(t)) {
        throw new Error(
          `${w} tags: '${t}' is a library tag; the library sets those (games only style them)`,
        );
      }
    }
  }
  if (seen.has(s.id)) throw new Error(`duplicate ${w}`);
  seen.add(s.id);
  if (!(Number.isFinite(s.x) && Number.isFinite(s.y) && s.x >= 0 && s.y >= 0)) {
    throw new Error(
      `${w} position must be finite and >= 0, got (${String(s.x)}, ${String(s.y)})`,
    );
  }
}

/** Throws unless slot `s` (named `w`) keeps the place rules. */
export function checkSlotPlace(
  s: SlotSpec,
  w: string,
  spec: Omit<TableSpec, 'variants'>,
): void {
  const scale = s.scale ?? 1;
  if (!(scale > 0 && scale <= 1)) {
    throw new Error(`${w} scale must be in (0, 1], got ${String(s.scale)}`);
  }
  // A centre-anchored slot's first card spans x ± scale/2; a start-anchored one x .. x + scale;
  // an end-anchored one x - scale .. x.
  const right =
    s.x +
    (s.anchor === 'center' ? scale / 2 : s.anchor === 'end' ? 0 : scale) +
    (s.cue === 'continues' ? CUE_WIDTH : 0);
  if (right > spec.width + 1e-9) {
    throw new Error(
      `${w} at x=${String(s.x)} does not fit a table ${String(spec.width)} wide`,
    );
  }
  if (s.y > spec.height) {
    throw new Error(
      `${w} at y=${String(s.y)} starts below a table ${String(spec.height)} high`,
    );
  }
}

/** Throws unless slot `s` (named `w`) keeps the anchor rules. */
export function checkSlotAnchor(s: SlotSpec, w: string, scale: number): void {
  if (s.rotate !== undefined && !RIGHT_ANGLES.includes(s.rotate)) {
    throw new Error(
      `${w} rotate must be one of ${RIGHT_ANGLES.join('/')}, got ${String(s.rotate)}`,
    );
  }
  const anchor: string | undefined = s.anchor;
  if (
    anchor !== undefined &&
    anchor !== 'start' &&
    anchor !== 'center' &&
    anchor !== 'end'
  ) {
    throw new Error(
      `${w} anchor must be 'start', 'center' or 'end', got ${anchor}`,
    );
  }
  if (s.anchor === 'end') {
    if (s.x < scale - 1e-9) {
      throw new Error(
        `${w} is end-anchored at x=${String(s.x)}: its first card (${String(scale)} wide) would start left of the table`,
      );
    }
    if ((s.rotate ?? 0) !== 0) {
      throw new Error(
        `${w} is end-anchored and turned ${String(s.rotate)}°: 'end' anchors an unturned slot only`,
      );
    }
    if (s.cue !== undefined) {
      throw new Error(
        `${w} is end-anchored with a cue: the cue bands are placed for a start-anchored line`,
      );
    }
  }
}

/** Throws unless slot `s` (named `w`) keeps the mode rules. */
export function checkSlotMode(s: SlotSpec, w: string): void {
  if (s.faceDown !== undefined && typeof s.faceDown !== 'boolean') {
    throw new Error(
      `${w} faceDown must be a boolean, got ${String(s.faceDown)}`,
    );
  }
  if (
    s.pile === 'free' &&
    (s.stagger !== undefined ||
      s.visibleCount !== undefined ||
      s.maxExtent !== undefined)
  ) {
    throw new Error(
      `${w} is 'free': stagger, visibleCount and maxExtent do not apply`,
    );
  }
  if (s.holeRoom !== undefined) {
    const holeRoom: string = s.holeRoom;
    if (holeRoom !== 'trailing' && holeRoom !== 'leading') {
      throw new Error(
        `${w} holeRoom must be 'trailing' or 'leading', got ${JSON.stringify(s.holeRoom)}`,
      );
    }
    if (isHandMode(s.pile) || s.pile === 'free' || s.pile === 'squared') {
      throw new Error(
        `${w} sets holeRoom on a '${s.pile}' slot; only fan slots take positional holes`,
      );
    }
  }
  if (s.places !== undefined) {
    if (s.pile !== 'free') {
      throw new Error(`${w} has places, which only 'free' slots take`);
    }
    for (const [name, p] of Object.entries(s.places) as Array<
      [string, {x: number; y: number} | null | undefined]
    >) {
      if (
        !p ||
        !Number.isFinite(p.x) ||
        !Number.isFinite(p.y) ||
        p.x < 0 ||
        p.y < 0
      ) {
        throw new Error(
          `${w} place '${name}' must be a finite { x, y } >= 0, got ${JSON.stringify(p)}`,
        );
      }
    }
  }
}

/** Throws unless slot `s` (named `w`) keeps the wrap rules. */
export function checkSlotWrap(s: SlotSpec, w: string): void {
  if (
    s.wrap !== undefined ||
    s.maxRows !== undefined ||
    s.rowGap !== undefined
  ) {
    if (!isHandMode(s.pile)) {
      throw new Error(
        `${w} sets wrap/maxRows/rowGap on a '${s.pile}' slot; only hand slots wrap`,
      );
    }
    if (s.wrap !== undefined && !HAND_WRAPS.includes(s.wrap)) {
      throw new Error(
        `${w} wrap must be one of ${HAND_WRAPS.join('/')}, got ${JSON.stringify(s.wrap)}`,
      );
    }
    if (
      s.maxRows !== undefined &&
      !(Number.isInteger(s.maxRows) && s.maxRows >= 1)
    ) {
      throw new Error(
        `${w} maxRows must be a whole number >= 1, got ${String(s.maxRows)}`,
      );
    }
    if (
      s.rowGap !== undefined &&
      !(Number.isFinite(s.rowGap) && s.rowGap >= 0)
    ) {
      throw new Error(
        `${w} rowGap must be a finite number >= 0, got ${String(s.rowGap)}`,
      );
    }
    if (
      s.wrap === 'never' &&
      (s.maxRows !== undefined || s.rowGap !== undefined)
    ) {
      throw new Error(
        `${w} sets maxRows/rowGap with wrap 'never'; a hand that never wraps has one row`,
      );
    }
    if (s.wrap === 'auto' && s.maxExtent === undefined) {
      throw new Error(
        `${w} wraps ('auto') but has no maxExtent to wrap against`,
      );
    }
  }
}

/** Throws unless slot `s` (named `w`) keeps the extras rules. */
export function checkSlotExtras(s: SlotSpec, w: string, scale: number): void {
  if (s.capacity !== undefined) {
    if (!isHandMode(s.pile)) {
      throw new Error(
        `${w} sets capacity on a '${s.pile}' slot; only hand slots have one`,
      );
    }
    if (!(Number.isInteger(s.capacity) && s.capacity >= 1)) {
      throw new Error(
        `${w} capacity must be a whole number >= 1, got ${String(s.capacity)}`,
      );
    }
  }
  if (s.arcRadius !== undefined && s.pile !== 'hand-arc') {
    throw new Error(
      `${w} sets arcRadius on a '${s.pile}' slot; only hand-arc slots have one`,
    );
  }
  if (s.maxExtent !== undefined && !(s.maxExtent >= scale)) {
    throw new Error(
      `${w} maxExtent must be >= one of its card widths (${String(scale)}), got ${String(s.maxExtent)}`,
    );
  }
  if (s.caption !== undefined && s.y < CAPTION_HEIGHT - 1e-9) {
    throw new Error(
      `${w} has a caption but y=${String(s.y)} leaves no ${String(CAPTION_HEIGHT)} band above it`,
    );
  }
  if (s.cue === 'continued' && s.x < CUE_WIDTH - 1e-9) {
    throw new Error(
      `${w} cue 'continued' needs x >= ${String(CUE_WIDTH)}, got ${String(s.x)}`,
    );
  }
  const cue: string | undefined = s.cue;
  if (cue !== undefined && cue !== 'continues' && cue !== 'continued') {
    throw new Error(`${w} cue must be 'continues' or 'continued', got ${cue}`);
  }
}
