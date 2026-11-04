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

// Validation of a TableSpec: every rule a spec must keep, checked once
// before it is drawn, so a malformed layout fails loud.
import {
  TABLE_PINS,
  turnsInPlace,
  type RegionSpec,
  type SlotBlock,
  type SlotSpec,
  type TableSpec,
  type TableVariant,
} from './table_spec.ts';
import {
  checkSlotAnchor,
  checkSlotExtras,
  checkSlotId,
  checkSlotMode,
  checkSlotPlace,
  checkSlotWrap,
} from './table_validate_slots.ts';

function checkVariant(
  spec: TableSpec,
  v: TableVariant,
  i: number,
  ids: readonly string[],
): void {
  const tag = `table variant ${String(i)}`;
  if ('variants' in v.spec) throw new Error(`${tag} has variants of its own`);
  if ('minWidth' in v.spec) {
    throw new Error(
      `${tag} sets minWidth inside its spec; a variant's threshold is TableVariant.minWidth`,
    );
  }
  if (
    v.minWidth !== undefined &&
    !(Number.isFinite(v.minWidth) && v.minWidth > 0)
  ) {
    throw new Error(
      `${tag} minWidth must be a finite px width > 0, got ${String(v.minWidth)}`,
    );
  }
  if ('minDrawnCardWidth' in v.spec) {
    throw new Error(
      `${tag} sets minDrawnCardWidth inside its spec; a variant's is TableVariant.minDrawnCardWidth`,
    );
  }
  if (
    v.minDrawnCardWidth !== undefined &&
    !(Number.isFinite(v.minDrawnCardWidth) && v.minDrawnCardWidth > 0)
  ) {
    throw new Error(
      `${tag} minDrawnCardWidth must be a finite px width > 0, got ${String(v.minDrawnCardWidth)}`,
    );
  }
  if (v.cost !== undefined && !(Number.isFinite(v.cost) && v.cost >= 1)) {
    throw new Error(
      `${tag} cost must be a finite number >= 1, got ${String(v.cost)}`,
    );
  }
  validateLayout(v.spec, `${tag}: `);
  const vids = v.spec.slots.map(s => s.id).sort();
  if (vids.length !== ids.length || vids.some((id, k) => id !== ids[k])) {
    const missing = ids.filter(id => !vids.includes(id));
    const extra = vids.filter(id => !ids.includes(id));
    throw new Error(
      `${tag} must name the base spec's slots: missing [${missing.join(', ')}], extra [${extra.join(', ')}]`,
    );
  }
  const rids = (spec.regions ?? []).map(r => r.id).sort();
  const vrids = (v.spec.regions ?? []).map(r => r.id).sort();
  if (vrids.length !== rids.length || vrids.some((id, k) => id !== rids[k])) {
    throw new Error(
      `${tag} must name the base spec's regions: base [${rids.join(', ')}], variant [${vrids.join(', ')}]`,
    );
  }
}

/** Throws on a malformed spec: duplicate ids, non-finite or negative
 *  positions, slots outside the table, rotations the check cannot map,
 *  captions or cues without room, variants that rename slots. */
export function validateTableSpec(spec: TableSpec): void {
  validateLayout(spec, '');
  if (spec.minWidth !== undefined) {
    if (!(Number.isFinite(spec.minWidth) && spec.minWidth > 0)) {
      throw new Error(
        `table minWidth must be a finite px width > 0, got ${String(spec.minWidth)}`,
      );
    }
    if (!spec.variants?.length) {
      throw new Error(
        `table minWidth ${String(spec.minWidth)} needs variants to draw below it`,
      );
    }
  }
  if (spec.minDrawnCardWidth !== undefined) {
    if (!(
      Number.isFinite(spec.minDrawnCardWidth) && spec.minDrawnCardWidth > 0
    )) {
      throw new Error(
        `table minDrawnCardWidth must be a finite px width > 0, got ${String(spec.minDrawnCardWidth)}`,
      );
    }
    if (!spec.variants?.length) {
      throw new Error(
        `table minDrawnCardWidth ${String(spec.minDrawnCardWidth)} needs variants to draw below it`,
      );
    }
  }
  const ids = spec.slots.map(s => s.id).sort();
  (spec.variants ?? []).forEach((v, i) => {
    checkVariant(spec, v, i, ids);
  });
}

const SIDES = ['top', 'bottom', 'left', 'right'];
const ALIGNS: readonly string[] = ['start', 'center', 'end'];
/** One or more CSS class names, single spaces between. */
const CLASS_NAMES = /^-?[_a-zA-Z][-_a-zA-Z0-9]*( -?[_a-zA-Z][-_a-zA-Z0-9]*)*$/;

function validateRegions(spec: Omit<TableSpec, 'variants'>, tag: string): void {
  const seen = new Set<string>();
  const slotIds = new Set(spec.slots.map(s => s.id));
  for (const r of spec.regions ?? []) {
    const where = `${tag}table region '${r.id}'`;
    if (!r.id) throw new Error(`${tag}table region without an id`);
    if (seen.has(r.id)) throw new Error(`duplicate ${where}`);
    if (slotIds.has(r.id)) throw new Error(`${where} has the id of a slot`);
    seen.add(r.id);
    if (!(
      Number.isFinite(r.width) &&
      Number.isFinite(r.height) &&
      r.width > 0 &&
      r.height > 0
    )) {
      throw new Error(
        `${where} size must be > 0, got ${String(r.width)} x ${String(r.height)}`,
      );
    }
    const fixed = r.x !== undefined || r.y !== undefined;
    if (fixed && r.attach) {
      throw new Error(`${where} sets both x/y and attach; give one`);
    }
    if (r.pin !== undefined) {
      if (!TABLE_PINS.includes(r.pin)) {
        throw new Error(
          `${where} pin must be one of ${TABLE_PINS.join('/')}, got ${JSON.stringify(r.pin)}`,
        );
      }
      if (!fixed) {
        throw new Error(
          `${where} is attached and pinned: an attached region moves with its slot`,
        );
      }
    }
    if (!fixed && !r.attach) throw new Error(`${where} needs x/y or attach`);
    if (fixed) {
      if (!(Number.isFinite(r.x) && Number.isFinite(r.y))) {
        throw new Error(
          `${where} needs both x and y, got (${String(r.x)}, ${String(r.y)})`,
        );
      }
      const x = r.x ?? 0;
      const y = r.y ?? 0;
      if (
        x < -1e-9 ||
        y < -1e-9 ||
        x + r.width > spec.width + 1e-9 ||
        y + r.height > spec.height + 1e-9
      ) {
        throw new Error(
          `${where} box (${String(x)}, ${String(y)}, ${String(r.width)} x ${String(r.height)}) lies outside the table [0, ${String(spec.width)}] x [0, ${String(spec.height)}]`,
        );
      }
    } else if (r.attach) {
      checkAttach(spec, r.attach, where, slotIds);
    }
    if (r.upright !== undefined && typeof r.upright !== 'boolean') {
      throw new Error(`${where} upright must be boolean`);
    }
    if (
      r.className !== undefined &&
      !(typeof r.className === 'string' && CLASS_NAMES.test(r.className))
    ) {
      throw new Error(
        `${where} className must be one or more space-separated class names, got ${JSON.stringify(r.className)}`,
      );
    }
  }
}

function checkAttach(
  spec: Omit<TableSpec, 'variants'>,
  a: NonNullable<RegionSpec['attach']>,
  where: string,
  slotIds: ReadonlySet<string>,
): void {
  if (!slotIds.has(a.slot)) {
    throw new Error(`${where} attaches to unknown slot '${a.slot}'`);
  }
  const alsoGiven: unknown = a.also;
  if (a.also !== undefined) {
    if (!Array.isArray(alsoGiven) || alsoGiven.length === 0) {
      throw new Error(
        `${where} attach.also must be a non-empty array of slot ids, got ${JSON.stringify(a.also)}`,
      );
    }
    const named: string[] = [a.slot, ...a.also];
    if (new Set(named).size !== named.length) {
      throw new Error(
        `${where} attach names a slot twice: ${named.join(', ')}`,
      );
    }
    for (const id of named) {
      const t = spec.slots.find(x => x.id === id);
      if (!t) {
        throw new Error(`${where} attach.also names unknown slot '${id}'`);
      }
      if (!turnsInPlace(t)) {
        throw new Error(
          `${where} attaches to several slots, but '${id}' is turned ${String(t.rotate)}°: a union is of unturned (or half-turned centred) slots`,
        );
      }
    }
  }
  if (!SIDES.includes(a.side)) {
    throw new Error(
      `${where} attach.side must be one of ${SIDES.join('/')}, got ${a.side}`,
    );
  }
  if (a.align !== undefined && !ALIGNS.includes(a.align)) {
    throw new Error(
      `${where} attach.align must be one of ${ALIGNS.join('/')}, got ${a.align}`,
    );
  }
  if (a.gap !== undefined && !(Number.isFinite(a.gap) && a.gap >= 0)) {
    throw new Error(`${where} attach.gap must be >= 0, got ${String(a.gap)}`);
  }
}

/** Throws unless SlotSpec.block is valid: a named block, centred on a finite axis or packed, of
 *  unturned non-free slots that all agree on it. */
export function validateBlock(
  s: SlotSpec,
  b: SlotBlock,
  where: string,
  blocks: Map<string, {slot: string; block: SlotBlock}>,
): void {
  const given: unknown = b;
  if (
    typeof given !== 'object' ||
    given === null ||
    typeof b.id !== 'string' ||
    b.id === ''
  ) {
    throw new Error(
      `${where} block must be { id, x?, y?, pack? } with a non-empty id, got ${JSON.stringify(b)}`,
    );
  }
  for (const axis of ['x', 'y'] as const) {
    const v = b[axis];
    if (v !== undefined && !(Number.isFinite(v) && v >= 0)) {
      throw new Error(
        `${where} block '${b.id}' ${axis} must be a finite number >= 0, got ${String(v)}`,
      );
    }
  }
  if (b.pack !== undefined && typeof b.pack !== 'boolean') {
    throw new Error(
      `${where} block '${b.id}' pack must be a boolean, got ${String(b.pack)}`,
    );
  }
  if (b.x === undefined && b.y === undefined && !b.pack) {
    throw new Error(
      `${where} block '${b.id}' centres on no axis and packs nothing`,
    );
  }
  for (const [axis, key] of [
    ['x', 'alignX'],
    ['y', 'alignY'],
  ] as const) {
    const a = b[key];
    if (a === undefined) continue;
    if (!ALIGNS.includes(a)) {
      throw new Error(
        `${where} block '${b.id}' ${key} must be one of ${ALIGNS.join('/')}, got ${a}`,
      );
    }
    if (b[axis] === undefined) {
      throw new Error(
        `${where} block '${b.id}' sets ${key} but no ${axis}: there is nothing to align on`,
      );
    }
  }
  if (s.pile === 'free') {
    throw new Error(
      `${where} is 'free' and in block '${b.id}': a block lays out piles and hands`,
    );
  }
  if (!turnsInPlace(s)) {
    throw new Error(
      `${where} is turned ${String(s.rotate)}° and in block '${b.id}': a block lays out unturned slots (or half-turned centred ones)`,
    );
  }
  const first = blocks.get(b.id);
  if (!first) {
    blocks.set(b.id, {slot: s.id, block: b});
  } else if (
    first.block.x !== b.x ||
    first.block.y !== b.y ||
    !!first.block.pack !== !!b.pack ||
    (first.block.alignX ?? 'center') !== (b.alignX ?? 'center') ||
    (first.block.alignY ?? 'center') !== (b.alignY ?? 'center')
  ) {
    throw new Error(
      `${where} block '${b.id}' is ${JSON.stringify(b)}, but slot '${first.slot}' has it as ${JSON.stringify(first.block)}: every slot of a block names the same centre, alignment and pack`,
    );
  }
}

function validateLayout(spec: Omit<TableSpec, 'variants'>, tag: string): void {
  if (!(spec.width > 0 && spec.height > 0)) {
    throw new Error(
      `table size must be > 0, got ${String(spec.width)} x ${String(spec.height)}`,
    );
  }
  validateRegions(spec, tag);
  if (spec.fillWidth !== undefined && typeof spec.fillWidth !== 'boolean') {
    throw new Error(
      `${tag}table fillWidth must be a boolean, got ${String(spec.fillWidth)}`,
    );
  }
  const seen = new Set<string>();
  const blocks = new Map<string, {slot: string; block: SlotBlock}>();
  for (const s of spec.slots) {
    const w = `${tag}table slot '${s.id}'`;
    const scale = s.scale ?? 1;
    checkSlotId(s, w, seen, blocks);
    checkSlotPlace(s, w, spec);
    checkSlotAnchor(s, w, scale);
    checkSlotMode(s, w);
    checkSlotWrap(s, w);
    checkSlotExtras(s, w, scale);
  }
}
