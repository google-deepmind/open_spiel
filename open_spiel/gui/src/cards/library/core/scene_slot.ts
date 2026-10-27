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

// One slot laid out at one card width: its box, and where each of its items
// is drawn (layoutScene, scene.ts).
import type {IndexGeometry} from '../types.ts';
import type {FreePlacement, SlotSpec} from '../table_spec.ts';
import {
  computePileGeometry,
  fanAxis,
  type PileGeometry,
} from '../pile_geometry.ts';
import {isHandMode, type HandGeometry} from '../hand_geometry.ts';
import {cornerIndex, quarterTurnIndex} from '../index_scale.ts';
import {
  isFaceUp,
  isToken,
  itemInteractive,
  type Rect,
  type TableCard,
  type TableItem,
} from './identity.ts';
import {isLibraryTag, resolveTagStyle, type TagStyle} from './tags.ts';
import {defined, definedFields} from './defined.ts';
import {
  badgeCorner,
  fanTurns,
  indexBounds,
  itemSize,
  rotateAbout,
  toTableBox,
  toTablePoint,
} from './scene_geometry.ts';
import {
  computeSlotBox,
  handGeometry,
  handWrapOf,
  toCapacity,
} from './scene_slot_box.ts';
import {
  VERTICAL,
  type Placement,
  type SceneEnv,
  type SlotBox,
  type SlotCardInfo,
  type SlotEntry,
  type SlotItems,
  type SlotLayout,
} from './scene_types.ts';

type Item = TableItem & Partial<FreePlacement>;

// What every step of laying one slot out shares.
interface SlotContext {
  s: SlotSpec;
  entryList: SlotItems;
  list: Item[];
  env: SceneEnv;
  cardWidth: number;
  cw: number;
  ch: number;
  scale: number;
  maxExtent: number | undefined;
  tags: Record<string, TagStyle>;
  baseIndex: IndexGeometry;
}

// The geometry of a laid-out slot, beyond its entries.
interface SlotShape {
  kind: SlotLayout['kind'];
  pile: PileGeometry | null;
  hand: HandGeometry | null;
  compressed: boolean;
  overflow: boolean;
  frame: {width: number; height: number};
}

function where(c: SlotContext, pos: number): string {
  return `slot '${c.s.id}' position ${String(pos)} ('${String(c.entryList[pos]?.id)}')`;
}

function gameStyle(c: SlotContext, it: TableItem): TagStyle {
  return resolveTagStyle(
    (it.tags ?? []).filter(t => !isLibraryTag(t)),
    c.tags,
  );
}

/** A free item's offset: its own, or the slot's place it names. */
function atOf(c: SlotContext, it: Item, pos: number): {x: number; y: number} {
  const at = it.at;
  if (typeof at === 'string') {
    const p = c.s.places?.[at];
    if (!p) {
      throw new Error(
        `${where(c, pos)} takes place '${at}', which slot '${c.s.id}' does not have (places: ${Object.keys(c.s.places ?? {}).join(', ') || 'none'})`,
      );
    }
    return p;
  }
  if (!at || !Number.isFinite(at.x) || !Number.isFinite(at.y)) {
    throw new Error(
      `${where(c, pos)} is in a 'free' slot and needs at: { x, y } or a place name`,
    );
  }
  return at;
}

// Positional holes (null): a fan keeps their room so later cards never
// move; nothing is drawn or hit-tested there, and positions count them.
function checkHoles(s: SlotSpec, entryList: SlotItems): void {
  const fan = !isHandMode(s.pile) && s.pile !== 'free' && s.pile !== 'squared';
  entryList.forEach((it, pos) => {
    if (it === null && !fan) {
      throw new Error(
        `slot '${s.id}' position ${String(pos)} is a positional hole (null); only fan slots take holes, not '${s.pile}'`,
      );
    }
  });
}

function checkItem(c: SlotContext, it: Item, pos: number): void {
  const {s} = c;
  if (s.pile === 'free') {
    atOf(c, it, pos);
  } else {
    if (it.at !== undefined) {
      throw new Error(
        `${where(c, pos)} has 'at', which only 'free' slots take`,
      );
    }
    if (isToken(it)) {
      throw new Error(`${where(c, pos)} is a token; tokens go in 'free' slots`);
    }
  }
  for (const t of it.tags ?? []) {
    if (!/^[a-z][a-z0-9-]*$/.test(t)) {
      throw new Error(
        `${where(c, pos)} has an invalid tag ${JSON.stringify(t)}`,
      );
    }
    if (isLibraryTag(t)) {
      throw new Error(
        `${where(c, pos)} carries the library tag '${t}'; the library sets those (games only style them)`,
      );
    }
  }
  if (!isToken(it) && it.turn && s.pile === 'hand-arc') {
    throw new Error(
      `${where(c, pos)} has turn ${String(it.turn)}, which a 'hand-arc' hand cannot carry (a turned card on an arc lies across its neighbours): turn cards in a 'hand-overlap' or 'hand-flat' hand, a fan, or a squared or free slot`,
    );
  }
}

function info(c: SlotContext, it: TableItem): SlotCardInfo {
  const lift = gameStyle(c, it).lift ?? 0;
  return {
    faceUp: isToken(it) ? false : isFaceUp(it),
    lifted: lift > 0 || (it.tags ?? []).includes('selected'),
    turn: isToken(it) ? 0 : (it.turn ?? 0),
  };
}

// The box round a free slot's items.
function freeBoxOf(c: SlotContext): {width: number; height: number} {
  const unit = c.cardWidth * c.scale;
  let w = c.cw;
  let h = c.ch;
  for (const [pos, it] of c.list.entries()) {
    const at = atOf(c, it, pos);
    const sz = itemSize(it, c.cw, c.ch, unit);
    w = Math.max(w, at.x * unit + sz.width);
    h = Math.max(h, at.y * unit + sz.height);
  }
  return {width: w, height: h};
}

// The visible index box of an entry, as fractions of its local rect.
function entryIndex(
  c: SlotContext,
  it: TableItem,
  drawnFaceUp: boolean,
  rotatedBase: boolean,
  corner: SlotEntry['corner'],
): IndexGeometry | null {
  const known = !isToken(it) && it.card !== null && drawnFaceUp;
  if (!known) return null;
  const upright =
    corner === 'top-left'
      ? c.baseIndex
      : cornerIndex(c.env.deck, c.env.style.indexScale, 'bottom-right');
  return rotatedBase ? quarterTurnIndex(c.baseIndex) : upright;
}

function makeEntry(
  c: SlotContext,
  box: SlotBox,
  it: TableItem,
  pos: number,
  local: Rect,
  localRotate: number,
  extra: Partial<SlotEntry>,
): SlotEntry {
  const lift = extra.lift ?? 0;
  const r = {...local, y: local.y - lift};
  const drawnFaceUp = extra.drawnFaceUp ?? (isToken(it) ? false : isFaceUp(it));
  const rotatedBase = extra.rotatedBase ?? false;
  const corner = extra.corner ?? 'top-left';
  const idx = entryIndex(c, it, drawnFaceUp, rotatedBase, corner);
  const anchorLocal = badgeCorner(r, idx);
  const placement: Placement = {
    id: it.id,
    slot: c.s.id,
    position: pos,
    item: it,
    box: toTableBox(box, r, localRotate),
    z: 0,
    interactive: itemInteractive(it),
    indexBox: idx ? indexBounds(box, r, localRotate, idx) : null,
    badgeAnchor: toTablePoint(
      box,
      rotateAbout(
        anchorLocal,
        {x: r.x + r.width / 2, y: r.y + r.height / 2},
        localRotate,
      ),
    ),
  };
  return {
    placement,
    local: r,
    localRotate,
    rotatedBase,
    drawnFaceUp,
    corner,
    lift,
    zBump: gameStyle(c, it).z ?? 0,
    covered: extra.covered ?? false,
    index: idx,
  };
}

function layFree(c: SlotContext, box: SlotBox, entries: SlotEntry[]): void {
  const unit = c.cardWidth * c.scale;
  c.list.forEach((it, pos) => {
    const at = atOf(c, it, pos);
    const sz = itemSize(it, c.cw, c.ch, unit);
    const turn = isToken(it) ? 0 : (it.turn ?? 0);
    const lift = isToken(it) ? 0 : (gameStyle(c, it).lift ?? 0);
    entries.push(
      makeEntry(
        c,
        box,
        it,
        pos,
        {x: at.x * unit, y: at.y * unit, ...sz},
        turn,
        {
          lift,
        },
      ),
    );
  });
}

function layHand(
  c: SlotContext,
  box: SlotBox,
  infos: readonly SlotCardInfo[],
  handLift: number,
  entries: SlotEntry[],
): SlotShape {
  const {s, env} = c;
  if (!isHandMode(s.pile)) throw new Error(`slot '${s.id}' is not a hand`);
  const hand = handGeometry(
    env.deck,
    env.style,
    toCapacity(s, infos),
    s.pile,
    c.cw,
    c.maxExtent,
    s.arcRadius,
    handLift,
    handWrapOf(s, env.style, c.cw),
  );
  c.list.forEach((it, pos) => {
    const p = defined(hand.cards[pos], `hand position ${String(pos)}`);
    // The lift is already in the hand geometry; a turned card keeps
    // the index corner the geometry cleared for it.
    entries.push(
      makeEntry(
        c,
        box,
        it,
        pos,
        {x: p.x, y: p.y, width: c.cw, height: hand.cardHeight},
        p.rotate,
        {corner: p.corner ?? 'top-left'},
      ),
    );
  });
  return {
    kind: 'hand',
    pile: null,
    hand,
    compressed: hand.compressed,
    overflow: !hand.fits && c.maxExtent !== undefined,
    frame: {width: hand.width, height: hand.height},
  };
}

// The positions a pile draws: a rotated base, then the visible run.
function pilePositions(s: SlotSpec, n: number, rotatedBase: boolean): number[] {
  const base = rotatedBase ? 1 : 0;
  let first = base;
  if (s.pile === 'squared') first = Math.max(base, n - 1 - (s.stagger ?? 0));
  else if (s.visibleCount !== undefined) {
    first = Math.max(base, n - s.visibleCount);
  }
  return [
    ...(rotatedBase ? [0] : []),
    ...Array.from({length: n - first}, (_, i) => first + i),
  ];
}

function pileGeometryOf(
  c: SlotContext,
  positions: readonly number[],
  infos: ReadonlyArray<SlotCardInfo | null>,
  drawnUp: ReadonlyArray<boolean | null>,
  rotatedBase: boolean,
): PileGeometry {
  const {s, env} = c;
  const mode = s.pile;
  if (mode === 'free' || isHandMode(mode)) {
    throw new Error(`slot '${s.id}' is not a pile`);
  }
  return computePileGeometry({
    deck: env.deck,
    cardWidth: c.cw,
    faceUp: [...drawnUp],
    ...definedFields({
      turns:
        mode === 'squared'
          ? undefined
          : fanTurns(
              positions.map(p => infos[p] ?? {}),
              rotatedBase,
            ),
      maxExtent: c.maxExtent,
    }),
    rotatedBase,
    mode,
    stagger: s.stagger ?? 0,
    spread:
      fanAxis(mode) === 'right' && mode !== 'squared'
        ? env.style.fan.horizontal
        : env.style.fan.faceUp,
    faceDownSpread: env.style.fan.faceDown,
    squaredStagger: env.style.squaredStagger,
    indexScale: env.style.indexScale,
  });
}

function layPile(
  c: SlotContext,
  box: SlotBox,
  infos: ReadonlyArray<SlotCardInfo | null>,
  entries: SlotEntry[],
): SlotShape {
  const {s, entryList} = c;
  const mode = s.pile;
  const first0 = entryList[0] as TableCard | null;
  const rotatedBase =
    first0 !== null &&
    VERTICAL.includes(mode) &&
    (first0.turn === 90 || first0.turn === 270);
  const positions = pilePositions(s, entryList.length, rotatedBase);
  const coveredAt = (p: number, i: number): boolean =>
    mode === 'squared' && i < positions.length - 1 && !(rotatedBase && p === 0);
  const drawnUp = positions.map((p, i) =>
    coveredAt(p, i) ? false : (infos[p]?.faceUp ?? null),
  );
  const pile = pileGeometryOf(c, positions, infos, drawnUp, rotatedBase);
  positions.forEach((p, i) => {
    const it = entryList[p] as TableCard | null;
    if (it === null) return; // a hole: its room is kept, nothing drawn
    const sl = defined(pile.slots[i], `pile position ${String(i)}`);
    const isBase = sl.rotated;
    entries.push(
      makeEntry(
        c,
        box,
        it,
        p,
        {x: sl.x, y: sl.y, width: sl.width, height: sl.height},
        isBase ? 0 : (it.turn ?? 0),
        {
          rotatedBase: isBase,
          drawnFaceUp: drawnUp[i] === true,
          corner: isBase ? 'top-left' : (sl.corner ?? pile.corner),
          lift: gameStyle(c, it).lift ?? 0,
          covered: coveredAt(p, i),
        },
      ),
    );
  });
  return {
    kind: 'pile',
    pile,
    hand: null,
    compressed: pile.compressed,
    overflow: !pile.fits && c.maxExtent !== undefined,
    frame: {width: pile.width, height: pile.height},
  };
}

/** One slot laid out at `cardWidth`: its box and its entries. */
export function layoutSlot(
  s: SlotSpec,
  entryList: SlotItems,
  env: SceneEnv,
  cardWidth: number,
  tags: Record<string, TagStyle>,
  baseIndex: IndexGeometry,
): SlotLayout {
  const scale = s.scale ?? 1;
  const cw = cardWidth * scale;
  checkHoles(s, entryList);
  const c: SlotContext = {
    s,
    entryList,
    list: entryList.filter((it): it is Item => it !== null),
    env,
    cardWidth,
    cw,
    ch: cw / env.deck.aspect,
    scale,
    maxExtent: s.maxExtent !== undefined ? s.maxExtent * cardWidth : undefined,
    tags,
    baseIndex,
  };
  entryList.forEach((it, pos) => {
    if (it !== null) checkItem(c, it, pos);
  });
  const infos = entryList.map(it => (it === null ? null : info(c, it)));
  const handLift = Math.max(
    env.style.selection.lift,
    ...c.list.map(it => gameStyle(c, it).lift ?? 0),
  );
  const freeBox = s.pile === 'free' ? freeBoxOf(c) : null;
  const box = computeSlotBox(
    s,
    infos,
    env.deck,
    env.style,
    cardWidth,
    freeBox,
    handLift,
  );
  const entries: SlotEntry[] = [];
  let shape: SlotShape = {
    kind: 'pile',
    pile: null,
    hand: null,
    compressed: false,
    overflow: false,
    frame: {width: box.width, height: box.height},
  };
  if (s.pile === 'free') {
    layFree(c, box, entries);
    shape = {...shape, kind: 'free'};
  } else if (isHandMode(s.pile)) {
    // No holes here (checked above): infos and list are the same cards.
    const known = infos.filter((x): x is SlotCardInfo => x !== null);
    shape = layHand(c, box, known, handLift, entries);
  } else if (entryList.length > 0) {
    shape = layPile(c, box, infos, entries);
  }
  return {
    spec: s,
    box,
    cw,
    ch: c.ch,
    entries,
    items: c.list,
    positions: entryList.length,
    ...shape,
  };
}
