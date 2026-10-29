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

// The types of a laid-out table (layoutScene, scene.ts) and the internals the
// React layer renders from.
import type {Deck, IndexGeometry} from '../types.ts';
import type {CardStyle} from '../style.ts';
import type {
  FreePlacement,
  RegionSpec,
  SlotSpec,
  TableFit,
  TableSpec,
} from '../table_spec.ts';
import type {PileGeometry} from '../pile_geometry.ts';
import type {HandGeometry} from '../hand_geometry.ts';
import type {IndexCorner} from '../index_scale.ts';
import type {Box, CardId, Point, Rect, TableItem} from './identity.ts';

/** Where one item is drawn on the table. */
export interface Placement {
  id: CardId;
  slot: string;
  position: number;
  item: TableItem;
  /** Table px, after turn, lift from game tags and slot rotation. */
  box: Box;
  /** Paint order. */
  z: number;
  /** Resolved (item default + policy). */
  interactive: boolean;
  /** Visible index box, table px (null: tokens and backs). */
  indexBox: Rect | null;
  /** Corner that is free of the visible index, table px. */
  badgeAnchor: Point;
}

/** Where one table region (a plate, a label box) is drawn. */
export interface RegionPlacement {
  id: string;
  box: Box;
  upright: boolean;
  occludes: boolean;
  droppable: boolean;
}

/** The computed geometry of a table, in table px. */
export interface Scene {
  cardWidth: number;
  width: number;
  height: number;
  /** Index into spec.variants (-1 = base). */
  variant: number;
  /** Paint order. */
  placements: readonly Placement[];
  regions: readonly RegionPlacement[];
  byId(id: CardId): Placement | undefined;
  inSlot(slot: string): readonly Placement[];
  /** Defined even when the slot is empty. */
  slotBox(slot: string): Box;
  /** The caret/gap box for 'insert' targets. */
  insertBox(slot: string, index: number): Box;
}

/** What a table is laid out for: the deck, the style and the container. */
export interface SceneEnv {
  deck: Pick<Deck, 'aspect' | 'index' | 'indexAt' | 'rotatedTwin'> & {
    id?: string;
  };
  style: CardStyle;
  /** The container, px (0 x 0: unmeasured; the base spec at its minimum). */
  box: {width: number; height: number};
  fit?: TableFit | undefined;
  cardWidth?: number | undefined;
  minCardWidth?: number | undefined;
  maxCardWidth?: number | undefined;
  slotInteractive?: ((slot: string) => boolean) | undefined;
  /** Keep this layout (an index into [spec, ...variants]) at this card
   *  width instead of fitting the box: <Table> holds the scene it drew
   *  while a drag is in progress, so a change of container size (or of
   *  what the container measures) never re-lays the table out under the
   *  pointer or the keyboard. Ignored if the spec has no such layout. */
  hold?: {variant: number; cardWidth: number; overflow: boolean} | undefined;
  /** A viewer's card size, x the fitted card width (1: as fitted). The
   *  layout is chosen and fitted to the box as usual, then drawn at this
   *  multiple of the width it fitted at: smaller never below minCardWidth,
   *  larger past maxCardWidth and the box, which then scrolls (overflow).
   *  Ignored with an explicit cardWidth. */
  cardScale?: number | undefined;
  /** A rectangle of the container (px, its top-left 0, 0) that no card may
   *  cover: the felt's info box corner (FeltTable, felt_keep_out.ts). A
   *  centred hand or fan that reaches into it is compressed about its own
   *  centre until it clears (never past the index floor), and the layout
   *  and card width are chosen among those whose every card lies clear of
   *  it; only when none does is the table drawn smaller until it is. What
   *  the rectangle does not reach is laid out as without it. Ignored with
   *  an explicit cardWidth and in an unmeasured box; under a hold the
   *  layout and card width are the held ones, and a hand still yields. */
  keepOut?:
    {x: number; y: number; width: number; height: number} | null | undefined;
}

/** The card-width bounds a fit honours (absent: the library's). */
export interface WidthBounds {
  min?: number | undefined;
  max?: number | undefined;
}

/** A layout of a spec: the spec without its variants. */
export type Layout = Omit<TableSpec, 'variants'>;

/** A slot's items: null is a positional hole in a fan. */
export type SlotItems = ReadonlyArray<
  (TableItem & Partial<FreePlacement>) | null
>;

/** Every slot's items, by slot id. */
export type Items = Readonly<Record<string, SlotItems>>;

/** Space reserved below an overflowing pile for its "scroll" marker, px. */
export const OVERFLOW_MARKER_PX = 20;

/** The unrotated box a slot draws on the table, and how it turns. */
export interface SlotBox {
  /** Unrotated top-left of everything the slot draws, px, on the table surface. */
  left: number;
  top: number;
  width: number;
  height: number;
  /** Clockwise degrees, about `origin` (relative to left/top). */
  rotate: number;
  origin: {x: number; y: number};
}

/** What a slot box needs of a card: its face and whether a tag lifts it. */
export interface SlotCardInfo {
  faceUp: boolean;
  lifted: boolean;
  turn?: number;
}

/** One drawn item of a slot, in the slot's own (unrotated) frame. */
export interface SlotEntry {
  placement: Placement;
  /** Top-left and size in the slot frame, px (a rotated base: the turned card's bounding box). */
  local: Rect;
  /** Degrees clockwise about the local rect's centre (arc hands, turn in play). */
  localRotate: number;
  /** A quarter-turned base under an upright pile (turn 90/270 at position 0 of a vertical fan). */
  rotatedBase: boolean;
  /** Face as drawn: a squared pile's depth layers are drawn as backs. */
  drawnFaceUp: boolean;
  /** The corner index this card keeps visible (top-left, or the twin). */
  corner: IndexCorner;
  /** Game-tag lift applied, px. */
  lift: number;
  /** Game-tag z bump. */
  zBump: number;
  /** Covered by a later card in this slot by design (squared depth layers). */
  covered: boolean;
  /** The visible index box as fractions of `local` (null: tokens and backs). */
  index: IndexGeometry | null;
}

/** One slot laid out: its box, its entries and its geometry. */
export interface SlotLayout {
  spec: SlotSpec;
  kind: 'pile' | 'hand' | 'free';
  box: SlotBox;
  /** This slot's card width and height, px. */
  cw: number;
  ch: number;
  entries: SlotEntry[];
  /** Every item in the slot (drawn or not), bottom first; positional holes
   *  are not items (an entry's position counts them). */
  items: readonly TableItem[];
  /** Positions in the slot: its items and its positional holes. A fan of
   *  holes only holds room (SlotSpec.block, RegionSpec.attach.also). */
  positions: number;
  pile: PileGeometry | null;
  hand: HandGeometry | null;
  /** Pile/hand geometry flags, as <Pile> and <Hand> report them. */
  compressed: boolean;
  /** The pile or hand cannot fit maxExtent: it scrolls, with a marker. */
  overflow: boolean;
  /** The frame's px size (piles and hands). */
  frame: {width: number; height: number};
}

/** What a Scene was drawn from; the React layer renders from it. */
export interface SceneInternals {
  spec: TableSpec;
  layout: Layout;
  /** Below the minimum card width: the table scrolls. */
  overflow: boolean;
  slots: SlotLayout[];
  slotById: Map<string, SlotLayout>;
  regionSpecs: RegionSpec[];
}

const INTERNALS = new WeakMap<Scene, SceneInternals>();

/** Records what `scene` was drawn from (layoutScene only). */
export function setSceneInternals(
  scene: Scene,
  internals: SceneInternals,
): void {
  INTERNALS.set(scene, internals);
}

/** What `scene` was drawn from. */
export function sceneInternals(scene: Scene): SceneInternals {
  const i = INTERNALS.get(scene);
  if (!i) throw new Error('sceneInternals: not a Scene made by layoutScene');
  return i;
}

// A quarter turn at position 0 of a vertical fan is a base lying turned under
// the upright pile; in a squared or free slot a turn rotates the card in
// place; anywhere else in a fan, and in a flat or overlapped hand, the card
// fans by its turned footprint and index (pileGeometry turns). Arc hands
// refuse turns.
/** The fans that run down the screen. */
export const VERTICAL: readonly string[] = ['fan-down', 'fan-up'];
