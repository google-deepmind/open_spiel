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

// Shared playing cards. Import from here.
export type {
  Card,
  ColorScheme,
  Deck,
  FaceOptions,
  IndexGeometry,
  Rank,
  Suit,
} from './types.ts';
export {registerDeck, getDeck, listDecks} from './registry.ts';
export {SuitGlyph} from './suit_glyph.tsx';
export {
  SUIT_GLYPH_SETS,
  SUIT_GLYPH_SET_SPECS,
  DEFAULT_SUIT_GLYPHS,
  assertSuitGlyphSet,
  isSuitGlyphSet,
  suitShapes,
  type SuitGlyphSet,
  type SuitGlyphSetSpec,
  type SuitShape,
} from './suit_glyphs/sets.ts';
export {CardDeckProvider, useCardDeck} from './card_deck_provider.tsx';
export {
  PlayingCard,
  CardBack,
  EmptySlot,
  DEFAULT_CARD_WIDTH,
  type PlayingCardProps,
} from './playing_card.tsx';
export {
  SUITS,
  RANKS,
  SUIT_GLYPH,
  SUIT_NAME,
  RANK_NAME,
  INK,
  JOKERS,
  suitColor,
  jokerColor,
  isJoker,
  cardKey,
  cardLabel,
  standardDeck,
} from './model.ts';
export {
  CardStyleProvider,
  useCardStyle,
  mergeCardStyle,
  diffCardStyle,
  assertCardStyle,
  backVariant,
  cardStyleVars,
  cardEdgeWidth,
  CARD_SHADOWS,
  DEFAULT_CARD_STYLE,
  DEFAULT_BACK,
  type CardStyle,
  type CardStyleOverride,
} from './style.ts';
export {
  clampIndexScale,
  cornerIndex,
  deckIndex,
  maxIndexScale,
  mirrorIndex,
  quarterTurnIndex,
  scaleIndex,
  MIN_INDEX_SCALE,
  MAX_INDEX_SCALE,
  type IndexCorner,
} from './index_scale.ts';
export type {
  LayoutFit,
  SlotBlock,
  SlotSpec,
  TableClearance,
  TableLayout,
  TablePin,
  TableSource,
  TableSpec,
  TableVariant,
} from './table_spec.ts';
export {
  validateTableSpec,
  rotateIndex,
  rowSlots,
  overlappedRows,
  pyramidColumns,
  radialSlots,
  orderedLine,
  orderedLineWidth,
  gridSlots,
  nestTable,
  selectTableVariant,
  minSlotScale,
  CAPTION_HEIGHT,
  CUE_WIDTH,
  resolveTableSpec,
  tableClearance,
  widenTableLayout,
  TABLE_PINS,
} from './table_spec.ts';
export {
  Table,
  fitTableCardWidth,
  type TableProps,
  type TableCollect,
  type CardRenderProps,
  type TableFit,
} from './react/table.tsx';
export {fitLayout} from './core/scene.ts';
export type {PileMode} from './pile_geometry.ts';
export {Hand, handGeometryFor, type HandProps} from './hand.tsx';
export {
  computeHandGeometry,
  handRowSizes,
  isHandMode,
  HAND_MODES,
  HAND_WRAPS,
  type HandMode,
  type HandWrap,
  type HandGeometry,
  type HandCardPlacement,
  type HandCardShape,
} from './hand_geometry.ts';
export {
  indexBoxes,
  measureHands,
  type HandMeasure,
  type HandMeasureOptions,
} from './hand_measure.ts';
export {
  HandTable,
  resolveHandTable,
  type HandTableProps,
  type HandTableRecipe,
} from './hand_table.tsx';
export {
  fourSeatTable,
  largerBox,
  ringRadii,
  ringSeatTable,
  seatBox,
  seatSize,
  seatWrap,
  SEAT_GAP,
  type FourSeatOptions,
  type HandSeat,
  type RingSeatOptions,
  type RingSeatTable,
  type SeatBox,
} from './seats.ts';
export {
  placeTable,
  placeScene,
  sceneLegibility,
  type SceneLegibilityReport,
} from './table_legibility.ts';
export {findCoveredIndexes} from './pile_geometry.ts';
// Card sets: deck composition as data, instance ids for duplicate faces.
export {
  CARD_SETS,
  makeCardSet,
  resolveCardSetSpec,
  instanceId,
  parseInstanceId,
  faceOf,
  sameFace,
  instanceItem,
  instanceLabel,
  countByRank,
  countBySuit,
  countByFace,
  type CardSet,
  type CardSetSpec,
  type CardSetPreset,
  type CardInstance,
  type CardInstanceId,
  type JokerColor,
  type ResolvedCardSetSpec,
} from './card_set.ts';
// Slot badges: pile information beside the pile, never on a card.
export {
  SlotBadge,
  slotBadgeRegion,
  slotBadges,
  slotBadgeId,
  SLOT_BADGE_WIDTH,
  SLOT_BADGE_HEIGHT,
  SLOT_BADGE_MIN_FONT_PX,
  SLOT_BADGE_PADDING_CSS,
  slotBadgePaddingPx,
  type SlotBadgeOptions,
  type SlotBadgeProps,
  type SlotBadgeTone,
} from './slot_badge.tsx';

// ---- Interaction and extensibility primitives ----
// core/: pure TypeScript, no React, no DOM.
export {
  isToken,
  assertItems,
  type CardId,
  type Point,
  type Rect,
  type Box,
  type TableCard,
  type TableToken,
  type TableItem,
  type SlotEntryItem,
} from './core/identity.ts';
export {
  targetKey,
  type ItemRef,
  type Target,
  type TargetKind,
} from './core/targets.ts';
export {
  handle,
  type Modifiers,
  type DragSource,
  type TableEvent,
  type TableEventType,
  type EventResult,
  type EventHandlers,
} from './core/events.ts';
export type {InteractionPolicy} from './core/policy.ts';
export {
  LIBRARY_TAGS,
  STYLED_TAGS,
  SYMBOL_CORNER_TAG,
  resolveTagStyle,
  type TagStyle,
} from './core/tags.ts';
export {
  layoutScene,
  type Placement,
  type RegionPlacement,
  type Scene,
  type SceneEnv,
} from './core/scene.ts';
export {
  hitTest,
  targetsAt,
  allTargets,
  targetBox,
  nearestInDirection,
} from './core/hit.ts';
export {a11yModel, type A11yNode} from './core/a11y.ts';
export {planTransitions, type Transition} from './core/motion_plan.ts';
export type {FreePlacement, RegionSpec} from './table_spec.ts';
export {DEFAULT_TAG_STYLES} from './style.ts';
// Additive: raise the playable cards, never dim the rest.
export {
  PLAYABLE_TAG,
  PLAYABLE_LIFT_PX,
  PLAYABLE_TAG_STYLE,
  playableTags,
  raisesPlayable,
} from './playable.ts';
// Additive: card codes in engine text as rank and suit glyph ("KS.1" -> "K♠"), for every visible string.
export {cardText, cardTokensText, hasCardCode} from './card_text.ts';
// motion/ (public surface; FLIP itself is internal).
export type {
  Frame,
  Pose,
  FreeCard,
  Run,
  MotionService,
} from './motion/service.ts';
// Additive (motion): the reduced-motion rule FLIP applies, for games' own frame loops.
export {prefersReducedMotion, motionDuration} from './motion/service.ts';
export {FreeLayer} from './motion/free_layer.tsx';
export {
  rafTicker,
  manualTicker,
  TickerProvider,
  type Ticker,
} from './motion/ticker.ts';
export {preloadCards, drawCard, cardImage, sizeCanvas} from './motion/paint.ts';
// react/: components, context and escape hatches.
export {
  CardScene,
  useCardScene,
  useCardWidth,
  type CardSceneApi,
  type CardSceneProps,
  type SyntheticTableEvent,
} from './react/card_scene.tsx';
export {CardView, type CardViewProps} from './react/card_view.tsx';
// The trick area: the trick in play laid out toward players, or overlapping.
export {
  TRICK_LAYOUTS,
  TRICK_LAYOUT_LABELS,
  TRICK_SEATS,
  TRICK_SLOT_PREFIX,
  OVERLAP_REFERENCE,
  overlapOffset,
  overlapPinwheel,
  pinwheelLegible,
  playOrder,
  towardOffset,
  trickSlotOf,
  trickSlotIds,
  trickAreaSize,
  trickAreaSlots,
  trickAreaItems,
  trickCardCentre,
  trickLayoutSpec,
  trickLayoutTable,
  trickLayoutRecipe,
  type TrickLayout,
  type TrickSeat,
  type TrickAreaGeometry,
  type TrickAreaCard,
  type TrickMeasure,
  type Pinwheel,
} from './trick_area.ts';
// Additive: the stacked trick, two seats or a ring, each card on the one before (element boards: <TrickStack>).
export {
  SEAT_TOWARD,
  PAIR_SEATS,
  PAIR_OVERLAP_REFERENCE,
  stackOffset,
  stackLegible,
  stackPinwheel,
  stackBox,
  pairStackSeats,
  stackTrickSpec,
  stackTrickTable,
  stackTrickRecipe,
  stackTrickItems,
  type Toward,
  type StackSeat,
  type StackTrickOptions,
} from './trick_area.ts';
export {
  TrickStack,
  type TrickStackCard,
  type TrickStackProps,
} from './react/trick_stack.tsx';
// Trick collect: a finished trick held, then collected toward its winner.
export {
  detectCompletedTrick,
  sameTrickFeed,
  trickHoldMs,
  collectGather,
  collectFrame,
  easeOutCubic,
  TRICK_HOLDS,
  TRICK_HOLD_MS,
  TRICK_HOLD_LABELS,
  DEFAULT_TRICK_HOLD,
  TRICK_COLLECT_MS,
  COLLECT_REACH,
  COLLECT_END_SCALE,
  COLLECT_FADE_FROM,
  type CompletedTrick,
  type TrickCardAt,
  type TrickFeed,
  type TrickHold,
} from './trick_collect.ts';
export {
  useTrickCollect,
  type TrickCollectState,
  type TrickCollectPhase,
  type TrickCollectOptions,
} from './react/use_trick_collect.ts';
export {
  collectElements,
  collectTableCards,
  type CollectRun,
} from './motion/collect.ts';
export {arriveFrom, type ArriveBox} from './motion/arrive.ts';
export {
  useCollectElements,
  type ElementCollect,
} from './react/use_collect_elements.ts';
export {useElementWidth, useElementSize} from './layouts.tsx';
// The felt fills the Gameplay panel; ONE info box on it, top-left, never over a card.
export {
  FeltTable,
  TableInfoBox,
  InfoSection,
  type FeltTableProps,
  type TableInfoBoxProps,
  type InfoSectionProps,
} from './felt_table.tsx';
export {
  FELT_KEEP_GAP,
  keepOutOver,
  useFeltInset,
  useFeltKeepOut,
  useSideYield,
  type KeepOut,
} from './felt_keep_out.ts';
