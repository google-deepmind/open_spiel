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

// Layout primitives over <Pile items>: TableauRow, FoundationRow, StockWaste,
// CardGrid. Each derives card width from the width it is given (or measures
// its own container), clamps to a readable minimum, and scrolls horizontally
// rather than shrink below it — the row-level twin of <Pile>'s legibility
// rule. Cards are TableCards (ids required, tags for state); interaction is
// one InteractionPolicy for the row, and its events name the pile by its id
// (the slot of every target).
import {
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type JSX,
  type ReactNode,
  type RefObject,
} from 'react';
import type {ColorScheme} from './types.ts';
import type {TableCard} from './core/identity.ts';
import type {InteractionPolicy} from './core/policy.ts';
import {Pile} from './pile.tsx';
import {CardStyleProvider, DEFAULT_CARD_STYLE, useCardStyle} from './style.ts';

import {pixels} from '../../styles/pixels.ts';
import {MAX_CARD_WIDTH, MIN_CARD_WIDTH} from './card_widths.ts';

export {MAX_CARD_WIDTH, MIN_CARD_WIDTH};

/** A fitted card width, with the gaps it implies. */
export interface CardWidthFit {
  cardWidth: number;
  /** Between piles in a row, px. */
  gap: number;
  /** Between rows of a layout, px (CardStyle.pileGap.y × card width). */
  rowGap: number;
  /** The row needs more width than is available at minimum card width. */
  overflow: boolean;
}

/** How a row's card width is fitted to its container. */
export interface FitOptions {
  /** Explicit px gap between piles; wins over pileGap. */
  gap?: number | undefined;
  /** Gaps as fractions of card width (CardStyle.pileGap); the default style's when omitted. */
  pileGap?: {x: number; y: number} | undefined;
  min?: number | undefined;
  max?: number | undefined;
}

/** Card width for `columns` cards (plus gaps) in `available` px. Gaps scale
 *  with the card (pileGap × card width) unless an explicit px gap is given. */
export function fitCardWidth(
  available: number,
  columns: number,
  opts: FitOptions = {},
): CardWidthFit {
  if (!(columns >= 1)) {
    throw new Error(`fitCardWidth needs columns >= 1, got ${String(columns)}`);
  }
  const min = opts.min ?? MIN_CARD_WIDTH;
  const max = opts.max ?? MAX_CARD_WIDTH;
  if (!(min > 0 && max >= min)) {
    throw new Error(
      `fitCardWidth bad bounds min=${String(min)} max=${String(max)}`,
    );
  }
  const pileGap = opts.pileGap ?? DEFAULT_CARD_STYLE.pileGap;
  for (const v of [pileGap.x, pileGap.y, opts.gap ?? 0]) {
    if (!(Number.isFinite(v) && v >= 0)) {
      throw new Error(
        `fitCardWidth gaps must be finite and >= 0, got ${JSON.stringify({pileGap, gap: opts.gap})}`,
      );
    }
  }
  const raw =
    opts.gap !== undefined
      ? Math.floor((available - opts.gap * (columns - 1)) / columns)
      : Math.floor(available / (columns + pileGap.x * (columns - 1)));
  const cardWidth = Math.max(min, Math.min(max, raw));
  const gap = opts.gap ?? Math.floor(pileGap.x * cardWidth);
  return {
    cardWidth,
    gap,
    rowGap: Math.floor(pileGap.y * cardWidth),
    overflow: raw < min,
  };
}

/** Width of an element, tracked across resizes. 0 until measured. */
export function useElementWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setWidth(e.contentRect.width);
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, [ref]);
  return width;
}

/** Content box of an element (width and height, padding excluded), tracked
 *  across resizes. 0 x 0 until measured. A felt's cards area sizes its cards
 *  from both (FeltTable). */
export function useElementSize(ref: RefObject<HTMLElement | null>): {
  width: number;
  height: number;
} {
  const [size, setSize] = useState({width: 0, height: 0});
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const cs = getComputedStyle(el);
    const px = (v: string): number => {
      const n = pixels(v);
      return Number.isFinite(n) ? n : 0;
    };
    setSize({
      width: el.clientWidth - px(cs.paddingLeft) - px(cs.paddingRight),
      height: el.clientHeight - px(cs.paddingTop) - px(cs.paddingBottom),
    });
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        setSize(o =>
          o.width === e.contentRect.width && o.height === e.contentRect.height
            ? o
            : {width: e.contentRect.width, height: e.contentRect.height},
        );
      }
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
    };
  }, [ref]);
  return size;
}

/**
 * Fit `columns` cards to the measured width of a container. Use this once per
 * layout and pass `cardWidth` to every row so rows share one card size.
 */
export function useCardWidth(
  columns: number,
  opts: Omit<FitOptions, 'pileGap'> & {width?: number | undefined} = {},
): CardWidthFit & {ref: RefObject<HTMLDivElement | null>; available: number} {
  const ref = useRef<HTMLDivElement | null>(null);
  const measured = useElementWidth(ref);
  const {pileGap} = useCardStyle();
  const available = opts.width ?? measured;
  return {
    ref,
    available,
    ...fitCardWidth(available, columns, {...opts, pileGap}),
  };
}

/** A row's resolved fit; `measured` rows fill their parent so they can be measured. */
type RowFit = CardWidthFit & {measured: boolean};

/** Measured rows fill the parent (their width is the input) and scroll when
 *  the cards cannot fit. Rows given a card width are exactly as wide as their
 *  cards and never shrink: a shrunk box with visible overflow would paint its
 *  cards over a sibling's, so the parent layout must scroll instead. */
function frameStyle(fit: RowFit): CSSProperties {
  return fit.measured
    ? {width: '100%', overflowX: fit.overflow ? 'auto' : 'visible'}
    : {width: 'max-content', flex: 'none'};
}

interface RowFrameProps {
  fit: RowFit;
  frameRef?: RefObject<HTMLDivElement | null> | undefined;
  testId?: string | undefined;
  children: ReactNode;
}

function RowFrame({
  fit,
  frameRef,
  testId,
  deck,
  scheme,
  children,
}: RowFrameProps & Pick<SharedRowProps, 'deck' | 'scheme'>) {
  return (
    <CardStyleProvider value={{deck, scheme}}>
      <div
        ref={frameRef}
        data-testid={testId}
        data-row-overflow={fit.overflow ? 'true' : 'false'}
        style={frameStyle(fit)}
      >
        <div
          style={{
            display: 'flex',
            gap: fit.gap,
            alignItems: 'flex-start',
            width: 'max-content',
          }}
        >
          {children}
        </div>
      </div>
    </CardStyleProvider>
  );
}

interface SharedRowProps {
  /** Deck id and scheme; default to the CardDeckProvider's. */
  deck?: string | undefined;
  scheme?: ColorScheme | undefined;
  /** One policy for every pile of the row; a target's slot is the pile's
   *  id. Absent: an inert display (no buttons, no tab stops). */
  policy?: InteractionPolicy | undefined;
  /** Explicit card width; omitted, the row fits its own container. */
  cardWidth?: number | undefined;
  gap?: number | undefined;
  minCardWidth?: number | undefined;
  maxCardWidth?: number | undefined;
  testId?: string | undefined;
}

/** Resolve a row's fit: explicit width wins, else measure the container. */
function useRowFit(
  columns: number,
  p: SharedRowProps,
): {fit: RowFit; ref: RefObject<HTMLDivElement | null>} {
  const auto = useCardWidth(columns, {
    gap: p.gap,
    min: p.minCardWidth,
    max: p.maxCardWidth,
  });
  const {pileGap} = useCardStyle();
  if (p.cardWidth !== undefined) {
    const gap = p.gap ?? Math.floor(pileGap.x * p.cardWidth);
    const rowGap = Math.floor(pileGap.y * p.cardWidth);
    return {
      fit: {
        cardWidth: p.cardWidth,
        gap,
        rowGap,
        overflow: false,
        measured: false,
      },
      ref: auto.ref,
    };
  }
  const {cardWidth, gap, rowGap, overflow} = auto;
  return {
    fit: {cardWidth, gap, rowGap, overflow, measured: true},
    ref: auto.ref,
  };
}

/** One pile of a row: its slot id and its cards. */
export interface TableauPile {
  /** The pile's slot id: unique in the row, and the slot its events name. */
  id: string;
  /** Bottom first (ids required). In a fanned column an entry may be null:
   *  a positional hole that keeps its room. */
  items: ReadonlyArray<TableCard | null>;
  emptyLabel?: string | undefined;
}

/** A row of fanned tableau columns. */
export interface TableauRowProps extends SharedRowProps {
  piles: readonly TableauPile[];
  /** Max column height in px; columns compress to index visibility, then scroll. */
  maxPileHeight?: number | undefined;
}

function assertPileIds(piles: readonly TableauPile[], where: string): void {
  const seen = new Set<string>();
  for (const p of piles) {
    if (typeof p.id !== 'string' || p.id === '') {
      throw new Error(`${where}: every pile needs an id`);
    }
    if (seen.has(p.id)) {
      throw new Error(`${where}: duplicate pile id '${p.id}'`);
    }
    seen.add(p.id);
  }
}

/** A row of fanned-down columns (a tableau). */
export function TableauRow(p: TableauRowProps): JSX.Element {
  assertPileIds(p.piles, '<TableauRow>');
  const {fit, ref} = useRowFit(p.piles.length, p);
  return (
    <RowFrame
      fit={fit}
      frameRef={ref}
      testId={p.testId}
      deck={p.deck}
      scheme={p.scheme}
    >
      {p.piles.map(pile => (
        <Pile
          key={pile.id}
          slotId={pile.id}
          mode="fan-down"
          items={pile.items}
          cardWidth={fit.cardWidth}
          maxExtent={p.maxPileHeight}
          emptyLabel={pile.emptyLabel}
          policy={p.policy}
        />
      ))}
    </RowFrame>
  );
}

/** A row of squared foundation piles. */
export interface FoundationRowProps extends SharedRowProps {
  piles: readonly TableauPile[];
  countBadge?: boolean | undefined;
}

/** Squared piles showing only their top card (foundations, free cells). */
export function FoundationRow(p: FoundationRowProps): JSX.Element {
  assertPileIds(p.piles, '<FoundationRow>');
  const {fit, ref} = useRowFit(p.piles.length, p);
  return (
    <RowFrame
      fit={fit}
      frameRef={ref}
      testId={p.testId}
      deck={p.deck}
      scheme={p.scheme}
    >
      {p.piles.map(pile => (
        <Pile
          key={pile.id}
          slotId={pile.id}
          mode="squared"
          items={pile.items}
          cardWidth={fit.cardWidth}
          countBadge={p.countBadge}
          emptyLabel={pile.emptyLabel}
          policy={p.policy}
        />
      ))}
    </RowFrame>
  );
}

/** A stock and its waste pile. */
export interface StockWasteProps extends SharedRowProps {
  /** Slot 'stock' (ids required). */
  stock: readonly TableCard[];
  /** Slot 'waste' (ids required). */
  waste: readonly TableCard[];
  /** Waste cards shown fanned: 1 for draw-one, 3 for draw-three. */
  wasteVisible?: number | undefined;
}

/** A face-down stock with count badge beside a fanned waste. Events name the
 *  slots 'stock' and 'waste'. */
export function StockWaste(p: StockWasteProps): JSX.Element {
  const visible = p.wasteVisible ?? 1;
  // Stock plus a waste fanned over up to three cards: budget three columns.
  const {fit, ref} = useRowFit(3, p);
  return (
    <RowFrame
      fit={fit}
      frameRef={ref}
      testId={p.testId}
      deck={p.deck}
      scheme={p.scheme}
    >
      <Pile
        slotId="stock"
        mode="squared"
        items={p.stock}
        cardWidth={fit.cardWidth}
        stagger={Math.min(2, p.stock.length)}
        countBadge
        emptyLabel="↻"
        policy={p.policy}
      />
      <Pile
        slotId="waste"
        mode="fan-right"
        items={p.waste}
        cardWidth={fit.cardWidth}
        visibleCount={visible}
        emptyLabel=""
        policy={p.policy}
      />
    </RowFrame>
  );
}

/** A grid of card cells. */
export interface CardGridProps extends SharedRowProps {
  /** Rows of cells (ids required); null is an empty cell. Cell (r, c) is
   *  the slot `cell-${r}-${c}`. */
  rows: ReadonlyArray<ReadonlyArray<TableCard | null>>;
}

/** A grid of single cards with no overlap. */
export function CardGrid(p: CardGridProps): JSX.Element {
  const columns = Math.max(1, ...p.rows.map(r => r.length));
  const {fit, ref} = useRowFit(columns, p);
  return (
    <CardStyleProvider value={{deck: p.deck, scheme: p.scheme}}>
      <div
        ref={ref}
        data-testid={p.testId}
        data-row-overflow={fit.overflow ? 'true' : 'false'}
        style={frameStyle(fit)}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${String(columns)}, ${String(fit.cardWidth)}px)`,
            columnGap: fit.gap,
            rowGap: fit.rowGap,
            width: 'max-content',
          }}
        >
          {p.rows.flatMap((row, r) =>
            Array.from({length: columns}, (_, c) => {
              const cell = row[c] ?? null;
              return (
                <Pile
                  key={`${String(r)}-${String(c)}`}
                  slotId={`cell-${String(r)}-${String(c)}`}
                  mode="squared"
                  items={cell ? [cell] : []}
                  cardWidth={fit.cardWidth}
                  policy={p.policy}
                />
              );
            }),
          )}
        </div>
      </div>
    </CardStyleProvider>
  );
}
