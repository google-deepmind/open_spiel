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

// <PlayingCard>: one card, purely visual. State tags (`tags`) style it through CardStyle.tags as --pcard-*
// variables. It takes no handlers: interaction is a <Table> policy, or
// <CardView> for a card drawn outside a table.
import type {AriaAttributes, CSSProperties, JSX} from 'react';
import {cardKey, cardLabel, isJoker} from './model.ts';
import {SuitGlyph} from './suit_glyph.tsx';
import type {SuitGlyphSet} from './suit_glyphs/sets.ts';
import {getDeck} from './registry.ts';
import {
  backVariant as styleBackVariant,
  cardStyleVars,
  tagStyles,
  tagStyleVars,
  useCardStyle,
  type CardStyle,
} from './style.ts';
import type {Card, ColorScheme, Deck} from './types.ts';
import {isLibraryTag, resolveTagStyle, TAG_NAME} from './core/tags.ts';
import './cards.css';

/** A card's width when none is given, px. */
export const DEFAULT_CARD_WIDTH = 80;

/** Deck from the prop, else the CardStyle's; plus the style itself. */
function useResolvedDeck(deckId?: string): {deck: Deck; style: CardStyle} {
  const style = useCardStyle();
  return {deck: getDeck(deckId ?? style.deck), style};
}

/** The back to draw: explicit variant wins, else the style's when this deck has it.
 *  A style is shared across decks whose back ids differ, so an id the deck
 *  lacks means its default rather than a crash in someone else's section. */
function resolveBack(
  deck: Deck,
  style: CardStyle,
  explicit?: string,
): string | undefined {
  if (explicit !== undefined) return explicit;
  const v = styleBackVariant(style.back);
  return v !== undefined && deck.backs?.includes(v) ? v : undefined;
}

/** The a11y attributes <Table>/<CardView> place on the card element, or
 *  null for none (the wrapper carries them). Internal. */
export interface CardA11yAttrs extends AriaAttributes {
  role?: string | undefined;
  tabIndex?: number | undefined;
  title?: string | undefined;
}

/** A card to draw: its face, deck, size, tags and turn. */
export interface PlayingCardProps {
  /** null: a card whose face the client does not know; always drawn as a back. */
  card: Card | null;
  /** Default: card !== null. */
  faceUp?: boolean | undefined;
  /** Deck id; defaults to the CardDeckProvider's. */
  deck?: string | undefined;
  scheme?: ColorScheme | undefined;
  /** Pixels; height follows the deck's aspect. */
  width?: number | undefined;
  backVariant?: string | undefined;
  /** State tags, styled by CardStyle.tags (classes pcard--t-<tag>). */
  tags?: readonly string[] | undefined;
  /** Quarter-turn about the card's centre (the box keeps its upright size). */
  turn?: 0 | 90 | 180 | 270 | undefined;
  /** Internal (<Table>, <CardView>): the a11y attributes of the card element;
   *  null for none (aria-hidden: the wrapper is the a11y node). */
  a11y?: CardA11yAttrs | null | undefined;
  /** Internal (<Table>): the lift is in the layout, so no CSS lift here. */
  noLift?: boolean | undefined;
}

// The card's tags, each once; throws on a malformed one.
function checkedTags(tags: readonly string[] | undefined): string[] {
  const out = [...new Set(tags ?? [])];
  for (const t of out) {
    if (!TAG_NAME.test(t)) {
      throw new Error(
        `<PlayingCard> tag ${JSON.stringify(t)} must match ${String(TAG_NAME)}`,
      );
    }
  }
  return out;
}

// The lift and the symbol-only look as CSS variables.
function symbolVars(
  lift: number | undefined,
  symbol: string | undefined,
  width: number,
): Record<string, string> {
  const vars: Record<string, string> = {};
  if (lift !== undefined) vars['--pcard-lift'] = `${String(lift)}px`;
  if (symbol !== undefined) {
    vars['--pcard-symbol-ink'] = symbol;
    vars['--pcard-symbol-font'] =
      `${String(Math.max(1, width * SYMBOL_FONT))}px`;
  }
  return vars;
}

/** One card, purely visual; interaction belongs to a <Table>. */
export function PlayingCard({
  card,
  faceUp: faceUpProp,
  deck: deckId,
  scheme: schemeProp,
  width = DEFAULT_CARD_WIDTH,
  backVariant,
  tags: tagsProp,
  turn,
  a11y,
  noLift,
}: PlayingCardProps): JSX.Element {
  const {deck, style} = useResolvedDeck(deckId);
  const scheme = schemeProp ?? style.scheme;
  const faceUp = faceUpProp ?? card !== null;
  if (faceUp && card === null) {
    throw new Error(
      '<PlayingCard> faceUp with card: null (an unknown face is always a back)',
    );
  }
  const tags = checkedTags(tagsProp);
  const gameTags = tags.filter(t => !isLibraryTag(t));
  const ts = resolveTagStyle(gameTags, tagStyles(style));
  const lifted = !noLift && (ts.lift ?? 0) > 0;
  // A symbol-only card (TagStyle.symbol): its known face as rank and suit
  // alone, no body. A face-down card has no face to name, so it is a back.
  const symbol =
    faceUp && card !== null && ts.symbol !== undefined ? ts.symbol : undefined;
  const classes = [
    'pcard',
    symbol !== undefined && 'pcard--symbol',
    deck.drawsOwnOutline && 'pcard--own-outline',
    ...tags.map(t => `pcard--t-${t}`),
    lifted && 'pcard--lifted',
    ts.className,
  ]
    .filter(Boolean)
    .join(' ');
  const css: CSSProperties = {
    width,
    height: width / deck.aspect,
    ...cardStyleVars(style, width),
    ...tagStyleVars(ts),
    ...(symbolVars(
      lifted ? ts.lift : undefined,
      symbol,
      width,
    ) as CSSProperties),
    ...(turn ? {transform: `rotate(${String(turn)}deg)`} : {}),
  };
  const label = faceUp && card ? cardLabel(card) : 'Face-down card';
  let own: CardA11yAttrs = {'aria-label': label, role: 'img'};
  if (a11y === null) own = {'aria-hidden': true};
  else if (a11y !== undefined) own = a11y;
  return (
    <div
      className={classes}
      style={css}
      data-card={faceUp && card ? cardKey(card) : 'back'}
      data-deck={deck.id}
      data-tags={tags.length ? tags.join(' ') : undefined}
      {...own}
    >
      {symbol !== undefined && card ? (
        <CardSymbol card={card} glyphs={style.suitGlyphs} />
      ) : faceUp && card ? (
        deck.renderFace(card, {
          scheme,
          indexScale: style.indexScale,
          suitGlyphs: style.suitGlyphs,
        })
      ) : (
        deck.renderBack({variant: resolveBack(deck, style, backVariant)})
      )}
    </div>
  );
}

/** A symbol-only card's font, × card width: larger than a deck's corner
 *  index (about 0.2), and '10' plus its suit still fits the width. */
export const SYMBOL_FONT = 0.34;

/** Rank and suit, centred and flat in --pcard-symbol-ink (cards.css). The
 *  suit is a drawn shape, never a text glyph a platform may paint as an
 *  emoji in its own colours. */
function CardSymbol({card, glyphs}: {card: Card; glyphs: SuitGlyphSet}) {
  if (isJoker(card)) {
    return (
      <span className="pcard-symbol" data-card-symbol>
        <span className="pcard-symbol__rank">JK</span>
      </span>
    );
  }
  return (
    <span className="pcard-symbol" data-card-symbol>
      <span className="pcard-symbol__rank">{card.rank}</span>
      <svg
        className="pcard-symbol__suit"
        viewBox="0 0 100 100"
        aria-hidden="true"
        focusable="false"
      >
        <SuitGlyph
          suit={card.suit}
          x={0}
          y={0}
          size={100}
          color="currentColor"
          set={glyphs}
        />
      </svg>
    </span>
  );
}

/** A card back, drawn outside a table. */
export function CardBack({
  deck: deckId,
  width = DEFAULT_CARD_WIDTH,
  variant,
}: {
  deck?: string | undefined;
  width?: number | undefined;
  variant?: string | undefined;
}): JSX.Element {
  const {deck, style} = useResolvedDeck(deckId);
  return (
    <div
      className={deck.drawsOwnOutline ? 'pcard pcard--own-outline' : 'pcard'}
      style={{
        width,
        height: width / deck.aspect,
        ...cardStyleVars(style, width),
      }}
      data-card="back"
      data-deck={deck.id}
      role="img"
      aria-label="Face-down card"
    >
      {deck.renderBack({variant: resolveBack(deck, style, variant)})}
    </div>
  );
}

/** Em widths per character assumed when fitting an empty place's label (a
 *  generous sans-serif average, so an estimate errs small, not over the edge). */
const LABEL_EM_PER_CHAR = 0.6;
/** The inset of the label inside the dashed box, px a side (border and air). */
const LABEL_INSET = 4;

/** The largest font, px, at which an empty place's label has its longest word
 *  inside the dashed box at this width. cards.css draws it at this size. */
export function emptyLabelMaxFont(label: string, width: number): number {
  const longest = Math.max(1, ...label.split(/\s+/).map(w => w.length));
  return (
    Math.floor(
      (Math.max(0, width - 2 * LABEL_INSET) / (longest * LABEL_EM_PER_CHAR)) *
        100,
    ) / 100
  );
}

function emptyFontVar(label: string, width: number): CSSProperties {
  const vars: Record<string, string> = {
    '--pcard-empty-font': `${String(emptyLabelMaxFont(label, width))}px`,
  };
  return vars;
}

/** An empty place's outline. `decorative`: drawn only (aria-hidden, no role,
 *  no name) — a <Table> slot is named by its slot node when the game names it. */
export function EmptySlot({
  width = DEFAULT_CARD_WIDTH,
  label,
  deck: deckId,
  decorative = false,
}: {
  width?: number | undefined;
  label?: string | undefined;
  deck?: string | undefined;
  decorative?: boolean | undefined;
}): JSX.Element {
  const {deck, style} = useResolvedDeck(deckId);
  const a11y = decorative
    ? {'aria-hidden': true as const}
    : {role: 'img', 'aria-label': label ? `Empty: ${label}` : 'Empty slot'};
  return (
    <div
      className="pcard-empty"
      style={{
        width,
        height: width / deck.aspect,
        ...(label ? emptyFontVar(label, width) : {}),
        ...cardStyleVars(style, width),
      }}
      data-card="empty"
      {...a11y}
    >
      {label}
    </div>
  );
}
