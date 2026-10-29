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

// CardStyle: every visual knob of the shared card scaffold as data. Tuning a
// finished game means changing values here or in a <CardStyleProvider>,
// never rebuilding its UI. Colours are token references
// (src/styles/tokens.css).
import {
  createContext,
  createElement,
  useContext,
  useMemo,
  type CSSProperties,
  type JSX,
  type ReactNode,
} from 'react';
import type {ColorScheme} from './types.ts';
import {HAND_WRAPS, type HandWrap} from './hand_geometry.ts';
import {
  assertTagName,
  assertTagStyle,
  isLibraryTag,
  LIBRARY_TAGS,
  TAG_STYLE_FIELDS,
  type TagStyle,
} from './core/tags.ts';
import './decks/index.ts';
import {getDeck} from './registry.ts';
import {
  assertSuitGlyphSet,
  DEFAULT_SUIT_GLYPHS,
  type SuitGlyphSet,
} from './suit_glyphs/sets.ts';

/** The TagStyle fields a library tag (focus, hover, …) can carry: the scoped
 *  sheet's variables (TAG_VARS below), a paint-only lift, and z. */
export const LIBRARY_TAG_FIELDS = [
  'ring',
  'glow',
  'tint',
  'band',
  'filter',
  'cursor',
  'lift',
  'z',
] as const;
const LIBRARY_TAG_UNSUPPORTED = TAG_STYLE_FIELDS.filter(
  f => !(LIBRARY_TAG_FIELDS as readonly string[]).includes(f),
);

/** The runtime sets library tags without a render, through the scoped sheet
 *  (libraryTagSheet): a field the sheet cannot carry would be silently
 *  ignored, so it is refused. Game tags keep every TagStyle field. */
export function assertLibraryTagStyle(t: string, ts: TagStyle): void {
  if (!isLibraryTag(t)) return;
  for (const f of LIBRARY_TAG_UNSUPPORTED) {
    if (ts[f] !== undefined) {
      throw new Error(
        `CardStyle.tags['${t}'].${f}: '${t}' is a library tag, which supports ${LIBRARY_TAG_FIELDS.join(', ')}; ${f} is for game tags`,
      );
    }
  }
}

/** Every visual knob of the card library, as data. */
export interface CardStyle {
  /** Registered deck id. */
  deck: string;
  scheme: ColorScheme;
  /** Suit-symbol set (suit_glyphs/sets.ts) for every deck that draws suit
   *  shapes with SuitGlyph (basic, simple, symbol-only cards); a deck whose
   *  art carries its own suits ignores it. */
  suitGlyphs: SuitGlyphSet;
  fan: {
    /** Face-up offset in a fan-down pile, × the legibility floor. Values < 1
     *  are clamped to 1: the index is never hidden. */
    faceUp: number;
    /** Face-down offset, × card height (the index-free cards). */
    faceDown: number;
    /** Face-up offset in a fan-right or fan-left pile, × the legibility floor; clamped like faceUp. */
    horizontal: number;
  };
  /** Squared piles: px of offset per depth layer. */
  squaredStagger: number;
  /** Corner index size, 1 = the deck's native index. Decks that can redraw
   *  their index honour it and their IndexGeometry scales with it. */
  indexScale: number;
  /** Gaps between piles in layouts, × card width. */
  pileGap: {x: number; y: number};
  /** Corner radius, × card width. */
  radius: number;
  shadow: 'none' | 'soft' | 'hard';
  /** Selected cards: px lifted towards the top of the screen, and the ring colour.
   *  A lift in a fan can cover the index of the card above; 0 by default. */
  selection: {lift: number; outline: string};
  /** The card's edge line, drawn by the shared container along the same
   *  rounded rect that clips the face, so the clip can never cut it.
   *  `width` is px at a 100px-wide card and scales with the card; any width
   *  above 0 is drawn at least 1px wide, so a small card keeps its edge.
   *  0 = no line. Decks with `drawsOwnOutline` get none. */
  outline: {width: number; color: string};
  /** Back variant id, passed to the deck's renderBack; 'default' is the deck's own. */
  back: string;
  /** State-tag styles: the style's
   *  own tags, merged tag by tag and field by field by mergeCardStyle, and
   *  over DEFAULT_TAG_STYLES wherever tags are resolved (tagStyles). Rendered
   *  as --pcard-* variables that cards.css reads. */
  tags: Record<string, TagStyle>;
  /** Motion (FLIP, flip). reducedMotion 'respect': duration 0 under
   *  prefers-reduced-motion. */
  motion: {
    durationMs: number;
    easing: string;
    flip: boolean;
    staggerMs: number;
    reducedMotion: 'respect' | 'ignore';
  };
  /** Gesture recognition thresholds. */
  interaction: {
    dragThresholdPx: number;
    touchDragThresholdPx: number;
    longPressMs: number;
    holdTimeoutMs: number;
  };
  /** Hand slots that cannot fit their maxExtent in one row (handGeometry.ts):
   *  `wrap` 'auto' wraps them onto up to `maxRows` rows, `rowGap` × card
   *  width apart; 'never' compresses them to the legibility floor, then
   *  scrolls. A SlotSpec's wrap, maxRows and rowGap override these. */
  hand: {wrap: HandWrap; maxRows: number; rowGap: number};
}

/** The `back` value meaning "the deck's default back" (ids differ per deck). */
export const DEFAULT_BACK = 'default';

/** The variant to hand renderBack: undefined for the deck default. */
export function backVariant(back: string): string | undefined {
  return back === DEFAULT_BACK ? undefined : back;
}

/** The built-in tag styles. `selected` takes its ring and lift from
 *  CardStyle.selection (its source of truth); see tagStyles(). */
export const DEFAULT_TAG_STYLES: Readonly<Record<string, TagStyle>> =
  Object.freeze({
    selected: {ring: 'var(--card-ring-selected)', lift: 0},
    highlighted: {ring: 'var(--card-ring-highlighted)'},
    'hide-empty': {empty: 'hidden'},
    focus: {ring: 'var(--card-ring-selected)'},
    hover: {},
    dragging: {z: 1000},
    'drop-legal': {ring: 'var(--card-ring-drop-legal)'},
    'drop-over': {
      ring: 'var(--card-ring-drop-over)',
      glow: 'var(--card-ring-drop-over)',
    },
    held: {},
  });

/** The style every card has unless a provider overrides it. */
export const DEFAULT_CARD_STYLE: CardStyle = {
  deck: 'basic',
  scheme: 'two',
  suitGlyphs: DEFAULT_SUIT_GLYPHS,
  fan: {faceUp: 1.3, faceDown: 0.1, horizontal: 1.3},
  squaredStagger: 2,
  indexScale: 1,
  pileGap: {x: 0.1, y: 0.15},
  radius: 0.05,
  shadow: 'none',
  selection: {lift: 0, outline: 'var(--card-ring-selected)'},
  // Near-black. The line lies inside the card, against its (white) face,
  // so one colour reads on light and dark surfaces alike.
  outline: {width: 1, color: 'var(--card-edge)'},
  back: DEFAULT_BACK,
  tags: {},
  // 240 ms at Normal, slow enough to follow a bot's draw and discard; the
  // viewer's Animation speed scales it.
  motion: {
    durationMs: 240,
    easing: 'cubic-bezier(.2,.8,.2,1)',
    flip: true,
    staggerMs: 0,
    reducedMotion: 'respect',
  }, // anim-scaled: motionDuration() and motionStagger() apply the speed
  interaction: {
    dragThresholdPx: 4,
    touchDragThresholdPx: 10,
    longPressMs: 500,
    holdTimeoutMs: 3000,
  },
  // Off by default: a wrapped hand is taller, and a layout opts in by
  // reserving the room (HandMeasure rows, HandSeat rows, SlotSpec maxRows).
  hand: {wrap: 'never', maxRows: 3, rowGap: 0.08},
};

/** The tag styles in effect: CardStyle.tags, with `selected` ring and lift
 *  taken from CardStyle.selection unless the style's tags set them. */
export function tagStyles(style: CardStyle): Record<string, TagStyle> {
  const own = style.tags;
  const out: Record<string, TagStyle> = {...DEFAULT_TAG_STYLES};
  for (const [t, s] of Object.entries(own)) out[t] = {...(out[t] ?? {}), ...s};
  const sel = own['selected'] ?? {};
  out['selected'] = {
    ...out['selected'],
    ring: sel.ring ?? style.selection.outline,
    lift: sel.lift ?? style.selection.lift,
  };
  return out;
}

/** Partial<CardStyle>, one level deeper: nested groups may be given in part,
 *  so `{ fan: { faceUp: 2 } }` keeps the parent's faceDown and horizontal. */
export type CardStyleOverride = {
  [K in keyof CardStyle]?:
    | (CardStyle[K] extends object ? Partial<CardStyle[K]> : CardStyle[K])
    | undefined;
};

const NESTED = [
  'fan',
  'pileGap',
  'selection',
  'outline',
  'motion',
  'interaction',
  'hand',
] as const;

/** `base` with `over` merged on top; nested groups merge field by field. */
export function mergeCardStyle(
  base: CardStyle,
  over: CardStyleOverride | undefined,
): CardStyle {
  if (!over) return base;
  const out = {...base} as unknown as Record<string, unknown>;
  for (const [k, v] of Object.entries(over) as Array<
    [keyof CardStyle, unknown]
  >) {
    if (v === undefined) continue;
    if (k === 'tags') {
      // Tag by tag, field by field.
      const merged: Record<string, TagStyle> = {...base.tags};
      for (const [t, ts] of Object.entries(v as Record<string, TagStyle>)) {
        assertTagName(t, 'CardStyle.tags');
        assertTagStyle(t, ts);
        assertLibraryTagStyle(t, ts);
        merged[t] = {...(merged[t] ?? {}), ...ts};
      }
      out['tags'] = merged;
    } else if ((NESTED as readonly string[]).includes(k)) {
      out[k] = {...(base[k] as object), ...(v as object)};
    } else {
      out[k] = v;
    }
  }
  return out as unknown as CardStyle;
}

const CardStyleContext = createContext<CardStyle>(DEFAULT_CARD_STYLE);

/** Merges `value` over the enclosing style; nests (app → game → region). */
export function CardStyleProvider({
  value,
  children,
}: {
  value: CardStyleOverride;
  children: ReactNode;
}): JSX.Element {
  const parent = useContext(CardStyleContext);
  // Fail at the provider, not at the first card, when the deck id is unknown.
  if (value.deck !== undefined) getDeck(value.deck);
  // Keyed on content, so an inline literal does not re-render every card.
  const key = JSON.stringify(value);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const merged = useMemo(() => mergeCardStyle(parent, value), [parent, key]);
  return createElement(CardStyleContext.Provider, {value: merged}, children);
}

/** The style in effect. Every shared component reads this; explicit props win. */
export function useCardStyle(): CardStyle {
  return useContext(CardStyleContext);
}

const SHADOWS: ReadonlyArray<CardStyle['shadow']> = ['none', 'soft', 'hard'];

function num(path: string, v: unknown): void {
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    throw new Error(
      `CardStyle.${path} must be a finite number, got ${JSON.stringify(v)}`,
    );
  }
}

function str(path: string, v: unknown): void {
  if (typeof v !== 'string' || v === '') {
    throw new Error(
      `CardStyle.${path} must be a non-empty string, got ${JSON.stringify(v)}`,
    );
  }
}

// Field `key` of `v` when `v` is an object: a style that arrives as data may
// lack a group its type promises.
function fieldOf(v: unknown, key: string): unknown {
  return v !== null && typeof v === 'object'
    ? (v as Record<string, unknown>)[key]
    : undefined;
}

/** Throws, naming the field, unless `s` is a complete, well-typed CardStyle.
 *  For styles that arrive as data (a URL, pasted JSON), not for literals. */
export function assertCardStyle(s: CardStyle): void {
  assertLook(s);
  assertBehaviour(s);
  const known = new Set(Object.keys(DEFAULT_CARD_STYLE));
  for (const k of Object.keys(s)) {
    if (!known.has(k)) throw new Error(`CardStyle has no field '${k}'`);
  }
}

function assertLook(s: CardStyle): void {
  str('deck', s.deck);
  getDeck(s.deck);
  const scheme: string = s.scheme;
  if (scheme !== 'two' && scheme !== 'four') {
    throw new Error(
      `CardStyle.scheme must be 'two' or 'four', got ${JSON.stringify(s.scheme)}`,
    );
  }
  assertSuitGlyphSet(s.suitGlyphs, 'CardStyle.suitGlyphs');
  num('fan.faceUp', fieldOf(s.fan, 'faceUp'));
  num('fan.faceDown', fieldOf(s.fan, 'faceDown'));
  num('fan.horizontal', fieldOf(s.fan, 'horizontal'));
  num('squaredStagger', s.squaredStagger);
  num('indexScale', s.indexScale);
  num('pileGap.x', fieldOf(s.pileGap, 'x'));
  num('pileGap.y', fieldOf(s.pileGap, 'y'));
  num('radius', s.radius);
  if (!SHADOWS.includes(s.shadow)) {
    throw new Error(
      `CardStyle.shadow must be one of ${SHADOWS.join(', ')}, got ${JSON.stringify(s.shadow)}`,
    );
  }
  num('selection.lift', fieldOf(s.selection, 'lift'));
  str('selection.outline', fieldOf(s.selection, 'outline'));
  num('outline.width', fieldOf(s.outline, 'width'));
  if (s.outline.width < 0) {
    throw new Error(
      `CardStyle.outline.width must be >= 0, got ${String(s.outline.width)}`,
    );
  }
  str('outline.color', fieldOf(s.outline, 'color'));
  str('back', s.back);
  const tags: unknown = s.tags;
  if (tags === null || typeof tags !== 'object') {
    throw new Error(
      `CardStyle.tags must be an object, got ${JSON.stringify(s.tags)}`,
    );
  }
  for (const [t, ts] of Object.entries(s.tags)) {
    assertTagName(t, 'CardStyle.tags');
    assertTagStyle(t, ts);
    assertLibraryTagStyle(t, ts);
  }
}

function assertBehaviour(s: CardStyle): void {
  num('motion.durationMs', fieldOf(s.motion, 'durationMs'));
  str('motion.easing', fieldOf(s.motion, 'easing'));
  num('motion.staggerMs', fieldOf(s.motion, 'staggerMs'));
  const flip: unknown = s.motion.flip;
  if (typeof flip !== 'boolean') {
    throw new Error(
      `CardStyle.motion.flip must be boolean, got ${JSON.stringify(s.motion.flip)}`,
    );
  }
  const reducedMotion: string = s.motion.reducedMotion;
  if (reducedMotion !== 'respect' && reducedMotion !== 'ignore') {
    throw new Error(
      `CardStyle.motion.reducedMotion must be 'respect' or 'ignore', got ${JSON.stringify(s.motion.reducedMotion)}`,
    );
  }
  for (const field of [
    'dragThresholdPx',
    'touchDragThresholdPx',
    'longPressMs',
    'holdTimeoutMs',
  ]) {
    num(`interaction.${field}`, fieldOf(s.interaction, field));
  }
  const wrap = fieldOf(s.hand, 'wrap');
  if (!(HAND_WRAPS as readonly unknown[]).includes(wrap)) {
    throw new Error(
      `CardStyle.hand.wrap must be one of ${HAND_WRAPS.join(', ')}, got ${JSON.stringify(wrap)}`,
    );
  }
  if (!(Number.isInteger(s.hand.maxRows) && s.hand.maxRows >= 1)) {
    throw new Error(
      `CardStyle.hand.maxRows must be a whole number >= 1, got ${JSON.stringify(s.hand.maxRows)}`,
    );
  }
  num('hand.rowGap', s.hand.rowGap);
  if (s.hand.rowGap < 0) {
    throw new Error(
      `CardStyle.hand.rowGap must be >= 0, got ${String(s.hand.rowGap)}`,
    );
  }
}

/** The fields of `s` that differ from `base`, as an override (nested groups field by field). */
export function diffCardStyle(
  s: CardStyle,
  base: CardStyle = DEFAULT_CARD_STYLE,
): CardStyleOverride {
  const out: Record<string, unknown> = {};
  for (const k of Object.keys(s) as Array<keyof CardStyle>) {
    if (k === 'tags') {
      const a = s.tags;
      const b = base.tags;
      const d: Record<string, TagStyle> = {};
      for (const [t, ts] of Object.entries(a)) {
        const bt = (b[t] ?? {}) as Record<string, unknown>;
        const f = Object.fromEntries(
          Object.entries(ts).filter(([fk, fv]) => fv !== bt[fk]),
        );
        if (Object.keys(f).length) d[t] = f;
      }
      if (Object.keys(d).length) out['tags'] = d;
    } else if ((NESTED as readonly string[]).includes(k)) {
      const a = s[k] as unknown as Record<string, unknown>;
      const b = base[k] as unknown as Record<string, unknown>;
      const d = Object.fromEntries(
        Object.entries(a).filter(([f, v]) => v !== b[f]),
      );
      if (Object.keys(d).length) out[k] = d;
    } else if (s[k] !== base[k]) {
      out[k] = s[k];
    }
  }
  return out;
}

/** Each CardStyle shadow as a CSS box-shadow. */
export const CARD_SHADOWS: Record<CardStyle['shadow'], string> = {
  none: 'none',
  soft: '0 1px 3px var(--card-shadow-near), 0 2px 6px var(--card-shadow-far)',
  hard: '2px 2px 0 var(--card-shadow-hard)',
};

/** Edge line width in px for a card `width` px wide: CardStyle.outline.width
 *  scaled from 100px, floored at 1px so it never fades to a sub-pixel smear. */
export function cardEdgeWidth(style: CardStyle, width: number): number {
  const w = style.outline.width;
  return w > 0 ? Math.max(1, (w * width) / 100) : 0;
}

/** The style's radius, shadow, selection and edge as CSS variables cards.css reads. */
export function cardStyleVars(style: CardStyle, width: number): CSSProperties {
  const vars: Record<string, string> = {
    '--pcard-edge-width': `${String(cardEdgeWidth(style, width))}px`,
    '--pcard-edge-color': style.outline.color,
    '--pcard-radius': `${String(Math.max(0, style.radius) * width)}px`,
    '--pcard-shadow': CARD_SHADOWS[style.shadow],
    // The same shadow as a box-shadow list entry ('none' cannot join a list).
    '--pcard-shadow-list':
      style.shadow === 'none'
        ? '0 0 0 0 var(--color-none)'
        : CARD_SHADOWS[style.shadow],
    '--pcard-outline': style.selection.outline,
    // Tag changes (a ring, a tint) ease with the style's motion, at the
    // viewer's Animation speed (--anim-scale on <html>, src/motion).
    '--pcard-motion-ms': `calc(${String(Math.max(0, style.motion.durationMs))}ms * var(--anim-scale, 1))`,
    '--pcard-motion-ease': style.motion.easing,
    '--pcard-lift': `${String(style.selection.lift)}px`,
  };
  return vars;
}

/** Tag-style fields that become CSS variables, and their names. Game tags
 *  write --pcard-<f>; library tags write --pcard-l-<f>, which cards.css reads
 *  first, so a library tag (focus, drop-over) wins over a game tag as the
 *  merge rule says, whichever element carries it. */
const TAG_VARS = {
  ring: 'ring',
  glow: 'glow',
  tint: 'tint',
  band: 'band',
  filter: 'filter',
  cursor: 'cursor',
} as const;

/** Inline CSS variables for a resolved game-tag style. */
export function tagStyleVars(s: TagStyle): CSSProperties {
  const out: Record<string, string> = {};
  for (const [f, v] of Object.entries(TAG_VARS)) {
    const val = s[f as keyof typeof TAG_VARS];
    if (val !== undefined) out[`--pcard-${v}`] = val;
  }
  return out;
}

/** One scoped rule per library tag: `[data-cs="<scope>"] .pcard--t-<tag>{--pcard-l-ring:…}`.
 *  The runtime toggles library-tag classes on card wrappers without a React
 *  render; these rules give them their look. Game tags are styled inline. */
export function libraryTagSheet(scope: string, style: CardStyle): string {
  const tags = tagStyles(style);
  const rules: string[] = [];
  for (const t of LIBRARY_TAGS) {
    const s = tags[t];
    if (!s) continue;
    assertLibraryTagStyle(t, s);
    // The focus ring is an offset outline (cards.css), not the tag ring:
    // a focused card keeps its selection or drop ring visible, and an
    // outline survives forced-colors mode where box-shadow is dropped.
    const decl = Object.entries(TAG_VARS)
      .filter(([f]) => s[f as keyof typeof TAG_VARS] !== undefined)
      .map(
        ([f, v]) =>
          `--pcard-l-${t === 'focus' && f === 'ring' ? 'focus' : v}:${cssValue(String(s[f as keyof typeof TAG_VARS]))}`,
      );
    if (s.z !== undefined) decl.push(`z-index:${String(s.z)}`);
    if (decl.length) {
      rules.push(`[data-cs="${scope}"] .pcard--t-${t}{${decl.join(';')}}`);
    }
    // A library tag's lift (a hover peek) is paint only: the runtime
    // toggles the class without a render, so nothing reflows and the
    // resting layout (what the legibility check measures) is unchanged.
    // `translate` composes with the wrapper's own transform (turn, arc
    // angle) and FLIP, and moves toward the slot's up. It lands on the
    // card wrapper carrying the tag, or on the cards and empty outline of
    // a slot carrying it; an untagged wrapper gets no translate at all.
    if (s.lift) {
      const at = `[data-cs="${scope}"]`;
      rules.push(
        `${at} [data-card-id].pcard--t-${t},${at} .pcard--t-${t} [data-card-id],${at} .pcard--t-${t} [data-pile-empty]{translate:0 ${String(-s.lift)}px}`,
      );
    }
  }
  return rules.join('\n');
}

/** Refuses values that could break out of a declaration. */
function cssValue(v: string): string {
  if (/[;{}<>]/.test(v)) {
    throw new Error(
      `CardStyle tag value ${JSON.stringify(v)} may not contain ; { } < >`,
    );
  }
  return v;
}

/** A stable scope id for a style's library-tag sheet. */
export function tagScope(style: CardStyle): string {
  const src = JSON.stringify(
    LIBRARY_TAGS.map(t => tagStyles(style)[t] ?? null),
  );
  let h = 5381;
  for (let i = 0; i < src.length; i++) {
    h = ((h << 5) + h + src.charCodeAt(i)) | 0;
  }
  return `cs${(h >>> 0).toString(36)}`;
}
