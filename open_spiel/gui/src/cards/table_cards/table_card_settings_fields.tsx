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

// The one settings section for the table surface and the cards, reused for
// the app-wide settings and each adopting game's View panel.
import {useId, useState, useSyncExternalStore, type JSX} from 'react';
import {
  getDeck,
  listDecks,
  SUIT_GLYPH_SETS,
  SUIT_GLYPH_SET_SPECS,
  type SuitGlyphSet,
} from '../library/index.ts';
import {
  MIN_TEXT_CONTRAST,
  SHADINGS,
  TEXTURES,
  whiteTextContrast,
  type Shading,
  type Texture,
} from './settings.ts';
import {SHADING_LABELS, SURFACE_SWATCHES, TEXTURE_LABELS} from './surface.ts';
import {readColourOverrides, subscribe} from './store.ts';
import {
  useTableCards,
  type TableCardScope,
  type TableCardsApi,
} from './table_cards.tsx';
import './table_cards.css';

/** A group of the settings fields. */
export type TableCardSection = 'surface' | 'cards';

function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
/** 'gin_rummy' -> 'Gin Rummy'. */
function gameName(id: string): string {
  return id.split('_').map(titleCase).join(' ');
}
function listNames(names: readonly string[]): string {
  return names.length <= 1
    ? names.join('')
    : `${names.slice(0, -1).join(', ')} and ${String(names.at(-1))}`;
}
/** What the custom input shows before any pick: Green, the first swatch. */
const CUSTOM_START = '#1a6b35';

type Pinned = (s: TableCardSection, f: string) => boolean;

function SurfaceFields({
  api,
  pinned,
  game,
}: {
  api: TableCardsApi;
  pinned: Pinned;
  game: string | undefined;
}): JSX.Element {
  const {value, set} = api;
  return (
    <>
      {TEXTURES.length > 1 && !pinned('surface', 'texture') && (
        <label className="ogs-config-field">
          <span>Table texture</span>
          <select
            value={value.surface.texture}
            onChange={e => {
              set({surface: {texture: e.target.value as Texture}});
            }}
          >
            {TEXTURES.map(t => (
              <option key={t} value={t}>
                {TEXTURE_LABELS[t]}
              </option>
            ))}
          </select>
        </label>
      )}
      {!pinned('surface', 'shading') && (
        <label className="ogs-config-field">
          <span>Table shading</span>
          <select
            value={value.surface.shading}
            onChange={e => {
              set({surface: {shading: e.target.value as Shading}});
            }}
          >
            {SHADINGS.map(t => (
              <option key={t} value={t}>
                {SHADING_LABELS[t]}
              </option>
            ))}
          </select>
        </label>
      )}
      {!pinned('surface', 'color') && <ColourField api={api} game={game} />}
    </>
  );
}

// A custom pick too light for the games' white table text is not stored.
// The native input keeps showing it (the open picker is never yanked back to
// the old colour mid-drag) and the refusal is said until the next pick or
// until the colour changes anywhere else, which drops it for good.
function useColourPick(api: TableCardsApi): {
  tooLight: string | null;
  pickCustom: (hex: string) => void;
  pickSwatch: (c: string | null) => void;
} {
  const {value, set} = api;
  const {color} = value.surface;
  const [refused, setRefused] = useState<string | null>(null);
  const [seenColor, setSeenColor] = useState(color);
  if (seenColor !== color) {
    setSeenColor(color);
    setRefused(null);
  }
  return {
    tooLight: seenColor === color ? refused : null,
    pickCustom: (hex: string) => {
      if (whiteTextContrast(hex) < MIN_TEXT_CONTRAST) {
        setRefused(hex);
        return;
      }
      setRefused(null);
      set({surface: {color: hex}});
    },
    pickSwatch: (c: string | null) => {
      setRefused(null);
      set({surface: {color: c}});
    },
  };
}

function Swatches({
  color,
  tooLight,
  pickCustom,
  pickSwatch,
  labelledBy,
}: {
  color: string | null;
  tooLight: string | null;
  pickCustom: (hex: string) => void;
  pickSwatch: (c: string | null) => void;
  labelledBy: string;
}): JSX.Element {
  const custom =
    color !== null && !SURFACE_SWATCHES.some(s => s.color === color);
  const shown = tooLight ?? color;
  return (
    <div
      className="table-cards-swatches"
      role="radiogroup"
      aria-labelledby={labelledBy}
    >
      <button
        type="button"
        role="radio"
        aria-checked={color === null}
        title="Theme default"
        aria-label="Theme default"
        className="table-cards-swatch table-cards-swatch--theme"
        onClick={() => {
          pickSwatch(null);
        }}
      />
      {SURFACE_SWATCHES.map(s => (
        <button
          key={s.color}
          type="button"
          role="radio"
          aria-checked={color === s.color}
          title={s.label}
          aria-label={s.label}
          className="table-cards-swatch"
          style={{background: s.color}}
          onClick={() => {
            pickSwatch(s.color);
          }}
        />
      ))}
      <label
        className={`table-cards-swatch table-cards-swatch--custom${custom ? ' is-on' : ''}${tooLight !== null ? ' is-refused' : ''}`}
        title="Custom colour"
        style={
          (custom || tooLight !== null) && shown !== null
            ? {background: shown}
            : undefined
        }
      >
        <input
          type="color"
          aria-label="Custom colour"
          value={shown ?? CUSTOM_START}
          aria-invalid={tooLight !== null}
          onChange={e => {
            pickCustom(e.target.value.toLowerCase());
          }}
        />
      </label>
    </div>
  );
}

function ColourField({
  api,
  game,
}: {
  api: TableCardsApi;
  game: string | undefined;
}): JSX.Element {
  const id = useId();
  const {color} = api.value.surface;
  const {tooLight, pickCustom, pickSwatch} = useColourPick(api);
  // App-wide, name the games whose own colour this does not reach.
  const overriding = useSyncExternalStore(
    subscribe,
    readColourOverrides,
    readColourOverrides,
  );
  const ownColour =
    game === undefined && overriding !== ''
      ? overriding.split(',').map(gameName)
      : [];
  return (
    <div className="ogs-config-field">
      <span id={`${id}-colour`}>
        Table colour
        {tooLight !== null && (
          <span className="table-cards-refused-tag"> (not applied)</span>
        )}
      </span>
      <Swatches
        color={color}
        tooLight={tooLight}
        pickCustom={pickCustom}
        pickSwatch={pickSwatch}
        labelledBy={`${id}-colour`}
      />
      {/* One line kept in place whether or not a pick is refused: a line
          that came and went would move the picker on every crossing. */}
      <span
        className={`table-cards-colour-hint${tooLight !== null ? ' is-refused' : ''}`}
        aria-live="polite"
        data-testid="table-colour-hint"
      >
        {tooLight === null
          ? `Custom colours must keep the table's white text readable (${String(MIN_TEXT_CONTRAST)}:1).`
          : `Not applied: ${tooLight} is too light (white text ${whiteTextContrast(tooLight).toFixed(1)}:1, needs ${String(MIN_TEXT_CONTRAST)}:1).`}
      </span>
      {ownColour.length > 0 && (
        <span className="ogs-config-note" data-testid="table-colour-scope">
          {listNames(ownColour)}{' '}
          {ownColour.length === 1 ? 'uses its' : 'use their'} own table colour
          (This game only), which this does not change.
        </span>
      )}
    </div>
  );
}

function CardFields({
  api,
  pinned,
}: {
  api: TableCardsApi;
  pinned: Pinned;
}): JSX.Element {
  const {value, set} = api;
  const deck = getDeck(value.cards.deck);
  const backs = deck.backs ?? [];
  return (
    <>
      {!pinned('cards', 'deck') && (
        <label className="ogs-config-field">
          <span>Card deck</span>
          <select
            value={deck.id}
            onChange={e => {
              set({cards: {deck: e.target.value}});
            }}
          >
            {listDecks().map(d => (
              <option key={d.id} value={d.id} title={d.description}>
                {d.label}
              </option>
            ))}
          </select>
          {deck.description !== undefined && (
            <span className="table-cards-hint" data-testid="deck-description">
              {deck.description}
            </span>
          )}
        </label>
      )}
      {deck.drawsSuitGlyphs &&
        SUIT_GLYPH_SETS.length > 1 &&
        !pinned('cards', 'suitGlyphs') && (
          <label className="ogs-config-field">
            <span>Suit symbols</span>
            <select
              value={value.cards.suitGlyphs}
              onChange={e => {
                set({cards: {suitGlyphs: e.target.value as SuitGlyphSet}});
              }}
            >
              {SUIT_GLYPH_SETS.map(g => (
                <option key={g} value={g}>
                  {SUIT_GLYPH_SET_SPECS[g].label}
                </option>
              ))}
            </select>
          </label>
        )}
      {backs.length > 0 && !pinned('cards', 'back') && (
        <label className="ogs-config-field">
          <span>Card back</span>
          <select
            value={value.cards.back}
            onChange={e => {
              set({cards: {back: e.target.value}});
            }}
          >
            <option value="default">Deck default</option>
            {backs.map(b => (
              <option key={b} value={b}>
                {deck.backLabels?.[b] ?? titleCase(b)}
              </option>
            ))}
          </select>
        </label>
      )}
    </>
  );
}

/** The table and card settings fields, for the app or one game. */
export function TableCardSettingsFields({
  sections = ['surface', 'cards'],
  ...scope
}: TableCardScope & {
  sections?: readonly TableCardSection[] | undefined;
}): JSX.Element {
  const api = useTableCards(scope);
  const pinned: Pinned = (s, f) =>
    (scope.pin?.[s] as Record<string, unknown> | undefined)?.[f] !== undefined;
  return (
    <div className="table-cards-fields" data-testid="table-card-settings">
      {scope.game !== undefined && (
        <label className="ogs-config-toggle">
          <input
            type="checkbox"
            checked={api.overridden}
            onChange={e => {
              api.setOverridden(e.target.checked);
            }}
          />
          <span>This game only</span>
        </label>
      )}
      {sections.includes('surface') && (
        <SurfaceFields api={api} pinned={pinned} game={scope.game} />
      )}
      {sections.includes('cards') && <CardFields api={api} pinned={pinned} />}
    </div>
  );
}
