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

// The table and card settings for React.
// Opt-in: a game that calls none of these sees no change.
import {
  useCallback,
  useMemo,
  useSyncExternalStore,
  type CSSProperties,
  type JSX,
  type ReactNode,
} from 'react';
import {CardStyleProvider} from '../library/index.ts';
import {
  resolveTableCards,
  type TableCardPatch,
  type TableCardSettings,
} from './settings.ts';
import {
  readApp,
  readOverride,
  setOverrideEnabled,
  subscribe,
  writeTableCards,
} from './store.ts';
import {surfaceVars} from './surface.ts';

/** Where a game's settings come from, and what it fixes. */
export interface TableCardScope {
  /** The game's id ([a-z0-9_]+). Absent: the app-wide value, no override. */
  game?: string | undefined;
  /** The game's own defaults, under every choice the user made. */
  defaults?: TableCardPatch | undefined;
  /** Fields the game fixes, over every choice (a game's own card face pins
   *  its deck). Their controls are hidden. */
  pin?: TableCardPatch | undefined;
}

/** The settings in effect for a game, and how to change them. */
export interface TableCardsApi {
  /** What the game shows: defaults, game defaults, app-wide, override, pin. */
  value: TableCardSettings;
  /** The game has its own override ("This game only"). */
  overridden: boolean;
  /** Sets fields where the game reads them from: its override when it has
   *  one, else the app-wide value. Throws on an invalid value. */
  set: (patch: TableCardPatch) => void;
  /** Turns the game's override on or off. Throws without a game. */
  setOverridden: (on: boolean) => void;
}

function stableKey(p: TableCardPatch | undefined): string {
  return p ? JSON.stringify(p) : '';
}

/** The settings in effect for a scope, and how to change them. */
export function useTableCards({
  game,
  defaults,
  pin,
}: TableCardScope = {}): TableCardsApi {
  const app = useSyncExternalStore(subscribe, readApp, readApp);
  const readGame = useCallback(
    () => (game === undefined ? null : readOverride(game)),
    [game],
  );
  const override = useSyncExternalStore(subscribe, readGame, readGame);
  const dKey = stableKey(defaults);
  const pKey = stableKey(pin);
  const value = useMemo(
    () => resolveTableCards(defaults, app, override, pin),
    // Keyed on content, so an inline literal does not recompute every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [app, override, dKey, pKey],
  );
  const set = useCallback(
    (patch: TableCardPatch) => {
      const own = game === undefined ? null : readOverride(game);
      writeTableCards(own === null ? undefined : game, patch, value.cards);
    },
    [game, value.cards],
  );
  const setOverridden = useCallback(
    (on: boolean) => {
      if (game === undefined) {
        throw new Error(
          'table-cards: only a game (useTableCards({ game })) can have an override',
        );
      }
      setOverrideEnabled(game, on);
    },
    [game],
  );
  return {value, overridden: override !== null, set, setOverridden};
}

/** The surface as CSS variables for the game's own root element (no wrapper,
 *  no layout change). The game paints with var(--color-felt*) and, to show a
 *  texture, `background: var(--table-texture, none), var(--color-felt)`.
 *  A game with a gradient or vignette of its own paints it through the
 *  shading variables (--table-shading, --table-shadow, --table-grain;
 *  surface.ts SHADING_VARS), so 'flat' turns it off. */
export function useTableSurface(game?: string): CSSProperties {
  const {value} = useTableCards({game});
  const {texture, shading, color} = value.surface;
  return useMemo(
    () => surfaceVars({texture, shading, color}),
    [texture, shading, color],
  );
}

/** A CardStyleProvider with the scope's deck, suit symbols and back. Adds no
 *  DOM. A CardStyleProvider nested inside (a game's own tweaks) still wins. */
export function TableCardStyle({
  children,
  ...scope
}: TableCardScope & {children: ReactNode}): JSX.Element {
  const {value} = useTableCards(scope);
  const {deck, suitGlyphs, back} = value.cards;
  const style = useMemo(
    () => ({deck, suitGlyphs, back}),
    [deck, suitGlyphs, back],
  );
  return <CardStyleProvider value={style}>{children}</CardStyleProvider>;
}
