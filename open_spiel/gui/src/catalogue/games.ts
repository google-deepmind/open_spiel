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

import type {CatalogueEntry} from '../contract/contract.ts';
import {RENDERERS} from '../renderers/registry.ts';
import type {Renderers} from '../renderers/renderer.ts';

const INFORMATION: Record<CatalogueEntry['information'], string> = {
  perfect_information: 'perfect information',
  imperfect_information: 'imperfect information',
  one_shot: 'one shot',
};

/**
 * How a game is drawn: `board` when the client has the renderer the
 * server's adapter names for it, `text` when it is played from its text,
 * and `unavailable` when the server cannot serve it at all. Every game is
 * exactly one, so the three filters' counts add up to the catalogue.
 */
export type GameView = 'board' | 'text' | 'unavailable';

/** Which games the catalogue lists: every one, or those drawn one way. */
export type ViewFilter = 'all' | GameView;

/**
 * How `entry` is drawn, read from the server's catalogue and the client's
 * own renderer registry, the two that decide it on the game page;
 * `unavailable` when the game cannot be played here.
 */
export function gameView(
  entry: CatalogueEntry,
  renderers: Renderers = RENDERERS,
): GameView {
  if (!entry.loadable) return 'unavailable';
  return entry.renderer !== null && renderers.get(entry.renderer) !== null
    ? 'board'
    : 'text';
}

/**
 * The games whose short or display name contains `query`, ignoring case and
 * the spaces around it, and that `view` admits, in catalogue order.
 */
export function filterGames(
  games: readonly CatalogueEntry[],
  query: string,
  view: ViewFilter = 'all',
  renderers: Renderers = RENDERERS,
): CatalogueEntry[] {
  const wanted = query.trim().toLowerCase();
  return games.filter(
    entry =>
      (entry.name.toLowerCase().includes(wanted) ||
        entry.long_name.toLowerCase().includes(wanted)) &&
      (view === 'all' || gameView(entry, renderers) === view),
  );
}

/** What kind of game `entry` is, such as "2 players, perfect information". */
export function gameFacts(entry: CatalogueEntry): string {
  const {min_seats: fewest, max_seats: most} = entry;
  const players =
    fewest === most
      ? `${String(fewest)} ${fewest === 1 ? 'player' : 'players'}`
      : `${String(fewest)} to ${String(most)} players`;
  return `${players}, ${INFORMATION[entry.information]}`;
}
