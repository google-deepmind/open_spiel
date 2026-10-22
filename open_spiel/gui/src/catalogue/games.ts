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

const INFORMATION: Record<CatalogueEntry['information'], string> = {
  perfect_information: 'perfect information',
  imperfect_information: 'imperfect information',
  one_shot: 'one shot',
};

/**
 * The games whose short or display name contains `query`, ignoring case and
 * the spaces around it, in catalogue order.
 */
export function filterGames(
  games: readonly CatalogueEntry[],
  query: string,
): CatalogueEntry[] {
  const wanted = query.trim().toLowerCase();
  return games.filter(
    ({name, long_name: longName}) =>
      name.toLowerCase().includes(wanted) ||
      longName.toLowerCase().includes(wanted),
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
