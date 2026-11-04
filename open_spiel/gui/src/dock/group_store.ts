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

// The section open in each panel group, remembered per game in the
// browser's local storage. Storage this page cannot read is left as it is:
// the groups then open at their first section and nothing is written.

/** The local storage key the open sections are kept under. */
export const GROUPS_STORAGE_KEY = 'open_spiel.panel_groups';

/** The version of the stored format this page writes. */
export const GROUPS_VERSION = 1;

/** For each group, by id, the id of its open section. */
export type ActiveSections = Readonly<Record<string, string>>;

type Stored = Record<string, Record<string, string>>;

// Every game's open sections, or null when storage holds something else.
function readAll(storage: () => Storage): Stored | null {
  let text: string | null;
  try {
    text = storage().getItem(GROUPS_STORAGE_KEY);
  } catch {
    return null;
  }
  if (text === null) return {};
  let stored: unknown;
  try {
    stored = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(stored) || stored['version'] !== GROUPS_VERSION) return null;
  const games = stored['games'];
  if (!isRecord(games)) return null;
  const all: Stored = {};
  for (const [game, groups] of Object.entries(games)) {
    if (!isRecord(groups)) return null;
    const sections: Record<string, string> = {};
    for (const [group, section] of Object.entries(groups)) {
      if (typeof section !== 'string') return null;
      sections[group] = section;
    }
    all[game] = sections;
  }
  return all;
}

/** The open sections remembered for `game`; none when unreadable. */
export function readActiveSections(
  storage: () => Storage,
  game: string,
): ActiveSections {
  return readAll(storage)?.[game] ?? {};
}

/**
 * Remembers `section` as open in `group` for `game`, over what storage holds
 * now. Returns false, writing nothing, when storage cannot be read or
 * written.
 */
export function writeActiveSection(
  storage: () => Storage,
  game: string,
  group: string,
  section: string,
): boolean {
  const all = readAll(storage);
  if (all === null) return false;
  const games = {...all, [game]: {...all[game], [group]: section}};
  try {
    storage().setItem(
      GROUPS_STORAGE_KEY,
      JSON.stringify({version: GROUPS_VERSION, games}),
    );
  } catch {
    return false;
  }
  return true;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
