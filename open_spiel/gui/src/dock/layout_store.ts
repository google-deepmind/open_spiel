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

// Saved dock layouts, kept in the browser's local storage. A layout is
// dockview's own serialisation of the dock and the tier it was saved at,
// saved for one game or for all games. Each game may name a default layout,
// and so may all games together; a game's own default comes first. A
// stored layout is checked against the dock's panels before it is offered,
// so a dock is only ever restored with panels it defines.

import type {GroupviewPanelState, SerializedDockview} from 'dockview-core';

import type {Tier} from './arrangement.ts';

/** The local storage key the saved layouts are kept under. */
export const LAYOUTS_STORAGE_KEY = 'open_spiel.layouts';

/** The version of the stored format this page writes. */
export const LAYOUTS_VERSION = 2;

/** The most characters a layout's name may have. */
export const MAX_NAME_LENGTH = 80;

const TIERS: ReadonlySet<string> = new Set<Tier>(['three', 'two', 'tabs']);

/** A dock's arrangement, and the tier of the dock it was taken from. */
export interface DockLayout {
  readonly tier: Tier;
  readonly dock: SerializedDockview;
}

/** A layout saved under a name, for one game or, when `game` is null, all. */
export interface SavedLayout extends DockLayout {
  /** Unique among the saved layouts; never shown. */
  readonly id: string;
  readonly name: string;
  readonly game: string | null;
}

/** The default layouts, by id: one for all games, and one per game. */
export interface Defaults {
  readonly all: string | null;
  readonly games: Readonly<Record<string, string>>;
}

/** Everything saved: the layouts, in the order they are shown, and defaults. */
export interface LayoutStore {
  readonly layouts: readonly SavedLayout[];
  readonly defaults: Defaults;
}

/** The store of a browser that has saved nothing. */
export const EMPTY_STORE: LayoutStore = {
  layouts: [],
  defaults: {all: null, games: {}},
};

/** The saved layouts, or why they cannot be used. */
export type LayoutsReading =
  | {readonly ok: true; readonly store: LayoutStore}
  | {readonly ok: false; readonly problem: string};

/**
 * What is saved in `storage`, each layout checked against `panels`, the ids
 * of the dock's panels. Nothing saved reads as the empty store. Layouts
 * written in version 1, before layouts had a game, read as layouts for all
 * games. Storage that holds anything this page did not write, or wrote in a
 * newer version, is refused whole, so that saving never overwrites it.
 */
export function readLayouts(
  storage: () => Storage,
  panels: ReadonlySet<string>,
): LayoutsReading {
  let text: string | null;
  try {
    text = storage().getItem(LAYOUTS_STORAGE_KEY);
  } catch (e) {
    return {ok: false, problem: `This browser keeps no layouts: ${String(e)}`};
  }
  if (text === null) return {ok: true, store: EMPTY_STORE};
  let stored: unknown;
  try {
    stored = JSON.parse(text);
  } catch {
    return {ok: false, problem: 'The saved layouts are not valid JSON.'};
  }
  if (!isRecord(stored) || typeof stored['version'] !== 'number') {
    return {ok: false, problem: 'The saved layouts have no version.'};
  }
  const {version} = stored;
  if (version !== 1 && version !== LAYOUTS_VERSION) {
    return {
      ok: false,
      problem: `The saved layouts are in version ${String(version)}, which this page does not read.`,
    };
  }
  const layouts = stored['layouts'];
  if (!Array.isArray(layouts)) {
    return {ok: false, problem: 'The saved layouts are not a list.'};
  }
  const read: SavedLayout[] = [];
  for (const [index, value] of layouts.entries()) {
    const layout: unknown = version === 1 ? fromVersion1(value) : value;
    const problem = layoutProblem(layout, panels);
    if (problem !== null) {
      return {
        ok: false,
        problem: `Saved layout ${String(index + 1)} ${problem}`,
      };
    }
    read.push(layout as SavedLayout);
  }
  const defaults = version === 1 ? EMPTY_STORE.defaults : stored['defaults'];
  const problem = storeProblem(read, defaults);
  if (problem !== null) return {ok: false, problem};
  return {ok: true, store: {layouts: read, defaults: defaults as Defaults}};
}

/** Stores `store`, replacing everything saved; throws if storage does. */
export function writeLayouts(storage: () => Storage, store: LayoutStore): void {
  storage().setItem(
    LAYOUTS_STORAGE_KEY,
    JSON.stringify({version: LAYOUTS_VERSION, ...store}),
  );
}

/** Removes every saved layout and default; throws if storage does. */
export function clearLayouts(storage: () => Storage): void {
  storage().removeItem(LAYOUTS_STORAGE_KEY);
}

/**
 * Why `name` cannot name a layout among `siblings`, the layouts it would sit
 * with, as a sentence, or null when it can. A layout being renamed, `except`,
 * may keep its own name.
 */
export function nameProblem(
  name: string,
  siblings: readonly SavedLayout[] = [],
  except?: string,
): string | null {
  const trimmed = name.trim();
  if (trimmed === '') return 'A layout needs a name.';
  if (trimmed.length > MAX_NAME_LENGTH) {
    return `A layout's name has at most ${String(MAX_NAME_LENGTH)} characters.`;
  }
  const taken = siblings.some(
    layout =>
      layout.id !== except &&
      layout.name.localeCompare(trimmed, undefined, {sensitivity: 'base'}) ===
        0,
  );
  return taken ? `A layout named "${trimmed}" is saved already.` : null;
}

/** The ids of the panels open in `layout`. */
export function layoutPanels(layout: DockLayout): string[] {
  return Object.keys(layout.dock.panels);
}

/**
 * What a dock shows, without its sizes: its groups as they are split, the
 * panels of each in order and the one in front. Two docks that differ only
 * in sizes have the same signature.
 */
export function layoutSignature(layout: DockLayout): string {
  return `${layout.tier}:${JSON.stringify(shape(layout.dock.grid.root))}`;
}

function shape(node: unknown): unknown {
  if (!isRecord(node)) return null;
  const {type, data} = node;
  if (type === 'branch' && Array.isArray(data)) return data.map(shape);
  if (type === 'leaf' && isRecord(data)) {
    return {views: data['views'], active: data['activeView']};
  }
  return null;
}

// A version 1 layout as version 2 keeps it: a layout for all games.
function fromVersion1(value: unknown): unknown {
  if (!isRecord(value) || typeof value['name'] !== 'string') return value;
  return {...value, id: `version-1:${value['name']}`, game: null};
}

// Why `layouts` and `defaults` do not make a store, as a sentence, or null.
function storeProblem(
  layouts: readonly SavedLayout[],
  defaults: unknown,
): string | null {
  const byId = new Map(layouts.map(layout => [layout.id, layout]));
  if (byId.size !== layouts.length) return 'Two saved layouts share an id.';
  for (const layout of layouts) {
    const siblings = layouts.filter(other => other.game === layout.game);
    if (nameProblem(layout.name, siblings, layout.id) !== null) {
      return `Two saved layouts share the name "${layout.name}".`;
    }
  }
  if (!isRecord(defaults) || !isRecord(defaults['games'])) {
    return 'The saved defaults are not readable.';
  }
  const all = defaults['all'];
  if (
    all !== null &&
    (typeof all !== 'string' || byId.get(all)?.game !== null)
  ) {
    return 'The default for all games is not a layout for all games.';
  }
  for (const [game, id] of Object.entries(defaults['games'])) {
    const layout = typeof id === 'string' ? byId.get(id) : undefined;
    if (layout === undefined || (layout.game ?? game) !== game) {
      return `The default for ${game} is not a layout it can use.`;
    }
  }
  return null;
}

// Why `value` is not a layout of the dock with `panels`, as the end of a
// sentence, or null when it is one.
function layoutProblem(
  value: unknown,
  panels: ReadonlySet<string>,
): string | null {
  if (!isRecord(value)) return 'is not an object.';
  const {id, name, game, tier, dock} = value;
  if (typeof name !== 'string' || nameProblem(name) !== null) {
    return 'has no valid name.';
  }
  if (typeof id !== 'string' || id === '') return `"${name}" has no id.`;
  if (game !== null && typeof game !== 'string') {
    return `"${name}" names no game.`;
  }
  if (typeof tier !== 'string' || !TIERS.has(tier)) {
    return `"${name}" has no valid tier.`;
  }
  if (!isRecord(dock) || !isRecord(dock['grid']) || !isRecord(dock['panels'])) {
    return `"${name}" holds no dock.`;
  }
  const popouts = dock['popoutGroups'];
  if (
    popouts !== undefined &&
    !(Array.isArray(popouts) && popouts.length === 0)
  ) {
    return `"${name}" opens windows of its own.`;
  }
  const ids = Object.keys(dock['panels']);
  if (ids.length === 0) return `"${name}" has no panels.`;
  for (const panel of ids) {
    if (!panels.has(panel)) {
      return `"${name}" has an unknown panel "${panel}".`;
    }
    if (!isPanelState(dock['panels'][panel], panel)) {
      return `"${name}" describes panel "${panel}" as another.`;
    }
  }
  return null;
}

// Whether `value` describes panel `id`, drawn by the component of that name.
function isPanelState(
  value: unknown,
  id: string,
): value is GroupviewPanelState {
  return (
    isRecord(value) && value['id'] === id && value['contentComponent'] === id
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
