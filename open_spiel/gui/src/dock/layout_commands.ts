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

// The commands of the layouts bar. Each says what it did, through the
// status, or why it failed, through the error, and returns whether it
// succeeded.

import {
  type LayoutStore,
  type LayoutsReading,
  type SavedLayout,
  EMPTY_STORE,
  clearLayouts,
  nameProblem,
  readLayouts,
  writeLayouts,
} from './layout_store.ts';
import type {
  LayoutActions,
  SavedLayoutsCommands,
  layoutsFor,
} from './use_saved_layouts.ts';

/** What the commands read and set. */
export interface CommandContext {
  readonly game: string;
  readonly panels: ReadonlySet<string>;
  readonly actions: LayoutActions | null;
  readonly storage: () => Storage;
  readonly active: SavedLayout | null;
  readonly lists: ReturnType<typeof layoutsFor>;
  readonly setReading: (reading: LayoutsReading) => void;
  readonly setActiveId: (id: string | null) => void;
  readonly setReset: (reset: boolean) => void;
  readonly setError: (error: string | null) => void;
  readonly setStatus: (status: string) => void;
  /** Takes the dock as it is now as the arrangement nothing has changed. */
  readonly markBuilt: () => void;
}

/** A change to the store: the new store, or why it cannot be made. */
type Change = (store: LayoutStore) => LayoutStore | string;

/** What every command uses: reporting, storing and the dock. */
interface Helpers {
  readonly fail: (problem: string) => false;
  readonly succeed: (done: string) => true;
  /** Applies a change to what storage holds now and stores the result. */
  readonly store: (change: Change, done: string) => boolean;
  readonly dock: () => LayoutActions;
  readonly requireActive: () => SavedLayout;
}

function helpers(c: CommandContext): Helpers {
  const fail = (problem: string): false => {
    c.setError(problem);
    c.setStatus(problem);
    return false;
  };
  const succeed = (done: string): true => {
    c.setError(null);
    c.setStatus(done);
    return true;
  };
  const store = (change: Change, done: string): boolean => {
    const current = readLayouts(c.storage, c.panels);
    if (!current.ok) {
      c.setReading(current);
      return fail(
        'Nothing was saved: the saved layouts can no longer be read.',
      );
    }
    const next = change(current.store);
    if (typeof next === 'string') return fail(next);
    try {
      writeLayouts(c.storage, next);
    } catch (e) {
      return fail(`The layouts could not be saved: ${String(e)}`);
    }
    c.setReading({ok: true, store: next});
    return succeed(done);
  };
  return {
    fail,
    succeed,
    store,
    dock: () => {
      if (c.actions === null) throw new Error('The dock is not ready.');
      return c.actions;
    },
    requireActive: () => {
      if (c.active === null) throw new Error('No layout is in use.');
      return c.active;
    },
  };
}

// `current` with layout `id`, or none, as the default for `game` or for all
// games.
function withDefault(
  current: LayoutStore,
  game: string,
  level: 'game' | 'all',
  id: string | null,
): LayoutStore {
  const {all, games} = current.defaults;
  if (level === 'all') return {...current, defaults: {all: id, games}};
  const rest = Object.fromEntries(
    Object.entries(games).filter(([other]) => other !== game),
  );
  return {
    ...current,
    defaults: {all, games: id === null ? rest : {...rest, [game]: id}},
  };
}

// Showing a layout, and putting the built-in or default one back.
function showCommands(
  c: CommandContext,
  h: Helpers,
): Pick<
  SavedLayoutsCommands,
  'apply' | 'resetToSaved' | 'resetToDefault' | 'resetToBuiltIn'
> {
  const apply = (layout: SavedLayout): boolean => {
    try {
      h.dock().restore(layout);
    } catch (e) {
      h.dock().reset();
      c.setActiveId(null);
      return h.fail(
        `The layout ‘${layout.name}’ could not be shown, so the built-in one is: ${String(e)}`,
      );
    }
    c.setActiveId(layout.id);
    c.setReset(false);
    return h.succeed(`Showing the layout ‘${layout.name}’.`);
  };
  return {
    apply,
    resetToSaved: () => apply(h.requireActive()),
    resetToDefault: () => {
      const {effective} = c.lists;
      if (effective === null) throw new Error('No default is set.');
      return apply(effective.layout);
    },
    resetToBuiltIn: () => {
      h.dock().reset();
      c.setActiveId(null);
      c.setReset(true);
      return h.succeed('Showing the built-in layout.');
    },
  };
}

// Saving the dock, as a new layout or over the one in use, and defaults.
function saveCommands(
  c: CommandContext,
  h: Helpers,
): Pick<
  SavedLayoutsCommands,
  'saveAs' | 'saveActive' | 'makeGameDefault' | 'makeAllDefault' | 'stopDefault'
> {
  const saveNew = (
    name: string,
    game: string | null,
    level: 'game' | 'all' | null,
    done: string,
  ): boolean => {
    const layout: SavedLayout = {
      id: crypto.randomUUID(),
      name: name.trim(),
      game,
      ...h.dock().capture(),
    };
    const siblings = (current: LayoutStore): SavedLayout[] =>
      current.layouts.filter(other => other.game === game);
    const saved = h.store(current => {
      const problem = nameProblem(name, siblings(current));
      if (problem !== null) return problem;
      const added = {...current, layouts: [...current.layouts, layout]};
      return level === null
        ? added
        : withDefault(added, c.game, level, layout.id);
    }, done);
    if (saved) {
      c.setActiveId(layout.id);
      c.markBuilt();
    }
    return saved;
  };
  const makeDefault = (level: 'game' | 'all', name?: string): boolean => {
    const where = level === 'game' ? 'this game' : 'all games';
    if (name !== undefined) {
      return saveNew(
        name,
        level === 'game' ? c.game : null,
        level,
        `Saved ‘${name.trim()}’ as the default for ${where}.`,
      );
    }
    const active = h.requireActive();
    return h.store(
      current => withDefault(current, c.game, level, active.id),
      `‘${active.name}’ is the default for ${where}.`,
    );
  };
  return {
    saveAs: name =>
      saveNew(name, c.game, null, `Saved the layout ‘${name.trim()}’.`),
    saveActive: () => {
      const active = h.requireActive();
      const {tier, dock} = h.dock().capture();
      return h.store(
        current => ({
          ...current,
          layouts: current.layouts.map(layout =>
            layout.id === active.id ? {...layout, tier, dock} : layout,
          ),
        }),
        `Saved the layout ‘${active.name}’.`,
      );
    },
    makeGameDefault: name => makeDefault('game', name),
    makeAllDefault: name => makeDefault('all', name),
    stopDefault: level =>
      h.store(
        current => withDefault(current, c.game, level, null),
        `No layout is the default for ${level === 'game' ? 'this game' : 'all games'}.`,
      ),
  };
}

// Renaming, moving and deleting layouts, and clearing storage.
function editCommands(
  c: CommandContext,
  h: Helpers,
): Pick<SavedLayoutsCommands, 'rename' | 'move' | 'remove' | 'clear'> {
  return {
    rename: (layout, name) =>
      h.store(
        current =>
          nameProblem(
            name,
            current.layouts.filter(other => other.game === layout.game),
            layout.id,
          ) ?? {
            ...current,
            layouts: current.layouts.map(other =>
              other.id === layout.id ? {...other, name: name.trim()} : other,
            ),
          },
        `Renamed ‘${layout.name}’ to ‘${name.trim()}’.`,
      ),
    move: (layout, index) =>
      h.store(
        current => moved(current, layout, index),
        `Moved ‘${layout.name}’.`,
      ),
    remove: layout => {
      const removed = h.store(
        current => withoutLayout(current, layout.id),
        `Deleted the layout ‘${layout.name}’.`,
      );
      if (removed && layout.id === c.active?.id) c.setActiveId(null);
      return removed;
    },
    clear: () => {
      try {
        clearLayouts(c.storage);
      } catch (e) {
        return h.fail(`The saved layouts could not be cleared: ${String(e)}`);
      }
      c.setReading({ok: true, store: EMPTY_STORE});
      c.setActiveId(null);
      return h.succeed('Cleared the saved layouts.');
    },
  };
}

/** The commands over the context `c`. */
export function layoutCommands(c: CommandContext): SavedLayoutsCommands {
  const h = helpers(c);
  return {
    ...showCommands(c, h),
    ...saveCommands(c, h),
    ...editCommands(c, h),
    dismissError: () => {
      c.setError(null);
    },
  };
}

// `store` with `layout` at position `index` among the layouts of its game.
function moved(
  store: LayoutStore,
  layout: SavedLayout,
  index: number,
): LayoutStore | string {
  const siblings = store.layouts.filter(other => other.game === layout.game);
  const current = siblings.find(other => other.id === layout.id);
  if (current === undefined) {
    return `The layout ‘${layout.name}’ is no longer saved.`;
  }
  if (index < 0 || index >= siblings.length) {
    return `‘${layout.name}’ cannot move further.`;
  }
  const order = siblings.filter(other => other.id !== layout.id);
  order.splice(index, 0, current);
  const next = order.values();
  return {
    ...store,
    layouts: store.layouts.map(other =>
      other.game === layout.game ? (next.next().value ?? other) : other,
    ),
  };
}

// `store` without layout `id`, and without any default that names it.
function withoutLayout(store: LayoutStore, id: string): LayoutStore {
  return {
    layouts: store.layouts.filter(layout => layout.id !== id),
    defaults: {
      all: store.defaults.all === id ? null : store.defaults.all,
      games: Object.fromEntries(
        Object.entries(store.defaults.games).filter(([, game]) => game !== id),
      ),
    },
  };
}
