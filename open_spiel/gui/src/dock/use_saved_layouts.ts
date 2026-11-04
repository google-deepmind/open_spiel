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

// What the saved layouts are and what can be done to them, for the layouts
// bar: the commands, the layout in use and the words that say so. Every
// change is made to what storage holds at the time, not to what this page
// last read, so a layout another tab saved meanwhile survives; storage that
// can no longer be read is refused, never overwritten.

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import {
  type DockLayout,
  type LayoutStore,
  type LayoutsReading,
  type SavedLayout,
  EMPTY_STORE,
  layoutSignature,
  readLayouts,
} from './layout_store.ts';
import {layoutCommands} from './layout_commands.ts';

/** What happened to the dock: built afresh, or changed by the user. */
export type DockEvent = 'build' | 'change';

/** What the saved layouts do to the dock they belong to. */
export interface LayoutActions {
  /** The dock's arrangement now. */
  readonly capture: () => DockLayout;
  /** Arranges the dock as `layout`; throws if dockview cannot. */
  readonly restore: (layout: DockLayout) => void;
  /** Arranges the dock as it was built. */
  readonly reset: () => void;
  /** Calls `listener` after each change to the dock; returns the undo. */
  readonly subscribe: (listener: (event: DockEvent) => void) => () => void;
}

/** A default and the level it is the default for. */
export interface EffectiveDefault {
  readonly layout: SavedLayout;
  readonly level: 'game' | 'all';
}

/** The layouts bar's model. */
export interface SavedLayoutsModel {
  readonly reading: LayoutsReading;
  readonly gameLayouts: readonly SavedLayout[];
  readonly allLayouts: readonly SavedLayout[];
  readonly active: SavedLayout | null;
  /** Whether the dock differs from the active layout as saved. */
  readonly modified: boolean;
  readonly gameDefault: SavedLayout | null;
  readonly allDefault: SavedLayout | null;
  /** The default this game opens with: its own, else that of all games. */
  readonly effective: EffectiveDefault | null;
  /** The level in use, as the menu's "Using: …" line says it. */
  readonly using: string;
  /** The level in use, as a few words on the bar; null with a layout. */
  readonly word: string | null;
  /** Why the last command failed; null when it did not. */
  readonly error: string | null;
  /** What the last command did, for screen readers. */
  readonly status: string;
}

/** The commands of the layouts bar; each returns whether it succeeded. */
export interface SavedLayoutsCommands {
  readonly apply: (layout: SavedLayout) => boolean;
  readonly saveAs: (name: string) => boolean;
  readonly saveActive: () => boolean;
  readonly makeGameDefault: (name?: string) => boolean;
  readonly makeAllDefault: (name?: string) => boolean;
  readonly stopDefault: (level: 'game' | 'all') => boolean;
  readonly resetToSaved: () => boolean;
  readonly rename: (layout: SavedLayout, name: string) => boolean;
  readonly move: (layout: SavedLayout, index: number) => boolean;
  readonly remove: (layout: SavedLayout) => boolean;
  readonly resetToDefault: () => boolean;
  readonly resetToBuiltIn: () => boolean;
  readonly clear: () => boolean;
  readonly dismissError: () => void;
}

/** The layouts bar's model and commands. */
export type SavedLayouts = SavedLayoutsModel & SavedLayoutsCommands;

/** The properties of useSavedLayouts. */
export interface SavedLayoutsOptions {
  /** The game whose layouts these are. */
  readonly game: string;
  /** The ids of the dock's panels; a saved layout must use only these. */
  readonly panels: ReadonlySet<string>;
  /** What the layouts do to the dock; null until it is ready. */
  readonly actions: LayoutActions | null;
  /** Where layouts are kept; the browser's local storage when omitted. */
  readonly storage?: () => Storage;
}

function browserStorage(): Storage {
  return window.localStorage;
}

/** The lists, defaults and the default in effect of `store` for `game`. */
export function layoutsFor(
  store: LayoutStore,
  game: string,
): Pick<
  SavedLayoutsModel,
  'gameLayouts' | 'allLayouts' | 'gameDefault' | 'allDefault' | 'effective'
> {
  const byId = (id: string | null | undefined): SavedLayout | null =>
    store.layouts.find(layout => layout.id === id) ?? null;
  const gameDefault = byId(store.defaults.games[game]);
  const allDefault = byId(store.defaults.all);
  return {
    gameLayouts: store.layouts.filter(layout => layout.game === game),
    allLayouts: store.layouts.filter(layout => layout.game === null),
    gameDefault,
    allDefault,
    effective:
      gameDefault !== null
        ? {layout: gameDefault, level: 'game'}
        : allDefault !== null
          ? {layout: allDefault, level: 'all'}
          : null,
  };
}

/** A default as the menu names it: ‘Wide’ (this game). */
export function defaultName({layout, level}: EffectiveDefault): string {
  return `‘${layout.name}’ (${level === 'game' ? 'this game' : 'all games'})`;
}

/** What the dock shows when no layout is active. */
export type Baseline = 'built-in' | 'reset' | 'unsaved';

/**
 * The "Using: …" line: the layout in use and what it is the default for, or
 * the unsaved arrangement or built-in layout on screen and, when a default
 * is set but not in use, which.
 */
export function usingText(
  model: Pick<
    SavedLayoutsModel,
    'active' | 'modified' | 'gameDefault' | 'allDefault'
  >,
  baseline: Baseline,
): string {
  const {active, modified, gameDefault, allDefault} = model;
  if (active !== null) {
    const where =
      gameDefault?.id === active.id
        ? ', default for this game'
        : allDefault?.id === active.id
          ? ', default for all games'
          : active.game === null
            ? ', a layout for all games'
            : '';
    return `Using: ‘${active.name}’${where}${modified ? ' (modified)' : ''}`;
  }
  const notUsed: EffectiveDefault[] = [];
  if (gameDefault !== null) notUsed.push({layout: gameDefault, level: 'game'});
  if (allDefault !== null) notUsed.push({layout: allDefault, level: 'all'});
  const set = notUsed.map(defaultName).join(' and ');
  const plural = notUsed.length > 1 ? 'Defaults' : 'Default';
  if (baseline === 'unsaved') {
    return notUsed.length > 0
      ? `Using: an unsaved arrangement you made here. ${plural} not in use: ${set}`
      : 'Using: an unsaved arrangement (no saved layout)';
  }
  if (baseline === 'reset') {
    return notUsed.length > 0
      ? `Using: this game's built-in layout, because you reset to it. ${plural} still set: ${set}`
      : "Using: this game's built-in layout, because you reset to it";
  }
  return notUsed.length > 0
    ? `Using: this game's built-in layout. ${plural} set but not in use: ${set}`
    : "Using: this game's built-in layout (no default is set)";
}

/** The words on the bar for the level in use; null with a layout active. */
export function stateWord(
  active: SavedLayout | null,
  baseline: Baseline,
): string | null {
  if (active !== null) return null;
  if (baseline === 'unsaved') return 'Unsaved arrangement';
  if (baseline === 'reset') return 'Built-in layout (you reset)';
  return 'Built-in layout';
}

/** The signature of the dock built for `actions`. */
interface Built {
  readonly actions: LayoutActions;
  readonly signature: string;
}

/**
 * The signature of the dock now, and of the dock as last built: as it was
 * when the dock became ready, and after each rebuild since.
 */
function useDockSignature(actions: LayoutActions | null): {
  readonly current: string | null;
  readonly built: string | null;
  readonly markBuilt: () => void;
} {
  const [built, setBuilt] = useState<Built | null>(null);
  if (actions !== null && built?.actions !== actions) {
    setBuilt({actions, signature: layoutSignature(actions.capture())});
  }
  const subscribe = useCallback(
    (listener: () => void) =>
      actions === null
        ? () => undefined
        : actions.subscribe(event => {
            if (event === 'build') {
              setBuilt({
                actions,
                signature: layoutSignature(actions.capture()),
              });
            }
            listener();
          }),
    [actions],
  );
  const current = useSyncExternalStore(subscribe, () =>
    actions === null ? null : layoutSignature(actions.capture()),
  );
  const markBuilt = useCallback(() => {
    if (actions !== null) {
      setBuilt({actions, signature: layoutSignature(actions.capture())});
    }
  }, [actions]);
  return {
    current,
    built: built?.actions === actions ? built.signature : null,
    markBuilt,
  };
}

/**
 * The saved layouts of `game`. Once the dock is ready, the default in effect
 * is applied, once.
 */
export function useSavedLayouts(options: SavedLayoutsOptions): SavedLayouts {
  const {game, panels, actions, storage = browserStorage} = options;
  const [reading, setReading] = useState<LayoutsReading>(() =>
    readLayouts(storage, panels),
  );
  const [activeId, setActiveId] = useState<string | null>(null);
  const [reset, setReset] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const signature = useDockSignature(actions);
  const store = reading.ok ? reading.store : null;
  const lists = layoutsFor(store ?? EMPTY_STORE, game);
  const active = store?.layouts.find(({id}) => id === activeId) ?? null;
  const modified =
    active !== null &&
    signature.current !== null &&
    signature.current !== layoutSignature(active);
  const baseline: Baseline =
    signature.built !== null && signature.current !== signature.built
      ? 'unsaved'
      : reset
        ? 'reset'
        : 'built-in';
  const model = {...lists, active, modified};
  const commands = layoutCommands({
    game,
    panels,
    actions,
    storage,
    active,
    lists,
    setReading,
    setActiveId,
    setReset,
    setError,
    setStatus,
    markBuilt: signature.markBuilt,
  });
  useApplyDefaultOnce(actions, lists.effective, commands.apply);
  return {
    reading,
    ...model,
    using: usingText(model, baseline),
    word:
      store !== null &&
      (store.layouts.length > 0 ||
        lists.gameDefault !== null ||
        lists.allDefault !== null)
        ? stateWord(active, baseline)
        : null,
    error,
    status,
    ...commands,
  };
}

function useApplyDefaultOnce(
  actions: LayoutActions | null,
  effective: EffectiveDefault | null,
  apply: (layout: SavedLayout) => boolean,
): void {
  const applied = useRef(false);
  useEffect(() => {
    if (actions === null || applied.current) return;
    applied.current = true;
    if (effective !== null) apply(effective.layout);
  }, [actions, effective, apply]);
}
