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

// aria-describedby text: ONE hidden node per card scene holds every
// description (policy.describe). Cards reference it by id, so the card
// subtree stays purely visual: a drag preview or a pinned free node that
// copies a card never duplicates an id, and no description text sits inside
// a button's content.
//
// <Table> computes its descriptions in render (the a11y model), names each
// one with describedById and publishes them after commit; the scene's
// <Descriptions> host renders them.
import {createElement, useSyncExternalStore, type JSX} from 'react';

type Listener = () => void;

/** The descriptions of a card scene, by DOM id, as its tables publish them. */
export interface DescriptionStore {
  /** Replace one table's descriptions: DOM id -> text. */
  publish: (tableId: string, entries: ReadonlyMap<string, string>) => void;
  /** Drop a table's descriptions (unmount). */
  remove: (tableId: string) => void;
  /** Every description, in table publish order. */
  snapshot: () => ReadonlyArray<readonly [string, string]>;
  subscribe: (l: Listener) => () => void;
  /** The DOM id prefix unique to this scene. */
  readonly prefix: string;
}

let sceneCounter = 0;
const stores = new WeakMap<object, DescriptionStore>();

function createStore(): DescriptionStore {
  const tables = new Map<string, ReadonlyMap<string, string>>();
  const listeners = new Set<Listener>();
  let snap: ReadonlyArray<readonly [string, string]> = [];
  const rebuild = () => {
    const out: Array<readonly [string, string]> = [];
    for (const m of tables.values()) for (const e of m) out.push(e);
    snap = out;
    for (const l of [...listeners]) l();
  };
  const same = (
    a: ReadonlyMap<string, string> | undefined,
    b: ReadonlyMap<string, string>,
  ) => {
    if (a?.size !== b.size) return false;
    for (const [k, v] of b) if (a.get(k) !== v) return false;
    return true;
  };
  return {
    prefix: `pcd${(sceneCounter++).toString(36)}`,
    publish: (tableId, entries) => {
      if (same(tables.get(tableId), entries)) return;
      if (entries.size === 0) {
        if (!tables.has(tableId)) return;
        tables.delete(tableId);
      } else {
        tables.set(tableId, new Map(entries));
      }
      rebuild();
    },
    remove: tableId => {
      if (tables.delete(tableId)) rebuild();
    },
    snapshot: () => snap,
    subscribe: l => {
      listeners.add(l);
      return () => {
        listeners.delete(l);
      };
    },
  };
}

/** The description store of a scene (keyed by its runtime object). */
export function descriptionStore(scene: object): DescriptionStore {
  let s = stores.get(scene);
  if (!s) {
    s = createStore();
    stores.set(scene, s);
  }
  return s;
}

/** An IDREF-safe token for any string (ids may hold spaces, colons, quotes). */
function token(s: string): string {
  return s.replace(
    /[^A-Za-z0-9_]/g,
    c => `.${(c.codePointAt(0) ?? 0).toString(16)}.`,
  );
}

/** The DOM id of a card's description within its scene. Injective in (tableId, id). */
export function describedById(
  store: Pick<DescriptionStore, 'prefix'>,
  tableId: string,
  id: string,
): string {
  return `${store.prefix}-${token(tableId)}--${token(id)}`;
}

const HIDDEN = {display: 'none'} as const;

/** The scene's one hidden description node. Rendered by <CardScene> and by an
 *  implicit (lone) <Table>, beside the live region. */
export function Descriptions({scene}: {scene: object}): JSX.Element {
  const store = descriptionStore(scene);
  const entries = useSyncExternalStore(
    store.subscribe,
    store.snapshot,
    store.snapshot,
  );
  return createElement(
    'div',
    {'data-card-descriptions': '', hidden: true, style: HIDDEN},
    entries.map(([id, text]) => createElement('span', {key: id, id}, text)),
  );
}
