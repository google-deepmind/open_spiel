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

import type {DockviewApi, IDockviewPanel} from 'dockview-core';
import {useEffect, useState} from 'react';

import type {Panels} from './panels.ts';

type Subscription = ReturnType<DockviewApi['onDidAddPanel']>;

/** The panels in the dock, and those of them on screen. */
export interface ShownPanels {
  readonly open: ReadonlySet<string>;
  /** The open panels whose content shows: each group's active panel. */
  readonly visible: ReadonlySet<string>;
}

/** The dock's `ShownPanels`, kept current; every panel until it is built. */
export function useShownPanels(
  api: DockviewApi | null,
  panels: Panels,
): ShownPanels {
  const [shown, setShown] = useState<ShownPanels>(() => {
    const all = new Set(panels.list.map(({id}) => id));
    return {open: all, visible: all};
  });
  useEffect(() => {
    if (api === null) {
      return undefined;
    }
    const sync = (): void => {
      const next = {
        open: openIds(api),
        visible: new Set(
          api.panels.filter(panel => panel.api.isVisible).map(({id}) => id),
        ),
      };
      setShown(previous =>
        sameIds(previous.open, next.open) &&
        sameIds(previous.visible, next.visible)
          ? previous
          : next,
      );
    };
    const watched = new Map<string, Subscription>();
    const watch = (panel: IDockviewPanel): void => {
      watched.set(panel.id, panel.api.onDidVisibilityChange(sync));
    };
    api.panels.forEach(watch);
    sync();
    const subscriptions = [
      api.onDidAddPanel(panel => {
        watch(panel);
        sync();
      }),
      api.onDidRemovePanel(({id}) => {
        watched.get(id)?.dispose();
        watched.delete(id);
        sync();
      }),
    ];
    return () => {
      for (const subscription of [...subscriptions, ...watched.values()]) {
        subscription.dispose();
      }
    };
  }, [api]);
  return shown;
}

function sameIds(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  return a.size === b.size && [...a].every(id => b.has(id));
}

/** The ids of the panels in the dock. */
export function openIds(api: DockviewApi): Set<string> {
  return new Set(api.panels.map(({id}) => id));
}
