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

import {
  arrange,
  panelOptions,
  placementOf,
  type Fit,
  type Size,
} from './arrangement.ts';
import type {Panels} from './panels.ts';

/**
 * Empties the dock and builds it afresh for `open` at `size`, returning how
 * it fits. Dockview is first laid out at `size`, so panels take their sizes
 * out of the size the dock has now rather than the size it last reported.
 * Then each column is given its width, and the groups of each stacked column
 * equal heights. Only here are group sizes set, so a size the user drags is
 * kept until the dock is rebuilt.
 */
export function buildDock(
  api: DockviewApi,
  panels: Panels,
  open: ReadonlySet<string>,
  size: Size,
): Fit {
  if (size.width > 0 && size.height > 0) {
    api.layout(size.width, size.height, true);
  }
  api.clear();
  const {tier, widthScale, heightScale, ...arrangement} = arrange(
    panels,
    open,
    size,
  );
  for (const options of arrangement.panels) {
    api.addPanel(options);
  }
  resizeGroups(
    api,
    arrangement.columns.map(({panel, width}) => [panel, {width}]),
  );
  for (const stack of arrangement.stacks) {
    const height = Math.round(size.height / stack.length);
    resizeGroups(
      api,
      stack.map(panel => [panel, {height}]),
    );
  }
  return {tier, widthScale, heightScale};
}

/**
 * Sizes the groups of the given panels, which lie side by side or one above
 * another, in that order. Each resize trades with the groups after it, so
 * all but the last are set, first to last, and the last takes the rest.
 */
function resizeGroups(
  api: DockviewApi,
  sizes: ReadonlyArray<readonly [string, {width: number} | {height: number}]>,
): void {
  for (const [id, size] of sizes.slice(0, -1)) {
    panelIn(api, id).group.api.setSize(size);
  }
}

/**
 * Brings panel `id` forward in a dock that fits as `fit`, adding it first if
 * it is closed, and returns how the dock fits then. A closed panel is added
 * where `placementOf` puts it, with the dock's scales; if the panels then
 * open need either scale smaller to fit `size`, the dock is instead rebuilt
 * for them, so that no group is laid out past its edge.
 */
export function openPanel(
  api: DockviewApi,
  panels: Panels,
  id: string,
  fit: Fit,
  size: Size,
): Fit {
  const existing = api.getPanel(id);
  if (existing !== undefined) {
    existing.api.setActive();
    return fit;
  }
  const open = new Set(api.panels.map(panel => panel.id));
  const reopened = new Set([...open, id]);
  const refit = arrange(panels, reopened, size);
  if (
    fit.tier !== 'tabs' &&
    (refit.widthScale < fit.widthScale || refit.heightScale < fit.heightScale)
  ) {
    const rebuilt = buildDock(api, panels, reopened, size);
    panelIn(api, id).api.setActive();
    return rebuilt;
  }
  const position = placementOf(panels, id, fit.tier, open);
  api.addPanel(panelOptions(panels.get(id), position, fit));
  return fit;
}

function panelIn(api: DockviewApi, id: string): IDockviewPanel {
  const panel = api.getPanel(id);
  if (panel === undefined) {
    throw new Error(`Panel "${id}" is not in the dock.`);
  }
  return panel;
}
