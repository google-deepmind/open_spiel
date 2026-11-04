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

/** Where a panel opens relative to its reference panel. */
export type Direction = 'right' | 'below' | 'within';

/** A dock panel: what it is called, where it opens and how small it may get. */
export interface PanelDefinition<Id extends string = string> {
  readonly id: Id;
  readonly title: string;
  /**
   * Where the panel opens: beside, below or as a tab of a panel defined
   * before it. Only the primary panel has none.
   */
  readonly position?: {
    readonly referencePanel: Id;
    readonly direction: Direction;
  };
  /** Pixels. */
  readonly minimumWidth: number;
  /** Pixels. */
  readonly minimumHeight: number;
  /**
   * Pixels; the width the panel asks for its column, `minimumWidth` if
   * unset. `arrange` decides what the column gets.
   */
  readonly initialWidth?: number;
  /** The panel the layout is built around. Exactly one panel is primary. */
  readonly primary?: true;
  /**
   * False for a panel the dock is built without: it is listed among the
   * closed panels until it is opened. Panels are open at first by default.
   */
  readonly initiallyOpen?: false;
}

/** A validated list of panel definitions. */
export interface Panels<Id extends string = string> {
  /** In definition order: every panel after the one it is positioned by. */
  readonly list: ReadonlyArray<PanelDefinition<Id>>;
  readonly primary: PanelDefinition<Id>;
  /** The ids of the panels the dock is built with. */
  readonly initial: ReadonlySet<Id>;
  /** The definition of `id`; throws if there is none. */
  get(id: Id): PanelDefinition<Id>;
}

/**
 * Validates `definitions` and returns them as `Panels`. Throws on a duplicate
 * id, on a blank title, on a reference to a panel not defined earlier, on
 * anything but exactly one primary panel, on a primary panel with a position
 * or another panel without one, on a primary panel that is not open at first,
 * and on a size that is not a positive number of pixels.
 */
export function definePanels<const Id extends string>(
  definitions: ReadonlyArray<PanelDefinition<Id>>,
): Panels<Id> {
  const byId = new Map<string, PanelDefinition<Id>>();
  for (const panel of definitions) {
    if (byId.has(panel.id)) {
      throw new Error(`Panel "${panel.id}" is defined twice.`);
    }
    if (panel.title.trim() === '') {
      throw new Error(`Panel "${panel.id}" has a blank title.`);
    }
    checkPosition(panel, byId);
    checkSizes(panel);
    byId.set(panel.id, panel);
  }
  const primaries = definitions.filter(panel => panel.primary);
  const [primary] = primaries;
  if (primary === undefined || primaries.length > 1) {
    throw new Error(
      `Exactly one panel must be primary, not ${String(primaries.length)}.`,
    );
  }
  if (primary.initiallyOpen === false) {
    throw new Error(`The primary panel "${primary.id}" is not open at first.`);
  }
  return {
    list: definitions,
    primary,
    initial: new Set(
      definitions
        .filter(panel => panel.initiallyOpen !== false)
        .map(({id}) => id),
    ),
    get(id: Id): PanelDefinition<Id> {
      const panel = byId.get(id);
      if (panel === undefined) {
        throw new Error(`There is no panel "${id}".`);
      }
      return panel;
    },
  };
}

function checkPosition(
  panel: PanelDefinition,
  earlier: ReadonlyMap<string, PanelDefinition>,
): void {
  if (panel.primary) {
    if (panel.position !== undefined) {
      throw new Error(`The primary panel "${panel.id}" has a position.`);
    }
    return;
  }
  if (panel.position === undefined) {
    throw new Error(`Panel "${panel.id}" has no position.`);
  }
  if (!earlier.has(panel.position.referencePanel)) {
    throw new Error(
      `Panel "${panel.id}" is positioned by "${panel.position.referencePanel}", ` +
        'which is not defined before it.',
    );
  }
}

function checkSizes(panel: PanelDefinition): void {
  const sizes = {
    minimumWidth: panel.minimumWidth,
    minimumHeight: panel.minimumHeight,
    initialWidth: panel.initialWidth ?? panel.minimumWidth,
  };
  for (const [name, size] of Object.entries(sizes)) {
    if (!(Number.isFinite(size) && size > 0)) {
      throw new Error(`Panel "${panel.id}" has ${name} ${String(size)}.`);
    }
  }
  if (sizes.initialWidth < panel.minimumWidth) {
    throw new Error(
      `Panel "${panel.id}" has an initialWidth below its minimumWidth.`,
    );
  }
}
