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

// Where each panel goes, as plain data: the dock builder hands these options
// to dockview unchanged, so every layout rule lives, and is tested, here.

import type {AddPanelOptions} from 'dockview-core';

import type {Direction, PanelDefinition, Panels} from './panels.ts';

/**
 * How the dock is laid out at its width: the panels' own columns, those
 * columns folded into two, or every panel a tab of one group.
 */
export type Tier = 'three' | 'two' | 'tabs';

/** Dock width, in pixels, from which the panels get their own columns. */
export const THREE_COLUMNS_FROM = 1600;

/** Dock width, in pixels, from which the panels are laid out in columns. */
export const TWO_COLUMNS_FROM = 900;

/** The least width a column after the first is built with, in pixels. */
export const SIDE_COLUMN_WIDTH = 340;

/** Where a panel is added: beside a panel in the dock, or at its left edge. */
export type Placement =
  | {
      readonly referencePanel: string;
      readonly direction: Direction;
      readonly index?: number;
    }
  | {readonly direction: 'left'};

/** The dock's size in pixels; 0 where it is not laid out yet. */
export interface Size {
  readonly width: number;
  readonly height: number;
}

/**
 * How a dock fits its size: its tier, and the factors its minimum widths and
 * heights are scaled by (see `fitScale`).
 */
export interface Fit {
  readonly tier: Tier;
  readonly widthScale: number;
  readonly heightScale: number;
}

/**
 * A dock built from scratch. A dock not laid out yet is given no group sizes
 * across that direction: no `columns` at width 0, no `stacks` at height 0.
 */
export interface Arrangement extends Fit {
  /** The panels to add, in order. */
  readonly panels: readonly AddPanelOptions[];
  /** Left to right, the first panel of each column and the column's width. */
  readonly columns: ReadonlyArray<{
    readonly panel: string;
    readonly width: number;
  }>;
  /** Top to bottom, the first panel of each group of each stacked column. */
  readonly stacks: ReadonlyArray<readonly string[]>;
}

/** The tier of a dock `width` pixels wide; an unmeasured (0) dock is wide. */
export function tierFor(width: number): Tier {
  if (!(width > 0) || width >= THREE_COLUMNS_FROM) {
    return 'three';
  }
  return width >= TWO_COLUMNS_FROM ? 'two' : 'tabs';
}

/** `panel`'s declared position, as the tier lays it out. */
function positionAt(
  panels: Panels,
  panel: PanelDefinition,
  tier: Tier,
): PanelDefinition['position'] {
  const {position} = panel;
  if (position === undefined) {
    return undefined;
  }
  if (tier === 'tabs') {
    return {referencePanel: panels.primary.id, direction: 'within'};
  }
  if (
    tier === 'two' &&
    position.direction === 'right' &&
    position.referencePanel !== panels.primary.id
  ) {
    return {...position, direction: 'below'};
  }
  return position;
}

/**
 * Where to add panel `id` to a dock holding `open`; undefined for the first
 * panel of an empty dock. The primary panel returns to the left edge, or in
 * the tabbed tier to the first tab. A panel whose place another panel took
 * while it was closed joins that panel as a tab. A panel whose reference
 * panel is closed goes where that panel would: beside the panel that took its
 * place, if one did, or else at its position, and so on up its references.
 */
export function placementOf(
  panels: Panels,
  id: string,
  tier: Tier,
  open: ReadonlySet<string>,
): Placement | undefined {
  if (open.size === 0) {
    return undefined;
  }
  const standIns = standInsFor(panels, tier, open);
  let position = positionAt(panels, panels.get(id), tier);
  if (position === undefined) {
    return tier === 'tabs'
      ? {
          referencePanel: standInFor(standIns, id),
          direction: 'within',
          index: 0,
        }
      : {direction: 'left'};
  }
  const own = standIns.get(id);
  if (own !== undefined) {
    return {referencePanel: own, direction: 'within'};
  }
  let {referencePanel, direction} = position;
  while (!open.has(referencePanel)) {
    const standIn = standIns.get(referencePanel);
    if (standIn !== undefined) {
      return {referencePanel: standIn, direction};
    }
    position = positionAt(panels, panels.get(referencePanel), tier);
    if (position === undefined) {
      throw missingStandIn(referencePanel);
    }
    ({referencePanel, direction} = position);
  }
  return {referencePanel, direction};
}

/**
 * For each closed panel that an open one took the place of, that open panel:
 * the earliest defined whose references pass through it. Every closed panel
 * a dock is built around, the primary panel included, has one.
 */
function standInsFor(
  panels: Panels,
  tier: Tier,
  open: ReadonlySet<string>,
): Map<string, string> {
  const standIns = new Map<string, string>();
  for (const panel of panels.list.filter(({id}) => open.has(id))) {
    let position = positionAt(panels, panel, tier);
    while (
      position !== undefined &&
      !open.has(position.referencePanel) &&
      !standIns.has(position.referencePanel)
    ) {
      standIns.set(position.referencePanel, panel.id);
      position = positionAt(panels, panels.get(position.referencePanel), tier);
    }
  }
  return standIns;
}

function standInFor(standIns: ReadonlyMap<string, string>, id: string): string {
  const standIn = standIns.get(id);
  if (standIn === undefined) {
    throw missingStandIn(id);
  }
  return standIn;
}

function missingStandIn(id: string): Error {
  return new Error(`No open panel stands in for the closed panel "${id}".`);
}

/**
 * The factor that scales `minimums`, laid end to end, to fit `length`
 * pixels: 1 when they already fit, or when `length` is not measured (0).
 */
export function fitScale(minimums: readonly number[], length: number): number {
  const total = sum(minimums);
  return length > 0 && total > length ? length / total : 1;
}

/**
 * The options that add `panel` at `position` to a dock that fits as `fit`.
 * Its minimum sizes are scaled by `fit`; in the tabbed tier it has none,
 * since the one group must fit any size.
 */
export function panelOptions(
  panel: PanelDefinition,
  position: Placement | undefined,
  fit: Fit,
): AddPanelOptions {
  return {
    id: panel.id,
    component: panel.id,
    title: panel.title,
    ...(position && {position}),
    ...(fit.tier !== 'tabs' && {
      minimumWidth: Math.floor(panel.minimumWidth * fit.widthScale),
      minimumHeight: Math.floor(panel.minimumHeight * fit.heightScale),
    }),
  };
}

/** The panels of a group of tabs, the one it was opened with first. */
type Group = [PanelDefinition, ...PanelDefinition[]];

interface Column {
  /** The panel it was opened with. */
  readonly head: string;
  /** Its groups, top first. */
  readonly groups: Group[];
}

/**
 * The dock for the `open` panels at `size`. Panels are added in definition
 * order, so each one's reference is placed before it, and the primary panel,
 * which every other panel is positioned by, comes first. A
 * panel that joins a group as a tab is added behind the group's first panel.
 * Minimum widths are scaled so the columns fit the width, and minimum heights
 * so the groups of every column fit the height. `columnWidths` sizes the
 * columns.
 */
export function arrange(
  panels: Panels,
  open: ReadonlySet<string>,
  {width, height}: Size,
): Arrangement {
  const tier = tierFor(width);
  const columnOf = new Map<string, Column>();
  const groupOf = new Map<string, Group>();
  const columns: Column[] = [];
  const placements: Array<{panel: PanelDefinition; position?: Placement}> = [];
  for (const panel of panels.list.filter(({id}) => open.has(id))) {
    const placed = new Set(columnOf.keys());
    const position = placementOf(panels, panel.id, tier, placed);
    const column = columnFor(panel.id, position, columnOf, columns);
    let group =
      position?.direction === 'within'
        ? groupOf.get(position.referencePanel)
        : undefined;
    if (group === undefined) {
      group = [panel];
      column.groups.push(group);
    } else {
      group.push(panel);
    }
    columnOf.set(panel.id, column);
    groupOf.set(panel.id, group);
    placements.push(position === undefined ? {panel} : {panel, position});
  }
  const fit = {
    tier,
    widthScale: fitScale(columns.map(widestMinimum), width),
    heightScale: Math.min(
      1,
      ...columns.map(({groups}) =>
        fitScale(groups.map(tallestMinimum), height),
      ),
    ),
  };
  return {
    ...fit,
    panels: placements.map(({panel, position}) => ({
      ...panelOptions(panel, position, fit),
      inactive: position?.direction === 'within',
    })),
    columns: width > 0 ? columnWidths(columns, width, fit.widthScale) : [],
    stacks:
      height > 0
        ? columns
            .filter(({groups}) => groups.length > 1)
            .map(({groups}) => groups.map(([first]) => first.id))
        : [],
  };
}

/**
 * The column panel `id` goes in at `position`: its reference's, or a new one
 * after it in `columns` when it opens to the right. Dockview splits only the
 * reference's group, so a column whose groups are stacked cannot take a
 * neighbour; that is refused.
 */
function columnFor(
  id: string,
  position: Placement | undefined,
  columnOf: ReadonlyMap<string, Column>,
  columns: Column[],
): Column {
  const reference =
    position !== undefined && 'referencePanel' in position
      ? columnOf.get(position.referencePanel)
      : undefined;
  if (reference !== undefined && position?.direction !== 'right') {
    return reference;
  }
  const column: Column = {head: id, groups: []};
  if (reference === undefined) {
    columns.push(column);
  } else if (reference.groups.length > 1) {
    throw new Error(
      `Panel "${id}" opens beside a panel whose column has stacked groups.`,
    );
  } else {
    columns.splice(columns.indexOf(reference) + 1, 0, column);
  }
  return column;
}

function widestMinimum({groups}: Column): number {
  return Math.max(...groups.flat().map(({minimumWidth}) => minimumWidth));
}

function tallestMinimum(group: Group): number {
  return Math.max(...group.map(({minimumHeight}) => minimumHeight));
}

/**
 * The width of each column, all scaled by `scale`. A column after the first
 * asks for the widest initial width of its panels, and at least
 * SIDE_COLUMN_WIDTH. The first takes the rest, and never less than its
 * minimum: where the others ask for more, each gives up the same fraction of
 * what it asks for above its own minimum.
 */
function columnWidths(
  columns: readonly Column[],
  width: number,
  scale: number,
): Arrangement['columns'] {
  const [first, ...rest] = columns;
  if (first === undefined) {
    return [];
  }
  const sides = rest.map(column => ({
    panel: column.head,
    minimum: Math.floor(scale * widestMinimum(column)),
    asked: Math.floor(
      scale *
        Math.max(
          SIDE_COLUMN_WIDTH,
          ...column.groups
            .flat()
            .map(member => member.initialWidth ?? member.minimumWidth),
        ),
    ),
  }));
  const spare =
    width -
    Math.floor(scale * widestMinimum(first)) -
    sum(sides.map(({minimum}) => minimum));
  const surplus = sum(sides.map(({minimum, asked}) => asked - minimum));
  const kept = surplus > spare ? spare / surplus : 1;
  const widths = sides.map(({panel, minimum, asked}) => ({
    panel,
    width: minimum + Math.floor(kept * (asked - minimum)),
  }));
  return [
    {panel: first.head, width: width - sum(widths.map(side => side.width))},
    ...widths,
  ];
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
