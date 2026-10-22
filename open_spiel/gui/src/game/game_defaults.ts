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

import type {
  CatalogueEntry,
  CreateSession,
  Viewer,
} from '../contract/contract.ts';
import {viewerParameter} from '../session/client.ts';
import {HUMAN_SEAT} from '../setup/parameters.ts';

// The parameters a game page opens a game with, where they are not the
// game's own defaults: Go opens on the 9×9 board.
const OPENING_PARAMETERS: Readonly<
  Record<string, CreateSession['parameters']>
> = {
  go: {board_size: 9},
};

/**
 * The request that opens a game of `entry` with its defaults: every seat
 * played by a person, and chance sampled by the server. A few games open
 * with parameters of their own, such as Go on its smallest board.
 *
 * @throws Error when the server cannot serve the game.
 */
export function defaultSession(entry: CatalogueEntry): CreateSession {
  if (entry.num_players === null) {
    throw new Error(`${entry.name} is not loadable, so it has no seats.`);
  }
  return {
    game: entry.name,
    parameters: {...OPENING_PARAMETERS[entry.name]},
    seats: Array.from({length: entry.num_players}, () => HUMAN_SEAT),
    chance_policy: 'auto',
    seed: null,
  };
}

/**
 * Who a game page shows the game to until told otherwise. A game of perfect
 * information hides nothing, so the oracle, who may act for every human
 * seat, sees what any seat sees; any other game opens as the first seat.
 */
export function defaultViewer(entry: CatalogueEntry): Viewer {
  return entry.information === 'perfect_information'
    ? {kind: 'oracle', seat: null}
    : {kind: 'seat', seat: 0};
}

/**
 * Who a game page opens for: `asked`, the viewer its address names, when the
 * game has that viewer, and otherwise `defaultViewer`.
 */
export function initialViewer(
  entry: CatalogueEntry,
  asked: Viewer | null,
): Viewer {
  if (asked === null) return defaultViewer(entry);
  if (asked.seat !== null && asked.seat >= (entry.num_players ?? 0)) {
    return defaultViewer(entry);
  }
  return asked;
}

/** A viewer a game page offers, with its name for people. */
export interface ViewerOption {
  /** The viewer's `viewer` query parameter. */
  readonly value: string;
  readonly label: string;
  readonly viewer: Viewer;
}

/**
 * The perspectives of a game of `numSeats` seats: each seat, then the oracle,
 * who sees everything, and a spectator, who sees only what is public.
 */
export function viewerOptions(numSeats: number): ViewerOption[] {
  const viewers: Viewer[] = [
    ...Array.from({length: numSeats}, (_, seat): Viewer => ({
      kind: 'seat',
      seat,
    })),
    {kind: 'oracle', seat: null},
    {kind: 'spectator', seat: null},
  ];
  return viewers.map(viewer => ({
    value: viewerParameter(viewer),
    label: viewerLabel(viewer),
    viewer,
  }));
}

/** `viewer`'s name for people. */
export function viewerLabel(viewer: Viewer): string {
  switch (viewer.kind) {
    case 'oracle':
      return 'Oracle (sees everything)';
    case 'spectator':
      return 'Spectator';
    case 'seat':
      return `Seat ${viewerParameter(viewer)}`;
    default: {
      const unreachable: never = viewer.kind;
      throw new Error(`Unhandled viewer kind: ${String(unreachable)}`);
    }
  }
}
