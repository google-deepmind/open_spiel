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
// game's own defaults: Go opens on the 9×9 board, and gin rummy on Oklahoma
// rules, where the first upcard sets the knock card.
const OPENING_PARAMETERS: Readonly<
  Record<string, CreateSession['parameters']>
> = {
  gin_rummy: {oklahoma: true},
  go: {board_size: 9},
};

/**
 * The request that opens a game of `entry` with its defaults: every seat
 * played by a person, and chance sampled by the server. A few games open
 * with parameters of their own, such as Go on its smallest board and gin
 * rummy on Oklahoma rules.
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
 * Who a game page shows the game to when its address names no one: the
 * oracle, so that the page opens in analysis. The oracle may act for every
 * human seat, and the seat the board is drawn from is chosen in View ›
 * Perspective.
 */
export function defaultViewer(): Viewer {
  return {kind: 'oracle', seat: null};
}

/**
 * Who a game page opens for: `defaultViewer` when its address names no one,
 * else `asked`. A seat the game does not have opens the spectator's view,
 * never the oracle's: an address meant to show one seat's hand must not
 * show every hand.
 */
export function initialViewer(
  entry: CatalogueEntry,
  asked: Viewer | null,
): Viewer {
  if (asked === null) return defaultViewer();
  if (asked.seat !== null && asked.seat >= (entry.num_players ?? 0)) {
    return {kind: 'spectator', seat: null};
  }
  return asked;
}

/**
 * The mode of a page shown to `viewer`, for people: analysis for the oracle,
 * a game at a seat or spectating for any other.
 */
export function modeLabel(viewer: Viewer): string {
  switch (viewer.kind) {
    case 'oracle':
      return 'Analysis';
    case 'spectator':
      return 'Game, spectating';
    case 'seat':
      return `Game, seat ${viewerParameter(viewer)}`;
    default: {
      const unreachable: never = viewer.kind;
      throw new Error(`Unhandled viewer kind: ${String(unreachable)}`);
    }
  }
}
