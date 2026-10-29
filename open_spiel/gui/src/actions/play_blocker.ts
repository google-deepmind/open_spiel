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

import type {Frame} from '../contract/contract.ts';
import {preparingMover, preparingText} from '../players/preparing.ts';

// The controller playing the seat to move, or null when no seat is.
function moverController({to_move: mover, seats}: Frame): string | null {
  if (mover.kind !== 'seat') return null;
  if (mover.seat === null) throw new Error('A seat is to move but unnamed.');
  const controller = seats[mover.seat]?.controller;
  if (controller === undefined) {
    throw new Error(`Seat ${String(mover.seat)} is to move but not seated.`);
  }
  return controller;
}

/** Why nobody plays at a terminal position. */
export const GAME_FINISHED = 'Game finished. No legal actions.';

/**
 * Why the viewer cannot play from the legal list at `frame`, as a sentence,
 * or null when it can. The server's rule: a seat acts only for itself and
 * only when a person plays it, the oracle acts for any such seat, and a
 * spectator never acts. An omniscient viewer may also play from an earlier
 * position, which takes back the later plies and resumes a paused session;
 * any other viewer plays only at the live position.
 *
 * @throws Error when the frame names a seat to move that it does not seat.
 */
export function playBlocker(frame: Frame): string | null {
  const {viewer, to_move: mover, session} = frame;
  const controller = moverController(frame);
  const live = frame.ply === frame.num_plies;
  if (!live && !frame.omniscient) {
    return 'Viewing an earlier position. Go to the live position to play.';
  }
  if (mover.kind === 'terminal') return GAME_FINISHED;
  const failed = frame.seats.find(seat => seat.failure !== null);
  if (live && failed !== undefined) {
    return `Paused: the controller of seat ${String(failed.seat)} failed.`;
  }
  if (controller === null) return 'Chance to move.';
  const preparing = preparingMover(frame);
  if (preparing !== null) {
    return preparingText(session.controllers, preparing);
  }
  if (viewer.kind === 'spectator') return 'Spectating. Claim a seat to act.';
  // The server lists the human controller first.
  if (controller !== session.controllers[0]?.name) {
    const label =
      session.controllers.find(known => known.name === controller)?.label ??
      controller;
    return viewer.seat === mover.seat
      ? `Seat controlled by ${label}.`
      : 'Waiting for opponent…';
  }
  if (viewer.kind === 'seat' && viewer.seat !== mover.seat) {
    return 'Waiting for your turn…';
  }
  return null;
}
