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

// What the Gameplay panel says and offers at a frame. Each is a hint; the
// server decides every request on its own rules.

import {returnsSentence} from '../board/board_text.ts';
import type {Frame} from '../contract/contract.ts';
import {preparingMover, preparingText} from '../players/preparing.ts';

// The seat to move at the live position of a game in play, or null.
function liveMover(frame: Frame): number | null {
  const {to_move: mover} = frame;
  if (frame.ply < frame.num_plies || frame.status !== 'playing') return null;
  return mover.kind === 'seat' ? mover.seat : null;
}

// Whether a person plays `seat`. The server lists the human controller first.
function isHuman(frame: Frame, seat: number): boolean {
  const controller = frame.seats[seat]?.controller;
  if (controller === undefined) {
    throw new Error(`Seat ${String(seat)} is to move but not seated.`);
  }
  return controller === frame.session.controllers[0]?.name;
}

/**
 * The seat a controller plays that is to move at the live position, which an
 * advance lets play, or null when there is none. A seat whose controller is
 * still being built is not one: it plays once it is ready.
 */
export function controllerToMove(frame: Frame): number | null {
  const seat = liveMover(frame);
  if (seat === null || isHuman(frame, seat)) return null;
  return preparingMover(frame) === null ? seat : null;
}

// Who acts at the position shown, as the status line names them.
function moverName({to_move: mover}: Frame): string {
  if (mover.kind === 'seat') return `Player ${String(mover.seat)}`;
  return mover.kind === 'chance' ? 'Chance' : 'Terminal';
}

/**
 * The Gameplay panel's status line for `frame`: the turn, who is to move,
 * and then, as they apply, that a controller is to play (an autoplaying one
 * is to move only where autoplay stopped, so it too waits for a step) or is
 * still being built ("Preparing MCTS…"), that the session is paused, that the game is simultaneous, the returns at the
 * end, and who the viewer is. "Turn 3 · Player 1 · Waiting for bot step… ·
 * Seat: Player 0".
 */
export function statusLine(frame: Frame): string {
  const parts = [`Turn ${String(frame.ply)}`, moverName(frame)];
  const controller = controllerToMove(frame);
  if (controller !== null) parts.push('Waiting for bot step…');
  const preparing = preparingMover(frame);
  if (preparing !== null) {
    parts.push(preparingText(frame.session.controllers, preparing));
  }
  if (frame.status === 'paused' && frame.ply === frame.num_plies) {
    parts.push('Paused');
  }
  if (frame.session.turn_based && frame.returns === null) {
    parts.push('Simultaneous, one seat at a time');
  }
  if (frame.returns !== null) parts.push(returnsSentence(frame.returns));
  const {viewer} = frame;
  if (viewer.kind === 'spectator') parts.push('Spectating');
  if (viewer.kind === 'seat') parts.push(`Seat: Player ${String(viewer.seat)}`);
  return parts.join(' · ');
}
