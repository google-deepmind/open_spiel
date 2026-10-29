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

// A seat whose controller the server is still building: it does not move
// until it is ready, and the client fetches the frame again until then.

import type {Controller, Frame, SeatView} from '../contract/contract.ts';

/** How long the client waits between fetches while a seat prepares. */
export const PREPARING_POLL_MS = 1000;

/** "Preparing MCTS…": what a preparing seat shows in place of its name. */
export function preparingText(
  controllers: readonly Controller[],
  seat: SeatView,
): string {
  const label =
    controllers.find(known => known.name === seat.controller)?.label ??
    seat.controller;
  return `Preparing ${label}…`;
}

/** Whether the server is building any seat's controller. */
export function anyPreparing(frame: Frame): boolean {
  return frame.seats.some(seat => seat.status === 'preparing');
}

/**
 * The seat to move at the live position of a game in play whose controller
 * is still being built, so that play waits on it; null when there is none.
 */
export function preparingMover(frame: Frame): SeatView | null {
  const {to_move: mover} = frame;
  if (frame.ply < frame.num_plies || frame.status !== 'playing') return null;
  if (mover.kind !== 'seat' || mover.seat === null) return null;
  const seat = frame.seats[mover.seat];
  return seat?.status === 'preparing' ? seat : null;
}
