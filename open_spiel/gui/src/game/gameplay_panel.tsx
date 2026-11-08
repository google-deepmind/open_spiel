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

import {type JSX, useState} from 'react';

import {GameplayBoard} from '../board/board.tsx';
import type {Frame} from '../contract/contract.ts';
import {preparingMover, preparingText} from '../players/preparing.ts';
import {RENDERERS} from '../renderers/registry.ts';
import {useGame} from './game_context.ts';
import {statusLine} from './gameplay_state.ts';

import './game.css';

/**
 * The Gameplay panel: one status line, unless the board draws its own, then
 * the board, and nothing else.
 * Moving is done in Actions; stepping through the positions, in Actions ›
 * History. The session's error is a notice over the top of the board, which
 * takes no room: showing one never moves the board. So is the seat to move
 * preparing, on a board that draws its own status line.
 */
export function GameplayPanel(): JSX.Element {
  const game = useGame();
  const {frame, error} = game.session;
  return (
    <div className="gameplay">
      <div className="gameplay__notices">
        {error !== null && <ErrorNotice message={error.message} />}
        {frame !== null && drawsStatus(frame) && (
          <PreparingNotice frame={frame} />
        )}
      </div>
      {frame !== null && !drawsStatus(frame) && (
        <p className="gameplay__status" role="status">
          {statusLine(frame)}
        </p>
      )}
      <div className="gameplay__board">
        <GameplayBoard session={game.session} />
      </div>
    </div>
  );
}

// Whether the board draws its own turn and status, so the panel shows none.
function drawsStatus(frame: Frame): boolean {
  const {presentation} = frame;
  if (presentation === null) return false;
  if (RENDERERS.get(presentation.renderer) === null) return false;
  return RENDERERS.options(presentation.renderer).drawsStatus === true;
}

/**
 * The session's error, until dismissed. A different message shows again
 * after a dismissal, and so does the same one once it has gone and come
 * back.
 */
function ErrorNotice({message}: {message: string}): JSX.Element | null {
  const [dismissed, setDismissed] = useState<string | null>(null);
  if (dismissed === message) return null;
  return (
    <div
      className="gameplay__notice"
      data-tone="error"
      role="alert"
      title={message}
    >
      <span className="gameplay__notice-text">{message}</span>
      <button
        type="button"
        className="gameplay__notice-dismiss"
        aria-label="Dismiss"
        onClick={() => {
          setDismissed(message);
        }}
      >
        ×
      </button>
    </div>
  );
}

// The seat to move, still being built, over a board that draws its status.
function PreparingNotice({frame}: {readonly frame: Frame}): JSX.Element | null {
  const seat = preparingMover(frame);
  if (seat === null) return null;
  const text = preparingText(frame.session.controllers, seat);
  return (
    <div
      className="gameplay__notice"
      data-tone="turn"
      role="status"
      title={seat.message ?? text}
    >
      <span className="gameplay__notice-text">{text}</span>
    </div>
  );
}
