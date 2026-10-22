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
import {type Game, useGame} from './game_context.ts';
import {seatToHandOver, statusLine} from './gameplay_state.ts';

import './game.css';

/**
 * The Gameplay panel: one status line, then the board, and nothing else.
 * Moving is done in Actions; stepping through the positions, in Actions ›
 * History. The session's error, and the offer to hand the device over when
 * another person's seat is to move in a game that hides information, are
 * notices over the top of the board, which take no room: showing one never
 * moves the board.
 */
export function GameplayPanel(): JSX.Element {
  const game = useGame();
  const {frame, error} = game.session;
  return (
    <div className="gameplay">
      <div className="gameplay__notices">
        {error !== null && <ErrorNotice message={error.message} />}
        {frame !== null && <HandOverNotice game={game} frame={frame} />}
      </div>
      {frame !== null && (
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

function HandOverNotice({
  game,
  frame,
}: {
  readonly game: Game;
  readonly frame: Frame;
}): JSX.Element | null {
  const seat = seatToHandOver(frame);
  if (seat === null) return null;
  const name = `Player ${String(seat)}`;
  const text = `${name} is to move, and plays from their own view.`;
  return (
    <div
      className="gameplay__notice"
      data-tone="turn"
      role="status"
      title={text}
    >
      <span className="gameplay__notice-text">{text}</span>
      <button
        type="button"
        className="gameplay__notice-action"
        onClick={() => {
          game.handOver(seat);
        }}
      >
        {`Hand over to ${name}`}
      </button>
    </div>
  );
}
