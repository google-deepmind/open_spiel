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

import type {JSX} from 'react';

import {NoFrame} from '../actions/note.tsx';
import type {Frame} from '../contract/contract.ts';
import type {Session} from '../session/use_session.ts';
import {boardText, missingTextReason} from './board_text.ts';

import './board.css';

/**
 * The gameplay board, the same for every game: the game's text for the
 * viewer, verbatim, in a card that fills the panel, and nothing else. The
 * viewer moves from Actions › Legal; whose turn it is, and the returns, are
 * on the Gameplay panel's status line; the text is copied from State.
 */
export function GameplayBoard({session}: {session: Session}): JSX.Element {
  const {frame} = session;
  if (frame === null) return <NoFrame />;
  return (
    <div className="board">
      <BoardTextCard frame={frame} />
    </div>
  );
}

/**
 * The text, kept as the game wrote it. It takes the focus so that the
 * keyboard can scroll it, and is named by what it is: the state, or the
 * viewer's observation.
 */
function BoardTextCard({frame}: {frame: Frame}): JSX.Element {
  const shown = boardText(frame);
  if (shown === null) {
    return <p className="board__missing">{missingTextReason(frame)}</p>;
  }
  return (
    <pre
      className="board__text"
      role="region"
      tabIndex={0}
      aria-label={shown.title}
    >
      {shown.text}
    </pre>
  );
}
