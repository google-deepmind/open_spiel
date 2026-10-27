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

import {createElement, type JSX} from 'react';

import {NoFrame} from '../actions/note.tsx';
import {playBlocker} from '../actions/play_blocker.ts';
import type {Frame, Presentation} from '../contract/contract.ts';
import {RENDERERS} from '../renderers/registry.ts';
import type {Renderers} from '../renderers/renderer.ts';
import {RendererBoundary} from '../renderers/renderer_boundary.tsx';
import type {Session} from '../session/use_session.ts';
import {boardText, missingTextReason} from './board_text.ts';

import './board.css';

/**
 * The gameplay board. When the frame carries a presentation whose renderer
 * the client has, that renderer draws it, and the viewer may also move on
 * it; otherwise the board is the game's text for the viewer, verbatim, in a
 * card that fills the panel. The viewer always moves from Actions › Legal
 * too; whose turn it is, and the returns, are on the Gameplay panel's status
 * line; the text is copied from State.
 */
export function GameplayBoard({
  session,
  renderers = RENDERERS,
}: {
  readonly session: Session;
  readonly renderers?: Renderers;
}): JSX.Element {
  const {frame} = session;
  if (frame === null) return <NoFrame />;
  const {presentation} = frame;
  const Renderer =
    presentation === null ? null : renderers.get(presentation.renderer);
  if (presentation === null || Renderer === null) {
    return (
      <div className="board">
        <BoardTextCard frame={frame} />
      </div>
    );
  }
  // A renderer is drawn afresh at each ply unless it keeps its state, in
  // which case only a new session or renderer draws it afresh, and a failure
  // is cleared when the position changes.
  const {keepsState = false} = renderers.options(presentation.renderer);
  const position = `${frame.session.id}/${String(frame.ply)}`;
  return (
    <RendererBoundary
      key={
        keepsState
          ? `${presentation.renderer}/${frame.session.id}`
          : `${presentation.renderer}/${position}`
      }
      resetKey={position}
      fallback={
        <div className="board">
          <BoardTextCard frame={frame} />
        </div>
      }
    >
      {
        // The registry's renderers are module-level components, so each
        // keeps its identity across renders.
        createElement(Renderer, {
          frame,
          presentation,
          play: playFor(frame, presentation, session),
          playSequence: playSequenceFor(frame, presentation, session),
          busy: session.busy,
        })
      }
    </RendererBoundary>
  );
}

// How the viewer plays on the renderer, or null when it cannot. A press
// while a request is in flight is ignored, as the legal list ignores it.
function playFor(
  frame: Frame,
  presentation: Presentation,
  session: Session,
): ((action: number) => void) | null {
  if (presentation.actions === null || playBlocker(frame) !== null) return null;
  const {busy, commands} = session;
  return action => {
    if (!busy) commands.act(action);
  };
}

// How the viewer plays several actions as one move, or null when it cannot.
function playSequenceFor(
  frame: Frame,
  presentation: Presentation,
  session: Session,
): ((actions: readonly number[]) => Promise<Frame>) | null {
  if (presentation.actions === null || playBlocker(frame) !== null) return null;
  return actions => session.commands.actSequence(actions);
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
