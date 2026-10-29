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

// The gin rummy Gameplay panel's content: the board for the session the
// renderer provides (gin_session.ts), with the notices over it. Its parts:
// gin_gameplay_view.ts (whose view), gin_gameplay_turn.ts (whose turn),
// gin_gameplay_moves.ts (the moves and the laydown), gin_gameplay_cards.ts
// (what each grid draws).

import {type JSX, useCallback, useState} from 'react';

import {
  composedDeadwood,
  laydownPrompt,
  layoffDropOf,
  opponentShown,
  useBoardCardStates,
  useHandOutcome,
  useNormalDiscard,
} from './gin_gameplay_cards.ts';
import type {BoardHandResult} from './gin_board_parts.ts';
import {
  useComposeCard,
  useGinCompose,
  useGinDone,
  useGinMoves,
  type GinCompose,
  type GinDone,
  type GinMoves,
} from './gin_gameplay_moves.ts';
import {
  ginStatusText,
  ginTurnOf,
  useMoveCue,
  type GinTurn,
} from './gin_gameplay_turn.ts';
import {useGinView, useSeatOptions, type GinView} from './gin_gameplay_view.ts';
import type {GinBoardCue} from './gin_move_cues.ts';
import {
  GinRummyBoardSurface,
  type GinRummyBoardSurfaceProps,
} from './gin_rummy_board_surface.tsx';
import type {GinCardStatesResult} from './hooks/use_gin_card_states.ts';
import {useGinSession} from './gin_session.ts';
import {
  useGinRummyDerivedState,
  type GinRummyDerivedState,
} from './use_gin_rummy_derived_state.ts';

import './gin_rummy.css';

/** A notice over the board: an error the viewer should see. */
function GinNotice({children}: {readonly children: string}): JSX.Element {
  return (
    <div className="gin-notice" role="alert">
      {children}
    </div>
  );
}

/** Pass and Knock: each one's handler, and whether it is disabled. */
export function passKnock(
  d: GinRummyDerivedState,
  moves: GinMoves,
  turn: GinTurn,
  isBusy: boolean,
  isAnalysisMode: boolean,
): {
  onPass: (() => void) | undefined;
  onKnock: (() => void) | undefined;
  passDisabled: boolean;
  knockDisabled: boolean;
} {
  const passEligible =
    isAnalysisMode ||
    turn.viewerIsActiveSeat ||
    d.normalizedPhase === 'firstupcard';
  const passDisabled = isBusy || moves.passHandler === null || !passEligible;
  const knockDisabled =
    isBusy || d.isTerminal || !moves.canAttemptKnock || d.knockAction === null;
  return {
    onPass: passDisabled ? undefined : (moves.passHandler ?? undefined),
    onKnock: knockDisabled ? undefined : moves.handleKnock,
    passDisabled,
    knockDisabled,
  };
}

/** Everything the board's props are built from. */
interface BoardInputs {
  d: GinRummyDerivedState;
  view: GinView;
  turn: GinTurn;
  compose: GinCompose;
  moves: GinMoves;
  done: GinDone;
  showOpponentCards: boolean;
  states: GinCardStatesResult;
  handResult: BoardHandResult | null;
  statusText: string;
  buttons: ReturnType<typeof passKnock>;
  moveCue: GinBoardCue | null;
  sortTrigger: number;
  onSort: () => void;
}

function boardProps(b: BoardInputs): GinRummyBoardSurfaceProps {
  const {d, turn, compose, moves} = b;
  return {
    showOpponentCards: b.showOpponentCards,
    opponentCardStates: b.states.opponentCardStates,
    opponentHandCount: d.opponentHandCount,
    stockCount: d.stockCount,
    discardCard: d.discardCard,
    knockCard: d.knockCard,
    playerDeadwood: composedDeadwood(d, compose) ?? d.playerDeadwood,
    handResult: b.handResult,
    statusText: b.statusText,
    laydown: compose.laydownActive
      ? {stage: d.laydownStage, onDone: b.done.doneHandler}
      : null,
    onDrawStock: moves.drawStockHandler ?? undefined,
    onDrawDiscard: moves.drawUpcardHandler ?? undefined,
    ...b.buttons,
    atWall: d.normalizedPhase === 'wall',
    waitingOnBot: turn.waitingOnBot,
    currentAutoPlay: turn.currentAutoPlay,
    interactionBlockReason: turn.interactionBlockReason,
    canInteract: turn.canInteract,
    playerCardStates: b.states.playerCardStates,
    playerHand: d.playerHand,
    // A spectator's table masks BOTH hands mid-hand (the server sends no
    // seat's cards to a seatless viewer).
    playerHandHidden:
      d.playerHand.length > 0 && d.playerHand.every(card => card === 'XX'),
    playerMelds: d.playerMelds,
    playerBestMeldIds: d.playerBestMeldIds,
    playerGhostCards: compose.composeOnOpponentGrid
      ? [...compose.stagedLayoffCards]
      : undefined,
    layoffDrop: layoffDropOf(d, compose),
    sortTrigger: b.sortTrigger,
    onSort: b.onSort,
    moveCue: b.moveCue,
    discardPile: d.discardPile,
  };
}

/** The gin board for the session of the renderer it is in. */
export function GinRummyGameplay(): JSX.Element {
  const s = useGinSession();
  const view = useGinView(s);
  const [sortTrigger, setSortTrigger] = useState(0);
  const handleSort = useCallback(() => {
    setSortTrigger(prev => prev + 1);
  }, []);
  const d = useGinRummyDerivedState(s.session, {
    preferObservationStruct: view.preferObservationStruct,
    viewerSeat: view.effectiveViewSeat,
  });
  useSeatOptions(s.session.players, d, view);
  const turn = ginTurnOf(s, d, view);
  const moveCue = useMoveCue(s, d, view, turn);
  const compose = useGinCompose(s, d, view, turn);
  const moves = useGinMoves(s, d, view, turn);
  const done = useGinDone(d, compose);
  const shown = opponentShown(d, view, turn);
  const normalDiscard = useNormalDiscard(d, moves);
  const composeCard = useComposeCard(d, compose);
  const states = useBoardCardStates(d, view, shown, compose, {
    normalDiscard,
    composeCard,
    canInteract: turn.canInteract,
  });
  const outcome = useHandOutcome(s, d, view);
  const inputs: BoardInputs = {
    d,
    view,
    turn,
    compose,
    moves,
    done,
    showOpponentCards: shown.showOpponentCards,
    states,
    handResult: outcome.value,
    statusText: laydownPrompt(d, compose, done) ?? ginStatusText(d, view, turn),
    buttons: passKnock(d, moves, turn, s.isBusy, view.isAnalysisMode),
    moveCue,
    sortTrigger,
    onSort: handleSort,
  };
  const submitError = compose.composition.submitError;
  return (
    <div className="ogs-panel ogs-panel--static gin-gameplay">
      <div className="gin-notices">
        {s.error !== null && <GinNotice>{s.error}</GinNotice>}
        {outcome.error !== null && (
          <GinNotice>{`The hand's result could not be read: ${outcome.error}`}</GinNotice>
        )}
        {submitError !== null && <GinNotice>{submitError}</GinNotice>}
      </div>
      {d.structRecord === null ? (
        <div className="ogs-status">
          Structured view unavailable for this state.
        </div>
      ) : (
        <GinRummyBoardSurface {...boardProps(inputs)} />
      )}
    </div>
  );
}
