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
// The euchre Gameplay panel: the meta line and the table for the session
// the renderer provides (euchre_session.ts), with an error notice over it.
// Its turn, moves and staging are euchre_gameplay_turn.ts.

import {type JSX, useState} from 'react';

import {useBottomSeat} from '../../game/bottom_seat.ts';
import {useViewSetting} from '../shared/view_settings.ts';
import {useSessionTrickCollect} from '../shared/trick_hold.ts';
import {
  collectOf,
  phaseLabelOf,
  flyingCard,
  statusLine,
  useEuchreMoves,
  useEuchreStaged,
  useEuchreTurn,
  type EuchreTurn,
} from './euchre_gameplay_turn.ts';
import {GameplayNotice} from './euchre_notice.tsx';
import type {SessionContextValue} from './euchre_session.ts';
import {euchreStagedView, type EuchreView} from './held_trick.ts';
import {useEuchreState, type EuchreUIState} from './hooks/use_euchre_state.ts';
import {TableLayout} from './sections/index.ts';
import './euchre_generic.css';
import './euchre.css';
import './euchre_overlays.css';

type UseSession = () => SessionContextValue;

const NO_CARDS = new Set<string>();

/** The line over the table: the phase, trump, the tricks, a bot's turn. */
function EuchreMeta({
  euchreState,
  view,
  turn,
}: {
  euchreState: EuchreUIState;
  view: EuchreView;
  turn: EuchreTurn;
}): JSX.Element {
  return (
    <div className="ogs-state-meta">
      <span>Phase: {phaseLabelOf(euchreState)}</span>
      {euchreState.trumpSuit && (
        <span>Trump: {euchreState.trumpSuit.toUpperCase()}</span>
      )}
      <span>
        Tricks N-S: {view.tricksWon.NS} | E-W: {view.tricksWon.EW}
      </span>
      {turn.waitingOnBot && (
        <span>
          {turn.currentAutoPlay ? 'Bot thinking...' : 'Waiting for bot...'}
        </span>
      )}
    </div>
  );
}

/** The table for the session `useSession` gives. */
function EuchreContent({useSession}: {useSession: UseSession}): JSX.Element {
  const s = useSession();
  const {session, error, viewer, replay} = s;

  // Config settings
  const [showAllCards] = useViewSetting<boolean>('euchre.showAllCards', false);

  // The seat at the bottom: the one chosen in View › Perspective in
  // analysis, the viewer's own seat in a game.
  const {seat: viewerSeat} = useBottomSeat(session, viewer);

  // Transform session state to UI state
  const euchreState = useEuchreState(session, viewerSeat);
  // A finished trick stays on the table, then is collected toward its
  // winner, while the live state is accepted underneath.
  // `settling` (from the play staging) holds the hold back until
  // the trick's last card has landed.
  const [settling, setSettling] = useState(false);
  const trickCollect = useSessionTrickCollect(
    session,
    replay,
    euchreState?.lastTrick ?? null,
    settling,
  );
  const turn = useEuchreTurn(s);
  const moves = useEuchreMoves(
    s,
    euchreState,
    turn,
    trickCollect.pending.length > 0,
  );
  const staged = useEuchreStaged(euchreState, trickCollect, setSettling);
  const {heldView} = staged;

  if (!euchreState || !heldView) {
    return <div className="ogs-status">Loading game state...</div>;
  }

  const view = euchreStagedView(heldView, staged.shown);
  // The table is behind the game while staged plays are still landing:
  // the turn is the next card's player, and nobody plays until it lands.
  const stagedBehind = staged.shown < heldView.trickPile.length;
  const nextStaged = stagedBehind
    ? (heldView.trickPile[staged.shown] ?? null)
    : null;
  return (
    <>
      {error && <GameplayNotice tone="error">{error}</GameplayNotice>}
      <EuchreMeta euchreState={euchreState} view={view} turn={turn} />
      <div
        className="euchre-gameplay-wrapper"
        data-trick-phase={trickCollect.phase ?? 'none'}
      >
        <TableLayout
          players={view.players}
          trickPile={view.trickPile}
          phase={view.phase}
          upcard={euchreState.upcard}
          trumpSuit={euchreState.trumpSuit}
          dealer={euchreState.dealer}
          declarer={euchreState.declarer}
          goingAlone={euchreState.goingAlone}
          currentPlayer={
            nextStaged ? nextStaged.position : euchreState.currentPlayer
          }
          tricksWon={view.tricksWon}
          teamOf={view.teamOf}
          isTerminal={view.isTerminal}
          outcome={euchreState.outcome}
          aloneSeat={euchreState.aloneSeat}
          sittingOut={euchreState.sittingOut}
          showOpponentCards={turn.isAnalysisMode && showAllCards}
          allHandsInteractive={turn.isAnalysisMode}
          playableCards={stagedBehind ? NO_CARDS : moves.playableCardsSet}
          biddingActions={euchreState.biddingActions}
          onCardClick={stagedBehind ? undefined : moves.handleCardClick}
          pickup={euchreState.pickup}
          flightMs={staged.flightMs}
          flightEasing={staged.easing}
          onBiddingAction={moves.handleBiddingAction}
          message={statusLine(
            euchreState,
            view,
            trickCollect,
            nextStaged ?? flyingCard(view, staged.settling, trickCollect),
            turn,
          )}
          collect={collectOf(trickCollect)}
          cardSize="md"
        />
      </div>
    </>
  );
}

/** The Gameplay panel's content for the session `useSession` gives. */
export function createEuchreGameplayPanel(
  useSession: UseSession,
): () => JSX.Element {
  function GameplayPanel(): JSX.Element {
    return (
      <div className="ogs-panel euchre-gameplay">
        <EuchreContent useSession={useSession} />
      </div>
    );
  }
  return GameplayPanel;
}
