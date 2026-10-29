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

// The hearts board's state, from the session: the table's look, the bottom
// seat, the hand as the UI reads it, whose turn it is, the pass, the card
// being played, and the trick hold and staged plays that decide what the
// table draws. The Gameplay panel (hearts_gameplay_panel.tsx) draws it.

import {
  type CSSProperties,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
} from 'react';

import {
  motionDuration,
  useCardStyle,
  type TrickCollectState,
} from '../../cards/library/index.ts';
import {DEFAULT_TABLE_CARDS} from '../../cards/table_cards/index.ts';
import {surfaceVars, svgLayer} from '../../cards/table_cards/surface.ts';
import {
  animationScale,
  scaleMs,
  useAnimationScale,
} from '../../motion/index.ts';
import {useAnalysisViewSeat, useBottomSeat} from '../../game/bottom_seat.ts';
import {useSessionTrickCollect} from '../shared/trick_hold.ts';
import {PLAY_REST_MS, useStagedPlays} from '../shared/use_staged_plays.ts';
import {useViewSetting} from '../shared/view_settings.ts';
import type {CardStyle} from '../shared/trick_view_section.tsx';
import {
  SHOW_ALL_CARDS_KEY,
  useHeartsCardStyle,
} from './hearts_gameplay_config_extras.tsx';
import {useHeartsPass, type HeartsPass} from './hearts_pass.ts';
import type {SessionContextValue, SessionPayload} from './hearts_session.ts';
import {
  actionFor,
  heartsStatusMessage,
  heartsTurn,
  type HeartsTurn,
} from './hearts_turn.ts';
import {heartsHeldView, heartsStagedView} from './held_trick.ts';
import {useHeartsState, type HeartsUIState} from './hooks/use_hearts_state.ts';
import type {TableLayoutProps} from './sections/table_layout.tsx';
import type {CardCode, PlayerPosition} from './types.ts';
import grain from './hearts_grain.svg?raw';

const NO_PLAYS: readonly string[] = [];

type Collect = TrickCollectState<CardCode, PlayerPosition>;
/** The felt's grain tile, as a CSS layer (hearts.css). */
const GRAIN = svgLayer(grain);

/** What the Gameplay panel draws. */
export interface HeartsBoard {
  readonly session: SessionPayload;
  readonly error: string | null;
  readonly heartsState: HeartsUIState | null;
  readonly turn: HeartsTurn;
  readonly pass: HeartsPass;
  readonly viewerSeat: number;
  readonly setAnalysisViewSeat: (seat: number | null) => void;
  /** The deck the cards are drawn with: View › Card style. */
  readonly deck: CardStyle;
  readonly wrapperStyle: CSSProperties;
  readonly trickCollect: Collect;
  /** The table's props; null until the state is read. */
  readonly tableProps: TableLayoutProps | null;
}

/** The board once the state is read. */
export type ReadyBoard = HeartsBoard & {readonly heartsState: HeartsUIState};

/** Your card on its way to the server, and the card click that sends it. */
function useCardPlay(
  s: SessionContextValue,
  heartsState: HeartsUIState | null,
  turn: HeartsTurn,
  pass: HeartsPass,
  tableBehind: boolean,
): {sending: CardCode | null; handleCardClick: (card: CardCode) => void} {
  const {session, submitAction} = s;
  // Your card on its way to the server: it stays lifted and the message
  // says so until the move lands (the history grows) or the request
  // settles. Never shown by dimming the rest of the hand.
  const [sending, setSending] = useState<CardCode | null>(null);
  const historyLength = session.state.history.length;
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a move that lands ends the sending cue.
    setSending(null);
  }, [historyLength]);
  const {canInteract, isAnalysisMode, notParticipant} = turn;
  const {togglePick} = pass;
  const handleCardClick = useCallback(
    (card: CardCode) => {
      // A spectator holds no cards: nothing to select or play.
      if (!heartsState || notParticipant) return;
      if (heartsState.phase === 'pass') {
        togglePick(card);
        return;
      }
      // Play the card
      if (sending || tableBehind || (!canInteract && !isAnalysisMode)) return;
      const action = actionFor(session.state.legalActions, card);
      if (action !== null) {
        setSending(card);
        void submitAction(action).finally(() => {
          setSending(prev => (prev === card ? null : prev));
        });
      }
    },
    [
      session,
      heartsState,
      notParticipant,
      canInteract,
      isAnalysisMode,
      submitAction,
      tableBehind,
      togglePick,
      sending,
    ],
  );
  return {sending, handleCardClick};
}

/** The table's surface: the theme's felt, as euchre's, with hearts' grain. */
const GRAIN_VARS: Record<string, string> = {'--hearts-grain': GRAIN};
const WRAPPER_STYLE: CSSProperties = {
  ...surfaceVars(DEFAULT_TABLE_CARDS.surface),
  ...GRAIN_VARS,
};

/**
 * The board's look: the deck View › Card style chose, on the default felt,
 * back and suit symbols. Your hand is flat and each trick card lies toward
 * its player, as at euchre's table.
 */
function useHeartsLook(): {deck: CardStyle; wrapperStyle: CSSProperties} {
  const [deck] = useHeartsCardStyle();
  return {deck, wrapperStyle: WRAPPER_STYLE};
}

/**
 * What the table draws while a finished trick is held and while plays that
 * arrived together are shown one at a time.
 */
function useTableView(
  s: SessionContextValue,
  heartsState: HeartsUIState | null,
): {
  trickCollect: Collect;
  view: ReturnType<typeof heartsHeldView> | null;
} {
  // A finished trick stays on the table, then is collected toward its
  // winner, while the live state is accepted underneath. `settling` (below,
  // from the play staging) holds the hold back until plays that arrived
  // together have all landed.
  const [settling, setSettling] = useState(false);
  const trickCollect = useSessionTrickCollect<CardCode, PlayerPosition>(
    s.session,
    s.replay,
    heartsState?.lastTrick ?? null,
    settling,
  );
  // Bot plays that arrive in one update join the trick one at a time,
  // each after the one before has landed (the viewer's Animation speed).
  const heldView = heartsState
    ? heartsHeldView(heartsState, trickCollect.trick, trickCollect.pending)
    : null;
  // Timed from the card style Hearts' table flies with.
  const cardMotion = useCardStyle().motion;
  useAnimationScale(); // re-render when the speed changes
  const staged = useStagedPlays(
    // Keyed by card, not by seat position: turning the table to another
    // seat is not a new play to fly in.
    heldView ? heldView.trickPile.map(t => t.card) : NO_PLAYS,
    {
      legMs: motionDuration(cardMotion),
      restMs: scaleMs(
        PLAY_REST_MS,
        animationScale({reducedMotion: cardMotion.reducedMotion}),
      ),
    },
  );
  useLayoutEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- the hold follows the staging's settling.
    setSettling(staged.settling);
  }, [staged.settling]);
  const view = heldView ? heartsStagedView(heldView, staged.shown) : null;
  return {trickCollect, view};
}

/** The board's state, from the session (see the file comment). */
export function useHeartsBoard(s: SessionContextValue): HeartsBoard {
  const {session, viewer, isBusy} = s;
  const [showAllCards] = useViewSetting<boolean>(SHOW_ALL_CARDS_KEY, false);
  const look = useHeartsLook();
  // The seat drawn at the bottom: the one chosen in analysis, your own seat
  // in a game.
  const {seat: viewerSeat} = useBottomSeat(session, viewer);
  const [, setAnalysisViewSeat] = useAnalysisViewSeat();
  // Transform session state to UI state
  const heartsState = useHeartsState(session, viewerSeat);
  const {trickCollect, view} = useTableView(s, heartsState);
  const turn = heartsTurn(session, viewer, heartsState, showAllCards, isBusy);
  // Your hand as the engine has it now.
  const myHand = useMemo(
    () => heartsState?.players.find(p => p.position === 'bottom')?.hand ?? [],
    [heartsState],
  );
  const pass = useHeartsPass({
    session,
    heartsState,
    isAnalysisMode: turn.isAnalysisMode,
    myHand,
    historyLength: session.state.history.length,
    viewerSeat,
    submitAtomicSequence: s.submitAtomicSequence,
  });
  // The table is a trick behind the game while a finished trick waits its
  // turn: no card play until it has caught up, so a move is never made on a
  // stale table.
  const tableBehind = trickCollect.pending.length > 0;
  const play = useCardPlay(s, heartsState, turn, pass, tableBehind);
  useClearPicks(heartsState, turn, pass);
  const tableProps =
    heartsState && view
      ? tablePropsOf({
          s,
          heartsState,
          view,
          turn,
          pass,
          play,
          tableBehind,
          showAllCards,
          trickCollect,
          look,
        })
      : null;
  return {
    session,
    error: s.error,
    heartsState,
    turn,
    pass,
    viewerSeat,
    setAnalysisViewSeat,
    deck: look.deck,
    wrapperStyle: look.wrapperStyle,
    trickCollect,
    tableProps,
  };
}

/** Clears the picks when the phase changes, and when the hand on move goes
 *  out of reach (Show all cards turned off over its picks). */
function useClearPicks(
  heartsState: HeartsUIState | null,
  turn: HeartsTurn,
  pass: HeartsPass,
): void {
  const {setSelectedCards} = pass;
  const phase = heartsState?.phase;
  const {outOfReach} = turn;
  useEffect(() => {
    if (phase === 'play' || outOfReach) {
      setSelectedCards(new Set());
    }
  }, [phase, outOfReach, setSelectedCards]);
}

interface TableInputs {
  readonly s: SessionContextValue;
  readonly heartsState: HeartsUIState;
  readonly view: NonNullable<ReturnType<typeof heartsHeldView>>;
  readonly turn: HeartsTurn;
  readonly pass: HeartsPass;
  readonly play: ReturnType<typeof useCardPlay>;
  readonly tableBehind: boolean;
  readonly showAllCards: boolean;
  readonly trickCollect: Collect;
  readonly look: ReturnType<typeof useHeartsLook>;
}

/** The table's props, from the board's inputs. */
function tablePropsOf(i: TableInputs): TableLayoutProps {
  const {heartsState, view, turn, pass, play, trickCollect, look} = i;
  const {isAnalysisMode, notParticipant, outOfReach, canInteract} = turn;
  // Get playable cards from the state
  const playableCards =
    play.sending || i.tableBehind || (!canInteract && !isAnalysisMode)
      ? new Set<string>()
      : heartsState.playableCards;
  const rewards = i.s.session.state.rewards;
  const collect =
    trickCollect.phase === 'collect' && trickCollect.trick && trickCollect.key
      ? {
          key: trickCollect.key,
          winner: trickCollect.trick.winner,
          durationMs: trickCollect.collectMs,
        }
      : null;
  return {
    players: view.players,
    trickPile: view.trickPile,
    phase: heartsState.phase,
    passDirection: heartsState.passDirection,
    // Nobody is to move once the game is over, the last trick's hold
    // included: no seat's badge stays lit under Game Over.
    currentPlayer: heartsState.isTerminal ? null : heartsState.currentPlayer,
    isTerminal: view.isTerminal,
    ...(rewards === null ? {} : {rewards: [...rewards]}),
    showOpponentCards: isAnalysisMode && i.showAllCards,
    canShowAll: isAnalysisMode,
    allHandsInteractive: isAnalysisMode,
    ...(heartsState.phase === 'pass'
      ? {selectedCards: pass.selectedCards}
      : {}),
    playableCards,
    sendingCard: play.sending,
    passPending: !!pass.sentPass,
    passFlight: pass.passFlight,
    // Out of reach, the bottom hand is not the one on move: picks there
    // could not be passed or played.
    ...(notParticipant || outOfReach
      ? {}
      : {onCardClick: play.handleCardClick}),
    // A spectator has no cards to pass: no button.
    ...(notParticipant ? {} : {onPassCards: pass.handlePassCards}),
    message: heartsStatusMessage(heartsState, turn, play.sending),
    collect,
    cardSize: 'md',
    deck: look.deck,
  };
}
