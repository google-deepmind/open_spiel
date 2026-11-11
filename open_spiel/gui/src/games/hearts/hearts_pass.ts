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

// The hearts pass, from the picks to the cards received: the three cards
// chosen (lifted until they leave the hand), the pass sent as one move, its
// flight to the seat it goes to, and the cards that fly back in.

import {useCallback, useEffect, useMemo, useState} from 'react';

import type {HeartsUIState} from './hooks/use_hearts_state.ts';
import type {SessionPayload} from './hearts_session.ts';
import {actionFor} from './hearts_turn.ts';
import {
  PASS_COUNT,
  passSeats,
  type PassFlight,
} from './sections/table_layout.tsx';
import type {CardCode, PlayerPosition} from './types.ts';

/** Your pass, from the click until the hand is over. */
export interface SentPass {
  cards: CardCode[];
  to: PlayerPosition;
  from: PlayerPosition;
  /** Your hand when you passed, to tell the cards you receive. */
  handBefore: CardCode[];
  /** The history length when you passed: a shorter history (a reset, an
   *  undo, a replay scrubbed back) forgets the pass. */
  atHistory: number;
  /** The cards you received, kept once all have arrived: the passer's
   *  backs keep their new ids for the rest of the hand (backId), even
   *  after you have played the cards. */
  received: CardCode[] | null;
}

/** What the pass hook reads. */
export interface HeartsPassInputs {
  readonly session: SessionPayload;
  readonly heartsState: HeartsUIState | null;
  readonly isAnalysisMode: boolean;
  readonly myHand: readonly CardCode[];
  readonly historyLength: number;
  readonly viewerSeat: number;
  readonly submitAtomicSequence: (
    actions: readonly number[],
  ) => Promise<SessionPayload>;
}

/** The pass's state, and how to pick and send it. */
export interface HeartsPass {
  readonly selectedCards: Set<string>;
  readonly setSelectedCards: (next: Set<string>) => void;
  readonly sentPass: SentPass | null;
  readonly passError: string | null;
  readonly setPassError: (next: string | null) => void;
  readonly passFlight: PassFlight | null;
  /** Toggles `card` among the picks (at most three). */
  readonly togglePick: (card: CardCode) => void;
  readonly handlePassCards: () => void;
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** The pass, once it has left your hand: the flight's data (heartsItems). */
function flightOf(
  sentPass: SentPass | null,
  myHand: readonly CardCode[],
): PassFlight | null {
  if (
    !sentPass ||
    (sentPass.received === null && sentPass.cards.some(c => myHand.includes(c)))
  ) {
    return null;
  }
  const arrived = myHand.filter(c => !sentPass.handBefore.includes(c));
  const received =
    sentPass.received ?? (arrived.length >= PASS_COUNT ? arrived : []);
  return {
    to: sentPass.to,
    from: sentPass.from,
    sent: sentPass.cards,
    received,
  };
}

/** The picks, the sent pass and its flight (see the file comment). */
export function useHeartsPass(p: HeartsPassInputs): HeartsPass {
  const {session, heartsState, isAnalysisMode, myHand, historyLength} = p;
  // Selected cards for pass phase
  const [selectedCards, setSelectedCards] = useState<Set<string>>(new Set());
  const [sentPass, setSentPass] = useState<SentPass | null>(null);
  const [passError, setPassError] = useState<string | null>(null);
  // A pass is forgotten with the session, or when the history goes back
  // past it.
  useEffect(() => {
    if (sentPass && historyLength < sentPass.atHistory) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- a pass is forgotten when the history goes back past it.
      setSentPass(null);
    }
  }, [sentPass, historyLength]);
  // So is a pass, and the picks, when the table turns to another seat:
  // they were made in the hand that was at the bottom.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a new session or bottom seat forgets the pass.
    setSentPass(null);
    setSelectedCards(new Set());
    setPassError(null);
  }, [session.sessionId, p.viewerSeat]);
  const passFlight = useMemo(
    () => flightOf(sentPass, myHand),
    [sentPass, myHand],
  );
  // Keep the received cards once they are all in (and the pass is
  // through: no error to show).
  const flightReceived = passFlight?.received;
  useEffect(() => {
    if (!flightReceived) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a pass that is through has no error to show.
    setPassError(null);
    if (flightReceived.length < PASS_COUNT) return;
    setSentPass(s =>
      s?.received === null ? {...s, received: [...flightReceived]} : s,
    );
  }, [flightReceived]);
  // Your picks stay lifted until they leave the hand; then the selection
  // is empty. In analysis a pick may be in the hand of whoever is
  // passing (show all cards), not only yours.
  const pickable = useMemo(
    () =>
      isAnalysisMode
        ? (heartsState?.players ?? []).flatMap(pl => pl.hand)
        : myHand,
    [isAnalysisMode, heartsState, myHand],
  );
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- a pick that left the hand is no longer picked.
    setSelectedCards(prev => {
      const kept = [...prev].filter(c => pickable.includes(c as CardCode));
      return kept.length === prev.size ? prev : new Set(kept);
    });
  }, [pickable]);
  const togglePick = useCallback(
    (card: CardCode) => {
      // Your pass is on its way or made: the picks are no longer yours to change.
      if (sentPass) return;
      // Toggle card selection for passing
      setSelectedCards(prev => {
        const next = new Set(prev);
        if (next.has(card)) {
          next.delete(card);
        } else if (next.size < PASS_COUNT) {
          next.add(card);
        }
        return next;
      });
    },
    [sentPass],
  );
  const handlePassCards = usePassSubmit(p, {
    selectedCards,
    sentPass,
    setSelectedCards,
    setSentPass,
    setPassError,
  });
  return {
    selectedCards,
    setSelectedCards,
    sentPass,
    passError,
    setPassError,
    passFlight,
    togglePick,
    handlePassCards,
  };
}

interface PassState {
  readonly selectedCards: Set<string>;
  readonly sentPass: SentPass | null;
  readonly setSelectedCards: (next: Set<string>) => void;
  readonly setSentPass: (
    next: SentPass | null | ((s: SentPass | null) => SentPass | null),
  ) => void;
  readonly setPassError: (next: string | null) => void;
}

/**
 * The pass button's handler. The three picks go in ONE request (all or
 * nothing), so they leave your hand together and fly to the seat they go
 * to; the picks stay lifted until then.
 */
function usePassSubmit(p: HeartsPassInputs, s: PassState): () => void {
  const {session, heartsState, isAnalysisMode, myHand, historyLength} = p;
  const {submitAtomicSequence} = p;
  const {selectedCards, sentPass, setSelectedCards, setSentPass} = s;
  const {setPassError} = s;
  return useCallback(() => {
    if (!heartsState || sentPass || selectedCards.size !== PASS_COUNT) return;
    const seats = passSeats(heartsState.passDirection);
    if (!seats) {
      throw new Error('hearts: a pass was submitted in a no-pass round');
    }
    const legalActions = session.state.legalActions;
    const actionsFor = (cards: readonly CardCode[]): number[] =>
      cards.map(cardCode => {
        const action = actionFor(legalActions, cardCode);
        if (action === null) {
          throw new Error(`hearts: no legal pass action for ${cardCode}`);
        }
        return action;
      });
    // Analysis, show all cards: you pass for whoever is passing, and
    // that need not be the bottom seat. Its picks go as one request
    // too; only your own pass flies (passFlight is the bottom seat's).
    const passer = isAnalysisMode ? heartsState.currentPlayer : 'bottom';
    if (passer !== 'bottom') {
      const seat = heartsState.players.find(pl => pl.position === passer);
      const cards = (seat?.hand ?? []).filter(c => selectedCards.has(c));
      if (cards.length !== PASS_COUNT) {
        const who = seat?.name ?? 'the passing player';
        setPassError(
          `Choose the ${String(PASS_COUNT)} cards from ${who}'s hand: it is their pass`,
        );
        return;
      }
      setPassError(null);
      submitAtomicSequence(actionsFor(cards)).then(
        () => {
          setSelectedCards(new Set());
        },
        (err: unknown) => {
          setPassError(errorText(err));
        },
      );
      return;
    }
    const cards = myHand.filter(c => selectedCards.has(c));
    const passActions = actionsFor(cards);
    const pass: SentPass = {
      cards,
      ...seats,
      handBefore: [...myHand],
      atHistory: historyLength,
      received: null,
    };
    setPassError(null);
    setSentPass(pass);
    submitAtomicSequence(passActions).then(
      () => undefined,
      (err: unknown) => {
        // Nothing was applied (all or nothing): the picks stay
        // chosen, to pass again.
        setSentPass(prev => (prev === pass ? null : prev));
        setPassError(errorText(err));
      },
    );
  }, [
    session,
    heartsState,
    sentPass,
    selectedCards,
    myHand,
    historyLength,
    submitAtomicSequence,
    isAnalysisMode,
    setSelectedCards,
    setSentPass,
    setPassError,
  ]);
}
