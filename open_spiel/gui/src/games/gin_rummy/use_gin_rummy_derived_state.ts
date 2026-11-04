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

// The gin board's state, derived from the session: whose hand is at the
// foot, each hand, the viewer's melds, the table (stock, upcard, discard
// pile), the knock and the lay-down, the status line, and the action ids of
// the common moves. Pure reading; nothing here draws.

import {useMemo} from 'react';

import {
  asNumberArray,
  asObject,
  asStringArray,
  formatCurrentPlayer,
  normaliseCardCode,
  normalisedCodes,
  normalizePlayerId,
  seatOf,
  turnPhaseLabel,
} from './gin_derived_parse.ts';
import {
  mergeHandWithLayed,
  playerMeldsOf,
  type PlayerMelds,
} from './gin_derived_melds.ts';
import {
  analysisBestIndicesOf,
  analysisBestMeldsOf,
  analysisMeldsOf,
  laydownOf,
  layedMeldsOf,
  meldCardTypes,
  visibleHandsOf,
  type Laydown,
} from './gin_derived_seats.ts';
import type {GinState, LegalAction, SessionPayload} from './gin_session.ts';
import {GIN_STATUS, GIN_STATUS_JOIN} from './gin_turn_line.ts';
import {useGinActions, type GinActionsState} from './hooks/use_gin_actions.ts';
import {asRecord, type GinStructRecord, type MeldType} from './types.ts';

export {formatNumber} from './gin_derived_parse.ts';

/** The seats, hands and melds part of the derived state. */
interface SeatState extends Laydown, PlayerMelds {
  structRecord: GinStructRecord | null;
  usingObservationStruct: boolean;
  ginPhase: string | null;
  revealHands: boolean;
  visibleHands: Set<number>;
  layedMeldCardTypesBySeat: Map<number, Map<string, MeldType>>;
  /** Each seat's laid melds (the knocker's, that lay-offs extend). */
  layedMeldsBySeat: Map<number, string[][]>;
  analysisMeldsBySeat: Map<number, string[][]>;
  analysisBestMeldsBySeat: Map<number, string[][]>;
  normalizedPhase: string;
  playerHand: string[];
  opponentHand: string[];
  playerHandSlot: number | null;
  opponentHandSlot: number | null;
}

/** The table part of the derived state. */
interface TableState {
  opponentHandCount: number;
  opponentHandMasked: boolean;
  playerHandSet: Set<string>;
  playerHandKey: string;
  discardPile: string[];
  discardPileNormalized: string[];
  discardCard: string | null;
  discardCardNormalised: string | null;
  prevUpcardNormalised: string | null;
  playerDeadwood: number | null;
  opponentDeadwood: number | null;
  stockCount: number;
  knockCard: number | null;
  layoffsCount: number;
  layoffCards: Set<string>;
  layoffCardList: string[];
}

/** The turn part of the derived state. */
interface TurnState {
  phaseLabel: string | null;
  knocked: boolean;
  knockerSeat: number | null;
  currentPlayerText: string | null;
  statusText: string;
  currentPlayerId: number | null;
  sessionText: string;
  lastActionText: string | null;
  turnNumber: number | null;
  isTerminal: boolean;
  legalActions: readonly LegalAction[];
  returns: readonly number[] | null;
}

type BaseState = SeatState &
  TableState &
  TurnState & {sessionId: string | null};

/** Everything the gin board reads from the session. */
export type GinRummyDerivedState = BaseState &
  GinActionsState & {
    discardHighlights: Set<string>;
    knownCardsBySeat: Map<number, Set<string>>;
  };

/** How to derive: whose seat is at the foot of the table. */
export interface DeriveOptions {
  preferObservationStruct?: boolean;
  viewerSeat?: number | null;
}

function seatsOf(viewerSeat: number | null | undefined): [number, number] {
  if (typeof viewerSeat === 'number' && viewerSeat >= 0) {
    return [viewerSeat, viewerSeat === 0 ? 1 : 0];
  }
  return [0, 1];
}

function pickStruct(
  state: GinState | undefined,
  prefer: boolean,
): {structRecord: GinStructRecord | null; usingObservationStruct: boolean} {
  const full = asRecord(state?.struct);
  const observation = asRecord(state?.observationStruct);
  const structRecord =
    prefer && observation !== null ? observation : (full ?? observation);
  return {
    structRecord,
    usingObservationStruct:
      structRecord !== null && structRecord === observation,
  };
}

function seatState(
  state: GinState | undefined,
  options: DeriveOptions | undefined,
): SeatState {
  const [viewer, opponent] = seatsOf(options?.viewerSeat);
  const visibleHands = visibleHandsOf(state);
  const layedMelds = layedMeldsOf(state);
  const revealHands = state?.ginRevealHands === true;
  const prefer =
    options?.preferObservationStruct === true &&
    !revealHands &&
    !visibleHands.has(opponent);
  const layedMeldCardTypesBySeat = meldCardTypes(layedMelds);
  const analysisMeldsBySeat = analysisMeldsOf(state);
  const analysisBest = analysisBestIndicesOf(state);
  const picked = pickStruct(state, prefer);
  const {structRecord} = picked;
  const laydown = laydownOf(state);
  const phase = structRecord?.['phase'];
  const hands = Array.isArray(structRecord?.['hands'])
    ? structRecord['hands'].map(asStringArray)
    : [];
  // The hands take the laid melds' cards before the struct's laid melds are
  // gathered with the viewer's melds below.
  const playerHand = mergeHandWithLayed(
    hands[viewer] ?? [],
    layedMelds.get(viewer),
  );
  const opponentHand = mergeHandWithLayed(
    hands[opponent] ?? [],
    layedMelds.get(opponent),
  );
  const melds = playerMeldsOf({
    struct: structRecord,
    viewer,
    analysisMelds: analysisMeldsBySeat,
    analysisBest,
    layedMelds,
    laydownSeat: laydown.laydownSeat,
    laydownPlans: laydown.laydownPlans,
  });
  return {
    ...picked,
    ...laydown,
    ...melds,
    ginPhase: typeof state?.ginPhase === 'string' ? state.ginPhase : null,
    revealHands,
    visibleHands,
    layedMeldCardTypesBySeat,
    layedMeldsBySeat: layedMelds,
    analysisMeldsBySeat,
    analysisBestMeldsBySeat: analysisBestMeldsOf(
      analysisMeldsBySeat,
      analysisBest,
    ),
    normalizedPhase: typeof phase === 'string' ? phase.toLowerCase() : '',
    playerHand,
    opponentHand,
    playerHandSlot: hands.length > viewer ? viewer : null,
    opponentHandSlot: hands.length > opponent ? opponent : null,
  };
}

function isMasked(card: string): boolean {
  const trimmed = card.trim();
  return trimmed.length > 0 && /^x+$/i.test(trimmed);
}

/** Every string in `value`, however deeply it nests in arrays. */
function collectStrings(value: unknown, into: string[] = []): string[] {
  if (typeof value === 'string') {
    into.push(value);
  } else if (Array.isArray(value)) {
    for (const item of value) collectStrings(item, into);
  }
  return into;
}

function numberOf(value: unknown): number | null {
  const n = typeof value === 'string' ? Number(value) : value;
  return typeof n === 'number' && Number.isFinite(n) ? n : null;
}

function supplyState(
  struct: GinStructRecord | null,
): Pick<
  TableState,
  | 'discardPile'
  | 'discardPileNormalized'
  | 'discardCard'
  | 'discardCardNormalised'
  | 'prevUpcardNormalised'
  | 'stockCount'
  | 'knockCard'
> {
  const discardPile = asStringArray(struct?.['discard_pile']);
  const upcard = struct?.['upcard'];
  const prevUpcard = struct?.['prev_upcard'];
  const knockCard = struct?.['knock_card'];
  const discardCard =
    typeof upcard === 'string' ? upcard : (discardPile.at(-1) ?? null);
  return {
    discardPile,
    discardPileNormalized: normalisedCodes(discardPile),
    discardCard,
    discardCardNormalised: normaliseCardCode(discardCard),
    prevUpcardNormalised: normaliseCardCode(
      typeof prevUpcard === 'string' ? prevUpcard : null,
    ),
    stockCount: numberOf(struct?.['stock_size']) ?? 0,
    knockCard: typeof knockCard === 'number' ? knockCard : null,
  };
}

function tableState(
  seats: SeatState,
  viewerSeat: number | null | undefined,
): TableState {
  const [viewer, opponent] = seatsOf(viewerSeat);
  const struct = seats.structRecord;
  const playerCards = seats.playerHand
    .map(card => normaliseCardCode(card))
    .filter((card): card is string => card !== null);
  const deadwood = asNumberArray(struct?.['deadwood']);
  // The engine marks a deadwood its viewer may not see with -1 (the
  // opponent's to a seat, both to a spectator): shown as unknown, never as
  // a number.
  const knownDeadwood = (seat: number): number | null => {
    const value = deadwood[seat];
    return value !== undefined && value >= 0 ? value : null;
  };
  const layoffsList = collectStrings(struct?.['layoffs']);
  const layoffCards = new Set(normalisedCodes(layoffsList));
  return {
    ...supplyState(struct),
    opponentHandCount: seats.opponentHand.length,
    opponentHandMasked:
      seats.opponentHand.length > 0 && seats.opponentHand.every(isMasked),
    playerHandSet: new Set(playerCards),
    playerHandKey: playerCards.join(','),
    playerDeadwood: knownDeadwood(viewer),
    opponentDeadwood: knownDeadwood(opponent),
    layoffsCount: layoffsList.length,
    layoffCards,
    layoffCardList: [...layoffCards],
  };
}

function knockOf(raw: unknown): {knocked: boolean; knockerSeat: number | null} {
  if (typeof raw === 'boolean') {
    return {knocked: raw, knockerSeat: raw ? 0 : null};
  }
  if (!Array.isArray(raw)) return {knocked: false, knockerSeat: null};
  const found = raw.findIndex(value => Boolean(value));
  return {knocked: found >= 0, knockerSeat: found >= 0 ? found : null};
}

const LAYOFF_PHASES = new Set(['layoff', 'show', 'showdown', 'score']);

function statusText(
  seats: SeatState,
  currentPlayerText: string | null,
  phaseLabel: string | null,
  isTerminal: boolean,
): string {
  if (isTerminal) return GIN_STATUS.gameOver;
  const parts: Array<string | null> = [currentPlayerText];
  if (seats.normalizedPhase === 'knock') {
    // The knocker's mandatory discard is the engine-derived stage, not a
    // scan of the legal actions' labels.
    parts.push(
      GIN_STATUS.knock,
      seats.laydownStage === 'discard'
        ? GIN_STATUS.discard
        : GIN_STATUS.pressDone,
    );
  } else if (LAYOFF_PHASES.has(seats.normalizedPhase)) {
    parts.push(phaseLabel ?? GIN_STATUS.layoff, GIN_STATUS.pressDone);
  } else {
    parts.push(phaseLabel);
  }
  const joined = parts
    .filter((part): part is string => part !== null && part !== '')
    .join(GIN_STATUS_JOIN);
  return joined === '' ? GIN_STATUS.ready : joined;
}

function turnState(seats: SeatState, state: GinState | undefined): TurnState {
  const struct = seats.structRecord;
  const phaseLabel = turnPhaseLabel(
    struct?.['phase'],
    state?.legalActions ?? [],
  );
  const currentPlayerText = formatCurrentPlayer(
    struct?.['current_player'],
    state?.currentPlayerLabel ?? null,
  );
  const isTerminal = state?.isTerminal ?? false;
  return {
    ...knockOf(struct?.['knocked'] ?? state?.ginKnocked ?? null),
    phaseLabel,
    currentPlayerText,
    statusText: statusText(seats, currentPlayerText, phaseLabel, isTerminal),
    currentPlayerId: normalizePlayerId(struct?.['current_player']),
    sessionText: state?.text ?? '',
    lastActionText: state?.lastAction ?? null,
    turnNumber: state?.turnNumber ?? null,
    isTerminal,
    legalActions: state?.legalActions ?? [],
    returns: state?.returns ?? null,
  };
}

/** The base derived state of `session` (everything but the action ids). */
export function deriveGinBase(
  session: SessionPayload | null | undefined,
  options?: DeriveOptions,
): BaseState {
  const state = session?.state;
  const seats = seatState(state, options);
  return {
    sessionId: session?.sessionId ?? null,
    ...seats,
    ...tableState(seats, options?.viewerSeat),
    ...turnState(seats, state),
  };
}

function knownCardsOf(raw: unknown): Map<number, Set<string>> {
  const map = new Map<number, Set<string>>();
  const record = asObject(raw);
  if (record === null) return map;
  for (const [key, cards] of Object.entries(record)) {
    const seat = seatOf(key);
    if (seat === null || !Array.isArray(cards)) continue;
    map.set(seat, new Set(normalisedCodes(cards)));
  }
  return map;
}

/** The gin board's state for `session`, seen from `options.viewerSeat`. */
export function useGinRummyDerivedState(
  session: SessionPayload | null | undefined,
  options?: DeriveOptions,
): GinRummyDerivedState {
  const state = session?.state;
  const prefer = options?.preferObservationStruct;
  const viewerSeat = options?.viewerSeat;
  const baseState = useMemo(
    () =>
      deriveGinBase(session, {
        ...(prefer === undefined ? {} : {preferObservationStruct: prefer}),
        ...(viewerSeat === undefined ? {} : {viewerSeat}),
      }),
    [session, prefer, viewerSeat],
  );
  const discardHighlights = useMemo(
    () => new Set(baseState.discardPileNormalized),
    [baseState.discardPileNormalized],
  );
  const knownCards = state?.knownCardsBySeat;
  const knownCardsBySeat = useMemo(
    () => knownCardsOf(knownCards),
    [knownCards],
  );
  const actions = useGinActions(
    baseState.legalActions,
    baseState.normalizedPhase,
  );
  return {...baseState, discardHighlights, knownCardsBySeat, ...actions};
}
