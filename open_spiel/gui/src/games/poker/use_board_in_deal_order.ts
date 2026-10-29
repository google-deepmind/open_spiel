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

// The board as it was dealt: the engine sends the board as a sorted
// card set, so this keeps the order the panel has shown and puts each new
// card on the end (boardInDealOrder). Remembered per session; a page opened
// mid-hand starts from the engine's order, where the flop's order carries
// nothing and a later card is still only ever appended.
import {useMemo, useState} from 'react';
import {boardInDealOrder, type PokerUiState} from './poker_state.ts';

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((code, index) => code === b[index]);
}

/** `state` with its board in the order it was dealt in this session. */
export function useBoardInDealOrder(
  state: PokerUiState | null,
  sessionId: string | undefined,
): PokerUiState | null {
  const [shown, setShown] = useState<{
    sessionId: string | undefined;
    codes: string[];
  }>({
    sessionId,
    codes: [],
  });
  const previous = shown.sessionId === sessionId ? shown.codes : [];
  // `previous` is read through its codes: the same order is no change.
  const previousCodes = previous.join(' ');
  const board = useMemo(
    () =>
      state
        ? boardInDealOrder(
            previousCodes === '' ? [] : previousCodes.split(' '),
            state.board,
          )
        : [],
    [state, previousCodes],
  );
  const codes = board.map(card => card.code);
  if (
    state &&
    (shown.sessionId !== sessionId || !sameOrder(codes, shown.codes))
  ) {
    // Derived state from the previous render (React's documented
    // "storing information from previous renders" pattern).
    setShown({sessionId, codes});
  }
  return useMemo(() => (state ? {...state, board} : null), [state, board]);
}
