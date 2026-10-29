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

import {useMemo} from 'react';
import type {LegalAction} from '../gin_session.ts';
import {findActionId, extractCardCodesFromLabel} from '../types.ts';

/** The action ids of gin's common moves at the position shown. */
export interface GinActionsState {
  drawUpcardAction: number | null;
  drawStockAction: number | null;
  /** Pass on the first upcard, or at the wall (a pass there ends the hand). */
  passAction: number | null;
  knockAction: number | null;
  discardActions: Map<string, number>;
}

/**
 * The phases whose Pass the board's Pass button plays: the first upcard
 * (decline it) and the wall (the stock is down to its last two cards; a pass
 * ends the hand as a draw, the only other move a knock). Every other pass is
 * a laydown commit, which goes through Done.
 */
export const PASS_BUTTON_PHASES: ReadonlySet<string> = new Set([
  'firstupcard',
  'wall',
]);

/** The Pass the board's Pass button plays at `normalizedPhase`, if any. */
export function passActionOf(
  legalActions: readonly LegalAction[],
  normalizedPhase: string,
): number | null {
  if (!PASS_BUTTON_PHASES.has(normalizedPhase)) return null;
  return findActionId(legalActions, label => label.includes('pass'));
}

/** Extracts action ids from the legal actions for gin's common moves. */
export function useGinActions(
  legalActions: readonly LegalAction[],
  normalizedPhase: string,
): GinActionsState {
  const drawUpcardAction = useMemo(
    () =>
      findActionId(
        legalActions,
        label => label.includes('draw') && label.includes('upcard'),
      ),
    [legalActions],
  );

  const drawStockAction = useMemo(
    () =>
      findActionId(
        legalActions,
        label => label.includes('draw') && label.includes('stock'),
      ),
    [legalActions],
  );

  const passAction = useMemo(
    () => passActionOf(legalActions, normalizedPhase),
    [normalizedPhase, legalActions],
  );

  const knockAction = useMemo(
    () => findActionId(legalActions, label => label.includes('knock')),
    [legalActions],
  );

  const discardActions = useMemo(() => {
    const map = new Map<string, number>();
    // Bare-card legal actions mean "discard" only during the normal
    // discard step and the knocker's mandatory post-knock discard. In the
    // layoff phase the bare-card actions are LAYOFFS with identical labels
    // ("Player: N Action: As") — treating them as discards made layoff
    // cards un-toggleable and let a double-click instantly play the layoff
    // outside the laydown transaction.
    if (normalizedPhase !== 'discard' && normalizedPhase !== 'knock') {
      return map;
    }
    for (const action of legalActions) {
      for (const card of extractCardCodesFromLabel(action.label)) {
        map.set(card, action.action);
      }
    }
    return map;
  }, [legalActions, normalizedPhase]);

  return {
    drawUpcardAction,
    drawStockAction,
    passAction,
    knockAction,
    discardActions,
  };
}
