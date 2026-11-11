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

// The check that holds Done back: the staged composition against the
// engine's rules (gin_laydown_logic.ts compositionProblem), read for the
// acting seat on every render.

import {useMemo} from 'react';

import {compositionProblem, serializeSignature} from './gin_laydown_logic.ts';
import type {GinLaydownCompositionInput} from './use_gin_laydown_composition.ts';

/** The staged composition checked against the engine's rules, or null. */
export function useDoneBlocker(
  input: GinLaydownCompositionInput,
  stagedMeldIds: ReadonlySet<string>,
  stagedDiscard: string | null,
  melded: ReadonlySet<string>,
): string | null {
  const {canAct, laydownStage, viewerIsKnocker, knockCard} = input;
  const {playerHand, laydownPlans, playerMelds} = input;
  return useMemo(() => {
    if (!canAct || laydownStage === null) return null;
    const meldSignatures = new Set<string>();
    const layoffSignatures = new Set<string>();
    for (const meld of playerMelds) {
      if (!stagedMeldIds.has(meld.id) || meld.locked) continue;
      const into = meld.cards.length >= 3 ? meldSignatures : layoffSignatures;
      into.add(serializeSignature(meld.cards));
    }
    return compositionProblem({
      stage: laydownStage,
      knocker: viewerIsKnocker,
      knockCard,
      hand: playerHand,
      melded,
      stagedDiscard,
      plans: laydownPlans,
      meldSignatures,
      layoffSignatures,
    });
  }, [
    canAct,
    laydownStage,
    viewerIsKnocker,
    knockCard,
    playerHand,
    laydownPlans,
    playerMelds,
    stagedMeldIds,
    stagedDiscard,
    melded,
  ]);
}
