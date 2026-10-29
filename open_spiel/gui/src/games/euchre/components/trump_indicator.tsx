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
import {type Suit, SUIT_META} from '../types.ts';

/** The properties of TrumpIndicator. */
export interface TrumpIndicatorProps {
  /** The current trump suit */
  trumpSuit: Suit | null;
  /** Whether the declarer is going alone */
  goingAlone?: boolean | undefined;
  /** The seat playing alone, by compass name ("East"); required when goingAlone */
  aloneSeat?: string | null | undefined;
  /** Additional CSS class */
  className?: string | undefined;
}

/**
 * Visual indicator for the current trump suit
 */
export function TrumpIndicator({
  trumpSuit,
  goingAlone = false,
  aloneSeat = null,
  className = '',
}: TrumpIndicatorProps): JSX.Element | null {
  if (!trumpSuit) {
    return null;
  }
  if (goingAlone && !aloneSeat) {
    throw new Error('euchre: a lone hand without the seat that plays alone');
  }

  const meta = SUIT_META[trumpSuit];
  const isRed = trumpSuit === 'h' || trumpSuit === 'd';

  return (
    <div className={`euchre-trump-indicator ${className}`}>
      <div className="euchre-trump-indicator__label">Trump</div>
      <div
        className={`euchre-trump-indicator__suit ${isRed ? 'euchre-trump-indicator__suit--red' : 'euchre-trump-indicator__suit--black'}`}
      >
        <span className="euchre-trump-indicator__symbol">{meta.symbol}</span>
        <span className="euchre-trump-indicator__name">{meta.name}</span>
      </div>
      {goingAlone && (
        <div
          className="euchre-trump-indicator__alone"
          data-testid="euchre-alone"
        >
          {aloneSeat} alone
        </div>
      )}
    </div>
  );
}
