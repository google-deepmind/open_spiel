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

// Euchre game UI components

// Components
export {
  Card,
  CardFan,
  TrickPile,
  BiddingOverlay,
  TrumpIndicator,
} from './components/index.ts';
export type {
  CardProps,
  CardSize,
  CardFanProps,
  FanOrientation,
  TrickPileProps,
  BiddingOverlayProps,
  BiddingAction,
  TrumpIndicatorProps,
} from './components/index.ts';

// Sections
export {PlayerSection, TableLayout} from './sections/index.ts';
export type {PlayerSectionProps, TableLayoutProps} from './sections/index.ts';

// Kernel renderer
export {EUCHRE_RENDERER} from './euchre_renderer.tsx';
export {createEuchreGameplayPanel} from './euchre_gameplay_panel.tsx';
export {EuchreGameplayConfigExtras} from './euchre_gameplay_config_extras.tsx';

// Hooks
export {useEuchreState} from './hooks/use_euchre_state.ts';
export type {EuchreUIState} from './hooks/use_euchre_state.ts';

// Types
export type {
  CardCode,
  Suit,
  EuchreRank,
  PlayerPosition,
  GamePhase,
  ActionType,
  TrickCard,
  Player,
  EuchreGameState,
} from './types.ts';

// Utilities
export {
  cardCodeToSvgId,
  parseCard,
  getCardDisplayName,
  getCardShortDisplay,
  getSameColorSuit,
  isRightBower,
  isLeftBower,
  getEffectiveSuit,
  generateDeck,
  sortCards,
  getTeam,
  arePartners,
  SUIT_META,
  RANK_META,
  RANKS,
  SUITS,
} from './types.ts';
