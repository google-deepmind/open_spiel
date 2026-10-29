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

// Gin's view settings: the hooks the board reads them with, and the View
// panel's two sections that set them, Visuals and Gin Rummy.

import {type JSX, useCallback} from 'react';

import {getDeck} from '../../cards/library/index.ts';
import {useIsAnalysisMode} from '../../game/bottom_seat.ts';
import {GinTableCardFields} from './gin_card_style.tsx';
import {GIN_CARD_FACES, type GinCardFace} from './gin_rummy_layout.ts';

import './gin_settings.css';
import {useViewSetting, usePersistedSetting} from '../shared/view_settings.ts';
import {
  CARD_SCALE_MAX,
  CARD_SCALE_MIN,
  CARD_SCALE_STEP,
  DEFAULT_GRID_STYLE,
  DEFAULT_HAND_VIEW,
  GRID_STYLES,
  GRID_STYLE_LABELS,
  HAND_VIEWS,
  isCardScale,
  readStoredCardScale,
  storedSettingKey,
  type GridStyle,
  type HandView,
} from './gin_view_settings.ts';

export type {GridStyle, HandView} from './gin_view_settings.ts';

/** The card face setting's key. */
export const CARD_STYLE_KEY = 'gin.cardFaceStyle';
/** The hand view setting's key. */
export const HAND_VIEW_KEY = 'gin.handView';
/** The grid style setting's key. */
export const GRID_STYLE_KEY = 'gin.gridStyle';
/** The card size setting's key. */
export const CARD_SCALE_KEY = 'gin.cardScale';
/** The analysis-only opponent reveal setting's key. */
export const SHOW_OPPONENT_KEY = 'gin.showOpponentHand';
/** The discard and known card hints setting's key. */
export const SHOW_KNOWLEDGE_KEY = 'gin.showKnowledge';

/**
 * The card face (gin_rummy_layout.ts GIN_FACE_STYLES); 'app', the shared
 * deck from the table and card settings, by default.
 */
export function useCardFaceStyle(): [
  GinCardFace,
  (next: GinCardFace | ((current: GinCardFace) => GinCardFace)) => void,
] {
  return useViewSetting<GinCardFace>(CARD_STYLE_KEY, 'app');
}

function isCardFace(value: string): value is GinCardFace {
  return GIN_CARD_FACES.some(face => face === value);
}

function CardStyleField(): JSX.Element {
  const [face, setFace] = useCardFaceStyle();
  return (
    <label className="ogs-config-field">
      <span>Card style</span>
      <select
        value={face}
        onChange={event => {
          if (isCardFace(event.target.value)) setFace(event.target.value);
        }}
      >
        <option value="app">As the card deck setting</option>
        <option value="modern" title={getDeck('simple').description}>
          {getDeck('simple').label}
        </option>
      </select>
      <span className="ogs-config-note" data-testid="gin-face-description">
        {face === 'modern'
          ? getDeck('simple').description
          : 'The face picked under Card deck, with its own "best for" line.'}
      </span>
    </label>
  );
}

/** The hand view, persisted on this browser; 'grid' by default. */
export function useHandView(): [HandView, (value: HandView) => void] {
  return usePersistedSetting<HandView>(
    HAND_VIEW_KEY,
    HAND_VIEWS,
    DEFAULT_HAND_VIEW,
  );
}

/** The grid style, persisted on this browser; 'normal' (spread) by default. */
export function useGridStyle(): [GridStyle, (value: GridStyle) => void] {
  return usePersistedSetting<GridStyle>(
    GRID_STYLE_KEY,
    GRID_STYLES,
    DEFAULT_GRID_STYLE,
  );
}

/**
 * The card size, percent of the fitted size, persisted on this browser; 100,
 * the size the board fits the panel at, by default.
 */
export function useCardScale(): [number, (value: number) => void] {
  const [value, set] = useViewSetting<number>(
    CARD_SCALE_KEY,
    readStoredCardScale(CARD_SCALE_KEY),
  );
  const setPersisted = useCallback(
    (next: number) => {
      if (!isCardScale(next)) {
        throw new Error(
          `gin setting ${CARD_SCALE_KEY}: ${String(next)} is not a card size (${String(CARD_SCALE_MIN)}-${String(CARD_SCALE_MAX)} in steps of ${String(CARD_SCALE_STEP)})`,
        );
      }
      window.localStorage.setItem(
        storedSettingKey(CARD_SCALE_KEY),
        String(next),
      );
      set(next);
    },
    [set],
  );
  return [value, setPersisted];
}

function isHandView(value: string): value is HandView {
  return HAND_VIEWS.some(view => view === value);
}

function isGridStyle(value: string): value is GridStyle {
  return GRID_STYLES.some(style => style === value);
}

function GridStyleField(): JSX.Element {
  const [gridStyle, setGridStyle] = useGridStyle();
  return (
    <label className="ogs-config-field">
      <span>Grid style</span>
      <select
        value={gridStyle}
        onChange={e => {
          if (isGridStyle(e.target.value)) setGridStyle(e.target.value);
        }}
      >
        {GRID_STYLES.map(g => (
          <option key={g} value={g}>
            {GRID_STYLE_LABELS[g]}
          </option>
        ))}
      </select>
    </label>
  );
}

function CardSizeField(): JSX.Element {
  const [cardScale, setCardScale] = useCardScale();
  return (
    <label className="ogs-config-field gin-card-size">
      <span>
        Card size{' '}
        <output className="gin-card-size__value">{`${String(cardScale)}%`}</output>
      </span>
      <input
        type="range"
        min={CARD_SCALE_MIN}
        max={CARD_SCALE_MAX}
        step={CARD_SCALE_STEP}
        value={cardScale}
        aria-valuetext={`${String(cardScale)}%`}
        onChange={e => {
          setCardScale(Number(e.target.value));
        }}
      />
    </label>
  );
}

/** Hand view, grid style (grid only) and card size, as config fields. */
export function GinViewFields(): JSX.Element {
  const [handView, setHandView] = useHandView();
  return (
    <>
      <label className="ogs-config-field">
        <span>Hand view</span>
        <select
          value={handView}
          onChange={e => {
            if (isHandView(e.target.value)) setHandView(e.target.value);
          }}
        >
          <option value="grid">Grid</option>
          <option value="flat">Flat</option>
          <option value="arc">Arc</option>
        </select>
      </label>
      {handView === 'grid' && <GridStyleField />}
      <CardSizeField />
    </>
  );
}

/** View › Visuals: the shared table and card settings, then Gin's own. */
export function GinRummyVisualsSettings(): JSX.Element {
  return (
    <div className="gin-config-section">
      <h3>Visuals</h3>
      <CardStyleField />
      <GinTableCardFields />
      <GinViewFields />
    </div>
  );
}

/**
 * View › Gin Rummy: the opponent reveal (analysis only, which is the
 * oracle's view) and the discard and known card hints.
 */
export function GinRummyViewSettings(): JSX.Element {
  const isAnalysisMode = useIsAnalysisMode();
  const [showOpponentHand, setShowOpponentHand] = useViewSetting<boolean>(
    SHOW_OPPONENT_KEY,
    false,
  );
  const [showKnowledge, setShowKnowledge] = useViewSetting<boolean>(
    SHOW_KNOWLEDGE_KEY,
    true,
  );
  return (
    <div className="gin-config-section">
      <h3>Gin Rummy</h3>
      <label className="ogs-config-toggle">
        <input
          type="checkbox"
          checked={showOpponentHand}
          disabled={!isAnalysisMode}
          onChange={event => {
            setShowOpponentHand(event.target.checked);
          }}
        />
        <span>Show opponent hand</span>
      </label>
      {!isAnalysisMode && (
        <p className="ogs-config-note">
          Opponent hands can only be revealed while in analysis mode.
        </p>
      )}
      <label className="ogs-config-toggle">
        <input
          type="checkbox"
          checked={showKnowledge}
          onChange={event => {
            setShowKnowledge(event.target.checked);
          }}
        />
        <span>Show discard/known card hints</span>
      </label>
    </div>
  );
}
