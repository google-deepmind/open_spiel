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

// A board's own History list: a renderer may draw the Actions › History
// section itself (chess pairs its moves a turn to a row). It is given the
// live game's actions, their labels, the position shown and how to show
// another.

import type {JSX} from 'react';

import type {Frame} from '../contract/contract.ts';
import {readableLabel} from './history_rows.ts';

/** What a board's History list is given. */
export interface HistoryRendererProps {
  /**
   * The frame shown, for a list that names the moves in the board's own
   * words (poker reads its table's history from the presentation).
   */
  readonly frame?: Frame;
  /** Each action of the live game; -1 for one withheld from the viewer. */
  readonly history: readonly number[];
  /** Each action's label as the viewer may read it; '' when withheld. */
  readonly labels: readonly string[];
  /** The labels to fall back on; the same as `labels` here. */
  readonly fallbackLabels: readonly string[];
  /** Each action's seat; -1 for chance. */
  readonly historyPlayers: readonly number[];
  /** The ply of the position shown: 0 is the initial position. */
  readonly currentStep: number;
  /** Whether a request is in flight, during which a press is ignored. */
  readonly isBusy: boolean;
  /** Whether the viewer may show any position (an omniscient viewer). */
  readonly isAnalysisMode: boolean;
  /** Shows the position after `step` plies. */
  readonly goToStep: (step: number, options?: {raw?: boolean}) => void;
  /** The label of the action at `index`, or its id when it has none. */
  readonly getActionLabel: (
    index: number,
    action: number,
    labels: readonly string[],
    fallbackLabels: readonly string[],
  ) => string;
}

/** A board's History list. */
export type HistoryRenderer = (props: HistoryRendererProps) => JSX.Element;

/** The label of an action, or its id when it has no label. */
export function historyActionLabel(
  index: number,
  action: number,
  labels: readonly string[],
  fallbackLabels: readonly string[],
): string {
  const label = labels[index] ?? fallbackLabels[index];
  if (label !== undefined && label.length > 0) return label;
  return Number.isFinite(action) && action >= 0 ? String(action) : '';
}

/** The props of a board's History list for `frame`. */
export function historyRendererProps(
  frame: Frame,
  busy: boolean,
  goTo: (ply: number) => void,
): HistoryRendererProps {
  const labels = frame.history.map(entry => readableLabel(entry) ?? '');
  return {
    frame,
    history: frame.history.map(entry => entry.action ?? -1),
    labels,
    fallbackLabels: labels,
    historyPlayers: frame.history.map(entry => entry.seat ?? -1),
    currentStep: frame.ply,
    isBusy: busy,
    isAnalysisMode: frame.omniscient,
    goToStep: step => {
      goTo(step);
    },
    getActionLabel: historyActionLabel,
  };
}
