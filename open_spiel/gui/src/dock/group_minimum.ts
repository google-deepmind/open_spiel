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

import type {DockviewGroupPanel, IDockviewPanelProps} from 'dockview-core';
import {useCallback, useEffect, useRef} from 'react';

/** Pixels the rest of a dock column keeps when a group asks to grow. */
export const COLUMN_RESERVE = 240;

/**
 * The minimum height, in pixels, of a group `height` pixels tall whose
 * content wants `overflow` more, kept between `floor` and what a column of
 * `columnHeight` pixels can give it after `COLUMN_RESERVE`; `floor` when
 * nothing is asked.
 */
export function groupMinimum(
  height: number,
  overflow: number | null,
  floor: number,
  columnHeight: number,
): number {
  if (overflow === null) return floor;
  const ceiling = Math.max(floor, columnHeight - COLUMN_RESERVE);
  return Math.min(Math.max(Math.ceil(height + overflow), floor), ceiling);
}

// A group made taller for a request: the height it had before, and the
// minimum that raised it.
interface Grant {
  readonly group: DockviewGroupPanel;
  readonly before: number;
  readonly minimum: number;
}

// Whether two heights are the same drawn height; the dock rounds them.
function same(a: number, b: number): boolean {
  return Math.abs(a - b) <= 1;
}

/**
 * Lets the content of the panel `props` draws ask its dock group to stay tall
 * enough to show it: the returned callback takes how many pixels more the
 * content wants, or null to ask for nothing. The request holds while the
 * panel is its group's visible tab, follows the panel to another group, and
 * otherwise leaves the group at `floor`, the panel's own minimum height.
 *
 * The dock enforces a raised minimum only at its next layout, so a request
 * resizes the group at once, taking the room from the rest of its column.
 * When the request goes, the group returns to the height it had before,
 * unless it has been resized since: a person's sizes win over the undo.
 */
export function useGroupMinimumHeight(
  props: IDockviewPanelProps,
  floor: number,
): (overflow: number | null) => void {
  const {api, containerApi} = props;
  const overflow = useRef<number | null>(null);
  const group = useRef<DockviewGroupPanel | null>(null);
  const grant = useRef<Grant | null>(null);
  const apply = useCallback(() => {
    const current = api.group;
    if (group.current !== null && group.current !== current) {
      group.current.api.setConstraints({minimumHeight: floor});
      grant.current = null;
    }
    group.current = current;
    const asked = api.isVisible ? overflow.current : null;
    const height = current.api.height;
    const minimum = groupMinimum(height, asked, floor, containerApi.height);
    current.api.setConstraints({minimumHeight: minimum});
    const granted = grant.current;
    if (asked !== null && minimum > height + 1) {
      grant.current = {
        group: current,
        before: granted?.before ?? height,
        minimum,
      };
      current.api.setSize({height: minimum});
    } else if (asked === null && granted !== null) {
      grant.current = null;
      if (granted.group === current && same(height, granted.minimum)) {
        current.api.setSize({height: granted.before});
      }
    }
  }, [api, containerApi, floor]);
  useEffect(() => {
    const visibility = api.onDidVisibilityChange(apply);
    const moved = api.onDidGroupChange(apply);
    return () => {
      visibility.dispose();
      moved.dispose();
      group.current?.api.setConstraints({minimumHeight: floor});
    };
  }, [api, apply, floor]);
  return useCallback(
    (next: number | null) => {
      overflow.current = next;
      apply();
    },
    [apply],
  );
}
