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

// Panel groups: one dock panel that holds named sections as inner tabs, one
// shown at a time. Any section can be popped out into a dock panel of its
// own, a member of its group, and docked back. This file is the definitions
// and the controller the dock gives every group; panel_group.tsx draws them.

import {createContext, useContext, type JSX} from 'react';

/** One section of a group. */
export interface SectionDefinition {
  /** Unique within its group. */
  readonly id: string;
  /** The tab's label, and the dock tab's title when popped out. */
  readonly title: string;
  /**
   * What the section draws. A section that draws nothing has no tab, so a
   * group never shows an empty section.
   */
  readonly Content: () => JSX.Element | null;
  /** The tab's label as it changes, "Legal (12)"; `title` when omitted. */
  readonly Label?: () => string;
  /**
   * Drawn under the section while it is shown, outside its scroller, such
   * as History's replay control.
   */
  readonly Footer?: () => JSX.Element | null;
}

/** A group: the dock panel `id`, holding `sections` in this order. */
export interface PanelGroupDefinition {
  readonly id: string;
  readonly title: string;
  readonly sections: readonly SectionDefinition[];
}

/** A section with the group it belongs to. */
export interface Member {
  readonly group: PanelGroupDefinition;
  readonly section: SectionDefinition;
  /** The id of its dock panel when it is popped out. */
  readonly id: string;
}

/** The dock panel id of `section` of group `groupId` when popped out. */
export function memberId(groupId: string, sectionId: string): string {
  return `${groupId}.${sectionId}`;
}

/** Every section of `groups`, by the id of its dock panel. */
export function membersOf(
  groups: readonly PanelGroupDefinition[],
): ReadonlyMap<string, Member> {
  const members = new Map<string, Member>();
  for (const group of groups) {
    if (group.sections.length === 0) {
      throw new Error(`Panel group "${group.id}" has no sections.`);
    }
    for (const section of group.sections) {
      const id = memberId(group.id, section.id);
      if (members.has(id)) {
        throw new Error(`Section "${id}" is defined twice.`);
      }
      members.set(id, {group, section, id});
    }
  }
  return members;
}

/** What the dock does for its groups. */
export interface PanelGroupController {
  /** The ids of the members popped out into the dock. */
  readonly popped: ReadonlySet<string>;
  /** The section remembered as open in group `groupId`, if any. */
  readonly active: (groupId: string) => string | undefined;
  readonly setActive: (groupId: string, sectionId: string) => void;
  /** Opens `member` as a panel of its own, beside its group. */
  readonly popOut: (member: Member) => void;
  /** Closes `member`'s panel and opens it in its group. */
  readonly dockBack: (member: Member) => void;
}

/** The controller of the dock a group is in; provided by `Dock`. */
export const PanelGroupContext = createContext<PanelGroupController | null>(
  null,
);

/** The controller of the dock, or null outside one. */
export function usePanelGroups(): PanelGroupController | null {
  return useContext(PanelGroupContext);
}

/** Whether the section rendered here is a hidden tab of its group. */
export const PaneHiddenContext = createContext(false);

/**
 * True while the section is a hidden tab of its group, so that a section
 * whose content costs work while unseen can skip it. False outside a group
 * and when popped out.
 */
export function usePaneHidden(): boolean {
  return useContext(PaneHiddenContext);
}
