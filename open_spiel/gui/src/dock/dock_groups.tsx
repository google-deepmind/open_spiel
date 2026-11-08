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

// The dock's panel groups: the members a dock adds when a section is popped
// out, the dock components that draw them, and the controller every group
// in the dock is given.

import type {DockviewApi, IDockviewPanelProps} from 'dockview-core';
import {
  type FunctionComponent,
  type JSX,
  type ReactNode,
  useCallback,
  useMemo,
  useState,
} from 'react';

import {readActiveSections, writeActiveSection} from './group_store.ts';
import {MemberPanel} from './panel_group.tsx';
import type {Member, PanelGroupController} from './panel_groups.ts';

/** What frames a panel's content: the page's panel frame, given its id. */
export type PanelFrame = FunctionComponent<{
  readonly id: string;
  readonly children: ReactNode;
}>;

/**
 * Adds `member` to the dock as a panel of its own: a tab of its group when
 * the group is open, else where dockview puts a new panel. Brings it
 * forward unless `inactive`; a member already in the dock is only brought
 * forward.
 */
export function addMember(
  api: DockviewApi,
  member: Member,
  inactive: boolean,
): void {
  const existing = api.getPanel(member.id);
  if (existing !== undefined) {
    if (!inactive) existing.api.setActive();
    return;
  }
  const home = api.getPanel(member.group.id) !== undefined;
  api.addPanel({
    id: member.id,
    component: member.id,
    title: member.section.title,
    ...(home && {
      position: {referencePanel: member.group.id, direction: 'within'},
    }),
    inactive: inactive && home,
  });
}

// The dock component that draws `member`, in `Frame` when there is one.
function memberComponent(
  member: Member,
  Frame: PanelFrame | undefined,
): FunctionComponent<IDockviewPanelProps> {
  const Component = ({api}: IDockviewPanelProps): JSX.Element => {
    const setTitle = useCallback(
      (title: string) => {
        api.setTitle(title);
      },
      [api],
    );
    const content = <MemberPanel member={member} setTitle={setTitle} />;
    return Frame === undefined ? (
      content
    ) : (
      <Frame id={member.id}>{content}</Frame>
    );
  };
  Component.displayName = `${member.id} member`;
  return Component;
}

/** A dock component for each member, keyed by its panel id. */
export function useMemberComponents(
  members: ReadonlyMap<string, Member>,
  Frame: PanelFrame | undefined,
): Readonly<Record<string, FunctionComponent<IDockviewPanelProps>>> {
  return useMemo(
    () =>
      Object.fromEntries(
        [...members.values()].map(member => [
          member.id,
          memberComponent(member, Frame),
        ]),
      ),
    [members, Frame],
  );
}

function browserStorage(): Storage {
  return window.localStorage;
}

/**
 * The controller of the dock's groups: which members are popped out (those
 * of `open` that are members), the open section of each group, kept for
 * `game`, and popping out and docking back.
 */
export function useGroupController(
  api: DockviewApi | null,
  members: ReadonlyMap<string, Member>,
  open: ReadonlySet<string>,
  game: string,
  storage: () => Storage = browserStorage,
): PanelGroupController {
  const [active, setActive] = useState(() => readActiveSections(storage, game));
  const remember = useCallback(
    (group: string, section: string) => {
      setActive(previous => ({...previous, [group]: section}));
      // A browser that keeps nothing still switches sections; it only
      // forgets them on reload.
      writeActiveSection(storage, game, group, section);
    },
    [storage, game],
  );
  return useMemo(
    () => ({
      popped: new Set([...open].filter(id => members.has(id))),
      active: group => active[group],
      setActive: remember,
      popOut: member => {
        if (api !== null) addMember(api, member, false);
      },
      dockBack: member => {
        if (api === null) return;
        api.getPanel(member.id)?.api.close();
        remember(member.group.id, member.section.id);
        api.getPanel(member.group.id)?.api.setActive();
      },
    }),
    [api, members, open, active, remember],
  );
}
