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

import 'dockview/dist/styles/dockview.css';
import './dock.css';

import {DockviewReact} from 'dockview';
import type {
  DockviewApi,
  DockviewReadyEvent,
  IDockviewPanelProps,
} from 'dockview-core';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FunctionComponent,
  type JSX,
  type RefObject,
} from 'react';
import {createPortal} from 'react-dom';

import {arrange, tierFor, type Fit, type Size} from './arrangement.ts';
import {buildDock, openPanel} from './build.ts';
import {DockContext, type DockState} from './dock_context.ts';
import {
  addMember,
  type PanelFrame,
  useGroupController,
  useMemberComponents,
} from './dock_groups.tsx';
import {DOCK_THEME} from './dock_theme.ts';
import {useHeaderActions} from './header_actions.ts';
import {layoutPanels, type DockLayout} from './layout_store.ts';
import {PanelDrawer} from './panel_drawer.tsx';
import {
  type Member,
  type PanelGroupDefinition,
  PanelGroupContext,
  membersOf,
} from './panel_groups.ts';
import type {Panels} from './panels.ts';
import {groupsOutside, settled} from './resize.ts';
import {SavedLayoutsBar} from './saved_layouts.tsx';
import {openIds, useShownPanels} from './shown_panels.ts';
import {watchTabStrips} from './tab_strip.ts';
import {
  type DockEvent,
  type LayoutActions,
  useSavedLayouts,
} from './use_saved_layouts.ts';

const NO_GROUPS: readonly PanelGroupDefinition[] = [];

interface DockProps<Id extends string> {
  /** Must not change while the dock is mounted. */
  readonly panels: Panels<Id>;
  /** What each panel draws. */
  readonly components: Readonly<
    Record<Id, FunctionComponent<IDockviewPanelProps>>
  >;
  /**
   * The panel groups among the panels, whose sections may be popped out.
   * Must not change while the dock is mounted.
   */
  readonly groups?: readonly PanelGroupDefinition[];
  /** The game the dock shows: its layouts and open sections are its own. */
  readonly game: string;
  /** What frames a popped-out section, as the panels are framed. */
  readonly frame?: PanelFrame;
}

/**
 * The panel dock, laid out for the dock's width with the panels that open
 * at first. Its controls sit in the page's header: the layouts bar, and the
 * "Panel options" drawer that reopens closed panels. When a resize settles
 * at a width of another tier, or leaves a group off screen, the dock is
 * rebuilt at its new size with the panels open at the time; popped out
 * sections are rebuilt with it.
 */
export function Dock<Id extends string>({
  panels,
  components,
  groups = NO_GROUPS,
  game,
  frame,
}: DockProps<Id>): JSX.Element {
  const area = useRef<HTMLDivElement>(null);
  const fit = useRef<Fit>({tier: 'three', widthScale: 1, heightScale: 1});
  const [api, setApi] = useState<DockviewApi | null>(null);
  const {open, visible} = useShownPanels(api, panels);
  const members = useMemo(() => membersOf(groups), [groups]);
  const events = useDockEvents(api);
  const build = useCallback(
    (dock: DockviewApi, ids: ReadonlySet<string>) => {
      fit.current = buildDock(dock, panels, ids, sizeOf(area.current));
      for (const id of ids) {
        const member = members.get(id);
        if (member !== undefined) addMember(dock, member, true);
      }
      events.emit('build');
    },
    [panels, members, events],
  );
  const onReady = useCallback(
    (event: DockviewReadyEvent) => {
      build(event.api, panels.initial);
      setApi(event.api);
    },
    [build, panels],
  );
  useRebuildOnResize(api, area, fit, build);
  useWatchTabStrips(area);
  const reopen = useCallback(
    (id: string) => {
      if (api !== null) {
        fit.current = openPanel(
          api,
          panels,
          id,
          fit.current,
          sizeOf(area.current),
        );
      }
    },
    [api, panels],
  );
  const refit = useCallback((next: Fit) => {
    fit.current = next;
  }, []);
  const actions = useLayoutActions(api, {panels, members, events}, area, {
    refit,
    build,
  });
  return (
    <DockContext value={useDockState(panels, visible)}>
      <PanelGroupContext value={useGroupController(api, members, open, game)}>
        <div className="dock">
          <DockActions
            panels={panels}
            members={members}
            actions={actions}
            open={open}
            reopen={reopen}
            game={game}
          />
          <div className="dock-area" ref={area}>
            <DockviewReact
              components={useAllComponents(components, members, frame)}
              onReady={onReady}
              theme={DOCK_THEME}
            />
          </div>
        </div>
      </PanelGroupContext>
    </DockContext>
  );
}

/** Keeps the tab strips under `area` fitted to their groups. */
function useWatchTabStrips(area: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const element = area.current;
    return element === null ? undefined : watchTabStrips(element);
  }, [area]);
}

/**
 * What a panel may know about the dock: the first panel on screen, in
 * definition order, shows the session's error, so Gameplay whenever it is
 * on screen. Unlike a rule that keeps the error in Gameplay while it is
 * open, this never hides the error behind a tab.
 */
function useDockState(panels: Panels, visible: ReadonlySet<string>): DockState {
  return useMemo(
    () => ({
      sessionErrorPanel: panels.list.find(({id}) => visible.has(id))?.id,
    }),
    [visible, panels],
  );
}

/** The panels' components and the popped out sections'. */
function useAllComponents<Id extends string>(
  components: Readonly<Record<Id, FunctionComponent<IDockviewPanelProps>>>,
  members: ReadonlyMap<string, Member>,
  frame: PanelFrame | undefined,
): Readonly<Record<string, FunctionComponent<IDockviewPanelProps>>> {
  const memberComponents = useMemberComponents(members, frame);
  return useMemo(
    () => ({...memberComponents, ...components}),
    [memberComponents, components],
  );
}

/**
 * The dock's controls, in the page's header when it has one and else above
 * the dock: the layouts bar, once the dock is ready, and the drawer that
 * reopens closed panels.
 */
function DockActions<Id extends string>({
  panels,
  members,
  actions,
  open,
  reopen,
  game,
}: {
  readonly panels: Panels<Id>;
  readonly members: ReadonlyMap<string, Member>;
  readonly actions: LayoutActions | null;
  readonly open: ReadonlySet<string>;
  readonly reopen: (id: string) => void;
  readonly game: string;
}): JSX.Element {
  const ids = useMemo(
    () => new Set([...panels.list.map(({id}) => id), ...members.keys()]),
    [panels, members],
  );
  const layouts = useSavedLayouts({game, panels: ids, actions});
  const content = (
    <>
      {actions !== null && <SavedLayoutsBar layouts={layouts} />}
      <PanelDrawer
        closed={panels.list.filter(({id}) => !open.has(id))}
        onOpen={reopen}
      />
    </>
  );
  const slot = useHeaderActions();
  return slot === null ? (
    <div className="dock__actions">{content}</div>
  ) : (
    createPortal(content, slot)
  );
}

/** Listeners to what happens to the dock, and the function that tells them. */
interface DockEvents {
  readonly emit: (event: DockEvent) => void;
  readonly subscribe: (listener: (event: DockEvent) => void) => () => void;
}

/** The dock's events: each build, and every change dockview reports. */
function useDockEvents(api: DockviewApi | null): DockEvents {
  const [events] = useState<DockEvents>(() => {
    const listeners = new Set<(event: DockEvent) => void>();
    return {
      emit: event => {
        for (const listener of listeners) listener(event);
      },
      subscribe: listener => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    };
  });
  useEffect(() => {
    if (api === null) return undefined;
    const subscription = api.onDidLayoutChange(() => {
      events.emit('change');
    });
    return () => {
      subscription.dispose();
    };
  }, [api, events]);
  return events;
}

/**
 * Rebuilds the dock in `area` with the panels open at the time, once a resize
 * settles at a width of a tier other than `fit`'s or with a group outside it.
 */
function useRebuildOnResize(
  api: DockviewApi | null,
  area: RefObject<HTMLElement | null>,
  fit: RefObject<Fit>,
  build: (api: DockviewApi, open: ReadonlySet<string>) => void,
): void {
  useEffect(() => {
    const element = area.current;
    if (api === null || element === null) {
      return undefined;
    }
    const rebuild = settled(() => {
      const {width} = sizeOf(element);
      if (
        width > 0 &&
        (tierFor(width) !== fit.current.tier || groupsOutside(element))
      ) {
        build(api, openIds(api));
      }
    });
    const observer = new ResizeObserver(rebuild.schedule);
    observer.observe(element);
    return () => {
      observer.disconnect();
      rebuild.cancel();
    };
  }, [api, area, fit, build]);
}

/**
 * What saved layouts do to the dock; null until it is ready. A layout saved
 * at the dock's tier is restored as it was saved; one saved at another tier
 * is built afresh with the panels it has open, as a resize to this tier
 * would. Every panel takes its title from its definition.
 */
function useLayoutActions<Id extends string>(
  api: DockviewApi | null,
  {
    panels,
    members,
    events,
  }: {
    readonly panels: Panels<Id>;
    readonly members: ReadonlyMap<string, Member>;
    readonly events: DockEvents;
  },
  area: RefObject<HTMLElement | null>,
  {
    refit,
    build,
  }: {
    readonly refit: (fit: Fit) => void;
    readonly build: (api: DockviewApi, open: ReadonlySet<string>) => void;
  },
): LayoutActions | null {
  return useMemo(() => {
    if (api === null) return null;
    const titleOf = (id: string): string =>
      members.get(id)?.section.title ?? panels.get(id as Id).title;
    return {
      capture: (): DockLayout => ({
        tier: tierFor(sizeOf(area.current).width),
        dock: api.toJSON(),
      }),
      restore: (layout: DockLayout): void => {
        const size = sizeOf(area.current);
        const open = new Set(layoutPanels(layout));
        if (layout.tier !== tierFor(size.width)) {
          build(api, open);
          return;
        }
        api.fromJSON(layout.dock);
        for (const panel of api.panels) {
          panel.api.setTitle(titleOf(panel.id));
        }
        const {tier, widthScale, heightScale} = arrange(panels, open, size);
        refit({tier, widthScale, heightScale});
      },
      reset: (): void => {
        build(api, panels.initial);
      },
      subscribe: events.subscribe,
    };
  }, [api, panels, members, area, events, refit, build]);
}

function sizeOf(element: HTMLElement | null): Size {
  return {
    width: element?.clientWidth ?? 0,
    height: element?.clientHeight ?? 0,
  };
}
