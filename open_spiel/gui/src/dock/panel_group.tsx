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

import {
  type JSX,
  type KeyboardEvent,
  type RefObject,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import {DockIcon, PopOutIcon} from '../controls/icons.tsx';
import {
  type Member,
  type PanelGroupDefinition,
  type SectionDefinition,
  PaneHiddenContext,
  memberId,
  usePanelGroups,
} from './panel_groups.ts';

import '../controls/controls.css';
import './panel_group.css';

// The key that moves the selection to the next tab, or back to the previous.
const STEPS: Readonly<Record<string, number>> = {
  ArrowRight: 1,
  ArrowDown: 1,
  ArrowLeft: -1,
  ArrowUp: -1,
};

/** Whether `pane` shows nothing: no element and no text. */
export function isPaneEmpty(pane: Element): boolean {
  return pane.children.length === 0 && pane.textContent.trim() === '';
}

/**
 * Calls `onChange` with whether the element `ref` holds is empty, now and
 * whenever what it holds changes.
 */
function usePaneEmpty(
  ref: RefObject<HTMLElement | null>,
  onChange: (empty: boolean) => void,
): void {
  const last = useRef<boolean | null>(null);
  const check = useCallback(() => {
    const pane = ref.current;
    if (pane === null) return;
    const empty = isPaneEmpty(pane);
    if (empty !== last.current) {
      last.current = empty;
      onChange(empty);
    }
  }, [ref, onChange]);
  useLayoutEffect(check);
  useEffect(() => {
    const pane = ref.current;
    if (pane === null) return undefined;
    const observer = new MutationObserver(check);
    observer.observe(pane, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => {
      observer.disconnect();
    };
  }, [ref, check]);
}

function TabLabel({
  section,
}: {
  readonly section: SectionDefinition;
}): JSX.Element {
  const {Label} = section;
  return Label === undefined ? <>{section.title}</> : <Label />;
}

interface PaneProps {
  readonly section: SectionDefinition;
  readonly id: string;
  readonly tabId: string;
  readonly hidden: boolean;
  readonly onEmpty: (sectionId: string, empty: boolean) => void;
}

function Pane({section, id, tabId, hidden, onEmpty}: PaneProps): JSX.Element {
  const pane = useRef<HTMLDivElement>(null);
  const report = useCallback(
    (empty: boolean) => {
      onEmpty(section.id, empty);
    },
    [onEmpty, section.id],
  );
  usePaneEmpty(pane, report);
  const {Content} = section;
  return (
    <div
      ref={pane}
      id={id}
      className="panel-group__pane"
      role="tabpanel"
      aria-labelledby={tabId}
      hidden={hidden}
    >
      <PaneHiddenContext value={hidden}>
        <Content />
      </PaneHiddenContext>
    </div>
  );
}

interface TabStripProps {
  readonly group: PanelGroupDefinition;
  readonly shown: readonly SectionDefinition[];
  readonly active: SectionDefinition;
  readonly select: (sectionId: string) => void;
  readonly tabId: (section: SectionDefinition) => string;
  readonly paneId: (section: SectionDefinition) => string;
}

/**
 * The group's tabs, following the ARIA tabs pattern: one tab stop, the
 * arrow keys moving between them and wrapping.
 */
function TabStrip({
  group,
  shown,
  active,
  select,
  tabId,
  paneId,
}: TabStripProps): JSX.Element {
  const tabs = useRef<Record<string, HTMLButtonElement | null>>({});
  const onKeyDown = (event: KeyboardEvent): void => {
    const step = STEPS[event.key];
    if (step === undefined) return;
    event.preventDefault();
    const count = shown.length;
    const next = shown[(shown.indexOf(active) + step + count) % count];
    if (next === undefined) throw new Error('No section to move to.');
    select(next.id);
    tabs.current[next.id]?.focus();
  };
  return (
    <div
      className="tabs panel-group__tabs"
      role="tablist"
      aria-label={`${group.title} sections`}
      onKeyDown={onKeyDown}
    >
      {shown.map(section => (
        <button
          key={section.id}
          ref={element => {
            tabs.current[section.id] = element;
          }}
          id={tabId(section)}
          type="button"
          role="tab"
          className="button tabs__tab"
          aria-selected={section === active}
          aria-controls={paneId(section)}
          tabIndex={section === active ? 0 : -1}
          onClick={() => {
            select(section.id);
          }}
        >
          <TabLabel section={section} />
        </button>
      ))}
    </div>
  );
}

/** Why the group shows no section, if it shows none. */
function GroupNote({
  group,
  docked,
  shown,
}: {
  readonly group: PanelGroupDefinition;
  readonly docked: number;
  readonly shown: number;
}): JSX.Element | null {
  if (docked === 0) {
    return (
      <p className="panel-group__note">
        {`Every ${group.title} section is open as its own panel.`}
      </p>
    );
  }
  if (shown > 0) return null;
  return (
    <p className="panel-group__note">
      {`Nothing to show in ${group.title} in this session.`}
    </p>
  );
}

/** The properties of a PanelGroup. */
export interface PanelGroupProps {
  readonly group: PanelGroupDefinition;
  /**
   * Outside a dock, the id of the section chosen, for a caller that keeps
   * the choice while the group is not drawn; the group keeps it itself when
   * undefined. Inside a dock, the dock remembers it for the game.
   */
  readonly chosen?: string | undefined;
  /** Outside a dock, called with the id of each section chosen. */
  readonly onChoose?: ((sectionId: string) => void) | undefined;
}

/**
 * A panel group: its sections as a strip of tabs, one shown at a time, with
 * a button that pops the open one out into a panel of its own. Every section
 * in the group stays mounted, the hidden ones hidden, so each keeps its
 * scroll and its drafts. A section that draws nothing has no tab. Inside a
 * dock the open tab is remembered for the game, and popping out is offered;
 * outside one the group works alone.
 */
export function PanelGroup({
  group,
  chosen: kept,
  onChoose,
}: PanelGroupProps): JSX.Element {
  const controller = usePanelGroups();
  const id = useId();
  const [empty, setEmpty] = useState<Readonly<Record<string, boolean>>>({});
  const [own, setOwn] = useState<string | undefined>(undefined);
  const chosen = kept ?? own;
  const setChosen = (sectionId: string): void => {
    setOwn(sectionId);
    onChoose?.(sectionId);
  };
  const docked = group.sections.filter(
    section => controller?.popped.has(memberId(group.id, section.id)) !== true,
  );
  const shown = docked.filter(section => empty[section.id] !== true);
  const wanted = controller === null ? chosen : controller.active(group.id);
  // A remembered section that is empty for now is not forgotten: it opens
  // again once it has content.
  const active = shown.find(section => section.id === wanted) ?? shown[0];
  const select = (sectionId: string): void => {
    if (controller === null) setChosen(sectionId);
    else controller.setActive(group.id, sectionId);
  };
  const onEmpty = useCallback((sectionId: string, isEmpty: boolean) => {
    setEmpty(previous =>
      previous[sectionId] === isEmpty
        ? previous
        : {...previous, [sectionId]: isEmpty},
    );
  }, []);
  const tabId = (section: SectionDefinition): string =>
    `${id}-${section.id}-tab`;
  const paneId = (section: SectionDefinition): string => `${id}-${section.id}`;
  return (
    <div className="panel-group" data-panel-group={group.id}>
      <h2 className="visually-hidden">{group.title}</h2>
      <div className="panel-group__bar">
        {active !== undefined && (
          <TabStrip
            group={group}
            shown={shown}
            active={active}
            select={select}
            tabId={tabId}
            paneId={paneId}
          />
        )}
        {active !== undefined && controller !== null && (
          <PopOut group={group} section={active} />
        )}
      </div>
      <GroupNote group={group} docked={docked.length} shown={shown.length} />
      {docked.map(section => (
        <Pane
          key={section.id}
          section={section}
          id={paneId(section)}
          tabId={tabId(section)}
          hidden={section !== active}
          onEmpty={onEmpty}
        />
      ))}
      {active !== undefined && <SectionFooter section={active} />}
    </div>
  );
}

// The footer of `section`, if it has one, under its pane.
function SectionFooter({
  section,
}: {
  readonly section: SectionDefinition;
}): JSX.Element | null {
  const {Footer} = section;
  if (Footer === undefined) return null;
  return (
    <div className="panel-group__footer">
      <Footer />
    </div>
  );
}

// The button that pops `section` out of `group` into a panel of its own.
function PopOut({
  group,
  section,
}: {
  readonly group: PanelGroupDefinition;
  readonly section: SectionDefinition;
}): JSX.Element | null {
  const controller = usePanelGroups();
  if (controller === null) return null;
  return (
    <button
      type="button"
      className="button button--icon panel-group__action"
      aria-label={`Pop out ${section.title}`}
      title={`Pop out ${section.title} into its own panel`}
      onClick={() => {
        controller.popOut({
          group,
          section,
          id: memberId(group.id, section.id),
        });
      }}
    >
      <PopOutIcon />
    </button>
  );
}

/** The properties of a MemberPanel. */
export interface MemberPanelProps {
  readonly member: Member;
  /** Gives the dock tab its title as the section's label changes. */
  readonly setTitle?: (title: string) => void;
}

/**
 * A section popped out of its group into a panel of its own: the group it
 * came from, a button that docks it back, then the section.
 */
export function MemberPanel({member, setTitle}: MemberPanelProps): JSX.Element {
  const controller = usePanelGroups();
  const {group, section} = member;
  const pane = useRef<HTMLDivElement>(null);
  const [empty, setEmpty] = useState(false);
  usePaneEmpty(pane, setEmpty);
  const {Content, Label} = section;
  return (
    <div className="panel-group panel-group--member">
      <h2 className="visually-hidden">{section.title}</h2>
      {Label !== undefined && setTitle !== undefined && (
        <LiveTitle Label={Label} setTitle={setTitle} />
      )}
      <div className="panel-group__bar">
        <span className="panel-group__from">{`From ${group.title}`}</span>
        {controller !== null && (
          <button
            type="button"
            className="button panel-group__dock"
            aria-label={`Dock ${section.title} into ${group.title}`}
            title={`Dock ${section.title} back into ${group.title}`}
            onClick={() => {
              controller.dockBack(member);
            }}
          >
            <DockIcon />
            <span>{`Dock into ${group.title}`}</span>
          </button>
        )}
      </div>
      {empty && (
        <p className="panel-group__note">
          {`Nothing to show in ${section.title} in this session.`}
        </p>
      )}
      <div ref={pane} className="panel-group__pane">
        <Content />
      </div>
      <SectionFooter section={section} />
    </div>
  );
}

// Keeps the dock tab's title the section's live label.
function LiveTitle({
  Label,
  setTitle,
}: {
  readonly Label: () => string;
  readonly setTitle: (title: string) => void;
}): null {
  const title = Label();
  useEffect(() => {
    setTitle(title);
  }, [setTitle, title]);
  return null;
}
